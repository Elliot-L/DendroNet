import os
import pandas as pd
import argparse
from tqdm import tqdm
import numpy as np
import json
from sklearn.metrics import roc_curve, auc

import torch
from torch.utils.data.dataloader import DataLoader
import torch.optim
import torch.nn as nn

# Local imports
from models.CT_conv_model import DendronetModule, SeqConvModule, FCModule
from utils.model_utils import split_indices, IndicesDataset, build_parent_path_mat
from Create_Tree_image import create_pc_mat

def get_one_hot_encoding(seq):
    # (A,G,T,C), ex: A = (1, 0, 0, 0), T = (0, 0, 1, 0)
    encoding = []
    for c in seq:
        if c == 'A':
            encoding.append([1, 0, 0, 0])
        elif c == 'G':
            encoding.append([0, 1, 0, 0])
        elif c == 'T':
            encoding.append([0, 0, 1, 0])
        elif c == 'C':
            encoding.append([0, 0, 0, 1])
    return encoding

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--feature', type=str, default='active')
    parser.add_argument('--LR', type=float, default=0.001)
    parser.add_argument('--DPF', type=float, default=0.01)
    parser.add_argument('--L1', type=float, default=0.01)
    parser.add_argument('--USE-CUDA', type=bool, default=False)
    parser.add_argument('--BATCH-SIZE', type=int, default=128)
    parser.add_argument('--whole-dataset', type=bool, default=False)
    parser.add_argument('--embedding-size', type=int, default=10)
    parser.add_argument('--cells_to_use', type=int, defautl=28)
    args = parser.parse_args()

    feature = args.feature
    LR = args.LR
    DPF = args.DPF
    L1 = args.L1
    BATCH_SIZE = args.BATCH_SIZE
    USE_CUDA = args.USE_CUDA
    whole_dataset = args.whole_dataset
    embedding_size = args.embedding_size

    device = torch.device("cuda:0" if (torch.cuda.is_available() and USE_CUDA) else "cpu")

    pc_mat, nodes = create_pc_mat()

    print(pc_mat)
    print(nodes)

    with open(os.path.join('data_files', 'enhancers_seqs.json'), 'r') as enhancers_file:
        enhancers_dict = json.load(enhancers_file)

    cell_names = []
    X = []
    y = []
    enhancers_list = list(enhancers_dict.keys())
    num_enhancers = len(enhancers_list)

    print(enhancers_list[0])
    print(len(enhancers_list))
    num_cells_used = 0
    for name in nodes:
        if name != 'internal node' and num_cells_used < args.cells_to_use:
            cell_names.append(name)
            num_cells_used += 1

    print(cell_names)

    for ct in cell_names:
        ct_df = pd.read_csv(os.path.join('data_files', 'CT_enhancer_features_matrices',
                                         ct + '_enhancer_features_matrix.csv'), index_col='cCRE_id')
        ct_df = ct_df.loc[enhancers_list]
        y.extend(list(ct_df.loc[:, feature]))

    print(len(y))
    print(y[0])
    print(len(enhancers_list))

    for enhancer in enhancers_list:
        X.append(get_one_hot_encoding(enhancers_dict[enhancer]))

    print(len(X))
    print(len(X[0]))

    # the list "samples" is a list of tuples each representing a sample. The first
    # entry is the row of the X matrix where the sequence and the label associated
    # to that samples can be found and the second entry represent the cell type of the samples,
    # or more precisely the index of the associated cell type in the cell_names list.
    # From these two information, the target value associated with a given sample, found in the
    # y vector can be extracted by multiplying the cell_type index by the number enhancers
    # and then adding the number of the row in X. This is similar to how you would access entries of
    # a 2D matrix when it is stored in a 1D format, in memory for example.

    samples = []

    for ct_idx in range(len(cell_names)):
        for enhancer_idx in range(len(enhancers_list)):
            samples.append((enhancer_idx, ct_idx))

    print(len(samples))

    parent_path_mat = build_parent_path_mat(pc_mat)
    num_edges = len(parent_path_mat)
    delta_mat = np.zeros(shape=(embedding_size, num_edges))
    root_vector = np.zeros(shape=embedding_size)

    # The three subparts of the model:
    dendronet = DendronetModule(device=device, root_weights=root_vector, delta_mat=delta_mat, path_mat=parent_path_mat)

    convolution = SeqConvModule(device=device, seq_lenght=501, kernel_sizes=(16, 3, 3), num_of_kernels=(64, 64, 32),
                                polling_windows=(3, 4), input_channels=4)

    fully_connected = FCModule(device=device, layer_sizes=(embedding_size + 32, 32, 1))

    train_idx, test_idx = split_indices(samples, seed=0)

    train_set = IndicesDataset(train_idx)
    test_set = IndicesDataset(test_idx)

    params = {'batch_size': BATCH_SIZE,
              'shuffle': True,
              'num_workers': 0}

    train_batch_gen = DataLoader(train_set, **params)
    test_batch_gen = DataLoader(test_set, **params)

    X = torch.tensor(X, dtype=torch.float, device=device)
    X = X.permute(0, 2, 1)
    y = torch.tensor(y, dtype=torch.float, device=device)

    loss_function = nn.BCELoss()
    if torch.cuda.is_available() and USE_CUDA:
        loss_function = loss_function.cuda()
    optimizer = torch.optim.Adam(list(dendronet.parameters()) + list(convolution.parameters())
                                 + list(fully_connected.parameters()), lr=LR)

    for epoch in range(10):
        print("Epoch " + str(epoch))
        for step, idx_batch in enumerate(tqdm(train_batch_gen)):
            optimizer.zero_grad()
            # print(y[idx_batch])
            X_idx = idx_batch[0]
            cell_idx = idx_batch[1]
            seq_features = convolution(X[X_idx])
            cell_embeddings = dendronet(cell_idx)
            y_hat = fully_connected(torch.cat((seq_features, cell_embeddings), 1))
            # print(y_hat)
            # error_loss = loss_function(y_hat, y[idx_batch])
            y_idx = []
            for x, c in zip(X_idx, cell_idx):
                y_idx.append(c*num_enhancers + x)
            y_idx = torch.tensor(y_idx, dtype=torch.long, device=device)
            error_loss = loss_function(y_hat, y[y_idx])
            delta_loss = dendronet.delta_loss(cell_idx)
            root_loss = 0.0
            for w in dendronet.root_weights:
                root_loss += abs(float(w))
            train_loss = error_loss + DPF*delta_loss + L1*root_loss
            # print("error loss on batch: " + str(float(error_loss)))
            train_loss.backward(retain_graph=True)
            optimizer.step()
            # print(CT_specific_conv.convLayer.weight)
            # print(torch.max(CT_specific_conv.convLayer.weight.grad))
        with torch.no_grad():
            print("Test performance on train set for epoch " + str(epoch))
            all_train_targets = []
            all_train_predictions = []
            train_error_loss = 0.0
            for step, idx_batch in enumerate(tqdm(train_batch_gen)):
                X_idx = idx_batch[0]
                cell_idx = idx_batch[1]
                seq_features = convolution(X[X_idx])
                cell_embeddings = dendronet(cell_idx)
                y_hat = fully_connected(torch.cat((seq_features, cell_embeddings), 1))
                y_idx = []
                for x, c in zip(X_idx, cell_idx):
                    y_idx.append(c * num_enhancers + x)
                y_idx = torch.tensor(y_idx, dtype=torch.long, device=device)
                train_error_loss += float(loss_function(y_hat, y[y_idx]))
                all_train_targets.extend(list(y[y_idx].detach().cpu().numpy()))
                all_train_predictions.extend(list(y_hat.detach().cpu().numpy()))

            print("average error loss on train set : " + str(float(train_error_loss) / (step + 1)))
            fpr, tpr, _ = roc_curve(all_train_targets, all_train_predictions)
            roc_auc = auc(fpr, tpr)
            print('ROC AUC on train set : ' + str(roc_auc))

            print("Test performance on test set for epoch " + str(epoch))
            all_test_targets = []
            all_test_predictions = []
            test_error_loss = 0.0
            for step, idx_batch in enumerate(tqdm(test_batch_gen)):
                X_idx = idx_batch[0]
                cell_idx = idx_batch[1]
                seq_features = convolution(X[X_idx])
                cell_embeddings = dendronet(cell_idx)
                y_hat = fully_connected(torch.cat((seq_features, cell_embeddings), 1))
                y_idx = []
                for x, c in zip(X_idx, cell_idx):
                    y_idx.append(c * num_enhancers + x)
                y_idx = torch.tensor(y_idx, dtype=torch.long, device=device)
                test_error_loss += float(loss_function(y_hat, y[y_idx]))
                all_test_targets.extend(list(y[y_idx].detach().cpu().numpy()))
                all_test_predictions.extend(list(y_hat.detach().cpu().numpy()))

            print("average error loss on test set : " + str(float(test_error_loss) / (step + 1)))
            fpr, tpr, _ = roc_curve(all_test_targets, all_test_predictions)
            roc_auc = auc(fpr, tpr)
            print('ROC AUC on test set : ' + str(roc_auc))

