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
    parser.add_argument('--seeds', type=int, nargs='+', default=[1])
    parser.add_argument('--early-stopping', type=int, default=3)
    parser.add_argument('--num-epoches', type=int, default=100)

    args = parser.parse_args()

    feature = args.feature
    LR = args.LR
    DPF = args.DPF
    L1 = args.L1
    BATCH_SIZE = args.BATCH_SIZE
    USE_CUDA = args.USE_CUDA
    whole_dataset = args.whole_dataset
    embedding_size = args.embedding_size
    seeds = args.seeds
    early_stop = args.early_stopping
    epoches = args.num_epoches

    device = torch.device("cuda:0" if (torch.cuda.is_available() and USE_CUDA) else "cpu")

    pc_mat, nodes = create_pc_mat()
    print(pc_mat.shape)

    print(pc_mat)
    print(nodes)

    with open(os.path.join('data_files', 'enhancers_seqs.json'), 'r') as enhancers_file:
        enhancers_dict = json.load(enhancers_file)

    cell_names = []
    num_internal_nodes = 0
    X = []
    y = []

    enhancers_list = list(enhancers_dict.keys())

    for name in nodes:
        if name == 'internal node':
            num_internal_nodes += 1
        else:
            cell_names.append(name)

    print(cell_names)

    parent_path_mat = build_parent_path_mat(pc_mat)
    num_edges = len(parent_path_mat)
    delta_mat = np.zeros(shape=(embedding_size, num_edges))
    root_vector = np.zeros(shape=embedding_size)

    if whole_dataset:  # if we want to use all the samples, usually leads to heavily unbalanced dataset
        print('Using whole dataset')
        for enhancer in enhancers_list:
            X.append(get_one_hot_encoding(enhancers_dict[enhancer]))

        # the list "samples" is a list of tuples each representing a sample. The first
        # entry is the row of the X matrix. The second is the index of the cell type in the
        # parent path matrix and the third is the index of the target in the y vector.

        samples = []

        for ct in cell_names:
            ct_df = pd.read_csv(os.path.join('data_files', 'CT_enhancer_features_matrices',
                                             ct + '_enhancer_features_matrix.csv'), index_col='cCRE_id')
            ct_df = ct_df.loc[enhancers_list]
            y.extend(list(ct_df.loc[:, feature]))

        for ct_idx in range(len(cell_names)):
            for enhancer_idx in range(len(enhancers_list)):
                samples.append((enhancer_idx, ct_idx + num_internal_nodes,
                                ct_idx * len(enhancers_list) + enhancer_idx))

    else:  # In this case, we make sure that for each tissue type, the number of positive and negative examples
           # is the same, which gives us a balanced dataset
        print('Using a balanced dataset')

        for enhancer in enhancers_list:
            X.append(get_one_hot_encoding(enhancers_dict[enhancer]))

        pos_count = {}
        neg_counter = {}

        for ct in cell_names:
            pos_count[ct] = 0
            neg_counter[ct] = 0

        samples = []

        # the list "samples" is a list of tuples each representing a sample. The first
        # entry is the row of the X matrix. The second is the index of the cell type in the
        # parent path matrix and the third is the index of the target in the y vector.

        for i, ct in enumerate(cell_names):
            ct_df = pd.read_csv(os.path.join('data_files', 'CT_enhancer_features_matrices',
                                             ct + '_enhancer_features_matrix.csv'), index_col='cCRE_id')
            ct_df = ct_df.loc[enhancers_list]

            for enhancer in enhancers_list:
                if ct_df.loc[enhancer, feature] == 1:
                    pos_count[ct] += 1

            for j, enhancer in enumerate(enhancers_list):
                if ct_df.loc[enhancer, feature] == 1:
                    samples.append((j, i, len(y)))
                    y.append(1)

                if ct_df.loc[enhancer, feature] == 0 and neg_counter[ct] < pos_count[ct]:
                    samples.append((j, i, len(y)))
                    y.append(0)
                    neg_counter[ct] += 1
        print(pos_count)
        print(neg_counter)

    print(len(X))
    print(len(y))

    output = {'train_auc': [], 'val_auc': [], 'test_auc': []}

    for seed in seeds:
        # The three subparts of the model:

        dendronet = DendronetModule(device=device, root_weights=root_vector, delta_mat=delta_mat, path_mat=parent_path_mat)

        convolution = SeqConvModule(device=device, seq_lenght=501, kernel_sizes=(16, 3, 3), num_of_kernels=(64, 64, 32),
                                    polling_windows=(3, 4), input_channels=4)

        fully_connected = FCModule(device=device, layer_sizes=(embedding_size + 32, 32, 1))

        train_idx, test_idx = split_indices(samples, seed=0)
        train_idx, val_idx = split_indices(train_idx, seed=seed)

        train_set = IndicesDataset(train_idx)
        test_set = IndicesDataset(test_idx)
        val_set = IndicesDataset(test_idx)

        params = {'batch_size': BATCH_SIZE,
                  'shuffle': True,
                  'num_workers': 0}

        train_batch_gen = DataLoader(train_set, **params)
        test_batch_gen = DataLoader(test_set, **params)
        val_batch_gen = DataLoader(val_set, **params)

        X = torch.tensor(X, dtype=torch.float, device=device)
        X = X.permute(0, 2, 1)
        y = torch.tensor(y, dtype=torch.float, device=device)

        loss_function = nn.BCELoss()
        if torch.cuda.is_available() and USE_CUDA:
            loss_function = loss_function.cuda()
        optimizer = torch.optim.Adam(list(dendronet.parameters()) + list(convolution.parameters())
                                     + list(fully_connected.parameters()), lr=LR)

        best_val_auc = 0
        early_stop_count = 0

        for epoch in range(epoches):
            print("Epoch " + str(epoch))
            for step, idx_batch in enumerate(tqdm(train_batch_gen)):
                optimizer.zero_grad()
                # print(y[idx_batch])
                X_idx = idx_batch[0]
                cell_idx = idx_batch[1]
                y_idx = idx_batch[2]
                seq_features = convolution(X[X_idx])
                cell_embeddings = dendronet(cell_idx)
                y_hat = fully_connected(torch.cat((seq_features, cell_embeddings), 1))
                # print(y_hat)
                # error_loss = loss_function(y_hat, y[idx_batch])
                error_loss = loss_function(y_hat, y[y_idx])
                delta_loss = dendronet.delta_loss(cell_idx)
                root_loss = dendronet.root_loss()
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
                    y_idx = idx_batch[2]
                    seq_features = convolution(X[X_idx])
                    cell_embeddings = dendronet(cell_idx)
                    y_hat = fully_connected(torch.cat((seq_features, cell_embeddings), 1))
                    train_error_loss += float(loss_function(y_hat, y[y_idx]))
                    all_train_targets.extend(list(y[y_idx].detach().cpu().numpy()))
                    all_train_predictions.extend(list(y_hat.detach().cpu().numpy()))

                print("average error loss on train set : " + str(float(train_error_loss) / (step + 1)))
                fpr, tpr, _ = roc_curve(all_train_targets, all_train_predictions)
                train_roc_auc = auc(fpr, tpr)
                print('ROC AUC on train set : ' + str(train_roc_auc))

                print("Test performance on validation set for epoch " + str(epoch))
                all_val_targets = []
                all_val_predictions = []
                val_error_loss = 0.0
                for step, idx_batch in enumerate(tqdm(val_batch_gen)):
                    X_idx = idx_batch[0]
                    cell_idx = idx_batch[1]
                    y_idx = idx_batch[2]
                    seq_features = convolution(X[X_idx])
                    cell_embeddings = dendronet(cell_idx)
                    y_hat = fully_connected(torch.cat((seq_features, cell_embeddings), 1))
                    val_error_loss += float(loss_function(y_hat, y[y_idx]))
                    all_val_targets.extend(list(y[y_idx].detach().cpu().numpy()))
                    all_val_predictions.extend(list(y_hat.detach().cpu().numpy()))

                print("average error loss on validation set : " + str(float(val_error_loss) / (step + 1)))
                fpr, tpr, _ = roc_curve(all_val_targets, all_val_predictions)
                val_roc_auc = auc(fpr, tpr)
                print('ROC AUC on validation set : ' + str(val_roc_auc))

                if val_roc_auc > best_val_auc:
                    best_val_auc = val_roc_auc
                    print('New best AUC on validation set: ' + str(best_val_auc))
                    early_stop_count = 0
                else:
                    early_stop_count += 1
                    print('The performance hasn\'t improved for ' + str(early_stop_count) + ' epoches')
                    print(' Best is :' + str(best_val_auc))

                if early_stop_count == early_stop:
                    print('Early Stopping!')
                    break

        with torch.no_grad():
            print("Test performance on test set for epoch " + str(epoch))
            all_test_targets = []
            all_test_predictions = []
            test_error_loss = 0.0
            for step, idx_batch in enumerate(tqdm(test_batch_gen)):
                X_idx = idx_batch[0]
                cell_idx = idx_batch[1]
                y_idx = idx_batch[2]
                seq_features = convolution(X[X_idx])
                cell_embeddings = dendronet(cell_idx)
                y_hat = fully_connected(torch.cat((seq_features, cell_embeddings), 1))
                test_error_loss += float(loss_function(y_hat, y[y_idx]))
                all_test_targets.extend(list(y[y_idx].detach().cpu().numpy()))
                all_test_predictions.extend(list(y_hat.detach().cpu().numpy()))

            print("average error loss on test set : " + str(float(test_error_loss) / (step + 1)))
            fpr, tpr, _ = roc_curve(all_test_targets, all_test_predictions)
            test_roc_auc = auc(fpr, tpr)
            print('ROC AUC on test set : ' + str(test_roc_auc))

        output['train'].append(train_roc_auc)
        output['val'].append(val_roc_auc)
        output['test'].append(test_roc_auc)

        print('All tissus encodings:')
        for i, tissue in enumerate(cell_names):
            print(tissue)
            print(dendronet.get_embedding([i]))

    dir_path = os.path.join('results', 'single_tissue_experiments', args.ct)
    os.makedirs(dir_path, exist_ok=True)
    if whole_dataset:
        filename = args.ct + '_' + feature + '_' + str(LR) + '_' + str(early_stop) + '_unbalanced'
    else:
        filename = args.ct + '_' + feature + '_' + str(LR) + '_' + str(early_stop) + '_balanced'

    with open(os.path.join(dir_path, filename), 'w') as outfile:
        json.dump(output, outfile)
