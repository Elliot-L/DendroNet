import os
import pandas as pd
import torch
import torch.nn as nn
import argparse
import json
import numpy as np
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc

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
    parser.add_argument('--model', type=str, default='multi', help='options are dendronet or multi')
    parser.add_argument('--feature', type=str, default='active')
    parser.add_argument('--tissue', type=str)
    parser.add_argument('--LR', type=float, default=0.0001)
    parser.add_argument('--DPF', type=float, default=0.001)
    parser.add_argument('--L1', type=float, default=0.001)
    parser.add_argument('--embedding-size', type=int, default=28)
    parser.add_argument('--GPU', default=True, action='store_true')
    parser.add_argument('--CPU', dest='GPU', action='store_false')
    parser.add_argument('--BATCH-SIZE', type=int, default=128)
    parser.add_argument('--balanced', default=True, action='store_true')
    parser.add_argument('--unbalanced', dest='balanced', action='store_false')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1])
    parser.add_argument('--early-stopping', type=int, default=3)
    parser.add_argument('--num-epochs', type=int, default=100)

    args = parser.parse_args()

    model = args.model
    tissue = args.tissue
    LR = args.LR
    USE_CUDA = args.GPU
    BATCH_SIZE = args.BATCH_SIZE
    balanced = args.balanced
    feature = args.feature
    seeds = args.seeds
    early_stop = args.early_stopping
    epochs = args.num_epochs

    data_type = '_balanced'
    if not balanced:
        data_type = '_unbalanced'

    device = torch.device("cuda:0" if (torch.cuda.is_available() and USE_CUDA) else "cpu")
    print('Using CUDA: ' + str(torch.cuda.is_available() and USE_CUDA))

    samples_df = pd.read_csv(os.path.join('data_files', 'CT_enhancer_features_matrices',
                                          tissue + '_enhancer_features_matrix.csv'))

    with open(os.path.join('data_files', 'enhancers_seqs.json'), 'r') as e_file:
        enhancers_dict = json.load(e_file)

    enhancer_list = enhancers_dict.keys()

    samples_df.set_index('cCRE_id', inplace=True)
    samples_df = samples_df.loc[enhancer_list]

    y = []
    X = []
    pos_ratio = 0

    valid_samples = 0
    for enhancer in enhancer_list:
        if samples_df.loc[enhancer, 'active'] == 1 or samples_df.loc[enhancer, 'repressed'] == 1:
            valid_samples += 1
            if samples_df.loc[enhancer, feature] == 1:
                pos_ratio += 1
    pos_ratio = pos_ratio / (valid_samples - pos_ratio)

    if not balanced:
        for enhancer in enhancer_list:
            if samples_df.loc[enhancer, 'active'] == 1 or samples_df.loc[enhancer, 'repressed'] == 1:
                if samples_df.loc[enhancer, feature] == 0:
                    y.append(0)
                    X.append(get_one_hot_encoding(enhancers_dict[enhancer]))
                if samples_df.loc[enhancer, feature] == 1:
                    y.append(1)
                    X.append(get_one_hot_encoding(enhancers_dict[enhancer]))
    else:
        for enhancer in enhancer_list:
            if samples_df.loc[enhancer, 'active'] == 1 or samples_df.loc[enhancer, 'repressed'] == 1:
                if samples_df.loc[enhancer, feature] == 1:
                    y.append(1)
                    X.append(get_one_hot_encoding(enhancers_dict[enhancer]))
                rand = np.random.uniform(0.0, 1.0)
                if samples_df.loc[enhancer, feature] == 0 and rand <= pos_ratio:
                    y.append(0)
                    X.append(get_one_hot_encoding(enhancers_dict[enhancer]))

    print(pos_ratio)
    print(len(X))
    print(len(X[0]))
    print(len(y))

    X = torch.tensor(X, dtype=torch.float, device=device)
    X = X.permute(0, 2, 1)
    y = torch.tensor(y, dtype=torch.float, device=device)

    params = {'batch_size': BATCH_SIZE,
              'shuffle': True,
              'num_workers': 0}

    test_idx = range(len(y))
    test_set = IndicesDataset(test_idx)
    test_batch_gen = DataLoader(test_set, **params)

    if model == 'dendronet':
        DPF = args.DPF
        L1 = args.L1
        embedding_size = args.embedding_size

        dir_name = feature + '_' + str(LR) + '_' + str(DPF) + '_' + str(L1) \
                    + '_' + str(embedding_size) + '_' + str(early_stop) + data_type
        print('Using Dendronet from dir: ' + dir_name)

        model_file = os.path.join('results', 'dendronet_embedding_experiments', dir_name, 'model.pt')

        model_dist = torch.load(model_file)

        convolution_state = model_dist['convolution']
        fully_connected_state = model_dist['fully_connected']
        delta_mat = model_dist['dendronet_delta_mat']
        root_vector = model_dist['dendronet_root']

        pc_mat, nodes = create_pc_mat()
        num_internal = 0
        tissue_names = []
        for node in nodes:
            if node == 'internal node':
                num_internal += 1
            else:
                tissue_names.append(node)
        tissue_idx = tissue_names.index(tissue) + num_internal

        parent_path_mat = build_parent_path_mat(pc_mat)

        convolution = SeqConvModule(device=device, seq_length=501, kernel_sizes=(16, 3, 3), num_of_kernels=(64, 64, 32),
                                    polling_windows=(3, 4), input_channels=4)
        dendronet = DendronetModule(device=device, root_weights=root_vector, delta_mat=delta_mat,
                                    path_mat=parent_path_mat)
        fully_connected = FCModule(device=device, layer_sizes=(embedding_size + 32, 32, 1))

        fully_connected.load_state_dict(fully_connected_state)
        convolution.load_state_dict(convolution_state)

        all_test_targets = []
        all_test_predictions = []
        # test_error_loss = 0.0
        for step, idx_batch in enumerate(tqdm(test_batch_gen)):
            # y_hat = CT_specific_conv(X[idx_batch])
            seq_features = convolution(X[idx_batch])
            tissue_embeddings = dendronet([tissue_idx for i in range(seq_features.size()[0])])
            y_hat = fully_connected(torch.cat((seq_features, tissue_embeddings), 1))
            # test_error_loss += float(loss_function(y_hat, y[idx_batch]))
            all_test_targets.extend(list(y[idx_batch].detach().cpu().numpy()))
            all_test_predictions.extend(list(y_hat.detach().cpu().numpy()))

        # print("average error loss on test set : " + str(float(test_error_loss) / (step + 1)))
        fpr, tpr, _ = roc_curve(all_test_targets, all_test_predictions)
        test_roc_auc = auc(fpr, tpr)
        print('ROC AUC on test set : ' + str(test_roc_auc))

    elif model == 'multi':
        dir_name = feature + '_' + str(LR) + '_' + str(early_stop) + data_type
        print('Using MultiTissue model from dir: ' + dir_name)

        tissue_names = []
        for t_file in os.listdir(os.path.join('data_files', 'CT_enhancer_features_matrices')):
            t_name = t_file[0:-29]
            tissue_names.append(t_name)

        tissue_encoding = [[1 if t == tissue else 0 for t in tissue_names]]
        tissue_encoding = torch.tensor(tissue_encoding, dtype=torch.float, device=device)

        model_file = os.path.join('results', 'multi_tissues_experiments', dir_name, 'model.pt')

        model_dist = torch.load(model_file)

        convolution_state = model_dist['convolution']
        fully_connected_state = model_dist['fully_connected']

        convolution = SeqConvModule(device=device, seq_length=501, kernel_sizes=(16, 3, 3), num_of_kernels=(64, 64, 32),
                                    polling_windows=(3, 4), input_channels=4)
        fully_connected = FCModule(device=device, layer_sizes=(len(tissue_names) + 32, 32, 1))

        convolution.load_state_dict(convolution_state)
        fully_connected.load_state_dict(fully_connected_state)

        all_test_targets = []
        all_test_predictions = []
        # test_error_loss = 0.0
        for step, idx_batch in enumerate(tqdm(test_batch_gen)):
            # y_hat = CT_specific_conv(X[idx_batch])
            seq_features = convolution(X[idx_batch])
            tissue_encodings = tissue_encoding[[0 for i in range(seq_features.size()[0])]]
            y_hat = fully_connected(torch.cat((seq_features, tissue_encodings), 1))
            # test_error_loss += float(loss_function(y_hat, y[idx_batch]))
            all_test_targets.extend(list(y[idx_batch].detach().cpu().numpy()))
            all_test_predictions.extend(list(y_hat.detach().cpu().numpy()))

        # print("average error loss on test set : " + str(float(test_error_loss) / (step + 1)))
        fpr, tpr, _ = roc_curve(all_test_targets, all_test_predictions)
        test_roc_auc = auc(fpr, tpr)
        print('ROC AUC on test set : ' + str(test_roc_auc))





