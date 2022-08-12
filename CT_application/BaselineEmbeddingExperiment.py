import os
import pandas as pd
import argparse
from tqdm import tqdm
import numpy as np
import json
import jsonpickle
from sklearn.metrics import roc_curve, auc
import copy

import torch
from torch.utils.data.dataloader import DataLoader
import torch.optim
import torch.nn as nn

# Local imports
from models.CT_conv_model import EmbeddingBaselineModule, SeqConvModule, FCModule
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
    parser.add_argument('--EL', type=float, default=0.0)
    # parser.add_argument('--USE-CUDA', type=bool, choices=[True, False], default=True)
    parser.add_argument('--GPU', default=True, action='store_true')
    parser.add_argument('--CPU', dest='GPU', action='store_false')
    parser.add_argument('--BATCH-SIZE', type=int, default=128)
    # parser.add_argument('--whole-dataset', type=bool, choices=[True, False], default=False)
    parser.add_argument('--balanced', default=True, action='store_true')
    parser.add_argument('--unbalanced', dest='balanced', action='store_false')
    parser.add_argument('--embedding-size', type=int, default=2)
    parser.add_argument('--seeds', type=int, nargs='+', default=[1])
    parser.add_argument('--early-stopping', type=int, default=3)
    parser.add_argument('--num-epochs', type=int, default=100)

    args = parser.parse_args()

    feature = args.feature
    LR = args.LR
    EL = args.EL
    BATCH_SIZE = args.BATCH_SIZE
    USE_CUDA = args.GPU
    balanced = args.balanced
    embedding_size = args.embedding_size
    seeds = args.seeds
    early_stop = args.early_stopping
    epochs = args.num_epochs

    print('Using CUDA: ' + str(torch.cuda.is_available() and USE_CUDA))
    device = torch.device("cuda:0" if (torch.cuda.is_available() and USE_CUDA) else "cpu")

    with open(os.path.join('data_files', 'enhancers_seqs.json'), 'r') as enhancers_file:
        enhancers_dict = json.load(enhancers_file)

    tissue_names = []
    X = []
    y = []

    for tissue_file in os.listdir(os.path.join('data_files', 'CT_enhancer_features_matrices')):
        tissue_names.append(tissue_file[0:-29])

    enhancers_list = list(enhancers_dict.keys())

    print(tissue_names)

    tissue_dfs = {}

    for t in tissue_names:
        t_df = pd.read_csv(os.path.join('data_files', 'CT_enhancer_features_matrices',
                                        t + '_enhancer_features_matrix.csv'), index_col='cCRE_id')
        t_df = t_df.loc[enhancers_list]
        tissue_dfs[t] = t_df

    for enhancer in enhancers_list:
        X.append(get_one_hot_encoding(enhancers_dict[enhancer]))

    pos_counts = {t: 0 for t in tissue_names}
    neg_counts = {t: 0 for t in tissue_names}
    valid_counts = {t: 0 for t in tissue_names}
    pos_counters = {t: 0 for t in tissue_names}
    neg_counters = {t: 0 for t in tissue_names}

    for t in tissue_names:
        for enhancer in enhancers_list:
            if tissue_dfs[t].loc[enhancer, 'active'] == 1 or tissue_dfs[t].loc[enhancer, 'repressed'] == 1:
                valid_counts[t] += 1
                if tissue_dfs[t].loc[enhancer, feature] == 1:
                    pos_counts[t] += 1

    for t in tissue_names:
        neg_counts[t] = valid_counts[t] - pos_counts[t]

    print(pos_counts)
    print(neg_counts)
    print(valid_counts)
    pos_ratios = {t: pos_counts[t]/valid_counts[t] for t in tissue_names}
    print(pos_ratios)

    if not balanced:  # if we want to use all the samples, usually leads to heavily unbalanced dataset
        print('Using whole dataset')
        for t in tissue_names:
            y.extend(list(tissue_dfs[t].loc[:, feature]))
            print(len(y))
        # the list "samples" is a list of tuples each representing a sample. The first
        # entry is the row of the X matrix. The second is the index of the cell type in the
        # parent path matrix and the third is the index of the target in the y vector.

        packed_samples = []

        for enhancer_idx in range(len(enhancers_list)):
            enhancer_samples = []
            for t_idx, t in enumerate(tissue_names):
                if tissue_dfs[t].loc[enhancer, 'active'] == 1 or tissue_dfs[t].loc[enhancer, 'repressed'] == 1:
                    y_idx = t_idx * len(enhancers_list) + enhancer_idx
                    enhancer_samples.append((enhancer_idx, t_idx, y_idx))
                    if y[y_idx] == 1:
                        pos_counters[t] += 1
                    else:
                        neg_counters[t] += 1
            packed_samples.append(enhancer_samples)

    else:  # In this case, we make sure that for each tissue type, the number of positive and negative examples
           # is the same, which gives us a balanced dataset
        print('Using a balanced dataset')

        packed_samples = []

        # the list "samples" is a list of tuples each representing a sample. The first
        # entry is the row of the X matrix. The second is the index of the cell type in the
        # parent path matrix and the third is the index of the target in the y vector.

        for j, enhancer in enumerate(enhancers_list):
            enhancer_samples = []
            for i, t in enumerate(tissue_names):
                if tissue_dfs[t].loc[enhancer, 'active'] == 1 or tissue_dfs[t].loc[enhancer, 'repressed'] == 1:
                    if pos_ratios[t] <= 0.5:
                        if tissue_dfs[t].loc[enhancer, feature] == 1:
                            enhancer_samples.append((j, i, len(y)))
                            y.append(1)
                            pos_counters[t] += 1
                        if tissue_dfs[t].loc[enhancer, feature] == 0 and neg_counters[t] < pos_counts[t]:
                            enhancer_samples.append((j, i, len(y)))
                            y.append(0)
                            neg_counters[t] += 1
                    else:
                        if tissue_dfs[t].loc[enhancer, feature] == 1 and pos_counters[t] < neg_counts[t]:
                            enhancer_samples.append((j, i, len(y)))
                            y.append(1)
                            pos_counters[t] += 1
                        if tissue_dfs[t].loc[enhancer, feature] == 0:
                            enhancer_samples.append((j, i, len(y)))
                            y.append(0)
                            neg_counters[t] += 1
            packed_samples.append(enhancer_samples)

    print(pos_counters)
    print(neg_counters)
    print(len(X))
    print(len(X[0]))
    print(len(y))

    X = torch.tensor(X, dtype=torch.float, device=device)
    X = X.permute(0, 2, 1)
    y = torch.tensor(y, dtype=torch.float, device=device)

    params = {'batch_size': BATCH_SIZE,
              'shuffle': True,
              'num_workers': 0}

    output = {'train_auc': [], 'val_auc': [], 'test_auc': [], 'epochs': [],
              'train_loss': [], 'val_loss': [], 'test_loss': []}
    embeddings_output = {t: [] for t in tissue_names}

    for seed in seeds:
        # The three subparts of the model:

        embeddings_mat = np.zeros(shape=(len(tissue_names), embedding_size))

        embedding_model = EmbeddingBaselineModule(device=device, embeddings_mat=embeddings_mat)

        convolution = SeqConvModule(device=device, seq_length=501, kernel_sizes=(16, 3, 3), num_of_kernels=(64, 64, 32),
                                    polling_windows=(3, 4), input_channels=4)

        fully_connected = FCModule(device=device, layer_sizes=(embedding_size + 32, 32, 1))

        packed_train_idx, packed_test_idx = split_indices(packed_samples, seed=0)
        packed_train_idx, packed_val_idx = split_indices(packed_train_idx, seed=seed)

        # unpacking
        train_idx = []
        val_idx = []
        test_idx = []

        for packet in packed_train_idx:
            for sample in packet:
                train_idx.append(sample)

        for packet in packed_val_idx:
            for sample in packet:
                val_idx.append(sample)

        for packet in packed_test_idx:
            for sample in packet:
                test_idx.append(sample)

        train_set = IndicesDataset(train_idx)
        test_set = IndicesDataset(test_idx)
        val_set = IndicesDataset(val_idx)

        train_batch_gen = DataLoader(train_set, **params)
        test_batch_gen = DataLoader(test_set, **params)
        val_batch_gen = DataLoader(val_set, **params)

        loss_function = nn.BCELoss()
        if torch.cuda.is_available() and USE_CUDA:
            loss_function = loss_function.cuda()
        optimizer = torch.optim.Adam(list(embedding_model.parameters()) + list(convolution.parameters())
                                     + list(fully_connected.parameters()), lr=LR)

        best_val_auc = 0
        early_stop_count = 0

        for epoch in range(epochs):
            print("Epoch " + str(epoch))
            for step, idx_batch in enumerate(tqdm(train_batch_gen)):
                optimizer.zero_grad()
                # print(y[idx_batch])
                X_idx = idx_batch[0]
                tissue_idx = idx_batch[1]
                y_idx = idx_batch[2]
                seq_features = convolution(X[X_idx])
                tissue_embeddings = embedding_model(tissue_idx)
                y_hat = fully_connected(torch.cat((seq_features, tissue_embeddings.float()), 1))
                # print(y_hat)
                # error_loss = loss_function(y_hat, y[idx_batch])
                error_loss = loss_function(y_hat, y[y_idx])
                embedding_loss = embedding_model.embedding_loss(tissue_idx)
                train_loss = error_loss + EL*embedding_loss
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
                    tissue_idx = idx_batch[1]
                    y_idx = idx_batch[2]
                    seq_features = convolution(X[X_idx])
                    tissue_embeddings = embedding_model(tissue_idx)
                    y_hat = fully_connected(torch.cat((seq_features, tissue_embeddings.float()), 1))
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
                    tissue_idx = idx_batch[1]
                    y_idx = idx_batch[2]
                    seq_features = convolution(X[X_idx])
                    tissue_embeddings = embedding_model(tissue_idx)
                    y_hat = fully_connected(torch.cat((seq_features, tissue_embeddings.float()), 1))
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
                    best_convolution_state = copy.deepcopy(convolution.state_dict())
                    best_fc_state = copy.deepcopy(fully_connected.state_dict())
                    best_embeddings_mat = embedding_model.embeddings_mat.clone().detach().cpu()

                else:
                    early_stop_count += 1
                    print('The performance hasn\'t improved for ' + str(early_stop_count) + ' epoches')
                    print(' Best is :' + str(best_val_auc))

                if early_stop_count == early_stop:
                    print('Early Stopping!')
                    break

        with torch.no_grad():
            # After training completed, we retrieve the model's components that led to the best performance on the
            # validation set
            embedding_model = EmbeddingBaselineModule(device, best_embeddings_mat)
            convolution.load_state_dict(best_convolution_state)
            fully_connected.load_state_dict(best_fc_state)

            print("Test performance on train set on best model: ")
            all_train_targets = []
            all_train_predictions = []
            train_error_loss = 0.0
            for step, idx_batch in enumerate(tqdm(train_batch_gen)):
                X_idx = idx_batch[0]
                tissue_idx = idx_batch[1]
                y_idx = idx_batch[2]
                seq_features = convolution(X[X_idx])
                tissue_embeddings = embedding_model(tissue_idx)
                y_hat = fully_connected(torch.cat((seq_features, tissue_embeddings.float()), 1))
                train_error_loss += float(loss_function(y_hat, y[y_idx]))
                all_train_targets.extend(list(y[y_idx].detach().cpu().numpy()))
                all_train_predictions.extend(list(y_hat.detach().cpu().numpy()))

            print("average error loss on train set : " + str(float(train_error_loss) / (step + 1)))
            fpr, tpr, _ = roc_curve(all_train_targets, all_train_predictions)
            train_roc_auc = auc(fpr, tpr)
            print('ROC AUC on train set : ' + str(train_roc_auc))
            train_steps = step + 1

            print("Test performance on validation set on best model:")
            all_val_targets = []
            all_val_predictions = []
            val_error_loss = 0.0
            for step, idx_batch in enumerate(tqdm(val_batch_gen)):
                X_idx = idx_batch[0]
                tissue_idx = idx_batch[1]
                y_idx = idx_batch[2]
                seq_features = convolution(X[X_idx])
                tissue_embeddings = embedding_model(tissue_idx)
                y_hat = fully_connected(torch.cat((seq_features, tissue_embeddings.float()), 1))
                val_error_loss += float(loss_function(y_hat, y[y_idx]))
                all_val_targets.extend(list(y[y_idx].detach().cpu().numpy()))
                all_val_predictions.extend(list(y_hat.detach().cpu().numpy()))

            print("average error loss on validation set : " + str(float(val_error_loss) / (step + 1)))
            fpr, tpr, _ = roc_curve(all_val_targets, all_val_predictions)
            val_roc_auc = auc(fpr, tpr)
            print('ROC AUC on validation set : ' + str(val_roc_auc))
            val_steps = step + 1

            print("Test performance on test set on best model:")
            all_test_targets = []
            all_test_predictions = []
            test_error_loss = 0.0
            for step, idx_batch in enumerate(tqdm(test_batch_gen)):
                X_idx = idx_batch[0]
                tissue_idx = idx_batch[1]
                y_idx = idx_batch[2]
                seq_features = convolution(X[X_idx])
                tissue_embeddings = embedding_model(tissue_idx)
                y_hat = fully_connected(torch.cat((seq_features, tissue_embeddings.float()), 1))
                test_error_loss += float(loss_function(y_hat, y[y_idx]))
                all_test_targets.extend(list(y[y_idx].detach().cpu().numpy()))
                all_test_predictions.extend(list(y_hat.detach().cpu().numpy()))

            print("average error loss on test set : " + str(float(test_error_loss) / (step + 1)))
            fpr, tpr, _ = roc_curve(all_test_targets, all_test_predictions)
            test_roc_auc = auc(fpr, tpr)
            print('ROC AUC on test set : ' + str(test_roc_auc))
            test_steps = step + 1

        output['train_auc'].append(train_roc_auc)
        output['val_auc'].append(val_roc_auc)
        output['test_auc'].append(test_roc_auc)
        output['train_loss'].append(train_error_loss / train_steps)
        output['val_loss'].append(val_error_loss / val_steps)
        output['test_loss'].append(test_error_loss / test_steps)
        output['epochs'].append(epoch + 1)

        for i, tissue in enumerate(tissue_names):
            embedding = (torch.squeeze(embedding_model.forward([i]))).cpu().tolist()
            embeddings_output[tissue].append(embedding)

    if not balanced:
        dir_name = feature + '_' + str(LR) + '_' + str(EL) \
                  + '_' + str(embedding_size) + '_' + str(early_stop) + '_unbalanced'
    else:
        dir_name = feature + '_' + str(LR) + '_' + str(EL) \
                  + '_' + str(embedding_size) + '_' + str(early_stop) + '_balanced'

    dir_path = os.path.join('results', 'baseline_embedding_experiments', dir_name)
    os.makedirs(dir_path, exist_ok=True)

    with open(os.path.join(dir_path, 'output.json'), 'w') as outfile:
        json.dump(output, outfile)

    torch.save({'convolution': convolution.state_dict(),
                'fully_connected': fully_connected.state_dict(),
                'embeddings_mat': embedding_model.embeddings_mat.clone().detach().cpu()},
                os.path.join(dir_path, 'model.pt'))

    with open(os.path.join(dir_path, 'embeddings.json'), 'w') as outfile:
        json.dump(embeddings_output, outfile)
