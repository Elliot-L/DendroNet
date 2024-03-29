import os
import pandas as pd
import argparse
from tqdm import tqdm
import numpy as np
import json
from sklearn.metrics import roc_curve, auc
import copy

import torch
from torch.utils.data.dataloader import DataLoader
import torch.optim
import torch.nn as nn

# Local imports
from models.CT_conv_model import SeqConvModule, FCModule, CTspecificConvNet, SimpleCTspecificConvNet
from utils.model_utils import split_indices, IndicesDataset


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
    parser.add_argument('--tissue', type=str, default='vagina')
    parser.add_argument('--feature', type=str, default='active')
    parser.add_argument('--LR', type=float, default=0.0001)
    parser.add_argument('--GPU', default=True, action='store_true')
    parser.add_argument('--CPU', dest='GPU', action='store_false')
    parser.add_argument('--BATCH-SIZE', type=int, default=128)
    # parser.add_argument('--whole-dataset', type=bool, choices=[True, False], default=False)
    parser.add_argument('--balanced', default=True, action='store_true')
    parser.add_argument('--unbalanced', dest='balanced', action='store_false')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1])
    parser.add_argument('--early-stopping', type=int, default=3)
    parser.add_argument('--num-epochs', type=int, default=100)

    args = parser.parse_args()

    tissue = args.tissue
    feature = args.feature
    LR = args.LR
    USE_CUDA = torch.cuda.is_available() and args.GPU
    BATCH_SIZE = args.BATCH_SIZE
    balanced = args.balanced
    seeds = args.seeds
    early_stop = args.early_stopping
    epochs = args.num_epochs

    print('Using CUDA: ' + str(USE_CUDA))
    device = torch.device("cuda:0" if USE_CUDA else "cpu")

    samples_df = pd.read_csv(os.path.join('data_files', 'CT_enhancer_features_matrices',
                                          tissue + '_enhancer_features_matrix.csv'))

    with open(os.path.join('data_files', 'enhancers_seqs.json'), 'r') as e_file:
        enhancers_dict = json.load(e_file)

    enhancer_list = list(enhancers_dict.keys())

    samples_df.set_index('cCRE_id', inplace=True)
    samples_df = samples_df.loc[enhancer_list]

    y = []
    X = []
    pos_count = 0
    pos_counter = 0
    neg_counter = 0
    total_valid = 0

    for enhancer in enhancer_list:
        if samples_df.loc[enhancer, 'active'] == 1 or samples_df.loc[enhancer, 'repressed'] == 1:
            total_valid += 1
            if samples_df.loc[enhancer, feature] == 1:
                pos_count += 1

    print(pos_count)
    print(total_valid)

    if not balanced:
        print('Unbalanced dataset')
        for enhancer in enhancer_list:
            if samples_df.loc[enhancer, 'active'] == 1 or samples_df.loc[enhancer, 'repressed'] == 1:
                if samples_df.loc[enhancer, feature] == 0:
                    y.append(0)
                    X.append(get_one_hot_encoding(enhancers_dict[enhancer]))
                    neg_counter += 1
                if samples_df.loc[enhancer, feature] == 1:
                    y.append(1)
                    X.append(get_one_hot_encoding(enhancers_dict[enhancer]))
                    pos_counter += 1
    else:
        print('Balanced dataset')
        if (pos_count / total_valid) <= 0.5:
            for enhancer in enhancer_list:
                if samples_df.loc[enhancer, 'active'] == 1 or samples_df.loc[enhancer, 'repressed'] == 1:
                    if samples_df.loc[enhancer, feature] == 1:
                        y.append(1)
                        X.append(get_one_hot_encoding(enhancers_dict[enhancer]))
                        pos_counter += 1
                    if samples_df.loc[enhancer, feature] == 0 and neg_counter < pos_count:
                        neg_counter += 1
                        y.append(0)
                        X.append(get_one_hot_encoding(enhancers_dict[enhancer]))
        else:
            neg_count = total_valid - pos_count
            for enhancer in enhancer_list:
                if samples_df.loc[enhancer, 'active'] == 1 or samples_df.loc[enhancer, 'repressed'] == 1:
                    if samples_df.loc[enhancer, feature] == 1 and pos_counter < neg_count:
                        pos_counter += 1
                        y.append(1)
                        X.append(get_one_hot_encoding(enhancers_dict[enhancer]))
                    if samples_df.loc[enhancer, feature] == 0:
                        y.append(0)
                        X.append(get_one_hot_encoding(enhancers_dict[enhancer]))
                        neg_counter += 1

    print(pos_counter)
    print(neg_counter)
    print(len(X))
    print(len(X[0]))
    print(len(y))
    """
    # TEST 1 : Adding motifs to the positive and/or negative sequences for testing purposes
    
    for i in range(len(X)):
        motif_pos = get_one_hot_encoding('AGAGAGAGAGAGAGAGAGAG')
        motif_neg = get_one_hot_encoding('TCTCTCTCTCTCTCTCTCTC')
        if y[i] == 1:
            X[i] = X[i][0:120] + motif_pos + X[i][140:501]
        if y[i] == 0:
            X[i] = X[i][0:347] + motif_neg + X[i][367:501]
    """
    """
    # TEST 2 : Adding motifs to the positive and/or negative sequences for testing purposes.
    # This time though, two different sequences, at random, are associated with each class

    motif_pos1 = get_one_hot_encoding('AGTCGCTAGATCGATCGGCA')
    motif_pos2 = get_one_hot_encoding('AGTCGCTAGATCGATCGGCA')
    motif_neg1 = get_one_hot_encoding('GATAGCTAGATGCTGGATGC')
    motif_neg2 = get_one_hot_encoding('TATATGATAGACTAGCTCAA')
    for i in range(len(X)):
        rand = np.random.uniform(0.0, 1.0)
        if y[i] == 1 and rand >= 0.5:
            X[i] = X[i][0:120] + motif_pos1 + X[i][140:501]
        if y[i] == 1 and rand < 0.5:
            X[i] = X[i][0:420] + motif_pos2 + X[i][440:501]
        if y[i] == 0 and rand >= 0.5:
            X[i] = X[i][0:247] + motif_neg1 + X[i][267:501]
        if y[i] == 0 and rand < 0.5:
            X[i] = X[i][0:347] + motif_neg2 + X[i][367:501]
    """
    """
    # Test 3 : Adding motifs, but also a specific syntax that determines the class of the sequence

    motif1 = get_one_hot_encoding('AGTTTGCTAG')  # len = 10
    motif2 = get_one_hot_encoding('AGTCGCCCTAGCA')  # len = 13
    motif3 = get_one_hot_encoding('GCTAG')  # len = 5
    motif4 = get_one_hot_encoding('AGGCATAAAGTGC')  # len = 13
    motif5 = get_one_hot_encoding('TATCCAG')  # len = 7
    motif6 = get_one_hot_encoding('ACCTACGCTAAA')  # len = 12
    
    for i in range(len(X)):
        rand = np.random.uniform(0.0, 1.0)
        if y[i] == 1 and rand >= 0.5:
            X[i] = X[i][0:30] + motif1 + X[i][40:501]
            X[i] = X[i][0:165] + motif2 + X[i][178:501]
            X[i] = X[i][0:250] + motif3 + X[i][255:501]
        if y[i] == 1 and rand < 0.5:
            X[i] = X[i][0:100] + motif3 + X[i][105:501]
            X[i] = X[i][0:130] + motif4 + X[i][143:501]
            X[i] = X[i][0:160] + motif5 + X[i][167:501]
            X[i] = X[i][0:190] + motif6 + X[i][202:501]
        if y[i] == 0 and rand >= 0.5:
            X[i] = X[i][0:30] + motif1 + X[i][40:501]
            X[i] = X[i][0:165] + motif3 + X[i][170:501]
            X[i] = X[i][0:250] + motif2 + X[i][263:501]
        if y[i] == 0 and rand < 0.5:
            X[i] = X[i][0:100] + motif3 + X[i][105:501]
            X[i] = X[i][0:130] + motif2 + X[i][143:501]
            X[i] = X[i][0:160] + motif5 + X[i][167:501]
            X[i] = X[i][0:190] + motif6 + X[i][202:501]
    """
    print(len(X))
    print(len(y))

    output = {'train_auc': [], 'val_auc': [], 'test_auc': [], 'pos_ratio': (pos_count/total_valid), 'epochs': [],
              'train_loss': [], 'val_loss': [], 'test_loss': []}

    X = torch.tensor(X, dtype=torch.float, device=device)
    X = X.permute(0, 2, 1)
    y = torch.tensor(y, dtype=torch.float, device=device)

    params = {'batch_size': BATCH_SIZE,
              'shuffle': True,
              'num_workers': 0}

    for seed in seeds:
        # CT_specific_conv = SimpleCTspecificConvNet(cell_type='test', device=device, seq_length=501, kernel_size=16,
        #                                num_of_kernels=128, polling_window=0, initial_channels=4)
        # CT_specific_conv = CTspecificConvNet(device=device, cell_type=args.ct, seq_length=501,
        #                                      kernel_size=(16, 3, 3), num_of_kernels=(64, 64, 32),
        #                                      polling_window=(3, 4))

        fully_connected = FCModule(device=device, layer_sizes=(32, 32, 1))

        convolution = SeqConvModule(device=device, seq_length=501, kernel_sizes=(16, 3, 3), num_of_kernels=(64, 64, 32),
                                    polling_windows=(3, 4), input_channels=4)

        train_idx, test_idx = split_indices(range(len(X)), seed=0)
        train_idx, val_idx = split_indices(train_idx, seed=seed)

        train_set = IndicesDataset(train_idx)
        test_set = IndicesDataset(test_idx)
        val_set = IndicesDataset(val_idx)

        train_batch_gen = DataLoader(train_set, **params)
        test_batch_gen = DataLoader(test_set, **params)
        val_batch_gen = DataLoader(val_set, **params)

        loss_function = nn.BCELoss()
        if USE_CUDA:
            loss_function = loss_function.cuda()
        optimizer = torch.optim.Adam(list(convolution.parameters()) + list(fully_connected.parameters())
                                     , lr=LR)

        best_val_auc = 0
        early_stop_count = 0

        for epoch in range(epochs):
            print("Epoch " + str(epoch))
            for step, idx_batch in enumerate(tqdm(train_batch_gen)):
                optimizer.zero_grad()
                # print(y[idx_batch])
                # y_hat = CT_specific_conv(X[idx_batch])
                seq_features = convolution(X[idx_batch])
                y_hat = fully_connected(seq_features)
                error_loss = loss_function(y_hat, y[idx_batch])
                error_loss.backward(retain_graph=True)
                optimizer.step()
            with torch.no_grad():
                print("Test performance on train set for epoch " + str(epoch))
                all_train_targets = []
                all_train_predictions = []
                train_error_loss = 0.0
                for step, idx_batch in enumerate(tqdm(train_batch_gen)):
                    # y_hat = CT_specific_conv(X[idx_batch])
                    seq_features = convolution(X[idx_batch])
                    y_hat = fully_connected(seq_features)
                    train_error_loss += float(loss_function(y_hat, y[idx_batch]))
                    all_train_targets.extend(list(y[idx_batch].detach().cpu().numpy()))
                    all_train_predictions.extend(list(y_hat.detach().cpu().numpy()))

                print("average error loss on train set : " + str(float(train_error_loss) / (step + 1)))
                fpr, tpr, _ = roc_curve(all_train_targets, all_train_predictions)
                train_roc_auc = auc(fpr, tpr)
                print('ROC AUC on train set : ' + str(train_roc_auc))

                print("Test performance on val set for epoch " + str(epoch))
                all_val_targets = []
                all_val_predictions = []
                val_error_loss = 0.0
                for step, idx_batch in enumerate(tqdm(val_batch_gen)):
                    # y_hat = CT_specific_conv(X[idx_batch])
                    seq_features = convolution(X[idx_batch])
                    y_hat = fully_connected(seq_features)
                    val_error_loss += float(loss_function(y_hat, y[idx_batch]))
                    all_val_targets.extend(list(y[idx_batch].detach().cpu().numpy()))
                    all_val_predictions.extend(list(y_hat.detach().cpu().numpy()))

                print("average error loss on val set : " + str(float(val_error_loss) / (step + 1)))
                fpr, tpr, _ = roc_curve(all_val_targets, all_val_predictions)
                val_roc_auc = auc(fpr, tpr)
                print('ROC AUC on validation set : ' + str(val_roc_auc))

                if val_roc_auc > best_val_auc:
                    best_val_auc = val_roc_auc
                    print('New best AUC on validation set: ' + str(best_val_auc))
                    early_stop_count = 0
                    best_convolution_state = copy.deepcopy(convolution.state_dict())
                    best_fc_state = copy.deepcopy(fully_connected.state_dict())
                else:
                    early_stop_count += 1
                    print('The performance hasn\'t improved for ' + str(early_stop_count) + ' epoches')
                    print(' Best is :' + str(best_val_auc))

                if early_stop_count == early_stop:
                    print('Early Stopping!')
                    break

        with torch.no_grad():
            print("Test performance on train set on best model:")

            convolution.load_state_dict(best_convolution_state)
            fully_connected.load_state_dict(best_fc_state)

            all_train_targets = []
            all_train_predictions = []
            train_error_loss = 0.0
            for step, idx_batch in enumerate(tqdm(train_batch_gen)):
                # y_hat = CT_specific_conv(X[idx_batch])
                seq_features = convolution(X[idx_batch])
                y_hat = fully_connected(seq_features)
                train_error_loss += float(loss_function(y_hat, y[idx_batch]))
                all_train_targets.extend(list(y[idx_batch].detach().cpu().numpy()))
                all_train_predictions.extend(list(y_hat.detach().cpu().numpy()))

            print("average error loss on train set : " + str(float(train_error_loss) / (step + 1)))
            fpr, tpr, _ = roc_curve(all_train_targets, all_train_predictions)
            train_roc_auc = auc(fpr, tpr)
            print('ROC AUC on train set : ' + str(train_roc_auc))
            train_steps = step + 1

            print("Test performance on val set for epoch on best model")
            all_val_targets = []
            all_val_predictions = []
            val_error_loss = 0.0
            for step, idx_batch in enumerate(tqdm(val_batch_gen)):
                # y_hat = CT_specific_conv(X[idx_batch])
                seq_features = convolution(X[idx_batch])
                y_hat = fully_connected(seq_features)
                val_error_loss += float(loss_function(y_hat, y[idx_batch]))
                all_val_targets.extend(list(y[idx_batch].detach().cpu().numpy()))
                all_val_predictions.extend(list(y_hat.detach().cpu().numpy()))

            print("average error loss on val set : " + str(float(val_error_loss) / (step + 1)))
            fpr, tpr, _ = roc_curve(all_val_targets, all_val_predictions)
            val_roc_auc = auc(fpr, tpr)
            print('ROC AUC on validation set : ' + str(val_roc_auc))
            val_steps = step + 1

            print("Test performance on test set on the trained model")
            all_test_targets = []
            all_test_predictions = []
            test_error_loss = 0.0
            for step, idx_batch in enumerate(tqdm(test_batch_gen)):
                # y_hat = CT_specific_conv(X[idx_batch])
                seq_features = convolution(X[idx_batch])
                y_hat = fully_connected(seq_features)
                test_error_loss += float(loss_function(y_hat, y[idx_batch]))
                all_test_targets.extend(list(y[idx_batch].detach().cpu().numpy()))
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

    if not balanced:
        dir_name = feature + '_' + str(LR) + '_' + str(early_stop) + '_unbalanced'
    else:
        dir_name = feature + '_' + str(LR) + '_' + str(early_stop) + '_balanced'

    dir_path = os.path.join('results', 'single_tissue_experiments', tissue, dir_name)
    os.makedirs(dir_path, exist_ok=True)

    with open(os.path.join(dir_path, 'output.json'), 'w') as outfile:
        json.dump(output, outfile)

    torch.save({'convolution': convolution.state_dict(),
                'fully_connected': fully_connected.state_dict()},
               os.path.join(dir_path, 'model.pt'))

