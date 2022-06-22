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
from models.CT_conv_model import SeqConvModule, FCModule, MultiCTConvNet
from utils.model_utils import split_indices, IndicesDataset


def get_one_hot_encoding(seq):
    #  (A,G,T,C), ex: A = (1, 0, 0, 0), T = (0, 0, 1, 0)
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
    """
    parser.add_argument('--cts', type=str, nargs='+', default=['vagina', 'adrenal_gland', 'prostate_gland',
                                                               'sigmoid_colon', 'testis', 'stomach',
                                                               'uterus', 'tibial_nerve', 'spleen'],
                        help='if empty, we take all available datasets')
    """
    parser.add_argument('--cts', type=str, nargs='+', default=[], help='if empty, we take all available datasets')
    parser.add_argument('--feature', type=str, default='active')
    parser.add_argument('--LR', type=float, default=0.001)
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

    feature = args.feature
    LR = args.LR
    BATCH_SIZE = args.BATCH_SIZE
    USE_CUDA = args.GPU
    balanced = args.balanced
    cell_names = args.cts
    seeds = args.seeds
    early_stop = args.early_stopping
    epochs = args.num_epochs

    if not cell_names:
        print('using all cell types')
        cell_names = []
        for ct_file in os.listdir(os.path.join('data_files', 'CT_enhancer_features_matrices')):
            ct_name = ct_file[0:-29]
            cell_names.append(ct_name)

    print('Using CUDA: ' + str(torch.cuda.is_available() and USE_CUDA))
    device = torch.device("cuda:0" if (torch.cuda.is_available() and USE_CUDA) else "cpu")

    with open(os.path.join('data_files', 'enhancers_seqs.json'), 'r') as enhancers_file:
        enhancers_dict = json.load(enhancers_file)

    X = []
    y = []
    cell_encodings = []

    enhancers_list = list(enhancers_dict.keys())
    num_enhancers = len(enhancers_list)
    print(len(enhancers_list))
    print(cell_names)

    for enhancer in enhancers_list:
        X.append(get_one_hot_encoding(enhancers_dict[enhancer]))

    for ct in range(len(cell_names)):
        cell_encodings.append([1 if j == ct else 0 for j in range(len(cell_names))])

    if not balanced:  # if we want to use all the samples, usually leads to unbalanced dataset
        print('Using whole dataset')
        samples = []

        # the list "samples" is a list of tuples each representing a sample. The first
        # entry is the row of the X matrix. The second is the index of the cell type.
        # the third is the index of the target in the y vector.

        for ct_idx, ct in enumerate(cell_names):
            ct_df = pd.read_csv(os.path.join('data_files', 'CT_enhancer_features_matrices',
                                             ct + '_enhancer_features_matrix.csv'), index_col='cCRE_id')
            ct_df = ct_df.loc[enhancers_list]
            y.extend(list(ct_df.loc[:, feature]))

            for enhancer_idx in range(len(enhancers_list)):
                if ct_df.loc[enhancer, 'active'] == 1 or ct_df.loc[enhancer, 'repressed'] == 1:
                    samples.append((enhancer_idx, ct_idx, ct_idx * len(enhancers_list) + enhancer_idx))

    else:  # In this case, we make sure that for each tissue type, the number of positive and negative examples
           # is the same, which gives us a balanced dataset
        print('Using a balanced dataset')

        pos_count = {}
        neg_counter = {}

        for ct in cell_names:
            pos_count[ct] = 0
            neg_counter[ct] = 0

        samples = []

        # the list "samples" is a list of tuples each representing a sample. The first
        # entry is the row of the X matrix. The second is the index of the cell type.
        # The third is the index of the target in the y vector.

        for i, ct in enumerate(cell_names):
            ct_df = pd.read_csv(os.path.join('data_files', 'CT_enhancer_features_matrices',
                                             ct + '_enhancer_features_matrix.csv'), index_col='cCRE_id')
            ct_df = ct_df.loc[enhancers_list]

            for enhancer in enhancers_list:
                if ct_df.loc[enhancer, feature] == 1:
                    pos_count[ct] += 1

            for j, enhancer in enumerate(enhancers_list):
                if ct_df.loc[enhancer, 'active'] == 1 or ct_df.loc[enhancer, 'repressed'] == 1:
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
    print(len(cell_encodings))
    print(len(samples))

    """
    # TEST 1 : Adding motifs to the positive and/or negative sequences for testing purposes
    motif_pos = get_one_hot_encoding('AGTCGCTAGATCGATCGGCA')
    motif_neg = get_one_hot_encoding('AGCGTGCTAGATGGCTGCTG')
    for i in range(len(X)):
        # if y[i] == 1:
        #    X[i] = X[i][0:120] + motif_pos + X[i][140:501]
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
    # TEST 3 : Adding motifs to the positive and/or negative sequences for testing purposes
    # In this case, the motifs are cell specific to test the capacity of the model
    # to detect such motifs.
    
    motif1 = get_one_hot_encoding('AGAGAGAGAGAGAGAGAGAG')
    motif2 = get_one_hot_encoding('TCTCTCTCTCTCTCTCTCTC')
    motif3 = get_one_hot_encoding('AAAAAAAAGGGAAAAAAAAA')
    motif4 = get_one_hot_encoding('GGGGGGGGTCTGGGGGGGGG')
    motif5 = get_one_hot_encoding('CCCCCCCCCCTTTTTTTTTT')
    motif6 = get_one_hot_encoding('AAGGAAGGAAGGAAGGTTTT')
    motif7 = get_one_hot_encoding('ACTGACTGACTGACTGACTG')
    motif8 = get_one_hot_encoding('ACGTGATAGCTAGCTACGTA')
    motif9 = get_one_hot_encoding('AGTACTCATAAACTGCTAGA')
    motif10 = get_one_hot_encoding('ACCCCCCCTCCCCCCCCCGC')
    for i in range(len(X)):
        if y[i] == 1 and cell_encodings[i] == [1, 0, 0, 0, 0]:
            X[i] = X[i][0:120] + motif1 + X[i][140:501]
        if y[i] == 0 and cell_encodings[i] == [1, 0, 0, 0, 0]:
            X[i] = X[i][0:347] + motif2 + X[i][367:501]
        if y[i] == 1 and cell_encodings[i] == [0, 1, 0, 0, 0]:
            X[i] = X[i][0:113] + motif3 + X[i][133:501]
        if y[i] == 0 and cell_encodings[i] == [0, 1, 0, 0, 0]:
            X[i] = X[i][0:453] + motif4 + X[i][473:501]
        if y[i] == 1 and cell_encodings[i] == [0, 0, 1, 0, 0]:
            X[i] = X[i][0:1] + motif5 + X[i][21:501]
        if y[i] == 0 and cell_encodings[i] == [0, 0, 1, 0, 0]:
            X[i] = X[i][0:233] + motif6 + X[i][253:501]
        if y[i] == 1 and cell_encodings[i] == [0, 0, 0, 1, 0]:
            X[i] = X[i][0:411] + motif7 + X[i][431:501]
        if y[i] == 0 and cell_encodings[i] == [0, 0, 0, 1, 0]:
            X[i] = X[i][0:17] + motif8 + X[i][37:501]
        if y[i] == 1 and cell_encodings[i] == [0, 0, 0, 0, 1]:
            X[i] = X[i][0:403] + motif9 + X[i][423:501]
        if y[i] == 0 and cell_encodings[i] == [0, 0, 0, 0, 1]:
            X[i] = X[i][0:76] + motif10 + X[i][96:501]
    """
    """
     # TEST 4 (THE ULTIMATE TEST!!!) : Adding motifs to the positive and/or negative sequences for testing purposes
    # In this case, the motifs are cell specific to test the capacity of the model
    # to detect such motifs. However, differently than in Test 3, the same motif can have different
    # meaning in different cell types. A motif that indicates a positive sequence in cell type 1 
    # could be an indicator that the sequence is negative in cell type 2.
    
    motif1 = get_one_hot_encoding('GATAGCTAGCTCGATAGCGT')  # same as motif 3 and 10
    motif2 = get_one_hot_encoding('ATGATAGCTAGATCGATAGA')  # same as motif 8 and 9
    motif3 = get_one_hot_encoding('GATAGCTAGCTCGATAGCGT')
    motif4 = get_one_hot_encoding('CTCCGATGACTCGGATGCAC')  # same as motif 7
    motif5 = get_one_hot_encoding('TTCGATCGCTGATCGATCGA')  # unique
    motif6 = get_one_hot_encoding('TAGCTGGCTCGGAAACGCTG')  # unique
    motif7 = get_one_hot_encoding('CTCCGATGACTCGGATGCAC')
    motif8 = get_one_hot_encoding('ATGATAGCTAGATCGATAGA')
    motif9 = get_one_hot_encoding('ATGATAGCTAGATCGATAGA')
    motif10 = get_one_hot_encoding('GATAGCTAGCTCGATAGCGT')
    for i in range(len(X)):
        if y[i] == 1 and cell_encodings[i] == [1, 0, 0, 0, 0]:
            X[i] = X[i][0:120] + motif1 + X[i][140:501]
        if y[i] == 0 and cell_encodings[i] == [1, 0, 0, 0, 0]:
            X[i] = X[i][0:347] + motif2 + X[i][367:501]
        if y[i] == 1 and cell_encodings[i] == [0, 1, 0, 0, 0]:
            X[i] = X[i][0:113] + motif3 + X[i][133:501]
        if y[i] == 0 and cell_encodings[i] == [0, 1, 0, 0, 0]:
            X[i] = X[i][0:453] + motif4 + X[i][473:501]
        if y[i] == 1 and cell_encodings[i] == [0, 0, 1, 0, 0]:
            X[i] = X[i][0:1] + motif5 + X[i][21:501]
        if y[i] == 0 and cell_encodings[i] == [0, 0, 1, 0, 0]:
            X[i] = X[i][0:233] + motif6 + X[i][253:501]
        if y[i] == 1 and cell_encodings[i] == [0, 0, 0, 1, 0]:
            X[i] = X[i][0:411] + motif7 + X[i][431:501]
        if y[i] == 0 and cell_encodings[i] == [0, 0, 0, 1, 0]:
            X[i] = X[i][0:17] + motif8 + X[i][37:501]
        if y[i] == 1 and cell_encodings[i] == [0, 0, 0, 0, 1]:
            X[i] = X[i][0:403] + motif9 + X[i][423:501]
        if y[i] == 0 and cell_encodings[i] == [0, 0, 0, 0, 1]:
            X[i] = X[i][0:76] + motif10 + X[i][96:501]
    """
    X = torch.tensor(X, dtype=torch.float, device=device)
    X = X.permute(0, 2, 1)
    y = torch.tensor(y, dtype=torch.float, device=device)
    cell_encodings = torch.tensor(cell_encodings, dtype=torch.float, device=device)

    params = {'batch_size': BATCH_SIZE,
              'shuffle': True,
              'num_workers': 0}

    output = {'train_auc': [], 'val_auc': [], 'test_auc': [], 'tissues_used': cell_names}

    for seed in seeds:

        # Multi_CT_conv = MultiCTConvNet(device=device, num_cell_types=len(cell_names), seq_length=501,
        #                                 kernel_size=26, number_of_kernels=64, polling_window=7)

        convolution = SeqConvModule(device=device, seq_length=501, kernel_sizes=(16, 3, 3), num_of_kernels=(64, 64, 32),
                                    polling_windows=(3, 4), input_channels=4)
        fully_connected = FCModule(device=device, layer_sizes=(len(cell_encodings[0]) + 32, 32, 1))

        train_idx, test_idx = split_indices(samples, seed=0)
        train_idx, val_idx = split_indices(train_idx, seed=seed)

        train_set = IndicesDataset(train_idx)
        test_set = IndicesDataset(test_idx)
        val_set = IndicesDataset(val_idx)

        train_batch_gen = DataLoader(train_set, **params)
        test_batch_gen = DataLoader(test_set, **params)
        val_batch_gen = DataLoader(val_set, **params)

        loss_function = nn.BCELoss()
        if torch.cuda.is_available() and USE_CUDA:
            loss_function = loss_function.cuda()
        optimizer = torch.optim.Adam(list(convolution.parameters()) + list(fully_connected.parameters())
                                     , lr=LR)

        best_val_auc = 0
        early_stop_count = 0

        for epoch in range(epochs):
            print("Epoch " + str(epoch))
            for step, idx_batch in enumerate(tqdm(train_batch_gen)):
                optimizer.zero_grad()
                X_idx = idx_batch[0]
                cell_idx = idx_batch[1]
                y_idx = idx_batch[2]
                #y_hat = Multi_CT_conv(X[X_idx], cell_encodings[cell_idx])
                seq_features = convolution(X[X_idx])
                y_hat = fully_connected(torch.cat((seq_features, cell_encodings[cell_idx]), 1))
                error_loss = loss_function(y_hat, y[y_idx])
                error_loss.backward(retain_graph=True)
                optimizer.step()
            with torch.no_grad():
                print("Test performance on train set for epoch " + str(epoch))
                train_error_loss = 0.0
                all_train_targets = []
                all_train_predictions = []
                for step, idx_batch in enumerate(tqdm(train_batch_gen)):
                    X_idx = idx_batch[0]
                    cell_idx = idx_batch[1]
                    y_idx = idx_batch[2]
                    # y_hat = Multi_CT_conv(X[X_idx], cell_encodings[cell_idx])
                    seq_features = convolution(X[X_idx])
                    y_hat = fully_connected(torch.cat((seq_features, cell_encodings[cell_idx]), 1))
                    train_error_loss += float(loss_function(y_hat, y[y_idx]))
                    all_train_targets.extend(list(y[y_idx].detach().cpu().numpy()))
                    all_train_predictions.extend(list(y_hat.detach().cpu().numpy()))

                print("average error loss on train set : " + str(float(train_error_loss) / (step + 1)))
                fpr, tpr, _ = roc_curve(all_train_targets, all_train_predictions)
                train_roc_auc = auc(fpr, tpr)
                print('ROC AUC on train set : ' + str(train_roc_auc))

                print("Test performance on validation set for epoch " + str(epoch))
                val_error_loss = 0.0
                all_val_targets = []
                all_val_predictions = []
                for step, idx_batch in enumerate(tqdm(val_batch_gen)):
                    X_idx = idx_batch[0]
                    cell_idx = idx_batch[1]
                    y_idx = idx_batch[2]
                    # y_hat = Multi_CT_conv(X[X_idx], cell_encodings[cell_idx])
                    seq_features = convolution(X[X_idx])
                    y_hat = fully_connected(torch.cat((seq_features, cell_encodings[cell_idx]), 1))
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
            convolution.load_state_dict(best_convolution_state)
            fully_connected.load_state_dict(best_fc_state)

            print("Test performance on train set on best model")
            train_error_loss = 0.0
            all_train_targets = []
            all_train_predictions = []
            for step, idx_batch in enumerate(tqdm(train_batch_gen)):
                X_idx = idx_batch[0]
                cell_idx = idx_batch[1]
                y_idx = idx_batch[2]
                # y_hat = Multi_CT_conv(X[X_idx], cell_encodings[cell_idx])
                seq_features = convolution(X[X_idx])
                y_hat = fully_connected(torch.cat((seq_features, cell_encodings[cell_idx]), 1))
                train_error_loss += float(loss_function(y_hat, y[y_idx]))
                all_train_targets.extend(list(y[y_idx].detach().cpu().numpy()))
                all_train_predictions.extend(list(y_hat.detach().cpu().numpy()))

            print("average error loss on train set : " + str(float(train_error_loss) / (step + 1)))
            fpr, tpr, _ = roc_curve(all_train_targets, all_train_predictions)
            train_roc_auc = auc(fpr, tpr)
            print('ROC AUC on train set : ' + str(train_roc_auc))

            print("Test performance on validation set on best model:")
            val_error_loss = 0.0
            all_val_targets = []
            all_val_predictions = []
            for step, idx_batch in enumerate(tqdm(val_batch_gen)):
                X_idx = idx_batch[0]
                cell_idx = idx_batch[1]
                y_idx = idx_batch[2]
                # y_hat = Multi_CT_conv(X[X_idx], cell_encodings[cell_idx])
                seq_features = convolution(X[X_idx])
                y_hat = fully_connected(torch.cat((seq_features, cell_encodings[cell_idx]), 1))
                val_error_loss += float(loss_function(y_hat, y[y_idx]))
                all_val_targets.extend(list(y[y_idx].detach().cpu().numpy()))
                all_val_predictions.extend(list(y_hat.detach().cpu().numpy()))

            print("average error loss on validation set : " + str(float(val_error_loss) / (step + 1)))
            fpr, tpr, _ = roc_curve(all_val_targets, all_val_predictions)
            val_roc_auc = auc(fpr, tpr)
            print('ROC AUC on validation set : ' + str(val_roc_auc))

            print("Test performance on test set on best model")
            test_error_loss = 0.0
            all_test_targets = []
            all_test_predictions = []
            for step, idx_batch in enumerate(tqdm(test_batch_gen)):
                X_idx = idx_batch[0]
                cell_idx = idx_batch[1]
                y_idx = idx_batch[2]
                # y_hat = Multi_CT_conv(X[X_idx], cell_encodings[cell_idx])
                seq_features = convolution(X[X_idx])
                y_hat = fully_connected(torch.cat((seq_features, cell_encodings[cell_idx]), 1))
                test_error_loss += float(loss_function(y_hat, y[y_idx]))
                all_test_targets.extend(list(y[y_idx].detach().cpu().numpy()))
                all_test_predictions.extend(list(y_hat.detach().cpu().numpy()))

            print("average error loss on test set : " + str(float(test_error_loss) / (step + 1)))
            fpr, tpr, _ = roc_curve(all_test_targets, all_test_predictions)
            test_roc_auc = auc(fpr, tpr)
            print('ROC AUC on test set : ' + str(test_roc_auc))

        output['train_auc'].append(train_roc_auc)
        output['val_auc'].append(val_roc_auc)
        output['test_auc'].append(test_roc_auc)

    if not balanced:
        dir_name = feature + '_' + str(LR) + '_' + str(early_stop) + '_unbalanced'
    else:
        dir_name = feature + '_' + str(LR) + '_' + str(early_stop) + '_balanced'
    dir_path = os.path.join('results', 'multi_tissues_experiments', dir_name)
    os.makedirs(dir_path, exist_ok=True)

    with open(os.path.join(dir_path, 'auc_results.json'), 'w') as outfile:
        json.dump(output, outfile)

    torch.save({'convolution': convolution.state_dict(),
                'fully_connected': fully_connected.state_dict()},
               os.path.join(dir_path, 'model.pt'))

    encodings_output = {}
    for tissue, encoding in zip(cell_names, cell_encodings):
        encodings_output[tissue] = list(cell_encodings)

    with open(os.path.join(dir_path, 'encoding.json'), 'w') as outfile:
        json.dump(encodings_output, outfile)
