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
from models.CT_conv_model import MultiCTConvNet
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
    parser.add_argument('--cts', type=str, nargs='+', default=['vagina', 'adrenal_gland', 'prostate_gland',
                                                               'sigmoid_colon', 'testis', 'stomach',
                                                               'uterus', 'tibial_nerve', 'spleen'],
                        help='if empty, we take all available datasets')
    parser.add_argument('--feature', type=str, default='active')
    parser.add_argument('--LR', type=float, default=0.001)
    parser.add_argument('--USE-CUDA', type=bool, default=False)
    parser.add_argument('--BATCH-SIZE', type=int, default=128)
    parser.add_argument('--whole-dataset', type=bool, default=False)
    args = parser.parse_args()

    feature = args.feature
    LR = args.LR
    BATCH_SIZE = args.BATCH_SIZE
    USE_CUDA = args.USE_CUDA
    whole_dataset = args.whole_dataset
    cell_names = args.cts

    if not cell_names:
        print('using all cell types')
        cell_names = []
        for ct_file in os.listdir(os.path.join('data_files', 'CT_enhancer_features_matrices')):
            ct_name = ct_file[0:-29]
            cell_names.append(ct_name)

    device = torch.device("cuda:0" if (torch.cuda.is_available() and USE_CUDA) else "cpu")

    with open(os.path.join('data_files', 'enhancers_seqs.json'), 'r') as enhancers_file:
        enhancers_dict = json.load(enhancers_file)

    X = []
    y = []
    cell_encodings = []

    enhancers_list = list(enhancers_dict.keys())
    num_enhancers = len(enhancers_list)
    print(enhancers_list[0])
    print(len(enhancers_list))
    print(cell_names)

    for ct in cell_names:
        ct_df = pd.read_csv(os.path.join('data_files', 'CT_enhancer_features_matrices',
                                         ct + '_enhancer_features_matrix.csv'), index_col='cCRE_id')
        ct_df = ct_df.loc[enhancers_list]
        y.extend(list(ct_df.loc[:, feature]))

    print(len(y))

    for enhancer in enhancers_list:
        X.append(get_one_hot_encoding(enhancers_dict[enhancer]))

    for ct in range(len(cell_names)):
        cell_encodings.append([1 if j == ct else 0 for j in range(len(cell_names))])

    print(cell_encodings)
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
    print(samples[0:100])

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

    Multi_CT_conv = MultiCTConvNet(device=device, num_cell_types=len(args.cts), seq_length=501,
                                   kernel_size=26, number_of_kernels=64, polling_window=7)

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
    print(torch.sum(y))
    cell_encodings = torch.tensor(cell_encodings, dtype=torch.float, device=device)

    loss_function = nn.BCELoss()
    if torch.cuda.is_available() and USE_CUDA:
        loss_function = loss_function.cuda()
    optimizer = torch.optim.Adam(Multi_CT_conv.parameters(), lr=LR)

    for epoch in range(5):
        print("Epoch " + str(epoch))
        for step, idx_batch in enumerate(tqdm(train_batch_gen)):
            optimizer.zero_grad()
            X_idx = idx_batch[0]
            cell_idx = idx_batch[1]
            y_hat = Multi_CT_conv(X[X_idx], cell_encodings[cell_idx])
            y_idx = []
            for x, c in zip(X_idx, cell_idx):
                y_idx.append(c * num_enhancers + x)
            y_idx = torch.tensor(y_idx, dtype=torch.long, device=device)
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
                y_hat = Multi_CT_conv(X[X_idx], cell_encodings[cell_idx])
                y_idx = []
                for x, c in zip(X_idx, cell_idx):
                    y_idx.append(c * num_enhancers + x)
                y_idx = torch.tensor(y_idx, dtype=torch.long, device=device)
                print(y_idx.size())
                train_error_loss += float(loss_function(y_hat, y[y_idx]))
                all_train_targets.extend(list(y[y_idx].detach().cpu().numpy()))
                all_train_predictions.extend(list(y_hat.detach().cpu().numpy()))

            print("average error loss on test set : " + str(float(train_error_loss) / (step + 1)))
            fpr, tpr, _ = roc_curve(all_train_targets, all_train_predictions)
            roc_auc = auc(fpr, tpr)
            print('ROC AUC on train set : ' + str(roc_auc))

            print("Test performance on test set for epoch " + str(epoch))
            test_error_loss = 0.0
            all_test_targets = []
            all_test_predictions = []
            for step, idx_batch in enumerate(tqdm(test_batch_gen)):
                X_idx = idx_batch[0]
                cell_idx = idx_batch[1]
                y_hat = Multi_CT_conv(X[X_idx], cell_encodings[cell_idx])
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
