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
from models.CT_conv_model import CTspecificConvNet
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
    parser.add_argument('--ct', type=str, default='active.distal.nonCTCF.nonAS-adrenal_gland')
    parser.add_argument('--LR', type=float, default=0.001)
    parser.add_argument('--USE-CUDA', type=bool, default=False)
    parser.add_argument('--BATCH-SIZE', type=int, default=128)
    args = parser.parse_args()

    LR = args.LR
    USE_CUDA = args.USE_CUDA
    BATCH_SIZE = args.BATCH_SIZE

    device = torch.device('cpu')

    samples_df = pd.read_csv(os.path.join('data_files', 'single_cell_datasets', args.ct + '_samples.csv'), dtype=str)

    with open(os.path.join('data_files', 'enhancers_seqs.json'), 'r') as e_file:
        enhancers_dict = json.load(e_file)

    y = []
    X = []
    pos_count = 0
    neg_counter = 0

    for row in range(samples_df.shape[0]):
        if int(samples_df.loc[row, 'labels']) == 1:
            pos_count += 1

    for row in range(samples_df.shape[0]):
        if int(samples_df.loc[row, 'labels']) == 0 and neg_counter < pos_count:
            y.append(int(samples_df.loc[row, 'labels']))
            X.append(get_one_hot_encoding(enhancers_dict[samples_df.loc[row, 'IDs']]))
            neg_counter += 1
        if int(samples_df.loc[row, 'labels']) == 1:
            y.append(int(samples_df.loc[row, 'labels']))
            X.append(get_one_hot_encoding(enhancers_dict[samples_df.loc[row, 'IDs']]))

    print(len(X))
    print(len(X[0]))
    print(len(X[1]))
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

    print(len(X))
    print(len(y))

    CT_specific_conv = CTspecificConvNet(device=device, cell_type=args.ct, seq_length=501,
                                         kernel_size=24, num_of_kernels=64)

    train_idx, test_idx = split_indices(range(0, len(X)), seed=0)

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
    optimizer = torch.optim.Adam(CT_specific_conv.parameters(), lr=LR)

    for epoch in range(10):
        print("Epoch " + str(epoch))
        for step, idx_batch in enumerate(tqdm(train_batch_gen)):
            optimizer.zero_grad()
            # print(y[idx_batch])
            y_hat = CT_specific_conv(X[idx_batch])
            # print(y_hat)
            # error_loss = loss_function(y_hat, y[idx_batch])
            error_loss = loss_function(y_hat, y[idx_batch])
            #print("error loss on batch: " + str(float(error_loss)))
            error_loss.backward(retain_graph=True)
            optimizer.step()
            # print(CT_specific_conv.convLayer.weight)
            # print(torch.max(CT_specific_conv.convLayer.weight.grad))

        print("Test performance for epoch " + str(epoch))
        all_test_targets = []
        all_test_predictions = []
        test_error_loss = 0.0
        for step, idx_batch in enumerate(tqdm(test_batch_gen)):
            y_hat = CT_specific_conv(X[idx_batch])
            test_error_loss += float(loss_function(y_hat, y[idx_batch]))
            all_test_targets.extend(list(y[idx_batch].detach().cpu().numpy()))
            all_test_predictions.extend(list(y_hat.detach().cpu().numpy()))

        print("average error loss on test set : " + str(float(test_error_loss) / (step + 1)))
        fpr, tpr, _ = roc_curve(all_test_targets, all_test_predictions)
        roc_auc = auc(fpr, tpr)

        print('ROC AUC for test set : ' + str(roc_auc))


