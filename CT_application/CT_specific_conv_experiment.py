import os
import pandas as pd
import argparse
from tqdm import tqdm
import numpy as np

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
    parser.add_argument('--ct', type=str, default='active.distal.CTCF-ascending_aorta')
    parser.add_argument('--LR', type=float, default=0.01)
    parser.add_argument('--USE-CUDA', type=bool, default=False)
    parser.add_argument('--BATCH-SIZE', type=int, default=8)
    args = parser.parse_args()

    LR = args.LR
    USE_CUDA = args.USE_CUDA
    BATCH_SIZE = args.BATCH_SIZE

    device = torch.device('cpu')

    samples_df = pd.read_csv(os.path.join('data_files', args.ct + '_samples.csv'), dtype=str)

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
            X.append(get_one_hot_encoding(samples_df.loc[row, 'sequences']))
            neg_counter += 1
        if int(samples_df.loc[row, 'labels']) == 1:
            y.append(int(samples_df.loc[row, 'labels']))
            X.append(get_one_hot_encoding(samples_df.loc[row, 'sequences']))
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
    CT_specific_conv = CTspecificConvNet(device=device, cell_type=args.ct, seq_length=501, kernel_size=24)

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

    for epoch in range(1):
        print("Epoch " + str(epoch))
        for step, idx_batch in enumerate(tqdm(train_batch_gen)):

            optimizer.zero_grad()
            # print(y[idx_batch])
            y_hat = CT_specific_conv(X[idx_batch])
            # print(y_hat)
            # error_loss = loss_function(y_hat, y[idx_batch])
            error_loss = loss_function(y_hat, y[idx_batch])

            print("error loss on batch: " + str(float(error_loss)))
            error_loss.backward(retain_graph=True)
            optimizer.step()
            # print(CT_specific_conv.convLayer.weight)
            print(torch.max(CT_specific_conv.convLayer.weight.grad))
            test_loss2 = loss_function(CT_specific_conv(X[test_idx]), y[test_idx])
            print("error loss on test set after optimization: " + str(float(test_loss2)))

    print(CT_specific_conv.convLayer.weight.size())
