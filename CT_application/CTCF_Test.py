import numpy as np
import os
from models.CT_conv_model import CTspecificConvNet, SimpleCTspecificConvNet
import torch
from utils.model_utils import split_indices, IndicesDataset
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc



if __name__ == "__main__":

    USE_CUDA = True
    BATCH_SIZE = 32
    LR = 0.001

    device = torch.device("cuda:0" if (torch.cuda.is_available() and USE_CUDA) else "cpu")

    feature_array = np.load(os.path.join('data_files', 'dataset_201_chr20_CTCF_H1-hESC_X_train.npy'))
    target_array = np.load(os.path.join('data_files', 'dataset_201_chr20_CTCF_H1-hESC_y_train.npy'))
    print(feature_array.shape)
    print(target_array.shape)
    print(target_array)

    X = []
    y = []

    i = 0
    for seqs in feature_array:
        X.append([])
        rep_seq = ''
        for nuc in seqs[0]:
            if nuc == 0:
                X[i].append([1, 0, 0, 0, 0, 0])
                rep_seq += '-'
            elif nuc == 1:
                X[i].append([0, 1, 0, 0, 0, 0])
                rep_seq += 'A'
            elif nuc == 2:
                X[i].append([0, 0, 1, 0, 0, 0])
                rep_seq += 'C'
            elif nuc == 3:
                X[i].append([0, 0, 0, 1, 0, 0])
                rep_seq += 'G'
            elif nuc == 4:
                X[i].append([0, 0, 0, 0, 1, 0])
                rep_seq += 'N'
            elif nuc == 5:
                X[i].append([0, 0, 0, 0, 0, 1])
                rep_seq += 'T'
        i += 1

    for target in target_array:
        y.append(target[1])

    model = SimpleCTspecificConvNet(cell_type='test', device=device, seq_length=201, kernel_size=16,
                                    num_of_kernels=128, polling_window=0, initial_channels=6)
    #model = CTspecificConvNet(cell_type='test', device=device, seq_length=201, kernel_size=(16, 3, 3),
    #                         num_of_kernels=(128, 64, 32), polling_window=(3, 3),
    #                          initial_channels=6)

    train_idx, test_idx = split_indices(range(len(X)), seed=0)

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

    print(X.size())
    print(y.size())

    loss_function = nn.BCELoss()
    if torch.cuda.is_available() and USE_CUDA:
        loss_function = loss_function.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(10):
        print("Epoch " + str(epoch))
        print('Train on all batches:')
        for step, idx_batch in enumerate(tqdm(train_batch_gen)):
            optimizer.zero_grad()
            # print(y[idx_batch])
            y_hat = model(X[idx_batch])
            # print(y_hat)
            # error_loss = loss_function(y_hat, y[idx_batch])
            error_loss = loss_function(y_hat, y[idx_batch])
            # print("error loss on batch: " + str(float(error_loss)))
            error_loss.backward(retain_graph=True)
            optimizer.step()
            # print(CT_specific_conv.convLayer.weight)
            # print(torch.max(CT_specific_conv.convLayer.weight.grad))
        with torch.no_grad():
            print("Test performance on train set for epoch " + str(epoch))
            all_train_targets = []
            all_train_predictions = []
            train_error_loss = 0.0
            for step, idx_batch in enumerate(tqdm(train_batch_gen)):
                y_hat = model(X[idx_batch])
                train_error_loss += float(loss_function(y_hat, y[idx_batch]))
                all_train_targets.extend(list(y[idx_batch].detach().cpu().numpy()))
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
                y_hat = model(X[idx_batch])
                test_error_loss += float(loss_function(y_hat, y[idx_batch]))
                all_test_targets.extend(list(y[idx_batch].detach().cpu().numpy()))
                all_test_predictions.extend(list(y_hat.detach().cpu().numpy()))

            print("average error loss on test set : " + str(float(test_error_loss) / (step + 1)))
            fpr, tpr, _ = roc_curve(all_test_targets, all_test_predictions)
            roc_auc = auc(fpr, tpr)
            print('ROC AUC on test set : ' + str(roc_auc))
