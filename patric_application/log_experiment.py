import math
import os
import json
import argparse
import jsonpickle
import pandas as pd
from build_parent_child_mat import build_pc_mat
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
import time

# imports from dag tutorial
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from models.baseline_models import LinRegModel
from utils.model_utils import build_parent_path_mat, split_indices, IndicesDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000, metavar='N')
    parser.add_argument('--early-stopping', type=int, default=5, metavar='E',
                        help='Number of epochs without improvement before early stopping')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4], metavar='S',
                        help='random seed for train/test/validation split (default: [0,1,2,3,4])')
    parser.add_argument('--save-seed', type=int, default=[], metavar='SS',
                        help='seeds for which the training (AUC score) will be plotted and saved')
    parser.add_argument('--validation-interval', type=int, default=1, metavar='VI')
    parser.add_argument('--p', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR')
    parser.add_argument('--l1', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--runtest', dest='runtest', action='store_true')
    parser.add_argument('--no-runtest', dest='runtest', action='store_false')
    parser.set_defaults(runtest=False)
    parser.add_argument('--group', type=str, default='Firmicutes')
    parser.add_argument('--antibiotic', type=str, default='erythromycin')
    parser.add_argument('--threshold', type=str, default='0.0')
    parser.add_argument('--output-path', type=str, default=os.path.join('data_files', 'output.json'),
                        metavar='OUT', help='file where the ROC AUC score of the model will be outputted')
    args = parser.parse_args()

    # annotating leaves with labels and features
    samples_file = args.group + '_' + args.antibiotic + '_' + args.threshold + '_samples.csv'
    samples_file = os.path.join('data_files', 'subproblems', args.group + '_' + args.antibiotic, samples_file)

    if os.path.isfile(samples_file):
        samples_df = pd.read_csv(samples_file, dtype=str)
    else:
        print('The samples file does not exist.')
        exit()

    samples_df = pd.read_csv(samples_file, dtype=str)

    # flag to use CUDA gpu if available
    USE_CUDA = True
    print('Using CUDA: ' + (str(torch.cuda.is_available() and USE_CUDA)))
    device = torch.device("cuda:0" if torch.cuda.is_available() and USE_CUDA else "cpu")

    # some other hyper-parameters for training
    LR = args.lr
    L1 = args.l1
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs

    X = []
    y = []

    for row in samples_df.itertuples():
        phenotype = eval(getattr(row, 'Phenotype'))[0]  # the y value
        y.append(phenotype)
        features = eval(getattr(row, 'Features'))  # the x value
        X.append(features)

    test_auc = []
    val_auc = []

    average_time_seed = 0  # to test time performance of the training of this model

    for s in args.seeds:
        init_time = time.time()
        # simple linear model to which a sigmoid will be applied in order to make it a logistic model
        # Used for comparison with DendroNet performance
        # We use a linear regression in order to be able to use BCEWithLogitsLoss as loss function,
        # a more stable version of BCEloss

        logistic = LinRegModel(len(X[0]))
        logistic.to(device)
        best_weights = logistic.lin_1

        train_idx, test_idx = split_indices(range(len(X)), seed=0)
        train_idx, val_idx = split_indices(train_idx, seed=s)

        # creating idx dataset objects for batching
        train_set = IndicesDataset(train_idx)
        val_set = IndicesDataset(val_idx)
        test_set = IndicesDataset(test_idx)

        # Setting some parameters for shuffle batch
        params = {'batch_size': BATCH_SIZE,
                  'shuffle': True,
                  'num_workers': 0}

        train_batch_gen = torch.utils.data.DataLoader(train_set, **params)
        val_batch_gen = torch.utils.data.DataLoader(val_set, **params)
        test_batch_gen = torch.utils.data.DataLoader(test_set, **params)

        # converting X and y to tensors, and transferring to GPU if the cuda flag is set
        X = torch.tensor(X, dtype=torch.double, device=device)
        y = torch.tensor(y, dtype=torch.double, device=device)

        # creating the loss function and optimizer
        loss_function = nn.BCEWithLogitsLoss()  # loss function for comparison model
        if torch.cuda.is_available() and USE_CUDA:
            loss_function = loss_function.cuda()
        optimizer = torch.optim.SGD(logistic.parameters(), lr=LR)

        # print("train:", 100*len(train_idx)/(len(train_idx)+len(test_idx)),"%")
        # print("val:", 100*len(val_idx)/(len(train_idx)+ len(val_idx)+ len(test_idx)),"%")
        # print("test:", 100*len(test_idx)/(len(train_idx)+ len(val_idx)+len(test_idx)),"%")

        best_auc = 0.0
        early_stopping_count = 0
        #aucs_for_plot = []

        all_y_train_idx = []
        for idx in train_idx:
            all_y_train_idx.append(idx)
        y_train_targets = y[all_y_train_idx].detach().cpu().numpy()  # target values for whole training set (useful to compute training AUC at each epoch)

        logistic = logistic.double()

        # running the training loop
        for epoch in range(EPOCHS):
            print('Train epoch ' + str(epoch))
            # we'll track the running loss over each batch so we can compute the average per epoch
            running_loss = 0.0
            # getting a batch of indices
            for step, idx_batch in enumerate(tqdm(train_batch_gen)):
                optimizer.zero_grad()
                y_hat = logistic.forward(X[idx_batch])
                error_loss = loss_function(y_hat, y[idx_batch].squeeze())  # idx_batch is also used to fetch the appropriate entries from y
                #computing L1 loss
                l1_loss = 0.0
                for w in logistic.lin_1.weight[0]:
                    l1_loss += abs(float(w))
                loss = error_loss + (L1*l1_loss)
                running_loss += float(loss)
                loss.backward(retain_graph=True)
                optimizer.step()
            step += 1
            print('Average training loss for epoch: ', str(running_loss / step))

            #Compute L1 after all batches of training of the epoch
            l1_loss = 0.0
            for w in logistic.lin_1.weight[0]:
                l1_loss += abs(float(w))

            # train set after weights are update
            y_train_pred = (torch.sigmoid(logistic.forward(X[all_y_train_idx]))).detach().cpu().numpy()  # predicted values (after sigmoid) for whole train set

            fpr, tpr, _ = roc_curve(y_train_targets, y_train_pred)
            roc_auc = auc(fpr, tpr)
            print("Training ROC AUC for epoch: ", roc_auc)

            # Test performance using validation set at each epoch
            with torch.no_grad():
                val_loss = 0.0
                y_true = []
                y_pred = []
                for step, idx_batch in enumerate(val_batch_gen):
                    y_hat = logistic.forward(X[idx_batch])
                    if y_hat.size() == torch.Size([]):
                        y_hat = torch.unsqueeze(y_hat, 0)
                    val_error_loss = loss_function(y_hat, y[idx_batch])
                    val_loss += val_error_loss + (L1*l1_loss)
                    y_t = list(y[idx_batch].detach().cpu().numpy())  # true values for this batch
                    y_p = list(torch.sigmoid(y_hat).detach().cpu().numpy())  # predictions for this batch
                    y_true.extend(y_t)
                    y_pred.extend(y_p)
                fpr, tpr, _ = roc_curve(y_true, y_pred)
                roc_auc = auc(fpr, tpr)
                step += 1
                print('Average loss on the validation set on this epoch: ', float(val_loss) / step)
                print("ROC AUC for epoch: ", roc_auc)

                #aucs_for_plot.append(roc_auc)

                if roc_auc > best_auc:  # Check if performance has increased on validation set (loss is decreasing)
                    best_auc = roc_auc
                    early_stopping_count = 0
                    best_weights = logistic.lin_1.weight.detach().clone()
                    print(best_weights)
                    print("Improvement!!!")
                else:
                    early_stopping_count += 1
                    print("Oups,... we are at " + str(early_stopping_count) + ", best: " + str(best_auc))

                if early_stopping_count > args.early_stopping:  # If performance has not increased for long enough, we stop training
                    print("EARLY STOPPING!")  # to avoid overfitting
                    break

        val_auc.append(roc_auc)

        # With training complete, we'll run the test set
        with torch.no_grad():
            print(logistic.lin_1.weight)
            if USE_CUDA and torch.cuda.is_available():
                del logistic.lin_1.weight
                torch.cuda.empty_cache()
            best_logistic = LinRegModel(len(X[0]), use_given_weights=True, input_weights=best_weights)
            best_logistic.lin_1.weight

            y_true = []
            y_pred = []
            test_loss = 0.0
            for step, idx_batch in enumerate(test_batch_gen):
                y_hat = best_logistic.forward(X[idx_batch])
                if y_hat.size() == torch.Size([]):
                    y_hat = torch.unsqueeze(y_hat, 0)
                test_error_loss = loss_function(y_hat, y[idx_batch])
                test_loss += test_error_loss + (L1*l1_loss)
                y_t = list(y[idx_batch].detach().cpu().numpy())
                y_p = list(torch.sigmoid(y_hat).detach().cpu().numpy())
                y_true.extend(y_t)
                y_pred.extend(y_p)

            fpr, tpr, _ = roc_curve(y_true, y_pred)
            roc_auc = auc(fpr, tpr)
            step += 1
            print("ROC AUC for test:", roc_auc)
            print('Average BCE loss on test set:', float(test_loss) / step)

            test_auc.append(roc_auc)

    output_dict = {'val_auc': val_auc, 'test_auc': test_auc}

    fileName = args.output_path
    os.makedirs(os.path.dirname(fileName), exist_ok=True)
    with open(fileName, 'w') as outfile:
        json.dump(output_dict, outfile)

    final_time = time.time() - init_time
    average_time_seed += final_time

    average_time_seed = average_time_seed / len(args.seeds)
    print('Average time to train a model: ' + str(average_time_seed) + 'seconds')

    os.makedirs(os.path.join('data_files', 'time_performances'), exist_ok=True)
    time_file = os.path.join('data_files', 'time_performances', 'logistic_' + args.group + '_' + args.antibiotic)
    with open(time_file, 'w') as file:
        json.dump({'average_per_seed': average_time_seed}, file)

    """"
            true_pos = 0
            false_pos = 0
            false_neg = 0
            true_neg = 0
            total = len(y_true)

            for i in range(total):
                pred = float(y_pred[i])
                real = float(y_true[i])
                if (pred > 0.5 and real == 1.0):
                    true_pos += 1
                elif (pred > 0.5 and real == 0.0):
                    false_pos += 1
                elif (pred < 0.5 and real == 0.0):
                    true_neg += 1
                elif (pred < 0.5 and real == 1.0):
                    false_neg += 1

            print("accuracy: ", (true_pos + true_neg) / total)
            print("sensitivity: ", true_pos / (true_pos + false_neg))
            print("specificity: ", true_neg / (true_neg + false_pos))
            print("true positives: ", true_pos)
            print("true negatives: ", true_neg)
            print("false positives: ", false_pos)
            print("false negatives: ", false_neg)
            

    
    
    plt.plot(aucs_for_plot)
    plt.show()
    _, file_info = os.path.split(args.label_file)
    antibiotic = file_info.split('_')[0]
    group = file_info.split('_')[1]
    os.makedirs(os.path.join('data_files', 'AUC_plots'), exist_ok=True)
    if s in args.save_seed:
        plt.savefig(os.path.join('data_files', 'AUC_plots', antibiotic + '_' + group + '_' \
                                 + str(args.lr) + '_' + str(args.dpf) + '_' + str(args.l1) + '_' + str(
            args.early_stopping) \
                                 + '_seed_' + str(s) + '.png'))
    """


