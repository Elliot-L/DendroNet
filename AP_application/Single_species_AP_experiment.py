import os
import pandas as pd
import json
import argparse
from tqdm import tqdm
import torch.nn as nn
import torch
from sklearn.metrics import roc_curve, auc
import torch.optim
from torch.utils.data.dataloader import DataLoader

from models.CT_conv_model import SeqConvModule, FCModule
from utils.model_utils import split_indices, IndicesDataset


def amino_acid_encoding(order, seq):
    new_seq = []
    for aa in seq:
        new_seq.append([1 if aa == aa_opt else 0 for aa_opt in order])
    return new_seq


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--species', type=str, default='Escherichia coli')
    parser.add_argument('--LR', type=float, default=0.001)
    parser.add_argument('--GPU', default=True, action='store_true')
    parser.add_argument('--CPU', dest='GPU', action='store_false')
    parser.add_argument('--BATCH-SIZE', type=int, default=128)
    parser.add_argument('--balanced', default=True, action='store_true')
    parser.add_argument('--unbalanced', dest='balanced', action='store_false')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1])
    parser.add_argument('--early-stopping', type=int, default=3)
    parser.add_argument('--num-epochs', type=int, default=100)

    args = parser.parse_args()

    species = args.species
    LR = args.LR
    BATCH_SIZE = args.BATCH_SIZE
    USE_CUDA = args.GPU
    balanced = args.balanced
    seeds = args.seeds
    early_stop = args.early_stopping
    epochs = args.num_epochs

    print('Using CUDA: ' + str(torch.cuda.is_available() and USE_CUDA))
    device = torch.device("cuda:0" if (torch.cuda.is_available() and USE_CUDA) else "cpu")

    with open(os.path.join('data_files', 'amino_acids.json'), 'r') as aa_file:
        aa_list = json.load(aa_file)
    print(aa_list)

    with open(os.path.join('data_files', 'peptide_seqs.json'), 'r') as p_file:
        peptide_dict = json.load(p_file)

    species_file = os.path.join('data_files', 'species_datasets',
                                species.split(' ')[0] + '_' + species.split(' ')[1] + '_dataset.csv')
    species_df = pd.read_csv(species_file)
    print(species_df)

    X = []
    y = []
    pos_count = 0
    total_count = 0

    for row in range(species_df.shape[0]):
        X.append(amino_acid_encoding(aa_list, peptide_dict[species_df.loc[row, 'IDs']]))
        y.append(species_df.loc[row, 'label'])
        if species_df.loc[row, 'label'] == 1:
            pos_count += 1
        total_count += 1
        
    print(len(X))
    print(X[0])
    print(len(y))

    output = {'train_auc': [], 'val_auc': [], 'test_auc': [], 'pos_ratio': (pos_count / total_count), 'epochs': []}

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

        convolution = SeqConvModule(device=device, seq_length=101, kernel_sizes=(5,), num_of_kernels=(128,),
                                    polling_windows=(), input_channels=len(aa_list))

        train_idx, test_idx = split_indices(range(len(X)), seed=0)
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
        optimizer = torch.optim.Adam(list(convolution.parameters()) + list(fully_connected.parameters()),
                                     lr=LR)

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

    dir_path = os.path.join('results', 'single_tissue_experiments', args.tissue, dir_name)
    os.makedirs(dir_path, exist_ok=True)

    with open(os.path.join(dir_path, 'output.json'), 'w') as outfile:
        json.dump(output, outfile)

    torch.save({'convolution': convolution.state_dict(),
                'fully_connected': fully_connected.state_dict()},
               os.path.join(dir_path, 'model.pt'))


