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
import copy

from models.CT_conv_model import SeqConvModule, FCModule
from utils.model_utils import split_indices, IndicesDataset
from build_pc_mat import build_pc_mat

def amino_acid_encoding(order, seq):
    new_seq = []
    for aa in seq:
        new_seq.append([1 if aa == aa_opt else 0 for aa_opt in order])
    return new_seq


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--species-list', type=str, nargs='+', default=[],
                        help='if empty, take all species')
    parser.add_argument('--LR', type=float, default=0.001)
    parser.add_argument('--GPU', default=True, action='store_true')
    parser.add_argument('--CPU', dest='GPU', action='store_false')
    parser.add_argument('--balanced', default=True, action='store_true')
    parser.add_argument('--unbalanced', dest='balanced', action='store_false')
    parser.add_argument('--BATCH-SIZE', type=int, default=128)
    parser.add_argument('--seeds', type=int, nargs='+', default=[1])
    parser.add_argument('--early-stopping', type=int, default=3)
    parser.add_argument('--num-epochs', type=int, default=100)

    args = parser.parse_args()

    species_list = args.species_list
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

    species_dfs = {}

    if not species_list:
        for species_file in os.listdir(os.path.join('data_files', 'species_datasets')):
            species_name = species_file[0:-12]
            species_name = species_name.split('_')[0] + ' ' + species_name.split('_')[1]
            species_list.append(species_name)
            df = pd.read_csv(os.path.join('data_files', 'species_datasets', species_file))
            species_dfs[species_name] = df
    else:
        for species_name in species_list:
            # species_name = species_name.split(' ')[0] + '_' + species_name.split(' ')[1]
            df = pd.read_csv(os.path.join('data_files', 'species_datasets',
                                          species_name.split(' ')[0] + '_' + species_name.split(' ')[1]
                                          + '_dataset.csv'))
            species_dfs[species_name] = df

    print(species_list)

    pc_mat, nodes = build_pc_mat(species_list)
    old_species_list = species_list
    species_list = []
    not_used_species = []

    num_internal_nodes = 0

    for species in nodes:
        if species in old_species_list:
            species_list.append(species)

    for species in old_species_list:
        if species not in species_list:
            not_used_species.append(species)
    print('Species not used: ')
    print(not_used_species)
    print('Species used: ')
    print(species_list)


    X = []
    y = []
    species_encodings = [[1 if species == s else 0 for s in species_list] for species in species_list]

    pos_count = {s: 0 for s in species_list}
    neg_count = {s: 0 for s in species_list}
    pos_counter = {s: 0 for s in species_list}
    neg_counter = {s: 0 for s in species_list}

    total_count = species_dfs[species_list[0]].shape[0]

    for species in species_list:
        for row in range(total_count):
            if species_dfs[species].loc[row, 'label'] == 1:
                pos_count[species] += 1
            else:
                neg_count[species] += 1

    print(pos_count)
    print(neg_count)
    print(total_count)

    for ID in list(species_dfs[species_list[0]].loc[:, 'IDs']):
        X.append(amino_acid_encoding(aa_list, peptide_dict[ID]))

    if not balanced:
        print('Unbalanced dataset')

        for s in species_list:
            y.extend(list(species_dfs[s].loc[:, 'label']))

        samples = []

        # the samples are typles of (row in X, row in y, encoding position)

        for i, species in enumerate(species_list):
            df = species_dfs[species]
            for row in range(total_count):
                samples.append((row, i, i*total_count + row))

        neg_counter = neg_count
        pos_counter = pos_count

    else:
        print('Balanced dataset')

        samples = []

        # the samples are typles of (row in X, encoding position, row in y)

        for i, species in enumerate(species_list):
            if pos_count[species]/total_count <= 0.5:
                for row in range(total_count):
                    if species_dfs[species].loc[row, 'label'] == 1:
                        samples.append((row, i, len(y)))
                        y.append(1)
                        pos_counter[species] += 1
                    if species_dfs[species].loc[row, 'label'] == 0 and neg_counter[species] < pos_count[species]:
                        samples.append((row, i, len(y)))
                        y.append(0)
                        neg_counter[species] += 1
            else:
                for row in range(total_count):
                    if species_dfs[species].loc[row, 'label'] == 1 and pos_counter[species] < neg_count[species]:
                        samples.append((row, i, len(y)))
                        y.append(1)
                        pos_counter[species] += 1
                    if species_dfs[species].loc[row, 'label'] == 0:
                        samples.append((row, i, len(y)))
                        y.append(0)
                        neg_counter[species] += 1

    print(pos_counter)
    print(neg_counter)
    print(len(X))
    print(len(X[0]))
    print(len(y))
    print(len(samples))

    output = {'train_auc': [], 'val_auc': [], 'test_auc': [],
              'train_loss': [], 'val_loss': [], 'test_loss': [],
              'epochs': [], 'species_used': species_list}

    X = torch.tensor(X, dtype=torch.float, device=device)
    X = X.permute(0, 2, 1)
    y = torch.tensor(y, dtype=torch.float, device=device)
    species_encodings = torch.tensor(species_encodings, dtype=torch.float, device=device)

    params = {'batch_size': BATCH_SIZE,
              'shuffle': True,
              'num_workers': 0}

    for seed in seeds:
        # CT_specific_conv = SimpleCTspecificConvNet(cell_type='test', device=device, seq_length=501, kernel_size=16,
        #                                num_of_kernels=128, polling_window=0, initial_channels=4)
        # CT_specific_conv = CTspecificConvNet(device=device, cell_type=args.ct, seq_length=501,
        #                                      kernel_size=(16, 3, 3), num_of_kernels=(64, 64, 32),
        #                                      polling_window=(3, 4))

        fully_connected = FCModule(device=device, layer_sizes=(64 + len(species_list), 32, 1))

        convolution = SeqConvModule(device=device, seq_length=101, kernel_sizes=(5,), num_of_kernels=(64,),
                                    polling_windows=(), input_channels=len(aa_list))

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
        optimizer = torch.optim.Adam(list(convolution.parameters()) + list(fully_connected.parameters()),
                                     lr=LR)

        best_val_auc = 0
        early_stop_count = 0

        for epoch in range(epochs):
            print("Epoch " + str(epoch))
            for step, idx_batch in enumerate(tqdm(train_batch_gen)):
                optimizer.zero_grad()
                X_idx = idx_batch[0]
                species_idx = idx_batch[1]
                y_idx = idx_batch[2]
                # print(y[idx_batch])
                # y_hat = CT_specific_conv(X[idx_batch])
                seq_features = convolution(X[X_idx])
                y_hat = fully_connected(torch.cat((seq_features, species_encodings[species_idx]), 1))
                error_loss = loss_function(y_hat, y[y_idx])
                error_loss.backward(retain_graph=True)
                optimizer.step()
            with torch.no_grad():
                print("Test performance on train set for epoch " + str(epoch))
                all_train_targets = []
                all_train_predictions = []
                train_error_loss = 0.0
                for step, idx_batch in enumerate(tqdm(train_batch_gen)):
                    # y_hat = CT_specific_conv(X[idx_batch])
                    X_idx = idx_batch[0]
                    species_idx = idx_batch[1]
                    y_idx = idx_batch[2]
                    seq_features = convolution(X[X_idx])
                    y_hat = fully_connected(torch.cat((seq_features, species_encodings[species_idx]), 1))
                    train_error_loss += float(loss_function(y_hat, y[y_idx]))
                    all_train_targets.extend(list(y[y_idx].detach().cpu().numpy()))
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
                    X_idx = idx_batch[0]
                    species_idx = idx_batch[1]
                    y_idx = idx_batch[2]
                    seq_features = convolution(X[X_idx])
                    y_hat = fully_connected(torch.cat((seq_features, species_encodings[species_idx]), 1))
                    val_error_loss += float(loss_function(y_hat, y[y_idx]))
                    all_val_targets.extend(list(y[y_idx].detach().cpu().numpy()))
                    all_val_predictions.extend(list(y_hat.detach().cpu().numpy()))

                print("average error loss on val set : " + str(float(val_error_loss) / (step + 1)))
                print(len(all_val_predictions))
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
                X_idx = idx_batch[0]
                species_idx = idx_batch[1]
                y_idx = idx_batch[2]
                # y_hat = CT_specific_conv(X[idx_batch])
                seq_features = convolution(X[X_idx])
                y_hat = fully_connected(torch.cat((seq_features, species_encodings[species_idx]), 1))
                train_error_loss += float(loss_function(y_hat, y[y_idx]))
                all_train_targets.extend(list(y[y_idx].detach().cpu().numpy()))
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
                X_idx = idx_batch[0]
                species_idx = idx_batch[1]
                y_idx = idx_batch[2]
                # y_hat = CT_specific_conv(X[idx_batch])
                seq_features = convolution(X[X_idx])
                y_hat = fully_connected(torch.cat((seq_features, species_encodings[species_idx]), 1))
                val_error_loss += float(loss_function(y_hat, y[y_idx]))
                all_val_targets.extend(list(y[y_idx].detach().cpu().numpy()))
                all_val_predictions.extend(list(y_hat.detach().cpu().numpy()))

            print("average error loss on val set : " + str(float(val_error_loss) / (step + 1)))
            print(len(all_val_predictions))
            fpr, tpr, _ = roc_curve(all_val_targets, all_val_predictions)
            val_roc_auc = auc(fpr, tpr)
            print('ROC AUC on validation set : ' + str(val_roc_auc))
            val_steps = step + 1

            print("Test performance on test set on the trained model")
            all_test_targets = []
            all_test_predictions = []
            test_error_loss = 0.0
            for step, idx_batch in enumerate(tqdm(test_batch_gen)):
                X_idx = idx_batch[0]
                species_idx = idx_batch[1]
                y_idx = idx_batch[2]
                # y_hat = CT_specific_conv(X[idx_batch])
                seq_features = convolution(X[X_idx])
                y_hat = fully_connected(torch.cat((seq_features, species_encodings[species_idx]), 1))
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

    if not balanced:
        dir_name = str(LR) + '_' + str(early_stop) + '_unbalanced'
    else:
        dir_name = str(LR) + '_' + str(early_stop) + '_balanced'

    dir_path = os.path.join('results', 'multi_species_experiments', dir_name)
    os.makedirs(dir_path, exist_ok=True)

    with open(os.path.join(dir_path, 'output.json'), 'w') as outfile:
        json.dump(output, outfile)

    torch.save({'convolution': convolution.state_dict(),
                'fully_connected': fully_connected.state_dict()},
               os.path.join(dir_path, 'model.pt'))


