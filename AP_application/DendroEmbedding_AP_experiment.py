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
from models.CT_conv_model import DendronetModule, SeqConvModule, FCModule
from utils.model_utils import split_indices, IndicesDataset, build_parent_path_mat
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
    parser.add_argument('--DPF', type=float, default=0.00001)
    parser.add_argument('--L1', type=float, default=0.01)
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

    species_list = args.species_list
    LR = args.LR
    DPF = args.DPF
    L1 = args.L1
    BATCH_SIZE = args.BATCH_SIZE
    USE_CUDA = args.GPU
    balanced = args.balanced
    embedding_size = args.embedding_size
    seeds = args.seeds
    early_stop = args.early_stopping
    epochs = args.num_epochs

    print('Using CUDA: ' + str(torch.cuda.is_available() and USE_CUDA))
    device = torch.device("cuda:0" if (torch.cuda.is_available() and USE_CUDA) else "cpu")

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
        else:
            num_internal_nodes += 1

    for species in old_species_list:
        if species not in species_list:
            not_used_species.append(species)
    print('Species not used: ')
    print(not_used_species)
    print('Species used: ')
    print(species_list)

    print(nodes)
    print(pc_mat.shape)
    print(pc_mat)

    with open(os.path.join('data_files', 'amino_acids.json'), 'r') as aa_file:
        aa_list = json.load(aa_file)
    print(aa_list)

    with open(os.path.join('data_files', 'peptide_seqs.json'), 'r') as p_file:
        peptide_dict = json.load(p_file)

    X = []
    y = []

    parent_path_mat = build_parent_path_mat(pc_mat)
    print(parent_path_mat)
    print(parent_path_mat.shape)
    num_edges = len(parent_path_mat)
    delta_mat = np.zeros(shape=(embedding_size, num_edges))
    root_vector = np.zeros(shape=embedding_size)

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
                samples.append((row, i + num_internal_nodes, i*total_count + row))

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
                        samples.append((row, i + num_internal_nodes, len(y)))
                        y.append(1)
                        pos_counter[species] += 1
                    if species_dfs[species].loc[row, 'label'] == 0 and neg_counter[species] < pos_count[species]:
                        samples.append((row, i + num_internal_nodes, len(y)))
                        y.append(0)
                        neg_counter[species] += 1
            else:
                for row in range(total_count):
                    if species_dfs[species].loc[row, 'label'] == 1 and pos_counter[species] < neg_count[species]:
                        samples.append((row, i + num_internal_nodes, len(y)))
                        y.append(1)
                        pos_counter[species] += 1
                    if species_dfs[species].loc[row, 'label'] == 0:
                        samples.append((row, i + num_internal_nodes, len(y)))
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
              'epochs': []}
    embeddings_output = {species: [] for species in species_list}

    X = torch.tensor(X, dtype=torch.float, device=device)
    X = X.permute(0, 2, 1)
    y = torch.tensor(y, dtype=torch.float, device=device)

    params = {'batch_size': BATCH_SIZE,
              'shuffle': True,
              'num_workers': 0}

    for seed in seeds:
        # The three subparts of the model:

        dendronet = DendronetModule(device=device, root_weights=root_vector, delta_mat=delta_mat,
                                    path_mat=parent_path_mat)

        convolution = SeqConvModule(device=device, seq_length=101, kernel_sizes=(5,), num_of_kernels=(64,),
                                    polling_windows=(), input_channels=len(aa_list))

        fully_connected = FCModule(device=device, layer_sizes=(embedding_size + 64, 32, 1))

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
        optimizer = torch.optim.Adam(list(dendronet.parameters()) + list(convolution.parameters())
                                     + list(fully_connected.parameters()), lr=LR)

        best_val_auc = 0
        early_stop_count = 0

        for epoch in range(epochs):
            print("Epoch " + str(epoch))
            for step, idx_batch in enumerate(tqdm(train_batch_gen)):
                optimizer.zero_grad()
                # print(y[idx_batch])
                X_idx = idx_batch[0]
                species_idx = idx_batch[1]
                y_idx = idx_batch[2]
                seq_features = convolution(X[X_idx])
                species_embeddings = dendronet(species_idx)
                y_hat = fully_connected(torch.cat((seq_features, species_embeddings.float()), 1))
                # print(y_hat)
                # error_loss = loss_function(y_hat, y[idx_batch])
                error_loss = loss_function(y_hat, y[y_idx])
                delta_loss = dendronet.delta_loss(species_idx)
                root_loss = dendronet.root_loss()
                train_loss = error_loss + DPF*delta_loss + L1*root_loss
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
                    species_idx = idx_batch[1]
                    y_idx = idx_batch[2]
                    seq_features = convolution(X[X_idx])
                    species_embeddings = dendronet(species_idx)
                    y_hat = fully_connected(torch.cat((seq_features, species_embeddings.float()), 1))
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
                    species_idx = idx_batch[1]
                    y_idx = idx_batch[2]
                    seq_features = convolution(X[X_idx])
                    species_embeddings = dendronet(species_idx)
                    y_hat = fully_connected(torch.cat((seq_features, species_embeddings.float()), 1))
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
                    best_root_state = dendronet.root_weights.clone().detach().cpu()
                    best_delta_state = dendronet.delta_mat.clone().detach().cpu()
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
            dendronet = DendronetModule(device, best_root_state, parent_path_mat, best_delta_state,
                                        init_root=False, init_deltas=False)
            convolution.load_state_dict(best_convolution_state)
            fully_connected.load_state_dict(best_fc_state)

            print("Test performance on train set on best model: ")
            all_train_targets = []
            all_train_predictions = []
            train_error_loss = 0.0
            for step, idx_batch in enumerate(tqdm(train_batch_gen)):
                X_idx = idx_batch[0]
                species_idx = idx_batch[1]
                y_idx = idx_batch[2]
                seq_features = convolution(X[X_idx])
                species_embeddings = dendronet(species_idx)
                y_hat = fully_connected(torch.cat((seq_features, species_embeddings.float()), 1))
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
                species_idx = idx_batch[1]
                y_idx = idx_batch[2]
                seq_features = convolution(X[X_idx])
                species_embeddings = dendronet(species_idx)
                y_hat = fully_connected(torch.cat((seq_features, species_embeddings.float()), 1))
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
                species_idx = idx_batch[1]
                y_idx = idx_batch[2]
                seq_features = convolution(X[X_idx])
                species_embeddings = dendronet(species_idx)
                y_hat = fully_connected(torch.cat((seq_features, species_embeddings.float()), 1))
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

        for i, species in enumerate(species_list):
            embedding = (torch.squeeze(dendronet.forward([i + num_internal_nodes]))).cpu().tolist()
            embeddings_output[species].append(embedding)

    if not balanced:
        dir_name = str(LR) + '_' + str(DPF) + '_' + str(L1) \
                  + '_' + str(embedding_size) + '_' + str(early_stop) + '_unbalanced'
    else:
        dir_name = str(LR) + '_' + str(DPF) + '_' + str(L1) \
                  + '_' + str(embedding_size) + '_' + str(early_stop) + '_balanced'

    dir_path = os.path.join('results', 'dendronet_embedding_experiments', dir_name)
    os.makedirs(dir_path, exist_ok=True)

    with open(os.path.join(dir_path, 'output.json'), 'w') as outfile:
        json.dump(output, outfile)

    torch.save({'convolution': convolution.state_dict(),
                'fully_connected': fully_connected.state_dict(),
                'dendronet_delta_mat': dendronet.delta_mat.clone().detach().cpu(),
                'dendronet_root': dendronet.root_weights.clone().detach().cpu()},
               os.path.join(dir_path, 'model.pt'))

    with open(os.path.join(dir_path, 'embeddings.json'), 'w') as outfile:
        json.dump(embeddings_output, outfile)