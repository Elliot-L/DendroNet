#  imports from outside the project
import os
import json
import argparse
import pandas as pd
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
import time
from tqdm import tqdm
import torch
import numpy as np
import torch.nn as nn

# imports from inside the project
from build_parent_child_mat import build_pc_mat
from models.dendronet_models import DendroMatrixLinReg
from utils.model_utils import build_parent_path_mat, split_indices, IndicesDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000, metavar='N')
    parser.add_argument('--early-stopping', type=int, default=5, metavar='E',
                        help='Number of epochs without improvement before early stopping')
    parser.add_argument('--seed', type=int, nargs='+', default=[1], metavar='S',
                        help='random seed for train/test/validation split (default: [0,1,2,3,4])')
    parser.add_argument('--save-seed', type=int, nargs='+', default=[], metavar='SS',
                        help='seeds for which the training (AUC score) will be plotted and saved')
    parser.add_argument('--validation-interval', type=int, default=1, metavar='VI')
    parser.add_argument('--dpf', type=float, default=1, metavar='D',
                        help='scaling factor applied to delta term in the loss (default: 1.0)')
    parser.add_argument('--l1', type=float, default=0.1)
    parser.add_argument('--p', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--runtest', dest='runtest', action='store_true')
    parser.add_argument('--no-runtest', dest='runtest', action='store_false')
    parser.set_defaults(runtest=False)
    parser.add_argument('--lineage-path', type=str, default=os.path.join('data_files', 'genome_lineage.csv', )
                        , help='file containing taxonomic classification for species from PATRIC')
    parser.add_argument('--samples-file', type=str,
                        default=os.path.join('data_files', 'subproblems', 'Firmicutes_betalactam',
                                             'Firmicutes_betalactam_0.0_samples.csv'),
                        metavar='LF', help='file to look in for samples')
    parser.add_argument('--output-path', type=str, default=os.path.join('data_files', 'output.json'),
                        metavar='OUT', help='file where the ROC AUC scores of the model will be outputted')
    parser.add_argument('--leaf-level', type=str, default='genome_id',
                        help='taxonomical level down to which the tree will be built')
    args = parser.parse_args()

    # We get the parent_child matrix using the prexisting file or by creating it
    file_info = os.path.split(args.samples_file)[1]
    antibiotic = file_info.split('_')[1]
    group = file_info.split('_')[0]
    threshold = file_info.split('_')[2]

    matrix_file = group + '_' + antibiotic + '_' + args.leaf_level + '.json'
    parent_child, topo_order, node_examples = build_pc_mat(genome_file=args.lineage_path,
                                                           label_file=args.samples_file,
                                                           leaf_level=args.leaf_level)
    # annotating leaves with labels and features
    if os.path.isfile(args.samples_file):
        samples_df = pd.read_csv(args.samples_file, dtype=str)
    else:
        print('The samples file does not exist.')
        exit()
    """
    There are 3 components we need:
    1. A matrix X, where each row contains the features for a bacteria
    2. A vector y, where each entry contains the phenotype for a bacteria. This should be in the same order as X; i.e.
    entry 0 in y is the phenotype for row 0 in X
    3. A parent-child matrix for the tree that is defined by the  structure 'data_tree', with some more details below:
        -This parent-child matrix has rows for all nodes in data_tree (including the internal ones)

        -Each row corresponds to a parent node, and each column to a child node, i.e. a 1 is entered at 
        position (row 1, col 2) if the node corresponding to row 1 is the parent of the node corresponding to column 2

        -The rows should be in descending order; i.e. row/col 0 is the root, row/col 1 and 2 are the first layer below the root

        -For each row, we need a mapping which tells us the appropriate entry in X that stores info for the relevant 
        species. This could be a list of tuples, i.e. (parent-child-row-index, entry-in-X-index). I would suggest using 
        the ID field to create this list of tuples as you are filling in the parent-child matrix
    """

    # flag to use cuda gpu if available
    USE_CUDA = True
    print('Using CUDA: ' + str(torch.cuda.is_available() and USE_CUDA))
    device = torch.device("cuda:0" if torch.cuda.is_available() and USE_CUDA else "cpu")

    PLOT = False
    SAVE_PLOT = False

    # some other hyper-parameters for training
    LR = args.lr
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    DPF = args.dpf
    L1 = args.l1

    mapping = []

    #  This will be the mapping between rows in the X and parent_child matrix. Only the features and target values
    #  of the leaves of the tree are added to the X matrix and y vector, respectively, while all nodes are added
    #  to the parent_child matrix. The list mapping contains a tuple for each leaf of the form
    #  (row_in_X, row_in_parent_child)
    #  This is done in order to save computer memory. This way the X matrix can be smaller
    #  as only the leaves of the tree have features.

    X = []
    y = []
    example_number = 0

    for row in samples_df.itertuples():
        added_in_X_and_y = False
        for i, example_list in enumerate(node_examples):
            if getattr(row, 'ID') in example_list:
                if not added_in_X_and_y:
                    phenotype = eval(getattr(row, 'Phenotype'))[0]  # the y value
                    features = eval(getattr(row, 'Features'))  # the x value
                    y.append(phenotype)
                    X.append(features)
                    added_in_X_and_y = True
                mapping.append((example_number, i))
        if added_in_X_and_y:
            example_number += 1

    parent_path_tensor = build_parent_path_mat(parent_child)
    num_features = len(X[0])
    num_nodes = len(parent_child[0])
    num_edges = len(parent_path_tensor)

    root_weights = np.zeros(shape=num_features)
    edge_tensor_matrix = np.zeros(shape=(num_features, num_edges))

    test_auc_output = []
    val_auc_output = []
    average_time_seed = 0

    for s in args.seed:
        init_time = time.time()

        dendronet = DendroMatrixLinReg(device, root_weights, parent_path_tensor, edge_tensor_matrix)
        best_root_weights = dendronet.root_weights
        best_edge_tensor_matrix = dendronet.delta_mat

        train_idx, test_idx = split_indices(mapping, seed=0)
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
        loss_function = nn.BCEWithLogitsLoss()  # note for posterity: can either use DendroLinReg with this loss, or DendroLogReg with BCELoss
        if torch.cuda.is_available() and USE_CUDA:
            loss_function = loss_function.cuda()
        optimizer = torch.optim.SGD(dendronet.parameters(), lr=LR)

        # shows ratios between datasets
        #  print("train:", 100*len(train_idx)/(len(train_idx)+len(test_idx)),"%")
        #  print("val:", 100*len(val_idx)/(len(train_idx)+ len(val_idx)+ len(test_idx)),"%")
        #  print("test:", 100*len(test_idx)/(len(train_idx)+ len(val_idx)+len(test_idx)),"%")

        best_auc = 0.0
        early_stopping_count = 0

        # Lists for final plots
        train_aucs_for_plot = []
        val_aucs_for_plot = []
        train_loss_for_plot = []
        val_loss_for_plot = []
        delta_loss_for_plot = []
        l1_loss_for_plot = []
        train_diff_loss_for_plot = []
        val_diff_loss_for_plot = []

        # Generate two lists containing 1) the index in the vector y of all the training example (whole train set)
        # 2) the corresponding positions of these training examples in the parent-path matrix.
        # This is done in order generate a list of all the phenotypes of the training set (train_set_targets), and to
        # compute the AUC score of the training set after training over all batches is completed.

        all_y_train_idx = []
        all_pp_train_idx = []
        for tup in train_idx:
            all_y_train_idx.append(tup[0])
            all_pp_train_idx.append(tup[1])

        all_train_targets = y[all_y_train_idx].detach().cpu().numpy()  # target values for whole training set

        # running the training loop
        for epoch in range(EPOCHS):
            print('Train epoch ' + str(epoch))
            # we'll track the running loss over each batch so we can compute the average per epoch
            total_train_loss = 0.0
            # getting a batch of indices
            for step, idx_batch in enumerate(tqdm(train_batch_gen)):
                optimizer.zero_grad()
                # separating corresponding rows in X (same as y) and parent_path matrix (same as parent_child order)
                idx_in_X = idx_batch[0]
                idx_in_pp_mat = idx_batch[1]
                # dendronet takes in a set of examples from X, and the corresponding column indices in the parent_path matrix
                y_hat = dendronet.forward(X[idx_in_X], idx_in_pp_mat)
                # collecting the loss terms
                delta_loss = dendronet.delta_loss()
                root_loss = 0.0
                for w in dendronet.root_weights:
                    root_loss += abs(float(w))
                # idx_in_X is also used to fetch the appropriate entries from y.
                diff_loss = loss_function(y_hat, y[idx_in_X])
                # A sigmoid is applied to the output of the model inside loss_function
                # to make them fit between 0 and 1.

                loss = diff_loss + (delta_loss * DPF) + (root_loss * L1)
                loss.backward(retain_graph=True)
                optimizer.step()
                total_train_loss += float(loss)
            step += 1
            print('Average loss per training batch for this epoch: ', str(total_train_loss / step))

            # predicted values (after sigmoid) for whole train set (in the same order as the train_set_targets list)
            y_train_predictions = torch.sigmoid(dendronet.forward(X[all_y_train_idx], all_pp_train_idx)).detach().cpu().numpy()

            fpr, tpr, _ = roc_curve(all_train_targets, y_train_predictions)
            roc_auc = auc(fpr, tpr)
            print("training ROC AUC for epoch: ", roc_auc)

            train_aucs_for_plot.append(roc_auc)
            train_loss_for_plot.append(total_train_loss / step)

            final_delta_loss_for_epoch = dendronet.delta_loss()
            delta_loss_for_plot.append(final_delta_loss_for_epoch)
            final_root_loss_for_epoch = 0.0
            for w in dendronet.root_weights:
                final_root_loss_for_epoch += abs(float(w))
            l1_loss_for_plot.append(final_root_loss_for_epoch)
            with torch.no_grad():
                final_train_diff_loss_for_epoch = loss_function(torch.tensor(y_train_predictions), torch.tensor(all_train_targets))
            train_diff_loss_for_plot.append(final_train_diff_loss_for_epoch)

            # Test performance using validation set at each epoch
            with torch.no_grad():
                total_val_loss = 0.0
                all_val_targets = []
                all_val_predictions = []
                # We compute the delta and root loss right away as it doesn't change anymore for this epoch
                for step, idx_batch in enumerate(val_batch_gen):
                    idx_in_X = idx_batch[0]
                    idx_in_pp_mat = idx_batch[1]
                    y_hat = dendronet.forward(X[idx_in_X], idx_in_pp_mat)
                    val_targets = list(y[idx_in_X].detach().cpu().numpy())  # target values for this batch
                    val_predictions = list(
                        torch.sigmoid(y_hat).detach().cpu().numpy())  # predictions (after sigmoid) for this batch
                    all_val_targets.extend(val_targets)
                    all_val_predictions.extend(val_predictions)
                    diff_loss = loss_function(y_hat, y[idx_in_X])

                    total_val_loss += float(diff_loss + (final_delta_loss_for_epoch * DPF) + (final_root_loss_for_epoch * L1))

                fpr, tpr, _ = roc_curve(all_val_targets, all_val_predictions)
                roc_auc = auc(fpr, tpr)
                step += 1
                print('Average loss on the validation set per batch on this epoch: ', str(total_val_loss / step))
                print("Validation ROC AUC for epoch: ", roc_auc)

                val_aucs_for_plot.append(roc_auc)
                val_loss_for_plot.append(total_val_loss / step)

                final_val_diff_loss_for_epoch = loss_function(torch.tensor(all_val_predictions), torch.tensor(all_val_targets))
                val_diff_loss_for_plot.append(final_val_diff_loss_for_epoch)

                if roc_auc > best_auc:  # Check if performance has increased on validation set (loss is decreasing)
                    best_auc = roc_auc
                    early_stopping_count = 0
                    print("Improvement!!!")
                    best_root_weights = dendronet.root_weights.detach().clone()
                    best_edge_tensor_matrix = dendronet.delta_mat.detach().clone()

                else:
                    early_stopping_count += 1
                    print("Oups,... we are at " + str(early_stopping_count) + ", best: " + str(best_auc))


                if early_stopping_count >= args.early_stopping:  # If performance has not increased for long enough, we stop training
                    print("EARLY STOPPING!")  # to avoid overfitting
                    break

        val_auc_output.append(roc_auc)

        # With training complete, we'll run the test set.
        with torch.no_grad():
            best_dendronet = DendroMatrixLinReg(device, best_root_weights, parent_path_tensor, best_edge_tensor_matrix,
                                                init_root=False)
            all_targets = []
            all_predictions = []
            all_bad_predictions = []
            bce_loss = 0.0
            for step, idx_batch in enumerate(test_batch_gen):
                idx_in_X = idx_batch[0]
                idx_in_pp_mat = idx_batch[1]
                y_hat = best_dendronet.forward(X[idx_in_X], idx_in_pp_mat)
                bad_y_hat = dendronet.forward(X[idx_in_X], idx_in_pp_mat)
                targets = list(y[idx_in_X].detach().cpu().numpy())
                pred = list(torch.sigmoid(y_hat).detach().cpu().numpy())
                bad_pred = list(torch.sigmoid(bad_y_hat).detach().cpu().numpy())
                bce_loss += loss_function(y_hat, y[idx_in_X])
                all_targets.extend(targets)
                all_predictions.extend(pred)
                all_bad_predictions.extend(bad_pred)
            delta_loss = best_dendronet.delta_loss()
            l1_loss = 0
            for w in best_dendronet.root_weights:
                l1_loss += abs(float(w))
            fpr, tpr, _ = roc_curve(all_targets, all_predictions)
            roc_auc = auc(fpr, tpr)
            bad_fpr, bad_tpr, _ = roc_curve(all_targets, all_bad_predictions)
            bad_roc_auc = auc(bad_fpr, bad_tpr)

            print("ROC AUC for test:", roc_auc)
            print('Final Delta loss:', float(delta_loss.detach().cpu().numpy()))
            print('L1 loss on test set is ', l1_loss)
            step += 1
            print('Average BCE loss per batch on test set:', float(bce_loss) / step)

            test_auc_output.append(roc_auc)

        final_time = time.time() - init_time
        average_time_seed += final_time

    average_time_seed = average_time_seed / len(args.seed)
    print('Average time to train a model: ' + str(average_time_seed) + ' seconds')

    os.makedirs(os.path.join('data_files', 'time_performances'), exist_ok=True)
    time_file = os.path.join('data_files', 'time_performances', 'experiment3_' + group + '_' + antibiotic + '_'
                             + args.leaf_level + '_' + threshold)
    with open(time_file, 'w') as file:
        json.dump({'average_per_seed': average_time_seed}, file)

    if PLOT:
        best_test_final_auc = []
        test_final_auc = []
        for i in range(len(train_aucs_for_plot)):
            test_final_auc.append(bad_roc_auc)
            best_test_final_auc.append(roc_auc)

        figure, axis = plt.subplots(2, 2)

        # Plot for AUC values
        axis[0, 0].plot(train_aucs_for_plot, c='r', label='T AUC')
        axis[0, 0].plot(val_aucs_for_plot, c='y', label='V AUC')
        axis[0, 0].plot(test_final_auc, c='b', label='F AUC')
        axis[0, 0].plot(best_test_final_auc, c='k', label='B AUC')

        axis[0, 0].set_title('(' + group + '_' + antibiotic + '_' + threshold + ') early-stop: ' + str(args.early_stopping)
                             + ' dpf: ' + str(DPF))
        #axis[0].xlabel('epochs')
        #axis[0].ylabel('ROC AUC')
        axis[0, 0].legend(loc='lower right')

        # Plot for average running train and validation loss :
        # Sum over all batches of diff, delta and l1 loss divided by number of batches
        axis[0, 1].plot(train_loss_for_plot, c='k', label='Train Loss')
        axis[0, 1].plot(val_loss_for_plot, c='m', label='Validation Loss')

        #axis[1].xlabel('epoch')
        #axis[1].ylabel('Loss')
        axis[0, 1].legend(loc='upper left')

        # Plot for Delta and L1 Loss
        axis[1, 0].plot(delta_loss_for_plot, c='r', label='D Loss')
        axis[1, 0].plot(l1_loss_for_plot, c='b', label='L1 Loss')

        #axis[2].xlabel('epoch')
        #axis[2].ylabel('Loss')
        axis[1, 0].legend(loc='upper left')

        # Plot for Diff Losses
        axis[1, 1].plot(train_diff_loss_for_plot, c='b', label='T Diff')
        axis[1, 1].plot(val_diff_loss_for_plot, c='k', label='V Diff')
        axis[1, 1].legend(loc='upper left')

        plt.show()

    if SAVE_PLOT:
        os.makedirs(os.path.join('data_files', 'AUC_plots'), exist_ok=True)
        file_name = group + '_' + antibiotic + '_' + threshold + '_' + str(args.lr) + '_' + str(args.dpf) + '_' \
                    + str(args.l1) + '_' + str(args.early_stopping) + '_seed_' + str(s) + '.png'
        if s in args.save_seed:
            plt.savefig(os.path.join('data_files', 'AUC_plots', file_name))

    output_dict = {'val_auc': val_auc_output, 'test_auc': test_auc_output}

    fileName = args.output_path
    os.makedirs(os.path.dirname(fileName), exist_ok=True)
    with open(fileName, 'w') as outfile:
        json.dump(output_dict, outfile)

