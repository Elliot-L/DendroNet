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
    parser.add_argument('--seeds', type=int, nargs='+', default=[0], metavar='S',
                        help='random seed for train/test/validation split (default: [0,1,2,3,4])')
    parser.add_argument('--save-seed', type=int, nargs='+', default=[0], metavar='SS',
                        help='seeds for which the training (AUC score) will be plotted and saved')
    parser.add_argument('--validation-interval', type=int, default=1, metavar='VI')
    parser.add_argument('--dpf', type=float, default=0.1, metavar='D',
                        help='scaling factor applied to delta term in the loss (default: 1.0)')
    parser.add_argument('--l1', type=float, default=0.001)
    parser.add_argument('--p', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--runtest', dest='runtest', action='store_true')
    parser.add_argument('--no-runtest', dest='runtest', action='store_false')
    parser.set_defaults(runtest=False)
    parser.add_argument('--lineage-path', type=str, default=os.path.join('data_files', 'genome_lineage.csv', )
                        , help='file containing taxonomic classification for species from PATRIC')
    parser.add_argument('--group', type=str, default='Firmicutes')
    parser.add_argument('--antibiotic', type=str, default='erythromycin')
    parser.add_argument('--threshold', type=str, default='0.0')
    parser.add_argument('--output-path', type=str, default=os.path.join('data_files', 'output.json'),
                        metavar='OUT', help='file where the ROC AUC scores of the model will be outputted')
    parser.add_argument('--leaf-level', type=str, default='genome_id',
                        help='taxonomical level down to which the tree will be built')
    parser.add_argument('--use-multi-gpus', type=str, default='n', help='options are yes (y) or no (n)')
    args = parser.parse_args()

    # We get the parent_child matrix using the prexisting file or by creating it

    samples_file = args.group + '_' + args.antibiotic + '_' + args.threshold + '_samples.csv'
    samples_file = os.path.join('data_files', 'subproblems', args.group + '_' + args.antibiotic + '_' + args.threshold,
                                samples_file)
    matrix_file = args.group + '_' + args.antibiotic + '_' + args.leaf_level + '.json'
    parent_child, topo_order, node_examples = build_pc_mat(genome_file=args.lineage_path,
                                                           label_file=samples_file,
                                                           leaf_level=args.leaf_level)
    # annotating leaves with labels and features
    if os.path.isfile(samples_file):
        samples_df = pd.read_csv(samples_file, dtype=str)
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
    device = torch.device("cuda:0" if (torch.cuda.is_available() and USE_CUDA) else "cpu")

    if (args.use_multi_gpus == 'y' or args.use_multi_gpus == 'yes') and USE_CUDA and torch.cuda.is_available() and (torch.cuda.device_count() > 1):
        MULTI_GPUs = True
    else:
        MULTI_GPUs = False

    PLOT = False
    PLOT_Batch = False
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
        genome_id = getattr(row, 'ID')
        for i, examples_list in enumerate(node_examples):
            if genome_id in examples_list:
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

    for s in args.seeds:

        print('New seed: ' + str(s))
        print(torch.cuda.memory_allocated())
        print(torch.cuda.memory_reserved())
        init_time = time.time()

        dendronet = DendroMatrixLinReg(device, root_weights, parent_path_tensor, edge_tensor_matrix)

        if MULTI_GPUs:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            dendronetParallel = nn.DataParallel(dendronet)

        best_root_weights = dendronet.root_weights
        best_delta_matrix = dendronet.delta_mat

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

        # shows ratios between single_cell_datasets
        #  print("train:", 100*len(train_idx)/(len(train_idx)+len(test_idx)),"%")
        #  print("val:", 100*len(val_idx)/(len(train_idx)+ len(val_idx)+ len(test_idx)),"%")
        #  print("test:", 100*len(test_idx)/(len(train_idx)+ len(val_idx)+len(test_idx)),"%")

        best_val_auc = 0.0
        early_stopping_count = 0
        best_epoch = 0

        if PLOT:
            # Lists and variables for final plots
            train_aucs_for_plot = []
            val_aucs_for_plot = []
            train_loss_for_plot = []
            val_loss_for_plot = []
            delta_loss_for_plot = []
            l1_loss_for_plot = []
            train_error_loss_for_plot = []
            val_error_loss_for_plot = []
            losses_per_batch = []
            error_per_batch = []
            new_error_per_batch = []
            delta_per_batch = []
            l1_per_batch = []
            first_epoch = True


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
            total_train_error_loss = 0.0
            # getting a batch of indices
            for step, idx_batch in enumerate(tqdm(train_batch_gen)):
                optimizer.zero_grad()
                # separating corresponding rows in X (same as y) and parent_path matrix (same as parent_child order)
                idx_in_X = idx_batch[0]
                idx_in_pp_mat = idx_batch[1]
                # dendronet takes in a set of examples from X, and the corresponding column indices in the parent_path matrix
                if MULTI_GPUs:
                    y_hat = dendronetParallel.forward(X[idx_in_X], idx_in_pp_mat)
                else:
                    y_hat = dendronet.forward(X[idx_in_X], idx_in_pp_mat)
                if y_hat.size() == torch.Size([]):
                    y_hat = torch.unsqueeze(y_hat, 0)
                # collecting the loss terms for this batch
                batch_delta_loss = dendronet.delta_loss()
                batch_root_loss = 0.0
                for w in dendronet.root_weights:
                    batch_root_loss += abs(float(w))
                # idx_in_X is also used to fetch the appropriate entries from y.
                batch_error_loss = loss_function(y_hat, y[idx_in_X])
                # A sigmoid is applied to the output of the model inside loss_function
                # to make them fit between 0 and 1.

                batch_loss = batch_error_loss + (batch_delta_loss * DPF) + (batch_root_loss * L1)
                batch_loss.backward(retain_graph=True)
                optimizer.step()
                new_y_hat = dendronet.forward(X[idx_in_X], idx_in_pp_mat)
                new_error_loss = loss_function(new_y_hat, y[idx_in_X])
                total_train_error_loss += float(batch_error_loss)
                total_train_loss += float(batch_loss)

                if PLOT and first_epoch:
                    losses_per_batch.append(float(batch_loss))
                    error_per_batch.append(float(error_loss))
                    new_error_per_batch.append(float(new_error_loss))
                    delta_per_batch.append(float(batch_delta_loss*DPF))
                    l1_per_batch.append(float(batch_root_loss*L1))
                    first_epoch = False
            print('Average error loss per training batch for this epoch: ', str(total_train_error_loss / (step + 1)))
            print('Average total loss per training batch for this epoch: ', str(total_train_loss / (step + 1)))

            # Here we will compute the loss terms for the whole train set after

            # predicted values (after sigmoid) for whole train set (in the same order as the train_set_targets list)
            all_train_predictions = torch.sigmoid(dendronet.forward(X[all_y_train_idx], all_pp_train_idx)).detach().cpu().numpy()

            fpr, tpr, _ = roc_curve(all_train_targets, all_train_predictions)
            roc_auc = auc(fpr, tpr)
            print("training ROC AUC for epoch: ", roc_auc)

            final_delta_loss_for_epoch = dendronet.delta_loss()
            final_root_loss_for_epoch = 0.0
            for w in dendronet.root_weights:
                final_root_loss_for_epoch += abs(float(w))
            with torch.no_grad():
                final_train_error_loss_for_epoch = loss_function(torch.tensor(all_train_predictions), torch.tensor(all_train_targets))

            if PLOT:
                delta_loss_for_plot.append(final_delta_loss_for_epoch*DPF)
                l1_loss_for_plot.append(final_root_loss_for_epoch*L1)
                train_error_loss_for_plot.append(final_train_error_loss_for_epoch)
                train_loss_for_plot.append(float(final_train_error_loss_for_epoch + final_delta_loss_for_epoch*DPF + final_root_loss_for_epoch*L1))
                train_aucs_for_plot.append(roc_auc)

            # Validate performance using validation set
            with torch.no_grad():
                total_val_loss = 0.0
                total_val_error_loss = 0.0
                all_val_targets = []
                all_val_predictions = []
                for step, idx_batch in enumerate(val_batch_gen):
                    idx_in_X = idx_batch[0]
                    idx_in_pp_mat = idx_batch[1]
                    if MULTI_GPUs:
                        y_hat = dendronetParallel.forward(X[idx_in_X], idx_in_pp_mat)
                    else:
                        y_hat = dendronet.forward(X[idx_in_X], idx_in_pp_mat)
                    if y_hat.size() == torch.Size([]):
                        y_hat = torch.unsqueeze(y_hat, 0)
                    # accumulate targets and prediction to compute AUC
                    val_targets = list(y[idx_in_X].detach().cpu().numpy())  # target values for this batch
                    val_predictions = list(
                        torch.sigmoid(y_hat).detach().cpu().numpy())  # predictions (after sigmoid) for this batch
                    all_val_targets.extend(val_targets)
                    all_val_predictions.extend(val_predictions)
                    error_loss = loss_function(y_hat, y[idx_in_X])
                    total_val_error_loss += float(error_loss)
                    # We use the delta/l1 loss computed earlier
                    total_val_loss += float(error_loss + (final_delta_loss_for_epoch * DPF) + (final_root_loss_for_epoch * L1))

                print('Average error loss on the validation set per batch on this epoch: ', str(total_val_error_loss / (step + 1)))
                print('Average loss on the validation set per batch on this epoch: ', str(total_val_loss / (step + 1)))

                fpr, tpr, _ = roc_curve(all_val_targets, all_val_predictions)
                roc_auc = auc(fpr, tpr)
                print("Validation ROC AUC for epoch: ", roc_auc)

                if PLOT:
                    final_val_error_loss_for_epoch = loss_function(torch.tensor(all_val_predictions), torch.tensor(all_val_targets))
                    val_error_loss_for_plot.append(final_val_error_loss_for_epoch)
                    val_loss_for_plot.append(float(final_val_error_loss_for_epoch + final_delta_loss_for_epoch*DPF + final_root_loss_for_epoch*L1))
                    val_aucs_for_plot.append(roc_auc)

                if roc_auc > best_val_auc:  # Check if performance has increased on validation set (loss is decreasing)
                    best_val_auc = roc_auc
                    early_stopping_count = 0
                    print("Improvement!!!")
                    best_epoch = epoch
                    best_root_weights = dendronet.root_weights.clone().detach().cpu()
                    best_delta_matrix = dendronet.delta_mat.clone().detach().cpu()
                else:
                    early_stopping_count += 1
                    print("Oups,... we are at " + str(early_stopping_count) + ", best: " + str(best_val_auc))

                if early_stopping_count >= args.early_stopping:  # If performance has not increased for long enough, we stop training
                    print("EARLY STOPPING!")                     # to avoid overfitting
                    break

        val_auc_output.append(best_val_auc)

        # With training complete, we'll run the test set.
        with torch.no_grad():
            X = X.cpu()
            y = y.cpu()
            # The best model obtained so far, based on AUC score on validation set will be used here
            best_dendronet = DendroMatrixLinReg(torch.device('cpu'), best_root_weights, parent_path_tensor,
                                                best_delta_matrix,
                                                init_root=False)
            all_test_targets = []
            all_best_model_predictions = []
            total_test_loss = 0.0
            total_test_error_loss = 0.0

            best_delta_loss = best_dendronet.delta_loss()
            best_l1_loss = 0
            for w in best_dendronet.root_weights:
                best_l1_loss += abs(float(w))

            for step, idx_batch in enumerate(test_batch_gen):
                idx_in_X = idx_batch[0]
                idx_in_pp_mat = idx_batch[1]
                best_model_y_hat = best_dendronet.forward(X[idx_in_X], idx_in_pp_mat)
                if best_model_y_hat.size() == torch.Size([]):
                    best_model_y_hat = torch.unsqueeze(best_model_y_hat, 0)
                test_targets = list(y[idx_in_X].detach().cpu().numpy())
                best_pred = list(torch.sigmoid(best_model_y_hat).detach().cpu().numpy())
                error_loss = loss_function(best_model_y_hat, y[idx_in_X])
                total_test_error_loss += float(error_loss)
                total_test_loss += float(error_loss + (best_delta_loss*DPF) + (best_l1_loss*L1))
                all_test_targets.extend(test_targets)
                all_best_model_predictions.extend(best_pred)

            fpr, tpr, _ = roc_curve(all_test_targets, all_best_model_predictions)
            roc_auc = auc(fpr, tpr)

            print("Best non-over-fitted model:")
            print("ROC AUC for test:", roc_auc)
            print('Delta loss:', float(best_delta_loss))
            print('L1 loss:', best_l1_loss)
            print('Average Error loss:', (total_test_error_loss / (step + 1)))
            print('Average total loss per batch on test set:', (total_test_loss / (step + 1)))

            test_auc_output.append(roc_auc)

        final_time = time.time() - init_time
        average_time_seed += final_time

        print('After training on seed: ' + str(s))
        print(torch.cuda.memory_allocated())
        print(torch.cuda.memory_reserved())

        if s in args.save_seed:
            models_output_dir = os.path.join('data_files', 'subproblems', args.group + '_' + args.antibiotic + '_'
                                             + args.threshold, 'best_models')
            os.makedirs(models_output_dir, exist_ok=True)
            root_file = str(DPF) + '_' + str(LR) + '_' + str(L1) + '_' + str(args.early_stopping) \
                        + '_' + args.leaf_level + '_seed_' + str(s) + '_root.pt'
            delta_file = str(DPF) + '_' + str(LR) + '_' + str(L1) + '_' + str(args.early_stopping) \
                         + '_' + args.leaf_level + '_seed_' + str(s) + '_delta.pt'
            torch.save(best_root_weights, os.path.join(models_output_dir, root_file))
            torch.save(best_delta_matrix, os.path.join(models_output_dir, delta_file))

    average_time_seed = average_time_seed / len(args.seeds)
    print('Average time to train a model: ' + str(average_time_seed) + ' seconds')

    os.makedirs(os.path.join('data_files', 'time_performances'), exist_ok=True)
    time_file = os.path.join('data_files', 'time_performances', 'experiment3_' + args.group + '_' + args.antibiotic + '_'
                             + args.leaf_level + '_' + args.threshold)
    with open(time_file, 'w') as file:
        json.dump({'average_per_seed': average_time_seed}, file)

    if PLOT:
        if PLOT_Batch:
            plt.plot(delta_per_batch, c='r', label='D')  # delta loss
            plt.plot(l1_per_batch, c='y', label='L1')  # L1 loss
            plt.plot(error_per_batch, c='k', label='E')  # error loss
            plt.plot(new_error_per_batch, c='g', label='N')  # error loss after optimizer step
            #plt.plot(losses_per_batch, c='b', label='T')  # total loss
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1)
            plt.suptitle('First batch ' + args.group + '_' + args.antibiotic + '_' + args.threshold + ' early-stop: '
                         + str(args.early_stopping) + ' dpf: ' + str(DPF))
            print(delta_per_batch)
            print(l1_per_batch)
            print(error_per_batch)
            print(new_error_per_batch)
            print(losses_per_batch)
        else:
            best_test_auc_for_plot = []
            final_test_auc_for_plot = []
            for i in range(len(train_aucs_for_plot)):
                best_test_auc_for_plot.append(roc_auc)

            figure, axis = plt.subplots(2, 1)
            figure.suptitle('Whole training' + args.group + '_' + args.antibiotic + '_' + args.threshold
                            + ' early-stop: ' + str(args.early_stopping) + ' dpf: ' + str(DPF))

            # Plot for AUC values
            axis[0].plot(train_aucs_for_plot, c='r', label='T AUC')  # train AUC
            axis[0].plot(val_aucs_for_plot, c='y', label='V AUC')  # validation AUC
            axis[0].plot(final_test_auc_for_plot, c='k', label='F AUC')  # AUC of final model on test
            axis[0].plot(best_test_auc_for_plot, c='b', label='B AUC')  # AUC of best model on test
            axis[0].legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1)
            axis[0].axvline(x=best_epoch)

            # Plot for test/val error, delta and L1 losses
            axis[1].plot(train_error_loss_for_plot, c='r', label='T error')  # train error
            axis[1].plot(val_error_loss_for_plot, c='y', label='V error')  # validation error
            axis[1].plot(delta_loss_for_plot, c='k', label='D loss')  # Delta loss
            axis[1].plot(l1_loss_for_plot, c='b', label='L1 loss')  # L1 loss
            axis[1].legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1)
            axis[1].axvline(x=best_epoch)

        """
        #axis[0].xlabel('epochs')
        #axis[0].ylabel('ROC AUC')

        # Plot for AUC values and total test/val losses at end of epoch
        axis[0, 1].plot(train_aucs_for_plot, c='r', label='T A')  # train AUC
        axis[0, 1].plot(val_aucs_for_plot, c='y', label='V A')  # validation AUC
        axis[0, 1].plot(train_loss_for_plot, c='k', label='T L')  # train loss
        axis[0, 1].plot(val_loss_for_plot, c='b', label='V L')  # validation loss
        #axis[0, 1].legend(loc='lower left', bbox_to_anchor=(0, 1), ncol=4)
        axis[0, 1].axvline(x=best_epoch)

        #axis[1].xlabel('epoch')
        #axis[1].ylabel('Loss')

        # Plot error, delta, l1 and total per batch, on first epoch
        axis[1, 1].plot(delta_per_batch, c='r', label='D')  # delta loss
        axis[1, 1].plot(l1_per_batch, c='y', label='L1')  # L1 loss
        axis[1, 1].plot(error_per_batch, c='k', label='E')  # error loss
        axis[1, 1].plot(losses_per_batch, c='b', label='T')  # total loss
        #axis[1, 1].legend(loc='upper left', bbox_to_anchor=(0, -0.5), ncol=4)
        """
    plt.show()

    if SAVE_PLOT and PLOT:
        os.makedirs(os.path.join('data_files', 'AUC_plots'), exist_ok=True)
        plot_file_name = args.group + '_' + args.antibiotic + '_' + args.threshold + '_' + str(args.dpf) + '_' \
                    + str(args.lr) + '_' + str(args.l1) + '_' + str(args.early_stopping) \
                    + '_seed_' + str(s) + '_' + args.leaf_level + '.png'
        if s in args.save_seed:
            plt.savefig(os.path.join('data_files', 'AUC_plots', plot_file_name))

    output_dict = {'val_auc': val_auc_output, 'test_auc': test_auc_output}

    dict_file_name = args.output_path
    os.makedirs(os.path.dirname(dict_file_name), exist_ok=True)
    with open(dict_file_name, 'w') as outfile:
        json.dump(output_dict, outfile)



