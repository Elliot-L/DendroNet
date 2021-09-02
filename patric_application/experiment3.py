import math
import os
import json
import argparse
import jsonpickle
import pandas as pd
from build_parent_child_mat import build_pc_mat
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt

#imports from dag tutorial
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from models.dendronet_models import DendroMatrixLinReg
from utils.model_utils import build_parent_path_mat, split_indices, IndicesDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200, metavar='N')
    parser.add_argument('--early-stopping', type=int, default=10, metavar='E',
                        help='Number of epochs without improvement before early stopping')
    parser.add_argument('--seed', type=int, default=[0, 1, 2, 3, 4], metavar='S',
                        help='random seed for train/test/validation split (default: [0,1,2,3,4])')
    parser.add_argument('--save-seed', type=int, default=[], metavar='SS',
                        help='seeds for which the training (AUC score) will be plotted and saved')
    parser.add_argument('--validation-interval', type=int, default=1, metavar='VI')
    parser.add_argument('--dpf', type=float, default=0.01, metavar='D',
                        help='scaling factor applied to delta term in the loss (default: 1.0)')
    parser.add_argument('--l1', type=float, default=0.1)
    parser.add_argument('--p', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR')
    parser.add_argument('--runtest', dest='runtest', action='store_true')
    parser.add_argument('--no-runtest', dest='runtest', action='store_false')
    parser.set_defaults(runtest=False)
    parser.add_argument('--lineage-path', type=str, default=os.path.join('data_files', 'genome_lineage.csv',)
                        , help='file containing taxonomic classification for species from PATRIC')
    parser.add_argument('--label-file', type=str, default=os.path.join('data_files', 'subproblems', 'firmicutes_betalactam', 'betalactam_firmicutes_samples.csv'),
                        metavar='LF', help='file to look in for labels')
    parser.add_argument('--output-path', type=str, default=os.path.join('data_files', 'output.json'),
                        metavar='OUT', help='file where the ROC AUC scores of the model will be outputted')
    parser.add_argument('--leaf-level', type=str, default='genome_id',
                        help='taxonomical level down to which the tree will be built')
    args = parser.parse_args()

    #We get the parent_child matrix using the prexisting file or by creating it
    antibiotic = os.path.split(args.label_file)[1].split('_')[0]
    group = os.path.split(args.label_file)[1].split('_')[1]
    matrix_file = antibiotic + '_' + group + '_' + args.leaf_level + '.json'
    parent_child, topo_order, leaves, node_examples = build_pc_mat(genome_file=args.lineage_path,
                                                                   label_file=args.label_file,
                                                                   leaf_level=args.leaf_level)

    # annotating leaves with labels and features
    labels_df = pd.read_csv(args.label_file, dtype=str)

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
    print('Using CUDA: ' + str(USE_CUDA))
    device = torch.device("cuda:0" if torch.cuda.is_available() and USE_CUDA else "cpu")

    # some other hyper-parameters for training
    LR = args.lr
    BATCH_SIZE = 8
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

    for row in labels_df.itertuples():
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
        example_number += 1

    """
    for row in labels_df.itertuples():
        for leaf in leaves:
            if leaf == getattr(row, 'ID'):  # we have matched a leaf to it's row in labels_df
                phenotype = eval(getattr(row, 'Phenotype'))[0]  # the y value
                y.append(phenotype)
                features = eval(getattr(row, 'Features'))  # the x value
                X.append(features)
                mapping.append((example_number, topo_order.index(leaf)))
                example_number += 1
    """

    parent_path_tensor = build_parent_path_mat(parent_child)
    num_features = len(X[0])
    num_nodes = len(parent_child[0])
    num_edges = len(parent_path_tensor)

    root_weights = np.zeros(shape=num_features)
    edge_tensor_matrix = np.zeros(shape=(num_features, num_edges))

    test_auc_output = []
    val_auc_output = []

    for s in args.seed:
        dendronet = DendroMatrixLinReg(device, root_weights, parent_path_tensor, edge_tensor_matrix)

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

        print(X)
        print(X.size())
        print(y)
        print(y.size())

        # creating the loss function and optimizer
        loss_function = nn.BCEWithLogitsLoss()  # note for posterity: can either use DendroLinReg with this loss, or DendroLogReg with BCELoss
        if torch.cuda.is_available() and USE_CUDA:
            loss_function = loss_function.cuda()
        optimizer = torch.optim.SGD(dendronet.parameters(), lr=LR)

        #  print("train:", 100*len(train_idx)/(len(train_idx)+len(test_idx)),"%")
        #  print("val:", 100*len(val_idx)/(len(train_idx)+ len(val_idx)+ len(test_idx)),"%")
        #  print("test:", 100*len(test_idx)/(len(train_idx)+ len(val_idx)+len(test_idx)),"%")

        best_auc = 0.0
        early_stopping_count = 0
        aucs_for_plot = []

        # Generate two lists containing 1) the index in the matrix y of all the training example (whole train set)
        # 2) the corresponding positions of these training examples in the parent-path matrix
        # This is done in order generate a list of all the phenotypes of the training set (train_set_targets), and to
        # compute the AUC score of the training set after training over all batches is completed
        all_y_train_idx = []
        all_pp_train_idx = []
        for tup in train_idx:
            all_y_train_idx.append(tup[0])
            all_pp_train_idx.append(tup[1])

        train_set_targets = y[all_y_train_idx].detach().cpu().numpy()  # target values for whole training set

        # running the training loop
        for epoch in range(EPOCHS):
            print('Train epoch ' + str(epoch))
            # we'll track the running loss over each batch so we can compute the average per epoch
            running_loss = 0.0

            # getting a batch of indices
            for step, idx_batch in enumerate(tqdm(train_batch_gen)):
                optimizer.zero_grad()
                # separating corresponding rows in X (same as y) and parent_path matrix (same as parent_child order)
                idx_in_X = idx_batch[0]
                idx_in_pp_mat = idx_batch[1]
                # dendronet takes in a set of examples from X, and the corresponding column indices in the parent_path matrix
                y_hat = dendronet.forward(X[idx_in_X], idx_in_pp_mat)
                # collecting the two loss terms
                delta_loss = dendronet.delta_loss()
                train_loss = loss_function(y_hat, y[idx_in_X])  # idx_in_X is also used to fetch the appropriate entries from y
                # a sigmoid is applied to the output of the model to make them fit between 0 and 1
                # Compute root loss (L1)
                root_loss = 0
                for w in dendronet.root_weights:
                    root_loss += abs(float(w))
                loss = train_loss + (delta_loss * DPF) + (root_loss * L1)
                running_loss += float(loss)
                loss.backward(retain_graph=True)
                optimizer.step()
            print('Average training loss per batch for this epoch: ', str(running_loss/step))

            # predicted values (after sigmoid) for whole train set (in the same order as the train_set_targets list)
            y_pred = torch.sigmoid(dendronet.forward(X[all_y_train_idx], all_pp_train_idx)).detach().cpu().numpy()

            fpr, tpr, _ = roc_curve(train_set_targets, y_pred)
            roc_auc = auc(fpr, tpr)
            print("training ROC AUC for epoch: ", roc_auc)

            # Test performance using validation set at each epoch
            with torch.no_grad():
                val_loss = 0.0
                all_targets = []
                all_pred = []
                # We compute the delta and root loss right away as it doesn't change anymore for this epoch
                delta_loss = dendronet.delta_loss()
                root_loss = 0
                for w in dendronet.root_weights:
                    root_loss += abs(float(w))

                for step, idx_batch in enumerate(val_batch_gen):
                    idx_in_X = idx_batch[0]
                    idx_in_pp_mat = idx_batch[1]
                    y_hat = dendronet.forward(X[idx_in_X], idx_in_pp_mat)
                    targets = list(y[idx_in_X])  # target values for this batch
                    pred = list(torch.sigmoid(y_hat))  # predictions (after sigmoid) for this batch
                    train_loss = loss_function(y_hat, y[idx_in_X])
                    all_targets.extend(targets)
                    all_pred.extend(pred)

                    val_loss += float(train_loss + (delta_loss * DPF) + (root_loss * L1))


                fpr, tpr, _ = roc_curve(all_targets, all_pred)
                roc_auc = auc(fpr, tpr)
                print('Average loss on the validation set per batch on this epoch: ', str(val_loss / step))
                print("ROC AUC for epoch: ", roc_auc)

                aucs_for_plot.append(roc_auc)

                if roc_auc > best_auc: #Check if performance has increased on validation set (loss is decreasing)
                    best_auc = roc_auc
                    early_stopping_count = 0
                    print("Improvement!!!")
                else:
                    early_stopping_count += 1
                    print("Oups,... we are at " + str(early_stopping_count) + ", best: " + str(best_auc))

                if early_stopping_count > args.early_stopping: # If performance has not increased for long enough, we stop training
                    print("EARLY STOPPING!")                   # to avoid overfitting
                    break
        val_auc_output.append(roc_auc)

        # With training complete, we'll run the test set.
        with torch.no_grad():
            all_targets = []
            all_pred = []
            bce_loss = 0.0
            for step, idx_batch in enumerate(test_batch_gen):
                idx_in_X = idx_batch[0]
                idx_in_pp_mat = idx_batch[1]
                y_hat = dendronet.forward(X[idx_in_X], idx_in_pp_mat)
                targets = y[idx_in_X]
                pred = torch.sigmoid(y_hat)
                bce_loss += loss_function(y_hat, targets)
                all_targets.extend(list(targets))
                all_pred.extend(list(pred))
            delta_loss = dendronet.delta_loss()
            l1_loss = 0
            for w in dendronet.root_weights:
                l1_loss += abs(float(w))
            fpr, tpr, _ = roc_curve(all_targets, all_pred)
            roc_auc = auc(fpr, tpr)

            
            
            true_pos = 0
            false_pos = 0
            false_neg = 0
            true_neg = 0
            total = len(all_targets)

            for i in range(total):
                pred = float(all_pred[i])
                target = float(all_targets[i])
                if (pred > 0.5 and target == 1.0):
                    true_pos += 1
                elif (pred > 0.5 and target == 0.0):
                    false_pos += 1
                elif (pred < 0.5 and target == 0.0):
                    true_neg += 1
                elif (pred < 0.5 and target == 1.0):
                    false_neg += 1

            print("ROC AUC for test:", roc_auc)
            print('Final Delta loss:', float(delta_loss.detach().cpu().numpy()))
            print('L1 loss on test set is ', l1_loss)
            print('Average BCE loss per batch on test set:', float(bce_loss)/step)

            test_auc_output.append(roc_auc)
    """
        plt.plot(aucs_for_plot)
        plt.show()
        _, file_info = os.path.split(args.label_file)
        antibiotic = file_info.split('_')[0]
        group = file_info.split('_')[1]
        os.makedirs(os.path.join('data_files', 'AUC_plots'), exist_ok=True)
        if s in args.save_seed:
            plt.savefig(os.path.join('data_files', 'AUC_plots', antibiotic + '_' + group + '_' \
                                     + str(args.lr) + '_' + str(args.dpf) + '_' + str(args.l1) + '_' + str(args.early_stopping) \
                                     + '_seed_' + str(s) + '.png'))
    """
    output_dict = {'val_auc': val_auc_output, 'test_auc': test_auc_output}

    fileName = args.output_path
    os.makedirs(os.path.dirname(fileName), exist_ok=True)
    with open(fileName, 'w') as outfile:
        json.dump(output_dict, outfile)

