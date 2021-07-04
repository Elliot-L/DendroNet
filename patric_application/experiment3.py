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
    parser.add_argument('--early-stopping', type=int, default=3, metavar='E',
                        help='Number of epochs without improvement before early stopping')
    parser.add_argument('--seed', type=int, default=[0, 1, 2, 3, 4], metavar='S',
                        help='random seed for train/test/validation split (default: [0,1,2,3,4])')
    parser.add_argument('--save-seed', type=int, default=[0], metavar='SS',
                        help='seeds for which the training (AUC score) will be plotted and saved')
    parser.add_argument('--validation-interval', type=int, default=1, metavar='VI')
    parser.add_argument('--dpf', type=float, default=1.0, metavar='D',
                        help='scaling factor applied to delta term in the loss (default: 1.0)')
    parser.add_argument('--l1', type=float, default=1.0)
    parser.add_argument('--p', type=int, default=1)
    parser.add_argument('--output-dir', type=str, default='data_files', metavar='O')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR')
    parser.add_argument('--runtest', dest='runtest', action='store_true')
    parser.add_argument('--no-runtest', dest='runtest', action='store_false')
    parser.set_defaults(runtest=False)
    parser.add_argument('--lineage-path', type=str, default=os.path.join('data_files', 'genome_lineage.csv',)
                        , help='file containing taxonomic classification for species from PATRIC')
    parser.add_argument('--tree-path', type=str, default=os.path.join('data_files', 'patric_tree_storage', 'erythromycin')
                        , help='folder to look in for a stored tree structure')
    parser.add_argument('--label-file', type=str, default=os.path.join('data_files', 'erythromycin_firmicutes_samples.csv'),
                        metavar='LF', help='file to look in for labels')
    parser.add_argument('--output_file', type=str, default=os.path.join('output.json'),
                        metavar='OUT', help='file where the ROC AUC score of the model will be outputted')
    parser.add_argument('--matrix-file', type=str, default=os.path.join('data_files', 'parent_child_matrices', 'erythromycin_firmicutes.json')
                        , help='File containing information about the parent-child matrix')
    args = parser.parse_args()

    #We get the parent_child matrix using the prexisting file or by creating it
    if os.path.isfile(args.matrix_file):
        with open(args.matrix_file) as file:
            matrix_data = json.load(file)
        file.close()
        matrix_dict = jsonpickle.decode(matrix_data)
        parent_child = matrix_dict['parent_child']
        topo_order = matrix_dict['nodes']
        leaves = matrix_dict['leaves']
    else:
        parent_child, topo_order, leaves = build_pc_mat(genome_file=args.lineage_path, label_file=args.label_file)

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
    #This will be the mapping between rows in the X and parent_child matrix. Only the features and target values
    #of the leaves of the tree are added to the X matrix and y vector, respectively, while all nodes are added
    #to the parent_child matrix. The list mapping contains a tuple for each leaf of the form 
    #(row_in_X, row_in_parent_child)

    X = []
    y = []
    feature_number = 0

    for row in labels_df.itertuples():
        for leaf in leaves:
            if leaf == getattr(row, 'ID'):  # we have matched a leaf to it's row in labels_df
                phenotype = eval(getattr(row, 'Phenotype'))[0]  # the y value
                y.append(phenotype)
                features = eval(getattr(row, 'Features'))  # the x value
                X.append(features)
                mapping.append((feature_number, topo_order.index(leaf)))
                feature_number += 1

    parent_path_tensor = build_parent_path_mat(parent_child)
    num_features = len(X[0])
    num_nodes = len(parent_child[0])
    num_edges = len(parent_path_tensor)

    root_weights = np.zeros(shape=num_features)
    edge_tensor_matrix = np.zeros(shape=(num_features, num_edges))

    test_auc_output = []
    val_auc_output = []
    specificity_output = []
    sensitivity_output = []

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

        # creating the loss function and optimizer
        loss_function = nn.BCEWithLogitsLoss()  # note for posterity: can either use DendroLinReg with this loss, or DendroLogReg with BCELoss
        if torch.cuda.is_available() and USE_CUDA:
            loss_function = loss_function.cuda()
        optimizer = torch.optim.SGD(dendronet.parameters(), lr=LR)

        #print("train:", 100*len(train_idx)/(len(train_idx)+len(test_idx)),"%")
        #print("val:", 100*len(val_idx)/(len(train_idx)+ len(val_idx)+ len(test_idx)),"%")
        #print("test:", 100*len(test_idx)/(len(train_idx)+ len(val_idx)+len(test_idx)),"%")

        # running the training loop
        best_auc = 0.0
        early_stopping_count = 0
        aucs_for_plot = []
        for epoch in range(EPOCHS):
            print('Train epoch ' + str(epoch))
            # we'll track the running loss over each batch so we can compute the average per epoch
            running_loss = 0.0
            #y_true = []
            #y_pred = []

            # getting a batch of indices
            for step, idx_batch in enumerate(tqdm(train_batch_gen)):
                optimizer.zero_grad()
                #separating corresponding rows in X (same as y) and parent_path matrix (same as parent_child order)
                idx_in_X = idx_batch[0]
                idx_in_pp_mat = idx_batch[1]
                # dendronet takes in a set of examples from X, and the corresponding column indices in the parent_path matrix
                y_hat = dendronet.forward(X[idx_in_X], idx_in_pp_mat)
                #y_t = y[idx_in_X].detach().cpu().numpy() #true values for this batch
                #y_p = torch.sigmoid(y_hat).detach().cpu().numpy()  #predicted values (after sigmoid) for this batch
                #The two above lines are used for calculation of AUC durint testing

                """"
                for i in range(len(y_t)):
                    y_pred.append(y_p[i])
                    y_true.append(y_t[i])
                """
                # collecting the two loss terms
                delta_loss = dendronet.delta_loss()
                train_loss = loss_function(y_hat, y[idx_in_X])  #idx_batch is also used to fetch the appropriate entries from y
                root_loss = 0
                for w in dendronet.root_weights:
                    root_loss += abs(float(w))
                loss = train_loss + (delta_loss * DPF) + (root_loss * L1)
                running_loss += float(loss)
                loss.backward(retain_graph=True)
                optimizer.step()
            print('Average training loss for epoch: ', str(running_loss/step))
            """
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            roc_auc = auc(fpr, tpr)
            print("ROC AUC for epoch: ", roc_auc)
            """

            #Test performance using validation set at each epoch
            with torch.no_grad():
                val_loss = 0.0
                y_true = []
                y_pred = []
                for step, idx_batch in enumerate(val_batch_gen):
                    idx_in_X = idx_batch[0]
                    idx_in_pp_mat = idx_batch[1]
                    y_hat = dendronet.forward(X[idx_in_X], idx_in_pp_mat)
                    y_t = list(y[idx_in_X]) #true values for this batch
                    y_p = list(torch.sigmoid(y_hat)) #predictions (after sigmoid) for this batch
                    delta_loss = dendronet.delta_loss()
                    train_loss = loss_function(y_hat, y[idx_in_X])
                    root_loss = 0
                    for w in dendronet.root_weights:
                        root_loss += abs(float(w))
                    val_loss += float(train_loss + (delta_loss * DPF) + (root_loss * L1))
                    y_true.extend(y_t)
                    y_pred.extend(y_p)

                fpr, tpr, _ = roc_curve(y_true, y_pred)
                roc_auc = auc(fpr, tpr)
                print('Average loss on the validation set on this epoch: ', str(val_loss / step))
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

        # With training complete, we'll run the test set. We could use batching here as well if the test set was large
        with torch.no_grad():
            y_true = []
            y_pred = []
            bce_loss = 0.0
            for step, idx_batch in enumerate(test_batch_gen):
                idx_in_X = idx_batch[0]
                idx_in_pp_mat = idx_batch[1]
                y_hat = dendronet.forward(X[idx_in_X], idx_in_pp_mat)
                y_t = y[idx_in_X]
                y_p = torch.sigmoid(y_hat)
                # y_pred = torch.cat((y_pred, y_p), 0)
                # y_true = torch.cat((y_true, y_t), 0)
                bce_loss += loss_function(y_hat, y_t)
                y_true.extend(list(y_t))
                y_pred.extend(list(y_p))
            delta_loss = dendronet.delta_loss()
            l1_loss = 0
            for w in dendronet.root_weights:
                l1_loss += abs(float(w))
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            roc_auc = auc(fpr, tpr)

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

            print("accuracy: ", (true_pos + true_neg)/total)
            print("sensitivity: ", true_pos/(true_pos + false_neg))
            print("specificity: ", true_neg/(true_neg + false_pos))
            print("true positives: ", true_pos)
            print("true negatives: ", true_neg)
            print("false positives: ", false_pos)
            print("false negatives: ", false_neg)
            print("ROC AUC for test:", roc_auc )

            print('Final Delta loss:', float(delta_loss.detach().cpu().numpy()))
            print('L1 loss on test set is ', l1_loss)
            print('Average BCE loss on test set:', float(bce_loss)/step)

            test_auc_output.append(roc_auc)
            specificity_output.append(true_neg/(true_neg + false_pos))
            sensitivity_output.append(true_pos/(true_pos + false_neg))

        output_dict = {'val_auc': val_auc_output, 'test_auc': test_auc_output, 'test_specificity': specificity_output,
                       'test_sensitivity': sensitivity_output}

        fileName = os.path.join(args.output_dir, args.output_file)
        os.makedirs(os.path.dirname(fileName), exist_ok=True)
        with open(os.path.join(fileName), 'w') as outfile:
            json.dump(output_dict, outfile)

        plt.plot(aucs_for_plot)
        _, file_info = os.path.split(args.label_file)
        antibiotic = file_info.split('_')[0]
        group = file_info.split('_')[1]
        os.makedirs(os.path.dirname(os.path.join('data_files', 'AUC_plots')), exist_ok=True)
        if s in args.save_seed:
            plt.savefig(os.path.join('data_files', 'AUC_plots', antibiotic + '_' + group + '_' \
                                     + str(args.lr) + '_' + str(args.dpf) + '_' + str(args.l1) + '_' + str(args.early_stopping) \
                                     + '_seed_' + str(s) + '.png'))



    """

        For Georgi:
        
        Let's make a dictionary to hold these results over multiple runs and then save the dict as JSON:
        {
        'test_auc': [0.88, 0.92...]
        }
    """



    
    