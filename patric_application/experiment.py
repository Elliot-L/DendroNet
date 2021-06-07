import math
import os
import json
import argparse
import pandas as pd
from parse_patric_tree import load_tree_and_leaves
from queue import Queue
from sklearn.metrics import roc_curve, auc

#imports from dag tutorial
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from models.dendronet_models import DendroMatrixLinReg
from utils.model_utils import build_parent_path_mat, split_indices, IndicesDataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10, metavar='N')
    parser.add_argument('--early-stopping', type=int, default=3, metavar='E',
                        help='Number of epochs without improvement before early stopping')
    parser.add_argument('--seed', type=int, default=[0,1,2,3,4], metavar='S',
                        help='random seed for train/valid split (default: [0,1,2,3,4])')
    parser.add_argument('--validation-interval', type=int, default=1, metavar='VI')
    parser.add_argument('--dpf', type=float, default=1.0, metavar='D',
                        help='scaling factor applied to delta term in the loss (default: 1.0)')
    parser.add_argument('--l1', type=float, default=1.0)
    parser.add_argument('--p', type=int, default=1)
    parser.add_argument('--output-dir', type=str, default='data_files', metavar='O')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR')
    parser.add_argument('--runtest', dest='runtest', action='store_true')
    parser.add_argument('--no-runtest', dest='runtest', action='store_false')
    parser.set_defaults(runtest=False)
    parser.add_argument('--tree-path', type=str, default=os.path.join('data_files', 'patric_tree_storage', 'erythromycin')
                        , help='folder to look in for a stored tree structure')
    parser.add_argument('--label-file', type=str, default=os.path.join('data_files', 'erythromycin_firmicutes_samples.csv'),
                        metavar='LF', help='file to look in for labels')
    parser.add_argument('--output_file', type=str, default=os.path.join('output.json'),
                        metavar='OUT', help='file where the ROC AUC score of the model will be outputted')
    args = parser.parse_args()

    data_tree, leaves = load_tree_and_leaves(args.tree_path)
    # annotating leaves with labels and features
    labels_df = pd.read_csv(args.label_file, dtype=str)

    
    for row in labels_df.itertuples():
        #print(row)
        for leaf in leaves:
            if leaf.name == getattr(row, 'ID'):  # we have matched a leaf to it's row in labels_df
                phenotype = eval(getattr(row, 'Phenotype'))[0]  # the y value
                leaf.y = phenotype
                features = eval(getattr(row, 'Features'))  # the x value
                leaf.x = features
            
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
    """
        We use a queue to do a breadth first search of the tree, thus creating the list topo_order where nodes are sorted
        the way they need to be in the matrices that are required for DendroNet. Using that list and its order, 
        I created the X matrix and the y vector, in addition to the parent-child matrix.
    """
    q = Queue(maxsize=0)
    topo_order = []
    q.put(data_tree) # inputing the root in the queue
    while not q.empty():
        curr = q.get() 
        topo_order.append(curr)
        if len(curr.descendants) > 0:
            for des in curr.descendants:
                q.put(des)
    
    parent_child = np.zeros(shape=(len(topo_order), len(topo_order)), dtype=np.int)

    mapping = []
    #This will be the mapping between rows in the X and parent_child matrix. Only the features and target values
    #of the leaves of the tree are added to the X matrix and y vector, respectively, while all nodes are added
    #to the parent_child matrix. The list mapping contains a tuple for each leaf of the form 
    #(row_in_X, row_in_parent_child)
        
    X = []
    y = []
    feature_index = 0
    
    #Filling the X matrix and the y vector with features and target values, respectively, from the leaves
    for index,node in enumerate(topo_order):
        if node in leaves:
            y.append(node.y)
            X.append(node.x)
            mapping.append((feature_index, index))
            feature_index += 1
        for child in node.descendants:
            parent_child[index][topo_order.index(child)] = 1        
        
    parent_path_tensor = build_parent_path_mat(parent_child)
    num_features = len(X[0])
    num_nodes = len(parent_child[0])
    num_edges = len(parent_path_tensor)

    root_weights = np.zeros(shape=num_features)
    edge_tensor_matrix = np.zeros(shape=(num_features, num_edges))

    auc_output = []
    specificity_output = []
    sensitivity_output = []

    for s in args.seed:

        dendronet = DendroMatrixLinReg(device, root_weights, parent_path_tensor, edge_tensor_matrix)

        train_idx, test_idx = split_indices(mapping, seed=s)

        # creating idx dataset objects for batching
        train_set = IndicesDataset(train_idx)
        test_set = IndicesDataset(test_idx)

        # Setting some parameters for shuffle batch
        params = {'batch_size': BATCH_SIZE,
              'shuffle': True,
              'num_workers': 0}

        train_batch_gen = torch.utils.data.DataLoader(train_set, **params)
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
        #print("test:", 100*len(test_idx)/(len(train_idx)+len(test_idx)),"%")

        # running the training loop
        for epoch in range(EPOCHS):
            print('Train epoch ' + str(epoch))
            # we'll track the running loss over each batch so we can compute the average per epoch
            running_loss = 0.0
            y_true = []
            y_pred = []
            # getting a batch of indices
            for step, idx_batch in enumerate(tqdm(train_batch_gen)):
                optimizer.zero_grad()
                #separating corresponding rows in X (same as y) and parent_path matrix (same as parent_child order)
                idx_in_X = idx_batch[0]
                idx_in_pp_mat = idx_batch[1]
                # dendronet takes in a set of examples from X, and the corresponding column indices in the parent_path matrix
                y_hat = dendronet.forward(X[idx_in_X], idx_in_pp_mat)
                y_t = y[idx_in_X].detach().cpu().numpy()
                y_p = torch.sigmoid(y_hat).detach().cpu().numpy()  # I would usually put these through a sigmoid first, I think this still works though
                for i in range(len(y_t)):
                    y_pred.append(y_p[i])
                    y_true.append(y_t[i])
                # collecting the two loss terms
                delta_loss = dendronet.delta_loss()
                # idx_batch is also used to fetch the appropriate entries from y
                train_loss = loss_function(y_hat, y[idx_in_X])
                running_loss += float(train_loss.detach().cpu().numpy())
                root_loss = 0
                for w in dendronet.root_weights:
                    root_loss += abs(float(w))
                loss = train_loss + (delta_loss * DPF) + (root_loss * L1)
                loss.backward(retain_graph=True)
                optimizer.step()

            print('Average BCE loss: ', str(running_loss / step))
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            roc_auc = auc(fpr, tpr)
            print("ROC AUC for epoch: ", roc_auc)

        # With training complete, we'll run the test set. We could use batching here as well if the test set was large
        with torch.no_grad():
            idx_in_X = []
            idx_in_pp_mat = []
            for idx in test_idx:
                idx_in_X.append(idx[0])
                idx_in_pp_mat.append(idx[1])
            y_hat = dendronet.forward(X[idx_in_X], idx_in_pp_mat)
            y_t = y[idx_in_X].detach().cpu().numpy()
            y_p = torch.sigmoid(y_hat).detach().cpu().numpy()

            true_pos = 0
            false_pos = 0
            false_neg = 0
            true_neg = 0
            total = len(y_hat)

            for i in range(len(y_hat)):
                pred = float(y_p[i])
                real = float(y_t[i])
                if (pred > 0.5 and real == 1.0):
                    true_pos += 1
                elif (pred > 0.5 and real == 0.0):
                    false_pos += 1
                elif (pred < 0.5 and real == 0.0):
                    true_neg += 1
                else:
                    false_neg += 1

            loss = loss_function(y_hat, y[idx_in_X])
            delta_loss = dendronet.delta_loss()
            fpr, tpr, _ = roc_curve(y_t, y_p)
            roc_auc = auc(fpr, tpr)

            print("accuracy: ", (true_pos + true_neg)/total)
            print("sensitivity: ", true_pos/(true_pos + false_neg))
            print("specificity: ", true_neg/(true_neg + false_pos))
            print("true positives: ",true_pos)
            print("true negatives: ", true_neg)
            print("false positives: ", false_pos)
            print("false negatives: ", false_neg)
            print("ROC AUC for test:", roc_auc )

            print('Final Delta loss:', float(delta_loss.detach().cpu().numpy()))
            print('Test set BCE:', float(loss.detach().cpu().numpy()))

            auc_output.append(roc_auc)
            specificity_output.append(true_neg/(true_neg + false_pos))
            sensitivity_output.append(true_pos/(true_pos + false_neg))

        output_dict = {'test_auc': auc_output, 'test_specificity': specificity_output,
                       'test_sensitivity': sensitivity_output}

        with open(os.path.join(args.output_dir, args.output_file), 'w') as outfile:
            json.dump(output_dict, outfile)

        """
        For Georgi:
        
        Let's make a dictionary to hold these results over multiple runs and then save the dict as JSON:
        {
        'test_auc': [0.88, 0.92...]
        }
        """



    
    