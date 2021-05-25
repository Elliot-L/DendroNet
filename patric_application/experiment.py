import os
import argparse
import numpy as np
import pandas as pd
from process_genome_lineage import load_tree_and_leaves
from queue import Queue

#Same imports as in dag_tutorial.py

import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from models.dendronet_models import DendroMatrixLinReg
from simulated_data_applications.generate_graph import gen_random_grid
from utils.model_utils import build_parent_path_mat, split_indices, IndicesDataset

"""
For Georgi: 
-The only relevant parameters for now are the last two (tree-path, label-file), which will point to the outputs from 
the preprocessor files
-Don't worry about the other parameters yet, we will use them once we are training DendroNet
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, metavar='N')
    parser.add_argument('--early-stopping', type=int, default=3, metavar='E',
                        help='Number of epochs without improvement before early stopping')
    parser.add_argument('--seed', type=int, default=[0], metavar='S',
                        help='random seed for train/valid split (default: 0)')
    parser.add_argument('--validation-interval', type=int, default=1, metavar='VI')
    parser.add_argument('--dpf', type=float, default=0.1, metavar='D',
                        help='scaling factor applied to delta term in the loss (default: 1.0)')
    parser.add_argument('--l1', type=float, default=1.0)
    parser.add_argument('--p', type=int, default=1)
    parser.add_argument('--output-dir', type=str, default='patric', metavar='O')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR')
    parser.add_argument('--runtest', dest='runtest', action='store_true')
    parser.add_argument('--no-runtest', dest='runtest', action='store_false')
    parser.set_defaults(runtest=False)
    parser.add_argument('--tree-path', type=str, default=os.path.join('data_files', 'patric_tree_storage', 'erythromycin')
                        , help='folder to look in for a stored tree structure')
    parser.add_argument('--label-file', type=str, default=os.path.join('data_files', 'erythromycin_firmicutes_samples.csv'),
                        metavar='LF', help='file to look in for labels')
    args = parser.parse_args()

    data_tree, leaves = load_tree_and_leaves(args.tree_path)
    # annotating leaves with labels and features
    labels_df = pd.read_csv(args.label_file, dtype=str)
    
    """
    Georgi:
    labels_df contains data about each species: it's ID, phenotype (y), and features (x) 
    The data is in an annoying format, here's a loop showing how to access it row-by-row
    and match it to a leaf in the 'leaves' object 
    """
    
    """
    For Elliot,
    I don't know if I was suppose to change something here, but I thought that we could initialize
    the features and target value of the nodes directly here. The rest of the code relies on that.
    Also, I wasn't sure if leaf.y is suppose to be a integer or a list. Here it is an integer.
    """
    
    for row in labels_df.itertuples():
        #print(row)
        for leaf in leaves:
            if leaf.name == getattr(row, 'ID'):  # we have matched a leaf to it's row in labels_df
                phenotype = eval(getattr(row, 'Phenotype'))[0]  # the y value
                leaf.y = phenotype
                features = eval(getattr(row, 'Features'))  # the x value
                leaf.x = features
            
    """
    For Georgi - here is where you can start the programming task we discussed. There are 3 components we need:
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
    
    """
    For Elliot,
    I use a queue to do a breadth first search of the tree, thus creating the list topo_order where nodes are sorted
    the way they need to be in the matrices that are required for DendroNet. Using that list and its order, 
    I created the X matrix and the y vector, in addition to the parent-child matrix. I didn't create the 
    mapping list that you asked, because the rows in the parent-child matrix correspond to the rows in X and y i.e. 
    if your node of interest in is the parent in row i of the parent-child matrix (or the child in column i),
    then its list of features are found at X[i] and its target value at y[i].
    """
    
    # flag to use cuda gpu if available
    USE_CUDA = True
    print('Using CUDA: ' + str(USE_CUDA))
    device = torch.device("cuda:0" if torch.cuda.is_available() and USE_CUDA else "cpu")

    # some other hyper-parameters for training
    LR = 0.001
    BATCH_SIZE = 8
    EPOCHS = 1000
    DPF = 0.1
    
    
    
    q = Queue(maxsize = 0)
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
    #This is the mapping between rows in the X and parent_child matrix. Only the features and target values
    #of the leaves of the tree are added to the X matrix and y vector, respectively, while all nodes are added
    #to the parent_child matrix. The list mapping contains a tuple for each leaf of the form 
    # (row_in_X, row_in_parent_child)
        
    X = []
    y = []
    feature_index = 0
    
    #Filling the X matrix and thre y vector with features and target values, respectively, from the leaves
    for index,node in enumerate(topo_order):
        if node in leaves:
            y.append(node.y)
            X.append(node.x)
            mapping.append((feature_index, index))
            feature_index += 1
        for child in node.descendants:
            parent_child[index][topo_order.index(child)] = 1        
        
    parent_path_tensor = build_parent_path_mat(parent_child)
    num_features = len(X[len(X)-1])
    num_nodes = len(parent_child[0])
    num_edges = len(parent_path_tensor)
    
    #print(y)
    #print(X)
    #print(parent_child)
    #print(parent_path_tensor)

    print('Data loaded')
    
    root_weights = np.zeros(shape=num_features)
    edge_tensor_matrix = np.zeros(shape=(num_features, num_edges))
    
    dendronet = DendroMatrixLinReg(device, root_weights, parent_path_tensor, edge_tensor_matrix)
    
    train_idx, test_idx = split_indices(mapping)

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
    loss_function = nn.BCEWithLogitsLoss()
    if torch.cuda.is_available() and USE_CUDA:
        loss_function = loss_function.cuda()
    optimizer = torch.optim.SGD(dendronet.parameters(), lr=LR)
    
    
    print("train:", 100*len(train_idx)/(len(train_idx)+len(test_idx)),"%")
    print("test:", 100*len(test_idx)/(len(train_idx)+len(test_idx)),"%")
    
        
    # running the training loop
    for epoch in range(EPOCHS):
        print('Train epoch ' + str(epoch))
        # we'll track the running loss over each batch so we can compute the average per epoch
        running_loss = 0.0
        # getting a batch of indices
        for step, idx_batch in enumerate(tqdm(train_batch_gen)):
            optimizer.zero_grad()
            #separating corresponding rows in X and parent_path matrix (same as parent_child order)
            idx1 = idx_batch[0]
            idx2 = idx_batch[1]
            # dendronet takes in a set of examples from X, and the corresponding column indices in the parent_path matrix
            y_hat = dendronet.forward(X[idx1], idx2)
            # collecting the two loss terms
            delta_loss = dendronet.delta_loss()
            # idx_batch is also used to fetch the appropriate entries from y
            train_loss = loss_function(y_hat, y[idx_batch[0]])
            running_loss += float(train_loss.detach().cpu().numpy())
            loss = train_loss + (delta_loss * DPF)
            loss.backward(retain_graph=True)
            optimizer.step()
        print('Average BCE loss: ', str(running_loss / step))

    # With training complete, we'll run the test set. We could use batching here as well if the test set was large
    with torch.no_grad():
        idx1 = []
        idx2 = []
        for idx in test_idx:
            idx1.append(idx[0])
            idx2.append(idx[1])
        y_hat = dendronet.forward(X[idx1], idx2)
        loss = loss_function(y_hat, y[idx1])
        delta_loss = dendronet.delta_loss()
        print('Final Delta loss:', float(delta_loss.detach().cpu().numpy()))
        print('Test set BCE:', float(loss.detach().cpu().numpy()))
        
    
    
    
    
    