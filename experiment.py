import os
import argparse
import numpy as np
import pandas as pd
from process_genome_lineage import load_tree_and_leaves
from queue import Queue

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
    parser.add_argument('--tree-path', type=str, default=os.path.join('data_files', 'patric_tree_storage', 'betalactam')
                        , help='folder to look in for a stored tree structure')
    parser.add_argument('--label-file', type=str, default=os.path.join('data_files', 'betalactam_firmicutes_samples.csv'),
                        metavar='LF', help='file to look in for labels')
    args = parser.parse_args()

    data_tree, leaves = load_tree_and_leaves(args.tree_path)
    # annotating leaves with labels and features
    labels_df = pd.read_csv(args.label_file, dtype=str)
    """
    print(data_tree.descendants)
    print(data_tree.descendants[0].level)
    print(data_tree.descendants[0].name)
    print(data_tree.descendants[0].descendants)
    print(data_tree.descendants[0].descendants[0].level)
    print(data_tree.descendants[0].descendants[0].name)
    print(data_tree.descendants[0].descendants[0].descendants)
    print(data_tree.descendants[0].descendants[0].descendants[0].level)
    print(data_tree.descendants[0].descendants[0].descendants[0].name)
    print(data_tree.descendants[0].descendants[0].descendants[0].descendants)
    print(data_tree.descendants[0].descendants[0].descendants[0].descendants[0].level)
    print(data_tree.descendants[0].descendants[0].descendants[0].descendants[0].name)
    print(data_tree.descendants[0].descendants[0].descendants[0].descendants[0].descendants)
    print(data_tree.descendants[0].descendants[0].descendants[0].descendants[0].descendants[0].level)
    print(data_tree.descendants[0].descendants[0].descendants[0].descendants[0].descendants[0].name)
    print(data_tree.descendants[0].descendants[0].descendants[0].descendants[0].descendants[0].descendants)
    print(data_tree.descendants[0].descendants[0].descendants[0].descendants[0].descendants[0].descendants[0].level)
    print(data_tree.descendants[0].descendants[0].descendants[0].descendants[0].descendants[0].descendants[0].name)
    print(data_tree.descendants[0].descendants[0].descendants[0].descendants[0].descendants[0].descendants[0].descendants)
    print(data_tree.descendants[0].descendants[0].descendants[0].descendants[0].descendants[0].descendants[0].descendants[0].level)
    print(data_tree.descendants[0].descendants[0].descendants[0].descendants[0].descendants[0].descendants[0].descendants[0].name)
    print(data_tree.descendants[0].descendants[0].descendants[0].descendants[0].descendants[0].descendants[0].descendants[0].descendants)
    print(data_tree.descendants[0].descendants[0].descendants[0].descendants[0].descendants[0].descendants[0].descendants[0].descendants[0].level)
    print(data_tree.descendants[0].descendants[0].descendants[0].descendants[0].descendants[0].descendants[0].descendants[0].descendants[0].name)
    """
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
    X = []
    y = []
    
    
    for index,node in enumerate(topo_order):
        y.append(node.y)
        X.append(node.x)
        for child in node.descendants:
            parent_child[index][topo_order.index(child)] = 1
            
            
    
    #printing the y vector and the X and parent-child matrix 
                
    print(y)
    print(X)
    print(parent_child)
    print('Data loaded')