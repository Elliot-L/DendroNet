
import pandas as pd
import numpy as np
import os
import json
import jsonpickle

"""
This function takes as input the file containing the taxonomic classification of all the species of patric and the 
file containing the species of interest for a given antibiotic and build a parent_child matrix for the species of interest alone.
If all_genomes is set to True, the function will use all the available genomes that are taxinomically classified.
If force_build is set to True, the function will build the matrix no matter if it already exists on the computer,
instead of just loading it.
If save_matrix is set to True, the function will save the matrix it has built. If force_build is set to False and 
the matrix exist already, the value of save_matrix doesn't make a difference. The matrix will not be saved.

"""
def build_pc_mat(genome_file='genome_lineage.csv', label_file='firmicutes_erythromycin_samples.csv', leaf_level='genome_id', all_genomes=False, force_build=False, save_matrix=True, new_method=False):
    file_name = os.path.split(label_file)[1]
    group = file_name.split('_')[0]
    antibiotic = file_name.split('_')[1]
    if os.path.isfile('data_files/parent_child_matrices/' + group + '_' + antibiotic + '_' + leaf_level + '.json') and not force_build:
        with open('data_files/parent_child_matrices/' + group + '_' + antibiotic + '_' + leaf_level + '.json') as file:
            js_string = json.load(file)
        jdict = jsonpickle.decode(js_string)
        print("We didn't need to build the matrix")
        return jdict['parent_child'], jdict['nodes'], jdict['leaves'], jdict['node_data']

    genome_df = pd.read_csv(genome_file, delimiter='\t', dtype=str)
    genome_df = genome_df.rename(columns={'class': 'safe_class'}) #class is a keyword in python
    genome_df = genome_df[genome_df['kingdom'] == 'Bacteria']
    genome_df = genome_df[(genome_df['kingdom'].notnull()) & (genome_df['phylum'].notnull()) & (genome_df['safe_class'].notnull())
                          & (genome_df['order'].notnull()) & (genome_df['family'].notnull()) & (genome_df['genus'].notnull())
                          & (genome_df['species'].notnull())& (genome_df['genome_id'].notnull())] # removing rows with missing data
    if not all_genomes:
        label_df = pd.read_csv(label_file, dtype=str)
        ids = list(set(label_df['ID'])) # These are the IDs of the species of interest (for which we have data for a specific antibiotic)
        genome_df = genome_df[genome_df['genome_id'].isin(ids)] # collecting taxonomic information only for species of interest

    new_idx = range(genome_df.shape[0])
    genome_df.set_index(pd.Index(new_idx), inplace=True) # Reindexing (part of the rows were removed)
    all_levels = ['kingdom', 'phylum', 'safe_class', 'order', 'family', 'genus', 'species', 'genome_id']

    levels = []
    for l in all_levels:
        levels.append(l)
        if l == leaf_level:
            break

    nodes = [] # list that will contain the nodes of the tree in a topological order (corresponding to rows and columns of the matrix)
    descendents = [] # list that will contain lists containing the direct descendents of the node at corresponding position
                     # in the lis "nodes". For leaves, the list will be empty.
    node_examples = []  # list of lists that contain the training examples associated with each leaf. If the leaves are
                        # at the level of the genome_id, there will be only one training example per leaf. However, if
                        # the leaves represent families for example, then all training examples being part of a family
                        # will be used at the same node during training. For inner nodes, the list is empty.
    if not new_method:
        for i, level in enumerate(levels):
            for j in range(genome_df.shape[0]): # iterating through each row and each column of the taxonomic information
                #print(genome_df[level][j])
                #print(level)
                if genome_df[level][j] not in nodes:
                    nodes.append(genome_df[level][j])
                    descendents.append([])
                    node_examples.append([])
                pos = nodes.index(genome_df[level][j])
                if level == leaf_level:
                    node_examples[pos].append(genome_df['genome_id'][j]) # adding the training examples, its id, to appropriate node
                if level != leaf_level:
                    if genome_df[levels[i+1]][j] not in descendents[pos]:
                        descendents[pos].append(genome_df[levels[i+1]][j])  # adding in the corresponding list in "descendents"
                                                                           # the name of a child (found at the same row, in the right column)
    else:
        for i, level in enumerate(levels):
            for j in range(genome_df.shape[0]):  # iterating through each row and each column of the taxonomic information
                # print(genome_df[level][j])
                # print(level)
                if genome_df[level][j] not in nodes:
                    nodes.append(genome_df[level][j])
                    descendents.append([])
                    node_examples.append([])
                pos = nodes.index(genome_df[level][j])
                #Here, we can see that all nodes, even inner ones, will have training examples associated to them. The root for examples, will have all of them.
                node_examples[pos].append(genome_df['genome_id'][j])  # adding the training examples, its id, to appropriate node
                if level != leaf_level:
                    if genome_df[levels[i + 1]][j] not in descendents[pos]:
                        descendents[pos].append(genome_df[levels[i + 1]][j]) # adding in the corresponding list in "descendents"
                                                                             # the name of a child (found at the same row, in the right column)

    parent_child = np.zeros(shape=(len(nodes), len(nodes)), dtype=np.int)
    for i, node in enumerate(nodes):
        for child in descendents[i]:  # enumerating through all the children of a given node
            parent_child[i][nodes.index(child)] += 1  # a 1 is written as entry where edges are present in the tree

    leaves = []
    for i, l in enumerate(descendents):
        if len(l) == 0:
            leaves.append(nodes[i]) #creating a list that identifies all leaves (useful for later)

    if save_matrix:
        jdict = {}
        jdict['parent_child'] = parent_child
        jdict['nodes'] = nodes
        jdict['node_data'] = node_examples
        jdict['leaves'] = leaves
        os.makedirs(os.path.join('data_files', 'parent_child_matrices'), exist_ok=True)
        with open(os.path.join('data_files', 'parent_child_matrices', group + '_' + antibiotic + '_' + leaf_level + '.json'), 'w') as outfile:
            frozen = jsonpickle.encode(jdict)
            json.dump(frozen, outfile)

    return parent_child, nodes, leaves, node_examples

if __name__ == "__main__":
    pc, nodes, leaves, node_examples = build_pc_mat(leaf_level='order', force_build=True, save_matrix=False, genome_file='data_files/genome_lineage.csv', label_file='data_files/subproblems/firmicutes_erythromycin/erythromycin_firmicutes_samples.csv')
    print(pc)
    print(pc.shape)
    print(nodes)
    print(len(nodes))
    print(leaves)
    print(len(leaves))
    print(node_examples)
    print(len(node_examples))
    print("done")