
import pandas as pd
import numpy as np
import os
import json
import jsonpickle




def build_pc_mat(genome_file='genome_lineage.csv', label_file='Firmicutes_erythromycin_samples.csv', leaf_level='genome_id', force_build=False, save_matrix=True, new_method=False):
    """
    Build a parent-child matrix for a given subproblem (this binary matrix describes a taxonomical tree, where a one
    indicates that the node represented by a row is the parent of the node represented by the column)
    Args:
        genome_file: file with all the taxonomical classifications of the species in the PATRIC database
        label_file: file containing of the IDs of the species of interest for a givne subproblem
        leaf_level: taxonomical deep of the tree (the matrix) that will be built
        force_build: Should the function build the matrix even if we already have it in memory
        save_matrix: Should the matrix be saved after being created for future use
        new_method: is not (should not) be used right now
    """

    file_name = os.path.split(label_file)[1]
    group = file_name.split('_')[0]
    antibiotic = file_name.split('_')[1]
    matrix_file = os.path.join('data_files', 'parent_child_matrices', group + '_' + antibiotic + '_'
                               + leaf_level + '.json')
    if os.path.isfile(matrix_file) and not force_build:
        with open(matrix_file) as file:
            js_string = json.load(file)
        jdict = jsonpickle.decode(js_string)
        print("We didn't need to build the matrix")
        return jdict['parent_child'], jdict['nodes'], jdict['node_data']

    print('Building the parent-child matrix')
    genome_df = pd.read_csv(genome_file, delimiter='\t', dtype=str)
    genome_df = genome_df.rename(columns={'class': 'safe_class'}) #class is a keyword in python
    genome_df = genome_df[genome_df['kingdom'] == 'Bacteria']
    genome_df = genome_df[(genome_df['kingdom'].notnull()) & (genome_df['phylum'].notnull()) & (genome_df['safe_class'].notnull())
                          & (genome_df['order'].notnull()) & (genome_df['family'].notnull()) & (genome_df['genus'].notnull())
                          & (genome_df['species'].notnull()) & (genome_df['genome_id'].notnull())]
    label_df = pd.read_csv(label_file, dtype=str)
    ids = list(set(label_df['ID']))
    genome_df = genome_df[genome_df['genome_id'].isin(ids)]
    new_idx = range(genome_df.shape[0])
    genome_df.set_index(pd.Index(new_idx), inplace=True) # Reindexing (part of the rows were removed)
    all_levels = ['kingdom', 'phylum', 'safe_class', 'order', 'family', 'genus', 'species', 'genome_id']

    levels = []
    for level in all_levels:
        levels.append(level)
        if level == leaf_level:
            break

    """"
    list which will contain the nodes of the tree in a topological order 
    (corresponding to rows and columns of the matrix)
    """
    nodes = []

    """
    list which will contain lists containing the direct descendents of the node at corresponding position
    in the lis "nodes". For leaves, the list will be empty.
    """
    descendents = []

    """
    list of lists which contain the training examples associated with each leaf. If the leaves are
    at the level of the genome_id, there will be only one training example per leaf. However, if
    the leaves represent families for example, then all training examples being part of a family
    will be used at the same node during training. For inner nodes, the list is empty.
    """
    node_examples = []

    if not new_method:
        for i, level in enumerate(levels):
            for j in range(genome_df.shape[0]):  # iterating through each row and each column
                if genome_df.loc[j, level] not in nodes:
                    nodes.append(genome_df.loc[j, level])
                    descendents.append([])
                    node_examples.append([])
                pos = nodes.index(genome_df.loc[j, level])
                if level == leaf_level:
                    node_examples[pos].append(genome_df.loc[j, 'genome_id'])  # adding the id, to appropriate node
                if level != leaf_level:
                    if genome_df.loc[j, levels[i+1]] not in descendents[pos]:
                        descendents[pos].append(genome_df.loc[j, levels[i+1]]) # adding in the corresponding list in "descendents"
                                                                           # the name of a child (found at the same row, in the right column)
    else:
        for i, level in enumerate(levels):
            for j in range(genome_df.shape[0]):  # iterating through each row and each column
                # print(genome_df[level][j])
                # print(level)
                if genome_df.loc[j, level] not in nodes:
                    nodes.append(genome_df.loc[j, level])
                    descendents.append([])
                    node_examples.append([])
                pos = nodes.index(genome_df.loc[j, level])
                #Here, we can see that all nodes, even inner ones, will have training examples associated to them. The root for examples, will have all of them.
                node_examples[pos].append(genome_df.loc[j, 'genome_id'])  # adding the id, to appropriate node
                if level != leaf_level:
                    if genome_df.loc[j, levels[i+1]] not in descendents[pos]:
                        descendents[pos].append(genome_df.loc[j, levels[i+1]]) # adding in the corresponding list in "descendents"
                                                                             # the name of a child (found at the same row, in the right column)

    parent_child = np.zeros(shape=(len(nodes), len(nodes)), dtype=np.int)
    for i, node in enumerate(nodes):
        for child in descendents[i]:  # enumerating through all the children of a given node
            parent_child[i][nodes.index(child)] += 1  # a 1 is written as entry where edges are present in the tree

    if save_matrix:
        jdict = {}
        jdict['parent_child'] = parent_child
        jdict['nodes'] = nodes
        jdict['node_data'] = node_examples
        os.makedirs(os.path.join('data_files', 'parent_child_matrices'), exist_ok=True)
        with open(os.path.join('data_files', 'parent_child_matrices', group + '_' + antibiotic
                               + '_' + leaf_level + '.json'), 'w') as outfile:
            frozen = jsonpickle.encode(jdict)
            json.dump(frozen, outfile)

    return parent_child, nodes, node_examples

if __name__ == "__main__":
    pc, n, node_examples = build_pc_mat(leaf_level='order', force_build=True, save_matrix=True, genome_file='data_files/genome_lineage.csv', label_file='data_files/subproblems/firmicutes_erythromycin/firmicutes_erythromycin_samples.csv')
    print(pc)
    print(pc.shape)
    print(n)
    print(len(n))
    print(node_examples)
    print(len(node_examples))
    print("done")