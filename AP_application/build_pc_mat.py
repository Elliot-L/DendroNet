import pandas as pd
import numpy as np
import os
import json
import jsonpickle


def build_pc_mat(species_list, genome_file=os.path.join('data_files', 'genome_lineage.csv'), force_build=False, save_matrix=True):

    matrix_file = os.path.join('data_files', 'parent_child_matrices', 'combined.json')

    if os.path.isfile(matrix_file) and not force_build:
        with open(matrix_file) as file:
            js_string = json.load(file)
        jdict = jsonpickle.decode(js_string)
        return jdict['parent_child'], jdict['nodes']

    genome_df = pd.read_csv(genome_file, delimiter='\t', dtype=str)
    genome_df = genome_df.rename(columns={'class': 'safeClass'}) #class is a keyword in python
    genome_df = genome_df.rename(columns={'genome_id': 'genomeID'})
    #genome_df = genome_df[genome_df['kingdom'] == 'Bacteria']
    genome_df = genome_df[(genome_df['kingdom'].notnull()) & (genome_df['phylum'].notnull()) & (genome_df['safeClass'].notnull())
                          & (genome_df['order'].notnull()) & (genome_df['family'].notnull()) & (genome_df['genus'].notnull())
                          & (genome_df['species'].notnull()) & (genome_df['genomeID'].notnull())]
    new_idx = range(genome_df.shape[0])
    genome_df.set_index(pd.Index(new_idx), inplace=True)  # Reindexing (part of the rows were removed)

    single_id_list = []
    already_used_species = []

    for row in range(genome_df.shape[0]):
        if genome_df.loc[row, 'species'] in species_list and genome_df.loc[row, 'species'] not in already_used_species:
            already_used_species.append(genome_df.loc[row, 'species'])
            single_id_list.append(genome_df.loc[row, 'genomeID'])

    genome_df = genome_df[genome_df['genomeID'].isin(single_id_list)]

    new_idx = range(genome_df.shape[0])
    genome_df.set_index(pd.Index(new_idx), inplace=True)  # Reindexing (part of the rows were removed)

    levels = ['kingdom', 'phylum', 'safeClass', 'order', 'family', 'genus', 'species']

    """
    list which will contain the nodes of the tree in a topological order 
    (corresponding to rows and columns of the matrix)
    """
    nodes = []

    """
    list which will contain lists containing the direct descendents of the node at corresponding position
    in the list "nodes". For leaves, the list will be empty.
    """
    descendents = []

    for i, level in enumerate(levels):
        for j in range(genome_df.shape[0]):  # iterating through each row and each column
            if genome_df.loc[j, level] not in nodes:
                nodes.append(genome_df.loc[j, level])
                descendents.append([])
            pos = nodes.index(genome_df.loc[j, level])
            if level != 'species':
                if genome_df.loc[j, levels[i + 1]] not in descendents[pos]:
                    descendents[pos].append(
                        genome_df.loc[j, levels[i + 1]])  # adding in the corresponding list in "descendents" the name of a child (found at the same row, in the right column)

    print(nodes)
    print(descendents)

    parent_child = np.zeros(shape=(len(nodes), len(nodes)), dtype=np.int)
    for i, node in enumerate(nodes):
        for child in descendents[i]:  # enumerating through all the children of a given node
            parent_child[i][nodes.index(child)] += 1  # a 1 is written as entry where edges are present in the tree

    for i, node in enumerate(nodes):
        parent_child[i][i] = 0

    if save_matrix:
        jdict = {'parent_child': parent_child, 'nodes': nodes}
        os.makedirs(os.path.join('data_files', 'parent_child_matrices'), exist_ok=True)
        with open(matrix_file, 'w') as outfile:
            frozen = jsonpickle.encode(jdict)
            json.dump(frozen, outfile)

    print('Done with the construction of the matrix')
    return parent_child, nodes

if __name__ == '__main__':
    name_list = []
    for species_file in os.listdir(os.path.join('data_files', 'species_datasets')):
        name_list.append(species_file.split('_')[0] + ' ' + species_file.split('_')[1])

    pc_mat, nodes = build_pc_mat(name_list)
    print(pc_mat)
    print(nodes)

