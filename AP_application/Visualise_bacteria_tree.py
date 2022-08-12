import os
import numpy as np
import pandas as pd
import json
import jsonpickle
from matplotlib import pyplot as plt
import ete3 as ete

from build_pc_mat import build_pc_mat

if __name__ == '__main__':

    species_list = []

    for species_file in os.listdir(os.path.join('data_files', 'species_datasets')):
        species_name = species_file[0:-12]
        species_name = species_name.split('_')[0] + ' ' + species_name.split('_')[1]
        species_list.append(species_name)

    print(species_list)

    pc_mat, nodes = build_pc_mat(species_list)
    old_species_list = species_list
    species_list = []
    not_used_species = []

    num_internal_nodes = 0

    for species in nodes:
        if species in old_species_list:
            species_list.append(species)
        else:
            num_internal_nodes += 1

    for species in old_species_list:
        if species not in species_list:
            not_used_species.append(species)

    print('Species not used: ')
    print(not_used_species)
    print('Species used: ')
    print(species_list)

    print(pc_mat)

    root = ete.Tree()
    tree_nodes_list = []
    for i in range(len(nodes)):
        tree_nodes_list.append('')
    tree_nodes_list[0] = root

    print(tree_nodes_list)

    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if pc_mat[i][j] == 1.0:
                if j >= num_internal_nodes:
                    name = species_list[j - num_internal_nodes]
                else:
                    name = 'internal node'
                tree_nodes_list[j] = tree_nodes_list[i].add_child(name=name)

    os.makedirs(os.path.join('data_files', 'Tree_visuals'), exist_ok=True)
    # tree_file = 'Tree_from_' + state_of_interest + '.png'
    tree_file = 'combined.png'
    root.render(os.path.join('data_files', 'Tree_visuals', tree_file), w=400, units='mm')
