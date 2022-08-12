import os
import numpy as np
import pandas as pd
import json
import jsonpickle

from Create_Tree_image import create_tree_image

if __name__ == '__main__':

    tissue_list = []
    for tissue_file in os.listdir(os.path.join('data_files', 'CT_enhancer_features_matrices')):
        tissue_list.append(tissue_file[0:-29])

    print(tissue_list)

    nodes = [tissue for tissue in tissue_list]
    descendents = [[] for tissue in tissue_list]
    available_children = [tissue for tissue in tissue_list]

    print(nodes)
    print(descendents)

    seed = 7
    np.random.seed(seed)

    i = 0

    while len(available_children) > 1:
        while True:
            r1 = np.random.randint(0, len(available_children))
            r2 = np.random.randint(0, len(available_children))
            if r1 == r2:
                continue
            print(r1)
            print(r2)
            nodes.append(str(i))
            descendents.append([available_children[r1], available_children[r2]])
            available_children.append(str(i))
            if r1 > r2:
                del available_children[r1]
                del available_children[r2]
            else:
                del available_children[r2]
                del available_children[r1]
            break
        print(nodes)
        print(descendents)
        print(available_children)
        i += 1

    print(nodes)
    print(descendents)
    print(available_children)

    nodes.reverse()
    descendents.reverse()
    print(nodes)
    print(descendents)

    pc_mat = np.zeros(shape=(len(nodes), len(nodes)), dtype=np.int)

    for i, children in enumerate(descendents):
        for child in children:
            pc_mat[i][nodes.index(child)] = 1

    print(pc_mat)
    print(pc_mat.shape)

    pc_dict = {'parent_child_matrix': pc_mat, 'nodes_names': nodes}
    print(pc_dict)
    with open(os.path.join('data_files', 'parent_child_matrices', 'random_pc_matrix_' + str(seed) + '.json'), 'w') as pc_file:
        pickle = jsonpickle.encode(pc_dict)
        json.dump(pickle, pc_file)

    create_tree_image(pc_mat, nodes, descendents, 'random_tree_' + str(seed))
