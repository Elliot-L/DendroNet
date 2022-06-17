import os
import numpy as np
import pandas as pd
import json
import jsonpickle
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
import ete3 as ete
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor

def create_pc_mat():
    os.makedirs(os.path.join('data_files', 'parent_child_matrices'), exist_ok=True)

    if os.path.isfile(os.path.join('data_files', 'parent_child_matrices', 'combined_pc_matrix.json')):
        with open(os.path.join('data_files', 'parent_child_matrices', 'combined_pc_matrix.json'), 'r') as pc_file:
            pickle = json.load(pc_file)
        pc_dict = jsonpickle.decode(pickle)
        return pc_dict['parent_child_matrix'], pc_dict['nodes_names']

    else:
        print('Building matrix')
        cell_names = []
        enhancer_activity_list = []

        for cell_file in os.listdir(os.path.join('data_files', 'CT_enhancer_features_matrices')):
            features_df = pd.read_csv(os.path.join('data_files', 'CT_enhancer_features_matrices', cell_file))

            cell_names.append(cell_file[0:-29])
            print(cell_names)
            feature_vector = []
            for feature in features_df.columns:
                if feature != 'cCRE_id':
                    feature_vector.extend(list(features_df.loc[:, feature]))

            enhancer_activity_list.append(feature_vector)

        n_samples = len(cell_names)
        distance_mat = np.zeros(shape=(n_samples, n_samples))
        print(len(enhancer_activity_list[0]))

        for i in range(n_samples):
            print(i)
            for j in range(n_samples):
                distance_mat[i][j] = hamming_distance(enhancer_activity_list[i], enhancer_activity_list[j])
        print(distance_mat)
        agg = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='average', distance_threshold=0)
        model = agg.fit(distance_mat)

        nodes = model.children_.tolist()[::-1]  # transform ndarray to list and reverse (root will be first)
        for i in range(n_samples):
            nodes.append([])
        print(nodes)
        parent_child_mat = np.zeros(shape=(len(nodes), len(nodes)), dtype=np.int)

        for i, children in enumerate(nodes):
            for child in children:
                parent_child_mat[i][int(len(nodes) - child - 1)] = 1

        node_names = []
        for i in range(len(nodes) - len(cell_names)):
            node_names.append('internal node')
        node_names.extend(cell_names[::-1])

        pc_dict = {'parent_child_matrix': parent_child_mat, 'nodes_names': node_names}
        print(pc_dict)
        with open(os.path.join('data_files', 'parent_child_matrices', 'combined_pc_matrix.json'), 'w') as pc_file:
            pickle = jsonpickle.encode(pc_dict)
            json.dump(pickle, pc_file)

        return parent_child_mat, node_names


def hamming_distance(l1, l2):
    #  inputs must be lists of 0 or 1 of same size
    distance = 0
    for i, j in zip(l1, l2):
        if i != j:
            distance += 1
    return distance


if __name__ == '__main__':

    cell_names = []
    enhancer_activity_list = []

    for cell_file in os.listdir(os.path.join('data_files', 'CT_enhancer_features_matrices')):
        features_df = pd.read_csv(os.path.join('data_files', 'CT_enhancer_features_matrices', cell_file))

        cell_names.append(cell_file[0:-29])

        feature_vector = []
        for feature in features_df.columns[1:]:
            feature_vector.extend(list(features_df.loc[:, feature]))

        enhancer_activity_list.append(feature_vector)

    print(cell_names)
    print(len(cell_names))

    print(len(enhancer_activity_list))
    print(len(enhancer_activity_list[0]))

    """
    activity_file = os.path.join('data_files', 'cCRE_decoration.matrix.1.csv')
    activity_df = pd.read_csv(activity_file, sep='\t', dtype=str)

    cell_names = []
    states_list = []

    activity_df.set_index(activity_df.loc[:, 'cCRE_id'], inplace=True)

    print(activity_df)
    
    state_of_interest = ['repressed.proximal.CTCF.nonAS', 'repressed.distal.CTCF.AS',
                         'active.proximal.nonCTCF.nonAS', 'active.distal.CTCF.AS']

    for state in activity_df.columns[1:]:
        for s in state_of_interest:
            if s in state:
                cell_name = state.split('-')[1]
                cell_names.append(cell_name + '/' + s)
                states_list.append(state)

    print(len(cell_names))
    print(len(states_list))
    print(cell_names)
    print(states_list)

    enhancer_activity_list = []

    for state in states_list:
        enhancer_activity_list.append(list(activity_df.loc[:, state]))

    print(len(enhancer_activity_list))
    print(len(enhancer_activity_list[0]))

    """
    n = len(cell_names)

    for i in range(n):
        print(cell_names[i] + ': ')
        print(hamming_distance(enhancer_activity_list[0], enhancer_activity_list[i]))
    
    distance_mat = np.zeros(shape=(n, n))

    for i in range(n):
        for j in range(n):
            distance_mat[i][j] = hamming_distance(enhancer_activity_list[i], enhancer_activity_list[j])

    print(distance_mat)
    np.save(os.path.join('data_files', 'complete_distance_matrix.npy', 'distance_mat'))

    agg = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='average', distance_threshold=0)
    model = agg.fit(distance_mat)
    print(model.n_clusters_)
    print(model.n_leaves_)
    print(model.children_)
    print(model.children_.shape)
    print(model.labels_)
    print(model.distances_.shape)

    n_samples = len(model.labels_)
    print(n_samples)

    """
    counts = np.zeros(model.children_.shape[0])

    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    dendrogram(linkage_matrix, p=10, truncate_mode='level')

    plt.show()
    """
    minim = 400000000000
    min_i = -1
    min_j = -1
    for i in range(n_samples):
        for j in range(n_samples):
            if distance_mat[i][j] < minim and distance_mat[i][j] > 0:
                minim = distance_mat[i][j]
                min_j = j
                min_i = i

    print(minim)
    print(min_j)
    print(min_i)
    print(cell_names[min_i])
    print(cell_names[min_j])

    nodes = model.children_.tolist()[::-1]  # transform ndarray to list and reverse (root will be first)
    for i in range(n_samples):
        nodes.append([])

    print(nodes)

    parent_child_mat = np.zeros(shape=(len(nodes), len(nodes)))

    for i, children in enumerate(nodes):
        for child in children:
            parent_child_mat[i][int(len(nodes) - child - 1)] = 1.0

    print(parent_child_mat)

    root = ete.Tree()
    tree_nodes_list = []
    for i in range(len(nodes)):
        tree_nodes_list.append('')
    tree_nodes_list[0] = root

    print(tree_nodes_list)

    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if parent_child_mat[i][j] == 1.0:
                # print(i)
                # print(j)
                if j >= (len(nodes) - n_samples):
                    name = cell_names[len(nodes) - j - 1]
                else:
                    name = 'internal node'
                tree_nodes_list[j] = tree_nodes_list[i].add_child(name=name)

    os.makedirs(os.path.join('data_files', 'Tree_visuals'), exist_ok=True)
    # tree_file = 'Tree_from_' + state_of_interest + '.png'
    tree_file = 'Tree_from_combined.png'
    root.render(os.path.join('data_files', 'Tree_visuals', tree_file), w=400, units='mm')

    # constructor = DistanceTreeConstructor()









