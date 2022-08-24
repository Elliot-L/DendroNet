import os
import math
import numpy as np
import pandas as pd
import json
import jsonpickle
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
import ete3 as ete
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor


def create_tree_image(pc_mat, nodes, descendents, matrix_name):
    num_internal_nodes = 0
    for children in descendents:
        if children:
            num_internal_nodes += 1

    root = ete.Tree()
    tree_nodes_list = []
    for i in range(len(nodes)):
        tree_nodes_list.append('')
    tree_nodes_list[0] = root

    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if pc_mat[i][j] == 1.0:
                if j >= num_internal_nodes:
                    name = nodes[j]
                else:
                    name = 'internal node'
                tree_nodes_list[j] = tree_nodes_list[i].add_child(name=name)

    os.makedirs(os.path.join('data_files', 'Tree_visuals'), exist_ok=True)
    # tree_file = 'Tree_from_' + state_of_interest + '.png'
    tree_file = matrix_name + '.png'
    root.render(os.path.join('data_files', 'Tree_visuals', tree_file), w=400, units='mm')

def create_pc_mat(type, origin_of_embedding):
    os.makedirs(os.path.join('data_files', 'parent_child_matrices'), exist_ok=True)

    if type == 'data_driven':
        if os.path.isfile(os.path.join('data_files', 'parent_child_matrices', 'data_driven_pc_matrix.json')):
            with open(os.path.join('data_files', 'parent_child_matrices', 'data_driven_pc_matrix.json'), 'r') as pc_file:
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
                feature_vector = []
                for feature in features_df.columns:
                    if feature != 'cCRE_id':
                        feature_vector.extend(list(features_df.loc[:, feature]))

                enhancer_activity_list.append(feature_vector)

            n_samples = len(cell_names)
            distance_mat = np.zeros(shape=(n_samples, n_samples))
            print(len(enhancer_activity_list[0]))

            for i in range(n_samples):
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
            with open(os.path.join('data_files', 'parent_child_matrices', 'data_driven_pc_matrix.json'), 'w') as pc_file:
                pickle = jsonpickle.encode(pc_dict)
                json.dump(pickle, pc_file)

            return parent_child_mat, node_names
    elif type == 'fromEmbedding':
        embedding_file = os.path.join('data_files', 'best_embeddings', origin_of_embedding + '.json')

        with open(embedding_file, 'r') as emb_file:
            emb_dict = json.load(emb_file)

        tissues_list = list(emb_dict.keys())
        emb_list = []

        for tissue in tissues_list:
            emb_list.append(emb_dict[tissue][0])

        distance_mat = np.zeros(shape=(len(tissues_list), len(tissues_list)))

        for t1 in range(len(tissues_list)):
            for t2 in range(len(tissues_list)):
                distance_mat[t1][t2] = vector_distance(emb_list[t1], emb_list[t2])

        print(distance_mat)
        n_samples = len(tissues_list)

        agg = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='average',
                                      distance_threshold=0)
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
        for i in range(len(nodes) - len(tissues_list)):
            node_names.append('internal node')
        node_names.extend(tissues_list[::-1])
        print(node_names)

        pc_dict = {'parent_child_matrix': parent_child_mat, 'nodes_names': node_names}
        print(pc_dict)
        with open(os.path.join('data_files', 'parent_child_matrices', origin_of_embedding + '_pc_matrix.json'), 'w') as pc_file:
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


def vector_distance(l1, l2):
    distance = 0
    for i, j in zip(l1, l2):
        distance += (i - j)**2
    distance = math.sqrt(distance)
    return distance


if __name__ == '__main__':

    pc_mat, nodes = create_pc_mat('fromEmbedding', 'baselineEmbedding')
    descendents = []
    for i in range(len(nodes)):
        des_list = []
        for j in range(len(nodes)):
            if pc_mat[i][j] == 1:
                des_list.append(j)
        descendents.append(des_list)

    create_tree_image(pc_mat, nodes, descendents, 'bestBaselineEmbedding')







