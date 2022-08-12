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

    nodes = [tissue for tissue in tissue_list]

    descendents = [[] for tissue in tissue_file]

    new_nodes = ['muscle', 'sexual', 'gland', 'skin', 'blood_vessel',
                 'immune', 'digestive', 'nerve', 'respiratory',
                 'blood_vessel/respiratory', 'digestive/muscle',
                 'gland/sexual', 'blood_vessel/respiratory/skin',
                 'blood_vessel/digestive/skin/muscle/respiratory',
                 'immune/sexual/gland', 'All']

    new_descendents = [['heart_left_ventricle',
                        'gastroesophageal_sphincter', 'right_atrium_auricular_region',
                        'vagina', 'gastrocnemius_medialis'],
                       ['uterus', 'ovary', 'testis'],
                       ['adrenal_gland', 'body_of_pancreas', 'prostate_gland',
                        'thyroid_gland', 'right_lobe_of_liver'],
                       ['breast_epithelium', 'esophagus_squamous_epithelium',
                        'suprapubic_skin'],
                       ['ascending_aorta', 'coronary_artery',
                        'thoracic_aorta', 'tibial_artery'],
                       ['spleen', 'Peyers_patch'],
                       ['transverse_colon', 'sigmoid_colon', 'stomach', 'esophagus_muscularis_mucosa'],
                       ['tibial_nerve'],
                       ['upper_lobe_of_left_lung'],
                       ['blood_vessel', 'respiratory'],
                       ['digestive', 'muscle'],
                       ['gland', 'sexual'],
                       ['blood_vessel/respiratory', 'skin'],
                       ['blood_vessel/respiratory/skin', 'digestive/muscle'],
                       ['immune', 'gland/sexual'],
                       ['nerve', 'immune/sexual/gland', 'blood_vessel/digestive/skin/muscle/respiratory']]

    descendents = descendents + new_descendents
    descendents.reverse()
    nodes = nodes + new_nodes
    nodes.reverse()

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
    with open(os.path.join('data_files', 'parent_child_matrices', 'custom_pc_matrix.json'),
              'w') as pc_file:
        pickle = jsonpickle.encode(pc_dict)
        json.dump(pickle, pc_file)

    create_tree_image(pc_mat, nodes, descendents, 'custom_tree')





