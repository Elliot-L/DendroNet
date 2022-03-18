import os
import pandas as pd
import json
import math
from build_parent_child_mat import build_pc_mat


def entropy(antibiotic, group, threshold, leaf_level):

    label_file = os.path.join('data_files', 'subproblems', group + '_' + antibiotic + '_' + threshold,
                              group + '_' + antibiotic + '_' + threshold + '_samples.csv')
    lineage_path = os.path.join('data_files', 'genome_lineage.csv')
    parent_child_matrix, topo_order, node_examples = build_pc_mat(genome_file=lineage_path,
                                                                  samples_file=label_file,
                                                                  leaf_level=leaf_level)
    proportions = []
    total_examples = 0
    for l in node_examples:
        total_examples += len(l)

    for l in node_examples:
        proportions.append((len(l) / total_examples))

    entropy = 0
    for p in proportions:
        if p > 0:
            entropy += p*(math.log2(p))

    return (-1)*entropy


def phylo_entropy(antibiotic, group, threshold, leaf_level):
    label_file = os.path.join('data_files', 'subproblems', group + '_' + antibiotic + '_' + threshold,
                              group + '_' + antibiotic + '_' + threshold + '_samples.csv')
    lineage_path = os.path.join('data_files', 'genome_lineage.csv')
    parent_child_matrix, topo_order, node_examples = build_pc_mat(genome_file=lineage_path,
                                                                  samples_file=label_file,
                                                                  leaf_level=leaf_level)
    proportions = []
    total_examples = 0
    for l in node_examples:
        total_examples += len(l)

    for l in node_examples:
        proportions.append((len(l)/total_examples))

    phylo_entropy_value = 0
    for p in range(len(topo_order)):
        for c in range(len(topo_order)):
            prop = 0
            if parent_child_matrix[p][c] == 1.0:
                prop = subtree_proportion(parent_child_matrix, c, proportions)
            if prop > 0:
                phylo_entropy_value += prop*(math.log2(prop))

    return (-1)*phylo_entropy_value

def subtree_proportion(mat, p, proportions):
    if proportions[p] > 0:
        return proportions[p]
    else:
        total_prop = 0
        for c in range(p + 1, mat.shape[0]):
            if mat[p][c] == 1.0:
                total_prop += subtree_proportion(mat, c, proportions)
        return total_prop

def quad_entropy(antibiotic, group, threshold, leaf_level):
    label_file = os.path.join('data_files', 'subproblems', group + '_' + antibiotic + '_' + threshold,
                              group + '_' + antibiotic + '_' + threshold + '_samples.csv')
    lineage_path = os.path.join('data_files', 'genome_lineage.csv')
    parent_child_matrix, topo_order, node_examples = build_pc_mat(genome_file=lineage_path,
                                                                  samples_file=label_file,
                                                                  leaf_level=leaf_level)
    proportions = []
    total_examples = 0
    for l in node_examples:
        total_examples += len(l)

    for l in node_examples:
        proportions.append((len(l) / total_examples))

    leaves = []
    for node in range(len(topo_order)):
        if len(node_examples[node]) > 0:
            leaves.append(topo_order[node])

    paths = []  # series of nodes, for each leaf, up to the root
    for i in range(len(topo_order)):
        if topo_order[i] in leaves:
            path = [topo_order[i]]
            curr = i
            for n in range(len(topo_order) - 1, -1, -1):
                if parent_child_matrix[n][curr] == 1.0:
                    path.append(topo_order[n])
                    curr = n
            paths.append(path)
        else:
            paths.append([])

    quad_entropy_value = 0
    for i in range(len(topo_order)):
        for j in range(len(topo_order)):
            if topo_order[i] in leaves and topo_order[j] in leaves and j > i:
                closest_common_ancestor = ''
                for ancestor in paths[i]:
                    if ancestor in paths[j]:
                        closest_common_ancestor = ancestor
                        break
                distance = paths[i].index(closest_common_ancestor) + paths[j].index(closest_common_ancestor)
                quad_entropy_value += distance*proportions[i]*proportions[j]

    return quad_entropy_value


if __name__ == "__main__":
    """
    for file_name in os.listdir(os.path.join('data_files', 'subproblems')):
        file_name = file_name.split('_')
        antibiotic = file_name[1]

        print('Firmicutes, ' +  antibiotic + ', 0.0 :')
        print("Shannon's index (entropy) :" + str(entropy(antibiotic, 'Firmicutes', '0.0', 'genome_id')))
        print('Quadratic entropy :' + str(quad_entropy(antibiotic, 'Firmicutes', '0.0', 'genome_id')))
        print('Phylogenetic entropy :' + str(phylo_entropy(antibiotic, 'Firmicutes', '0.0', 'genome_id')))
        print('')
    
    print('Firmicutes, antibiotic, 0.0 :')
    print("Shannon's index (entropy) :" + str(entropy(antibiotic, 'Firmicutes', 'genome_id')))
    print('Quadratic entropy :' + str(quad_entropy(antibiotic, 'Firmicutes', 'genome_id')))
    print('Phylogenetic entropy :' + str(phylo_entropy(antibiotic, 'Firmicutes', 'genome_id')))
    """


    data1 = {}
    data1['Group'] = []
    data1['Antibiotic'] = []
    data1['Threshold'] = []
    data1['Leaf level'] = []
    data1['AUC on train'] = []
    data1['AUC on train (log)'] = []
    data1['AUC on val'] = []
    data1['AUC on val (log)'] = []
    data1['AUC on test'] = []
    data1['AUC on test (log)'] = []
    data1['DPF'] = []
    data1['LR'] = []
    data1['L1'] = []

    data2 = {}
    data2['Group'] = []
    data2['Antibiotic'] = []
    data2['Threshold'] = []
    data2['Leaf level'] = []
    data2["Shannon's index (entropy)"] = []
    data2['Quadratic entropy'] = []
    data2['Phylogenetic entropy'] = []

    for result in os.listdir(os.path.join('data_files', 'Results')):
        elements = result.split(sep='_')
        if 'refined' in result and 'dendronet' in result:
            group = elements[2]
            antibiotic = elements[3]
            threshold = elements[4]
            leaf_level = elements[6].split(sep='.')[0]
            with open(os.path.join('data_files', 'Results', result)) as file:
                dendro_dict = json.load(file)
            log_file = os.path.join('data_files', 'Results', 'refined_results_'
                                    + group + '_' + antibiotic + '_' + threshold
                                    + '_logistic.json')

            if os.path.isfile(log_file):
                with open(log_file) as file:
                    log_dict = json.load(file)
            else:
                log_dict = {}

            data1['Group'].append(group)
            data1['Antibiotic'].append(antibiotic)
            data1['Leaf level'].append(leaf_level)
            data1['Threshold'].append(threshold)
            """
            data2['Group'].append(group)
            data2['Antibiotic'].append(antibiotic)
            data2['Leaf level'].append(leaf_level)
            data2['Threshold'].append(threshold)
            """
            data1['AUC on train'].append(dendro_dict['train_average'])
            data1['AUC on val'].append(dendro_dict['validation_average'])
            if 'test_average' in dendro_dict.keys():
                data1['AUC on test'].append(dendro_dict["test_average"])
            else:
                data1['AUC on test'].append(0.0)
            data1['DPF'].append(dendro_dict["best_combinations"]["DPF"])
            data1['LR'].append(dendro_dict["best_combinations"]["LR"])
            data1['L1'].append(dendro_dict["best_combinations"]["L1"])
            if len(log_dict) > 1:
                if 'test_average' in dendro_dict.keys():
                    data1['AUC on test (log)'].append(log_dict["test_average"])
                else:
                    data1['AUC on test (log)'].append(0.0)
                data1['AUC on val (log)'].append(log_dict['validation_average'])
                data1['AUC on test (log)'].append(log_dict['test_average'])
            else:
                data1['AUC on train (log)'].append('-')
                data1['AUC on val (log)'].append('-')
                data1['AUC on test (log)'].append('-')
            """
            data2["Shannon's index (entropy)"].append(entropy(antibiotic, group, threshold, leaf_level))
            data2['Quadratic entropy'].append(quad_entropy(antibiotic, group, threshold, leaf_level))
            data2['Phylogenetic entropy'].append(phylo_entropy(antibiotic, group, threshold, leaf_level))
            """
    df1 = pd.DataFrame(data=data1)
    #df2 = pd.DataFrame(data=data2)
    print(df1.sort_values(by=['Group']))
    #print(df2.sort_values(by=['Group']))





