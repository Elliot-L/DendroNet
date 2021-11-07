import os
import pandas as pd
import json
import math
from build_parent_child_mat import build_pc_mat


def entropy(antibiotic, group, leaf_level):

    label_file = os.path.join('data_files', 'subproblems', group + '_' + antibiotic,
                              group + '_' + antibiotic + '_samples.csv')
    lineage_path = os.path.join('data_files', 'genome_lineage.csv')
    parent_child_matrix, topo_order, node_examples = build_pc_mat(genome_file=lineage_path,
                                                                  label_file=label_file,
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


def phylo_entropy(antibiotic, group, leaf_level):
    label_file = os.path.join('data_files', 'subproblems', group + '_' + antibiotic,
                              group + '_' + antibiotic + '_samples.csv')
    lineage_path = os.path.join('data_files', 'genome_lineage.csv')
    parent_child_matrix, topo_order, node_examples = build_pc_mat(genome_file=lineage_path,
                                                                  label_file=label_file,
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
                prop = subtree(parent_child_matrix, c, proportions)
            if prop > 0:
                phylo_entropy_value += prop*(math.log2(prop))

    return (-1)*phylo_entropy_value

def subtree(mat, p, prop):
    if prop[p] > 0:
        return prop[p]
    else:
        total_prop = 0
        for c in range(p + 1, mat.shape[0]):
            if mat[p][c] == 1.0:
                total_prop += subtree(mat, c, prop)
        return total_prop


def quad_entropy(antibiotic, group, leaf_level):
    label_file = os.path.join('data_files', 'subproblems', group + '_' + antibiotic,
                              group + '_' + antibiotic + '_samples.csv')
    lineage_path = os.path.join('data_files', 'genome_lineage.csv')
    parent_child_matrix, topo_order, node_examples = build_pc_mat(genome_file=lineage_path,
                                                                  label_file=label_file,
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

    paths = []
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

    data1 = {}
    data1['Group'] = []
    data1['Antibiotic'] = []
    data1['Leaf level'] = []
    data1['AUC on val'] = []
    data1['AUC on val (log)'] = []
    data1['AUC on test'] = []
    data1['AUC on test (log)'] = []
    data1['# of Examples'] = []
    data1['# of Features'] = []
    data1['Threshold'] = []
    data2 = {}
    data2['Group'] = []
    data2['Antibiotic'] = []
    data2['Leaf level'] = []
    data2["Shannon's index (entropy)"] = []
    data2['Quadratic entropy'] = []
    data2['Phylogenetic entropy'] = []

    for result in os.listdir(os.path.join('data_files', 'Results')):
        elements = result.split(sep='_')
        if elements[0] == 'refined' and elements[4] == 'dendronet':
            group = elements[2]
            antibiotic = elements[3]
            if len(elements) == 6:
                leaf_level = elements[5].split(sep='.')[0]
            elif len(elements) == 7:
                leaf_level = elements[5] + '_' + elements[6].split(sep='.')[0]
            with open(os.path.join('data_files', 'Results', result)) as file:
                dendro_dict = json.load(file)
            log_file = os.path.join('data_files', 'Results', 'refined_results_' + group + '_' + antibiotic + '_logistic.json')
            info_file = os.path.join('data_files', 'subproblems', group + '_' + antibiotic,
                                     'subproblem_infos.json')
            if os.path.isfile(log_file):
                with open(log_file) as file:
                    log_dict = json.load(file)
            else:
                log_dict = {}
            if os.path.isfile(info_file):
                with open(info_file) as info:
                    info_dict = json.load(info)
            else:
                info_dict = {}

            data1['Group'].append(group)
            data1['Antibiotic'].append(antibiotic)
            data1['Leaf level'].append(leaf_level)
            data2['Group'].append(group)
            data2['Antibiotic'].append(antibiotic)
            data2['Leaf level'].append(leaf_level)
            data1['AUC on val'].append(dendro_dict['validation_average'])
            data1['AUC on test'].append(dendro_dict["test_average"])
            if len(log_dict) > 1:
                data1['AUC on val (log)'].append(log_dict['validation_average'])
                data1['AUC on test (log)'].append(log_dict['test_average'])
            else:
                data1['AUC on val (log)'].append('-')
                data1['AUC on test (log)'].append('-')
            if len(info_dict) > 0:
                data1['# of Examples'].append(info_dict['number of examples:'])
                data1['# of Features'].append(info_dict['number of features:'])
            else:
                data1['# of Examples'].append('-')
                data1['# of Features'].append('-')
            if 'threshold:' in info_dict.keys:
                data1['Threshold'].append(info_dict['threshold'])
            else:
                data1['Threshold'].append('-')
            data2["Shannon's index (entropy)"].append(entropy(antibiotic, group, leaf_level))
            data2['Quadratic entropy'].append(quad_entropy(antibiotic, group, leaf_level))
            data2['Phylogenetic entropy'].append(phylo_entropy(antibiotic, group, leaf_level))

    df1 = pd.DataFrame(data=data1)
    df2 = pd.DataFrame(data=data2)
    print(df1.sort_values(by=['Group']))
    print(df2.sort_values(by=['Group']))







