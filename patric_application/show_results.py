import os
import pandas as pd
import json
import math
from build_parent_child_mat import build_pc_mat

def entropy(antibiotic, group, leaf_level):
    label_file = os.path.join('data_file', 'subproblems', group + '_' + antibiotic,
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
    label_file = os.path.join('data_file', 'subproblems', group + '_' + antibiotic,
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

    leafs = []
    for node in range(len(topo_order)):
        if len(node_examples[node]) > 0:
            leafs.append(topo_order[node])


def quad_entropy(antibiotic, group, leaf_level):
    label_file = os.path.join('data_file', 'subproblems', group + '_' + antibiotic,
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

    leafs = []
    for node in range(len(topo_order)):
        if len(node_examples[node]) > 0:
            leafs.append(topo_order[node])

    paths = []
    for i in range(len(topo_order)):
        if topo_order[i] in leafs:
            path = [topo_order[i]]
            curr = i
            for n in range(len(topo_order) - 1, -1, -1):
                if parent_child_matrix[n][curr] == 1.0:
                    path.append(topo_order[n])
                    curr = n
            paths.append(path)
        else:
            path.append([])

    quad_entropy_value = 0
    for i in range(len(topo_order)):
        for j in range(len(topo_order)):
            if topo_order[i] in leafs and topo_order[j] in leafs and j > i:
                closest_common_ancestor = ''
                for ancestor in paths[i]:
                    if ancestor in paths[j]:
                        closest_common_ancestor = path[i]
                        break
                distance = paths[i].index(closest_common_ancestor) + paths[j].index(closest_common_ancestor)
                quad_entropy_value += distance*proportions[i]*proportions[j]

    return quad_entropy_value




if __name__ == "__main__":

    data = {}
    data['Group'] = []
    data['Antibiotic'] = []
    data['Leaf level'] = []
    data['AUC on val'] = []
    data['AUC on val (log)'] = []
    data['AUC on test'] = []
    data['AUC on test (log)'] = []
    data["Shannon's index (entropy)"] = []
    data['Quadratic entropy'] = []
    data['Phylogenetic entropy'] = []

    for result in os.listdir(os.path.join('data_files', 'Results')):
        elements = result.split(sep='_')
        if elements[0] == 'refined' and elements[4] == 'dendronet':
            group = elements[2]
            antibiotic = elements[3]
            leaf_level = elements[5].split(sep='.')[0]
            with open(os.path.join('data_files', 'Results', result)) as file:
                dendro_dict = json.load(file)
            log_file = os.path.join('data_files', 'Results', 'refined_results_' + group + '_' + antibiotic + '_logistic.json')
            if os.path.isfile(log_file):
                with open(log_file) as file:
                    log_dict = json.load(file)

            data['Group'].append(group)
            data['Antibiotic'].append(antibiotic)
            data['Leaf level'].append(leaf_level)
            data['AUC on val'].append(dendro_dict['validation_average'])
            data['AUC on test'].append(dendro_dict["test_average"])
            if os.path.isfile(log_file) and len(log_dict) > 1:
                data['AUC on val (log)'].append(log_dict['validation_average'])
                data['AUC on test (log)'].append(log_dict['test_average'])
            else:
                data['AUC on val (log)'].append('-')
                data['AUC on test (log)'].append('-')
            data["Shannon's index (entropy)"].append(entropy(antibiotic, group, leaf_level))
            data['Quadratic entropy'].append(quad_entropy(antibiotic, group, leaf_level))
            data['Phylogenetic entropy'].append('-')

    df = pd.DataFrame(data=data)

    print(df)






