import argparse
import os
import pandas as pd
from build_parent_child_mat import build_pc_mat
#  import networkx as nx
#  import matplotlib.pyplot as plt
import ete3 as ete

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', type=str, default='Bacteria')
    parser.add_argument('--antibiotic', type=str, default='gentamicin')
    parser.add_argument('--leaf-level', type=str, default='species')
    args = parser.parse_args()

    samples_file = args.group + '_' + args.antibiotic + '_0.0_samples.csv'
    samples_file = os.path.join('data_files', 'subproblems', args.group + '_' + args.antibiotic + '_0.0',
                                samples_file)
    parent_child, topo_order, node_examples = build_pc_mat(samples_file=samples_file,
                                                           leaf_level=args.leaf_level)
    print(parent_child)
    print(topo_order)

    all_levels = ['kingdom', 'phylum', 'safe_class', 'order', 'family', 'genus', 'species', 'genomeID']
    levels = []
    for level in all_levels:
        levels.append(level)
        if level == args.leaf_level:
            break
    cur_level_pos = [0]
    next_level_pos = []
    levels_members_pos = {}
    levels_members_names = {}
    for level in levels:
        for parent in cur_level_pos:
            for child in range(len(topo_order)):
                if parent_child[parent][child] == 1:
                    next_level_pos.append(child)
        levels_members_pos[level] = cur_level_pos
        levels_members_names[level] = []
        for i in cur_level_pos:
            levels_members_names[level].append(topo_order[i])
        cur_level_pos = next_level_pos
        next_level_pos = []

    samples_df = pd.read_csv(samples_file, dtype=str)

    pheno_dict = {}
    for row in range(samples_df.shape[0]):
        pheno_dict[samples_df.loc[row, 'ID']] = samples_df.loc[row, 'Phenotype']

    pos_neg = {}
    for leaf in levels_members_pos[args.leaf_level]:
        leaf_name = topo_order[leaf]
        samples = node_examples[leaf]
        pos = 0
        neg = 0
        print(leaf_name)
        print(samples)
        for sample in samples:
            if pheno_dict[sample] == '[1]':
                pos += 1
            else:
                neg += 1
        pos_neg[leaf_name] = (pos, neg)

    root = ete.Tree()
    curr_node = root
    node_dict = {}
    node_dict[topo_order[0]] = curr_node.add_child(name=topo_order[0])
    for i, level in enumerate(levels):
        for parent in levels_members_pos[level]:
            curr_node = node_dict[topo_order[parent]]
            for child in range(len(topo_order)):
                if parent_child[parent][child] == 1:
                    """
                    parent_name = topo_order[parent].split(' ')
                    if len(parent_name) >= 2:
                        parent_name = ' '.join(parent_name[1])
                    else:
                        parent_name = topo_order[parent]
                    child_name = topo_order[child].split(' ')
                    if len(child_name) >= 2:
                        child_name = ' '.join(child_name[1:])
                    else:
                        child_name = topo_order[child]
                    if i + 1 < len(levels) and levels[i + 1] == args.leaf_level:
                        child_name += ' ' + str(pos_neg[topo_order[child]][0]) + '/' \
                                      + str(pos_neg[topo_order[child]][1])
                    """
                    if i + 1 < len(levels) and levels[i + 1] == args.leaf_level:
                        node_dict[topo_order[child]] = curr_node.add_child(name=topo_order[child]
                                                                           + ' ' + str(pos_neg[topo_order[child]][0])
                                                                           + '/' + str(pos_neg[topo_order[child]][1]))
                    else:
                        node_dict[topo_order[child]] = curr_node.add_child(name=topo_order[child])

    os.makedirs(os.path.join('data_files', 'Tree_visuals'), exist_ok=True)
    tree_file = 'Tree_of_' + args.group + '_' + args.antibiotic + '_' + args.leaf_level + '.png'
    root.render(os.path.join('data_files', 'Tree_visuals', tree_file), w=400, units='mm')

    """
    total = 0
    for examples_list in node_examples:
        total += len(examples_list)
    print(total)
    for level in levels:
        print(level)
        for member in levels_members_pos[level]:
            print(topo_order[member] + ' : ' + str(len(node_examples[member])/total))
        if level == args.leaf_level:
            break
    
    # Defining a Class
    class TreeVisualization:

        def __init__(self):
            # visual is a list which stores all
            # the set of edges that constitutes a
            # graph
            self.visual = []

        # addEdge function inputs the vertices of an
        # edge and appends it to the visual list
        def addEdge(self, a, b):
            temp = [a, b]
            self.visual.append(temp)

        # In visualize function G is an object of
        # class Graph given by networkx G.add_edges_from(visual)
        # creates a graph with a given list
        # nx.draw_networkx(G) - plots the graph
        # plt.show() - displays the graph
        def visualize(self):
            G = nx.Graph()
            G.add_edges_from(self.visual)
            nx.draw_networkx(G)
            plt.show()


    # Driver code
    G = TreeVisualization()

    for i, level in enumerate(levels):
        for parent in levels_members_pos[level]:
            for child in range(len(topo_order)):
                if parent_child[parent][child] == 1:
                    parent_name = topo_order[parent]
                    if len(parent_name) == 2:
                        parent_name = parent_name[1]
                    else:
                        parent_name = topo_order[parent]
                    child_name = topo_order[child].split(' ')
                    if len(child_name) == 2:
                        child_name = child_name[1]
                    else:
                        child_name = topo_order[child]
                    if i + 1 < len(levels) and levels[i + 1] == args.leaf_level:
                        child_name += '\n' + str(pos_neg[topo_order[child]][0]) + '/' + str(pos_neg[topo_order[child]][1])
                    G.addEdge(parent_name, child_name)

    G.visualize()
    os.makedirs(os.path.join('data_files', 'Tree_visuals'), exist_ok=True)
    tree_file = 'Tree_of_' + args.group + '_' + args.antibiotic + '_' + args.leaf_level
    plt.savefig(os.path.join('data_files', 'Tree_visuals', tree_file))

    G.addEdge(0, 2)
    G.addEdge(1, 2)
    G.addEdge(1, 3)
    G.addEdge(5, 3)
    G.addEdge(3, 4)
    G.addEdge(1, 0)
    G.visualize()

    """