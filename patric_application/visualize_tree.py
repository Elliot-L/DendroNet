import argparse
import os
from build_parent_child_mat import build_pc_mat
import networkx as nx
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', type=str, default='Firmicutes')
    parser.add_argument('--antibiotic', type=str, default='betalactam')
    parser.add_argument('--leaf-level', type=str, default='genomeID')
    args = parser.parse_args()

    samples_file = args.group + '_' + args.antibiotic + '_0.0_samples.csv'
    samples_file = os.path.join('data_files', 'subproblems', args.group + '_' + args.antibiotic + '_0.0',
                                samples_file)
    parent_child, topo_order, node_examples = build_pc_mat(label_file=samples_file,
                                                           leaf_level=args.leaf_level)
    print(parent_child)
    print(topo_order)

    levels = ['kingdom', 'phylum', 'safe_class', 'order', 'family', 'genus', 'species', 'genomeID']
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


    print(levels_members_pos)
    print(levels_members_names)

    total = 0
    for examples_list in node_examples:
        total += len(examples_list)
    print(total)
    for level in levels:
        print(level)
        if level != 'genomeID':
            for member in levels_members_pos[level]:
                print(topo_order[member] + ' : ' + str(len(node_examples[member])/total))

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

    for level in levels:
        if level == 'species':
            break
        else:
            for parent in levels_members_pos[level]:
                for child in range(len(topo_order)):
                    if parent_child[parent][child] == 1:
                        G.addEdge(topo_order[parent], topo_order[child])
    G.visualize()
    os.makedirs(os.path.join('data_files', 'Tree_visuals'), exist_ok=True)
    tree_file = 'Tree_of_' + args.group + '_' + args.antibiotic + '_' + args.leaf_level
    plt.savefig(os.path.join('data_files', 'Tree_visuals', tree_file))

"""
    G.addEdge(0, 2)
    G.addEdge(1, 2)
    G.addEdge(1, 3)
    G.addEdge(5, 3)
    G.addEdge(3, 4)
    G.addEdge(1, 0)
    G.visualize()
"""
