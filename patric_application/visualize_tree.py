import argparse
import os
from build_parent_child_mat import build_pc_mat

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', type=str, default='Firmicutes')
    parser.add_argument('--antibiotic', type=str, default='erythromycin')
    parser.add_argument('--leaf-level', type=str, default='genome_id')
    args = parser.parse_args()

    samples_file = args.group + '_' + args.antibiotic + '_0.0_samples.csv'
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





