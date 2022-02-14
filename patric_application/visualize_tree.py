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

    levels = ['kingdom', 'phylum', 'safe_class', 'order', 'family', 'genus', 'species', 'genome_id']
    cur_level_pos = [0]
    next_level_pos = []
    levels_members = {}
    for i, level in enumerate(levels):
        for pos in cur_level_pos:
            for row in range(len(topo_order)):
                if parent_child[pos][row] == 1:
                    next_level_pos.append(row)

        cur_level_pos = next_level_pos
        next_level_pos = []
        levels_members[level] = topo_order[]

    print(levels_members)





