import argparse
import os
from build_parent_child_mat import build_pc_mat

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', type=str, default='Firmicutes')
    parser.add_argument('--antibiotic', type=str, default='erythromycin')
    parser.add_argument('--leaf-level', type=str, default='genome_id')
    args = parser.parse_args()

    matrix_file_name = args.group + '_' + args.antibiotic + '_(' + args.leaf_level + ').json'
    os.makedirs(os.path.join('data_files', 'parent_child_matrices'), exist_ok=True)

    samples_file = args.group + '_' + args.antibiotic + '_0.0_samples.csv'
    parent_child, topo_order, node_examples = build_pc_mat(label_file=samples_file,
                                                           leaf_level=args.leaf_level)

    print(parent_child)
    print(topo_order)

    """
    levels = ['kingdom', 'phylum', 'safe_class', 'order', 'family', 'genus', 'species', 'genome_id']
    next_parents = [0]
    row = 0
    for i, level in enumerate(levels):
        for parent in next_parents:
            for k, child in enumerate(topo_order):
                if parent_child[parent][k] == 1:
                    pass
    """


