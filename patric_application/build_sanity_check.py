
import os
import argparse
from build_parent_child_mat import build_pc_mat
import pandas as pd
import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--group', type=str)
    parser.add_argument('--antibiotic', type=str)
    parser.add_argument('--threshold', type=str)
    args = parser.parse_args()

    samples_file = args.group + '_' + args.antibiotic + '_' + args.threshold + '_samples.csv'
    samples_file = os.path.join('data_files', 'subproblems', args.group + '_' + args.antibiotic, samples_file)
    matrix_file = args.group + '_' + args.antibiotic + '_' + args.leaf_level + '.json'
    parent_child, topo_order, node_examples = build_pc_mat(genome_file=args.lineage_path,
                                                           label_file=samples_file,
                                                           leaf_level=args.leaf_level)

    if os.path.isfile(samples_file):
        new_samples_df = pd.read_csv(samples_file, dtype=str)
    else:
        print('The samples file does not exist.')
        exit()

    np.random.seed(1)
    pos_leaves = []
    neg_leaves = []
    for i, ls in enumerate(node_examples):
        if len(ls) > 0:
            if np.random.uniform(0.0, 1.0) > 0.5:
                pos_leaves.append(topo_order[i])
            else:
                neg_leaves.append(topo_order[i])

    for row in range(new_samples_df.shape[0]):
        if new_samples_df.loc[row, 'ID'] in pos_leaves:









