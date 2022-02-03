
import os
import argparse
from build_parent_child_mat import build_pc_mat
import pandas as pd
import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--group', type=str, default='Bacteria')
    parser.add_argument('--antibiotic', type=str, default='erythromycin')
    parser.add_argument('--threshold', type=str, default='0.0')
    parser.add_argument('--leaf-level', type=str, default='order')
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
    pos_examples = []
    neg_examples = []
    for i, ls in enumerate(node_examples):
        if len(ls) > 0:
            if np.random.uniform(0.0, 1.0) > 0.5:
                pos_examples.extend(ls)
                pos_leaves.append(topo_order[i])
            else:
                neg_examples.extend(ls)
                neg_leaves.append(topo_order[i])
    print(new_samples_df)
    for row in range(new_samples_df.shape[0]):
        if new_samples_df.loc[row, 'ID'] in pos_examples:
            new_samples_df.at[row, 'Phenotype'] = [1]
        elif new_samples_df.loc[row, 'ID'] in neg_examples:
            new_samples_df.at[row, 'Phenotype'] = [0]

    print(new_samples_df)











