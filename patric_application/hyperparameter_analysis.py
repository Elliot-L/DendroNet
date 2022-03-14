import os
import argparse
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--antibiotic', type=str)
    parser.add_argument('--group', type=str)
    parser.add_argument('--threshold', type=str)
    parser.add_argument('--leaf-level', type=str)
    args = parser.parse_args()

    dendronet_results_file = os.path.join('data_files', 'Results', 'brute_results_' + args.group + '_' + args.antibiotic
                                          + '_' + args.threshold + '_dendronet_' + args.leaf_level + '.csv')
    logistic_results_file = os.path.join('data_files', 'Results', 'brute_results_' + args.group
                                          + '_' + args.antibiotic + '_' + args.threshold + '_logistic.csv')

    dendronet_df = pd.read_csv(dendronet_results_file, dtype=str)
    logistic_df = pd.read_csv(logistic_results_file, dtype=str)

    for row in range(dendronet_df.shape[0]):




