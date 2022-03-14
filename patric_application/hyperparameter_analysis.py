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

    dpfs = []
    lrs = []
    l1s = []
    early_stops = []

    for row in range(dendronet_df.shape[0]):
        if dendronet_df.loc[row, 'DPF'] not in dpfs:
            dpfs.append(dendronet_df.loc[row, 'DPF'])
        if dendronet_df.loc[row, 'LR'] not in lrs:
            lrs.append(dendronet_df.loc[row, 'LR'])
        if dendronet_df.loc[row, 'L1'] not in l1s:
            l1s.append(dendronet_df.loc[row, 'L1'])
        if dendronet_df.loc[row, 'Early Stopping'] not in early_stops:
            early_stops.append(dendronet_df.loc[row, 'Early Stopping'])







