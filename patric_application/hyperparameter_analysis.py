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

    print(dpfs)
    print(lrs)
    print(l1s)
    print(early_stops)

    data= {}
    data['DPF'] = []
    data['LR'] = []
    data['L1'] = []
    data['Early Stop'] = []
    data['Train mean'] = []
    data['Train variance'] = []
    data['Val mean'] = []
    data['Val variance'] = []
    data['Test mean'] = []
    data['Test variance'] = []

    #print("DPFs: ")
    for dpf in dpfs:
        train_list = []
        test_list = []
        val_list = []
        for row in range(dendronet_df.shape[0]):
            if dendronet_df.loc[row, 'DPF'] == dpf:
                train_list.append(float(dendronet_df.loc[row, 'Train AUC']))
                test_list.append(float(dendronet_df.loc[row, 'Test AUC']))
                val_list.append(float(dendronet_df.loc[row, 'Val AUC']))
        train_average = 0
        test_average = 0
        val_average = 0
        for t in train_list:
            train_average += t
        for t in test_list:
            test_average += t
        for v in val_list:
            val_average += v
        train_average = train_average / len(train_list)
        test_average = test_average / len(test_list)
        val_average = val_average / len(val_list)
        train_variance = 0
        test_variance = 0
        val_variance = 0
        for t in train_list:
            train_variance += (t - train_average) ** 2
        for t in test_list:
            test_variance += (t - test_average) ** 2
        for v in val_list:
            val_variance += (v - val_average) ** 2

        train_variance = train_variance / (len(train_list) - 1)
        test_variance = test_variance / (len(test_list) - 1)
        val_variance = val_variance / (len(val_list) - 1)

        data['DPF'].append(dpf)
        data['LR'].append('-')
        data['L1'].append('-')
        data['Early Stop'].append('-')
        data['Train mean'].append(train_average)
        data['Train variance'].append(train_variance)
        data['Val mean'].append(val_average)
        data['Val variance'].append(val_variance)
        data['Test mean'].append(test_average)
        data['Test variance'].append(test_variance)

    #print("LRs: ")
    for lr in lrs:
        train_list = []
        test_list = []
        val_list = []
        for row in range(dendronet_df.shape[0]):
            if dendronet_df.loc[row, 'LR'] == lr:
                train_list.append(float(dendronet_df.loc[row, 'Train AUC']))
                test_list.append(float(dendronet_df.loc[row, 'Test AUC']))
                val_list.append(float(dendronet_df.loc[row, 'Val AUC']))
        train_average = 0
        test_average = 0
        val_average = 0
        for t in train_list:
            train_average += t
        for t in test_list:
            test_average += t
        for v in val_list:
            val_average += v
        train_average = train_average / len(train_list)
        test_average = test_average / len(test_list)
        val_average = val_average / len(val_list)
        train_variance = 0
        test_variance = 0
        val_variance = 0
        for t in train_list:
            train_variance += (t - train_average) ** 2
        for t in test_list:
            test_variance += (t - test_average) ** 2
        for v in val_list:
            val_variance += (v - val_average) ** 2

        train_variance = train_variance / (len(train_list) - 1)
        test_variance = test_variance / (len(test_list) - 1)
        val_variance = val_variance / (len(val_list) - 1)

        data['DPF'].append('-')
        data['LR'].append(lr)
        data['L1'].append('-')
        data['Early Stop'].append('-')
        data['Train mean'].append(train_average)
        data['Train variance'].append(train_variance)
        data['Val mean'].append(val_average)
        data['Val variance'].append(val_variance)
        data['Test mean'].append(test_average)
        data['Test variance'].append(test_variance)

    #print("L1s: ")
    for l1 in l1s:
        train_list = []
        test_list = []
        val_list = []
        for row in range(dendronet_df.shape[0]):
            if dendronet_df.loc[row, 'L1'] == l1:
                train_list.append(float(dendronet_df.loc[row, 'Train AUC']))
                test_list.append(float(dendronet_df.loc[row, 'Test AUC']))
                val_list.append(float(dendronet_df.loc[row, 'Val AUC']))
        train_average = 0
        test_average = 0
        val_average = 0
        for t in train_list:
            train_average += t
        for t in test_list:
            test_average += t
        for v in val_list:
            val_average += v
        train_average = train_average / len(train_list)
        test_average = test_average / len(test_list)
        val_average = val_average / len(val_list)
        train_variance = 0
        test_variance = 0
        val_variance = 0
        for t in train_list:
            train_variance = (t - train_average) ** 2
        for t in test_list:
            test_variance += (t - test_average) ** 2
        for v in val_list:
            val_variance += (v - val_average) ** 2
        train_variance = train_variance / (len(train_list) - 1)
        test_variance = test_variance / (len(test_list) - 1)
        val_variance = val_variance / (len(val_list) - 1)

        data['DPF'].append('-')
        data['LR'].append('-')
        data['L1'].append(l1)
        data['Early Stop'].append('-')
        data['Train mean'].append(train_average)
        data['Train variance'].append(train_variance)
        data['Val mean'].append(val_average)
        data['Val variance'].append(val_variance)
        data['Test mean'].append(test_average)
        data['Test variance'].append(test_variance)

    #print("Early Stops: ")
    for early in early_stops:
        train_list = []
        test_list = []
        val_list = []
        for row in range(dendronet_df.shape[0]):
            if dendronet_df.loc[row, 'Early Stopping'] == early:
                train_list.append(float(dendronet_df.loc[row, 'Train AUC']))
                test_list.append(float(dendronet_df.loc[row, 'Test AUC']))
                val_list.append(float(dendronet_df.loc[row, 'Val AUC']))
        train_average = 0
        test_average = 0
        val_average = 0
        for t in train_list:
            train_average += t
        for t in test_list:
            test_average += t
        for v in val_list:
            val_average += v
        train_average = train_average / len(train_list)
        test_average = test_average / len(test_list)
        val_average = val_average / len(val_list)
        train_variance = 0
        test_variance = 0
        val_variance = 0
        for t in train_list:
            train_variance += (t - train_average) ** 2
        for t in test_list:
            test_variance += (t - test_average) ** 2
        for v in val_list:
            val_variance += (v - val_average) ** 2
        train_variance = train_variance / (len(train_list) - 1)
        test_variance = test_variance / (len(test_list) - 1)
        val_variance = val_variance / (len(val_list) - 1)

        data['DPF'].append('-')
        data['LR'].append('-')
        data['L1'].append('-')
        data['Early Stop'].append(early)
        data['Train mean'].append(train_average)
        data['Train variance'].append(train_variance)
        data['Val mean'].append(val_average)
        data['Val variance'].append(val_variance)
        data['Test mean'].append(test_average)
        data['Test variance'].append(test_variance)

    df = pd.DataFrame(data=data)
    print(df)
    """
    os.makedirs(os.path.join('data_files', 'HyperAnalysis'), exist_ok=True)
    df_file = os.path.join('data_files', 'HyperAnalysis', args.group + '_' + args.antibiotic + '_'
                           + args.threshold + '_' + args.leaf_level + '.csv')
    df.to_csv(df_file, index=False)

    """