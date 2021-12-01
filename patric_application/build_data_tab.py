import pandas as pd
import os
import jsonpickle
import json
import argparse


def build_tab(antibiotic, group, model, threshold, leaf_level, seeds=[0, 1, 2, 3, 4]):
    if model == 'dendronet':
        data = {}
        data['LR'] = []
        data['DPF'] = []
        data['L1'] = []
        data['Early Stopping'] = []
        data['Seed'] = []
        data['Val AUC'] = []
        data['Test AUC'] = []

        for directory in os.listdir(os.path.join('data_files', 'patric_tuning')):
            if antibiotic in directory and group in directory and model in directory and leaf_level in directory and str(threshold) in directory:
                print(directory)
                with open(os.path.join('data_files', 'patric_tuning', directory, 'output.json')) as file:
                    JSdict = json.load(file)
                    for i, seed in enumerate(seeds):
                        data['LR'].append(directory.split("_")[4])
                        data['DPF'].append(directory.split("_")[3])
                        data['L1'].append(directory.split("_")[5])
                        data['Early Stopping'].append(float(directory.split("_")[6]))
                        data['Seed'].append(seed)
                        data['Val AUC'].append(JSdict['val_auc'][i])
                        data['Test AUC'].append(JSdict['test_auc'][i])

        df = pd.DataFrame(data=data)
        # print(df)

        results = {'validation_average': -1}

        for row in range(0, df.shape[0], len(seeds)):
            val_average_auc = 0.0
            test_average_auc = 0.0
            for seed in range(len(seeds)):
                val_average_auc += df['Val AUC'][row + seed]
                test_average_auc += df['Test AUC'][row + seed]
            val_average_auc = val_average_auc / len(seeds)
            test_average_auc = test_average_auc / len(seeds)

            if val_average_auc > results['validation_average']:
                results['best_combinations'] = ["LR:", df['LR'][row], "DPF:", df['DPF'][row], "L1", df['L1'][row],
                                                'Early Stop:', df['Early Stopping'][row]]
                results['validation_average'] = val_average_auc
                results['test_average'] = test_average_auc

    elif model == 'logistic':
        data = {}
        data['LR'] = []
        data['Early Stopping'] = []
        data['Seed'] = []
        data['Val AUC'] = []
        data['Test AUC'] = []

        for directory in os.listdir(os.path.join('data_files', 'patric_tuning')):
            if antibiotic in directory and group in directory and model in directory and leaf_level == 'none':
                print(directory)
                with open(os.path.join('data_files', 'patric_tuning', directory, 'output.json')) as file:
                    JSdict = json.load(file)
                    for i, seed in enumerate(seeds):
                        data['LR'].append(directory.split("_")[3])
                        data['Early Stopping'].append(float(directory.split("_")[4]))
                        data['Seed'].append(seed)
                        data['Val AUC'].append(JSdict['val_auc'][i])
                        data['Test AUC'].append(JSdict['test_auc'][i])

        df = pd.DataFrame(data=data)
        # print(df)

        results = {'validation_average': -1}

        for row in range(0, df.shape[0], len(seeds)):
            val_average_auc = 0.0
            test_average_auc = 0.0
            for seed in range(len(seeds)):
                val_average_auc += df['Val AUC'][row + seed]
                test_average_auc += df['Test AUC'][row + seed]
            val_average_auc = val_average_auc / len(seeds)
            test_average_auc = test_average_auc / len(seeds)

            if val_average_auc > results['validation_average']:
                results['best_combinations'] = ["LR:", df['LR'][row], 'Early Stop:', df['Early Stopping'][row]]
                results['validation_average'] = val_average_auc
                results['test_average'] = test_average_auc

    print(results)
    print(df)

    if leaf_level == 'none':
        df_file = os.path.join('data_files', 'Results', 'brute_results_' + group
                               + '_' + antibiotic + '_' + model + '_' + threshold + '.csv')
        refined_file = 'refined_results_' + group + '_' + antibiotic + '_' + model + '_' + threshold + '.json'
    else:
        df_file = os.path.join('data_files', 'Results', 'brute_results_' + group
                               + '_' + antibiotic + '_' + model + '_' + leaf_level + '_' + threshold + '.csv')
        refined_file = 'refined_results_' + group + '_' + antibiotic + '_' + model + '_' + leaf_level + '_' + threshold + '.json'

    os.makedirs(os.path.join('data_files', 'Results'), exist_ok=True)
    df.to_csv(df_file, index=False)

    with open(os.path.join('data_files', 'Results', refined_file), 'w') as outfile:
        json.dump(results, outfile)

    return df, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--antibiotic', type=str)
    parser.add_argument('--group', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--leaf-level', type=str)
    args = parser.parse_args()
    build_tab(antibiotic=args.antibiotic, group=args.group, model=args.model, leaf_level=args.leaf_level)
