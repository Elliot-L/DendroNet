import pandas as pd
import os
import jsonpickle
import json
import argparse


def build_tab(antibiotic, group, threshold, model, leaf_level='none', seeds=[0, 1, 2, 3, 4]):
    """
    Build a csv file containing all the results for a given subproblem (brute results), in addition to a file
    containing the selected best results for the subproblem (refined results). These output files are places
    in the Results directory. The functions uses all the json files (found in the patric_tuning directory) produced
    during the hyper-parameter tuning of the given subproblem.
        Args:
            antibiotic, group, threshold leaf_level: Those are the main characteristics of a subproblem.
            model: Allow to select if we want to analyse the data produces from the tuning of the dendronet model or
                the logistic regression (baseline model)
            seeds: random seeds used to separate training from validation set (see experiment file)
    """

    if model == 'dendronet':
        data = {}
        data['LR'] = []
        data['DPF'] = []
        data['L1'] = []
        data['Early Stopping'] = []
        data['Seed'] = []
        data['Val AUC'] = []
        data['Test AUC'] = []

        #  Create a DataFrame with all the brute results (brute) for the given subproblem

        for directory in os.listdir(os.path.join('data_files', 'patric_tuning')):
            if antibiotic in directory and group in directory and model in directory and leaf_level in directory and str(threshold) in directory:
                with open(os.path.join('data_files', 'patric_tuning', directory, 'output.json')) as file:
                    JSdict = json.load(file)
                    for i, seed in enumerate(seeds):
                        data['DPF'].append(directory.split("_")[4])
                        data['LR'].append(directory.split("_")[5])
                        data['L1'].append(directory.split("_")[6])
                        data['Early Stopping'].append(directory.split("_")[7])
                        data['Seed'].append(seed)
                        data['Val AUC'].append(JSdict['val_auc'][i])
                        data['Test AUC'].append(JSdict['test_auc'][i])

        df = pd.DataFrame(data=data)
        print(df)

        #  Selecting best results

        best_results = {'validation_average': -1}

        for row in range(0, df.shape[0], len(seeds)):
            val_average_auc = 0.0
            test_average_auc = 0.0
            for seed in range(len(seeds)):
                val_average_auc += df.loc[row + seed, 'Val AUC']
                test_average_auc += df.loc[row + seed, 'Test AUC']
            val_average_auc = val_average_auc / len(seeds)
            test_average_auc = test_average_auc / len(seeds)

            if val_average_auc > best_results['validation_average']:
                best_results['best_combinations'] = {"DPF": df['DPF'][row], "LR": df['LR'][row], "L1": df['L1'][row],
                                                'Early Stop': df['Early Stopping'][row]}
                best_results['validation_average'] = val_average_auc
                best_results['test_average'] = test_average_auc

    elif model == 'logistic':
        data = {}
        data['LR'] = []
        data['L1'] = []
        data['Early Stopping'] = []
        data['Seed'] = []
        data['Val AUC'] = []
        data['Test AUC'] = []

        for directory in os.listdir(os.path.join('data_files', 'patric_tuning')):
            if antibiotic in directory and group in directory and model in directory and leaf_level == 'none' and str(threshold) in directory:
                with open(os.path.join('data_files', 'patric_tuning', directory, 'output.json')) as file:
                    JSdict = json.load(file)
                    for i, seed in enumerate(seeds):
                        data['LR'].append(directory.split("_")[4])
                        data['L1'].append(directory.split("_")[5])
                        data['Early Stopping'].append(directory.split("_")[6])
                        data['Seed'].append(seed)
                        data['Val AUC'].append(JSdict['val_auc'][i])
                        data['Test AUC'].append(JSdict['test_auc'][i])

        df = pd.DataFrame(data=data)
        print(df)

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
                results['best_combinations'] = {"LR:": df['LR'][row], "L1:": df['L1'][row],
                                                'Early Stop:': df['Early Stopping'][row]}
                results['validation_average'] = val_average_auc
                results['test_average'] = test_average_auc

    if leaf_level == 'none':
        df_file = os.path.join('data_files', 'Results', 'brute_results_' + group
                               + '_' + antibiotic + '_' + threshold + '_' + model + '.csv')
        refined_file = 'refined_results_' + group + '_' + antibiotic + '_' + threshold + '_' + model + '_'  + '.json'
    else:
        df_file = os.path.join('data_files', 'Results', 'brute_results_' + group
                               + '_' + antibiotic + '_' + threshold + '_' + model + '_' + leaf_level + '.csv')
        refined_file = 'refined_results_' + group + '_' + antibiotic + '_' + threshold + \
                       '_' + model + '_' + leaf_level + '.json'

    os.makedirs(os.path.join('data_files', 'Results'), exist_ok=True)
    df.to_csv(df_file, index=False)

    with open(os.path.join('data_files', 'Results', refined_file), 'w') as outfile:
        json.dump(best_results, outfile)

    return df, best_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--antibiotic', type=str)
    parser.add_argument('--group', type=str)
    parser.add_argument('--threshold', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--leaf-level', type=str)
    args = parser.parse_args()
    build_tab(antibiotic=args.antibiotic, group=args.group,
              model=args.model, leaf_level=args.leaf_level,
              threshold=args.threshold)
