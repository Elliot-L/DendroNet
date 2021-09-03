import pandas as pd
import os
import jsonpickle
import json

def build_tab(antibiotic, group, model, leaf_level, seeds=[0, 1, 2, 3, 4]):
    df_file = os.path.join('data_files', 'Results', 'brute_results_' + group
                           + '_' + antibiotic + '_' + leaf_level + '_' + model + '.csv')

    data = {}
    data['LR'] = []
    data['DPF'] = []
    data['L1'] = []
    data['Early Stopping'] = []
    data['Seed'] = []
    data['Val AUC'] = []
    data['Test AUC'] = []

    for dir in os.listdir(os.path.join('data_files', 'patric_tuning')):
        if (antibiotic in dir and group in dir and model in dir and leaf_level in dir) or (antibiotic in dir and group in dir and model in dir and leaf_level == 'none'):
            print(dir)
            with open(os.path.join('data_files', 'patric_tuning', dir, 'output.json')) as file:
                JSdict = json.load(file)
                for i, seed in enumerate(seeds):
                    data['LR'].append(dir.split("_")[4])
                    data['DPF'].append(dir.split("_")[3])
                    data['L1'].append(dir.split("_")[5])
                    data['Early Stopping'].append(float(dir.split("_")[6]))
                    data['Seed'].append(seed)
                    data['Val AUC'].append(JSdict['val_auc'][i])
                    data['Test AUC'].append(JSdict['test_auc'][i])
    df = pd.DataFrame(data=data)

    #print(df)

    results = { 'validation_average': -1}

    for row in range(0, df.shape[0], len(seeds)):
        val_average_auc = 0.0
        test_average_auc = 0.0
        for seed in range(len(seeds)):
            val_average_auc += df['Val AUC'][row + seed]
            test_average_auc += df['Test AUC'][row + seed]
        val_average_auc = val_average_auc / len(seeds)
        test_average_auc = test_average_auc / len(seeds)


        if val_average_auc > results['validation_average']:
            results['best_combinations'] = ["LR:", df['LR'][row], "DPF:", df['DPF'][row], "L1", df['L1'][row], 'Early Stop:', df['Early Stopping'][row]]
            results['validation_average'] = val_average_auc
            results['test_average'] = test_average_auc

    print(results)
    print(df)

    os.makedirs(os.path.join('data_files', 'Results'), exist_ok=True)
    df.to_csv(df_file, index=False)

    with open(os.path.join('data_files', 'Results', 'refined_results_'
                                                    + antibiotic + '_' + group + '_' + model
                                                    + '_' + leaf_level + '.json'), 'w') as outfile:
        json.dump(results, outfile)

    return df, results


if __name__ == "__main__":
    build_tab(antibiotic='erythromycin', group='firmicutes', model='logistic', leaf_level='none')
