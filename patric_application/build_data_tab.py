import pandas as pd
import os
import jsonpickle
import json

def build_tab(antibiotic, seeds=[0, 1, 2, 3, 4]):
    df_file = os.path.join('data_files', 'Results', 'brute_results_' + antibiotic + '.csv')
    if os.path.isfile(df_file):
        df = pd.read_csv(df_file)
    else:
        data = {}
        data['antibiotic'] = []
        data['group'] = []
        data['LR'] = []
        data['DPF'] = []
        data['L1'] = []
        data['Early Stopping'] = []
        data['Seed'] = []
        data['Val AUC'] = []
        data['Test AUC'] = []
        data['Sensitivity'] = []
        data['Specificity'] = []
        for dir in os.listdir(os.path.join('data_files', 'patric_tuning')):
            with open(os.path.join('data_files', 'patric_tuning', dir, 'output.json')) as file:
                JSdict = json.load(file)
                for i, seed in enumerate(seeds):
                    data['antibiotic'].append(dir.split("_")[0])
                    data['group'].append(dir.split("_")[1])
                    data['LR'].append(dir.split("_")[4])
                    data['DPF'].append(dir.split("_")[3])
                    data['L1'].append(dir.split("_")[5])
                    data['Early Stopping'].append(float(dir.split("_")[6]))
                    data['Seed'].append(seed)
                    data['Val AUC'].append(JSdict['val_auc'][i])
                    data['Test AUC'].append(JSdict['test_auc'][i])
                    data['Sensitivity'].append(JSdict['test_sensitivity'][i])
                    data['Specificity'].append(JSdict['test_specificity'][i])
        df = pd.DataFrame(data=data)

    print(df)

    best_combs = {}
    val_averages = {}
    test_averages = {}

    for row in range(0, df.shape[0], len(seeds)):
        val_average_auc = 0.0
        test_average_auc = 0.0
        for seed in range(len(seeds)):
            val_average_auc += df['Val AUC'][row + seed]
            test_average_auc += df['Test AUC'][row + seed]
        val_average_auc = val_average_auc / len(seeds)
        test_average_auc = test_average_auc / len(seeds)

        name = df['antibiotic'][row] + '_' + df['group'][row]

        if name in best_combs:
            if val_average_auc > val_averages[name]:
                best_combs[name] = ["LR:", df['LR'][row], "DPF:", df['DPF'][row], "L1", df['L1'][row], 'Early Stop:', df['Early Stopping'][row]]
                val_averages[name] = val_average_auc
                test_averages[name] = test_average_auc
        else:
            best_combs[name] = ["LR:", df['LR'][row], "DPF:", df['DPF'][row], "L1", df['L1'][row], 'Early Stop:', df['Early Stopping'][row]]
            val_averages[name] = val_average_auc
            test_averages[name] = test_average_auc

    print(best_combs)
    print(val_averages)
    print(test_averages)

    os.makedirs(os.path.join('data_files', 'Results'), exist_ok=True)
    df.to_csv(os.path.join('data_files', 'Results', 'brute_results_' + antibiotic + '.csv'), index=False)

    output_list = [best_combs, val_averages, test_averages]

    with open(os.path.join('data_files', 'Results', 'refined_results_' + antibiotic + '.json'), 'w') as outfile:
        json.dump(output_list, outfile)

    return df, best_combs, val_averages, test_averages


if __name__ == "__main__":
    build_tab('betalactam')