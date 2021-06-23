import pandas as pd
import os
import jsonpickle
import json

def build_tab(seeds=[0,1,2,3,4]):
    data = {}
    data['antibiotic'] = []
    data['group'] = []
    data['LR'] = []
    data['DPF'] = []
    data['L1'] = []
    data['Seed'] = []
    data['Val AUC'] = []
    data['Test AUC'] = []
    data['Sensitivity'] = []
    data['Specificity'] = []
    for dir in os.listdir('data_files\patric_tuning'):
        with open(os.path.join('data_files', 'patric_tuning', dir, 'output.json')) as file:
            JSdict = json.load(file)
            for i, seed in enumerate(seeds):
                data['antibiotic'].append(dir.split("_")[0])
                data['group'].append(dir.split("_")[1])
                data['LR'].append(dir.split("_")[4])
                data['DPF'].append(dir.split("_")[3])
                data['L1'].append(dir.split("_")[5])
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

    for row in range(0, df.shape[0], 5):
        val_average_auc = 0.0
        test_average_auc = 0.0
        for seed in range(len(seeds)):
            val_average_auc += df['Val AUC'][row + seed]
            test_average_auc += df['Test AUX'][row + seed]
        val_average_auc = val_average_auc / len(seeds)
        test_average_auc = test_average_auc / len(seeds)

        name = df['antibiotic'][row] + '_' +df['group'][row]

        if name in best_combs:
            if val_average_auc > val_averages[name]:
                best_combs[name] = (df['LR'][row], df['DPF'][row], df['L1'][row])
                val_averages[name] = val_average_auc
        else:
            best_combs[name] = (df['LR'][row], df['DPF'][row], df['L1'][row])
            val_averages[name] = val_average_auc
            test_averages[name] = test_average_auc

    print(best_combs)
    print(val_averages)
    print(test_averages)
    return df







if __name__ == "__main__":
    build_tab()