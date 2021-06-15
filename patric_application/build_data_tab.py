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
    data['AUC score'] = []
    data['Sensitivity'] = []
    data['Specificity'] = []
    for dir in os.listdir('data_files\patric_tuning'):
        with open(os.path.join('data_files', 'patric_tuning', dir, 'output2.json')) as file:
            JSdict = json.load(file)
            for i, seed in enumerate(seeds):
                data['antibiotic'].append(dir.split("_")[0])
                data['group'].append(dir.split("_")[1])
                data['LR'].append(dir.split("_")[4])
                data['DPF'].append(dir.split("_")[3])
                data['L1'].append(dir.split("_")[5])
                data['Seed'].append(seed)
                data['AUC score'].append(JSdict['test_auc'][i])
                data['Sensitivity'].append(JSdict['test_sensitivity'][i])
                data['Specificity'].append(JSdict['test_specificity'][i])
    df = pd.DataFrame(data=data)

    return df







if __name__ == "__main__":
    build_tab()