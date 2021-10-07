import os
import pandas as pd
import json

if __name__ == "__main__":

    data = {}
    data['Group'] = []
    data['Antibiotic'] = []
    data['Leaf level'] = []
    data['AUC on val'] = []
    data['AUC on val (log)'] = []
    data['AUC on test'] = []
    data['AUC on test (log)'] = []


    for result in os.listdir(os.path.join('data_files', 'Results')):
        elements = result.split(sep='_')
        if elements[0] == 'refined' and elements[4] == 'dendronet':
            print(result)
            group = elements[2]
            antibiotic = elements[3]
            leaf_level = elements[5]
            with open(os.path.join('data_files', 'Results', result)) as file:
                dendro_dict = json.load(file)
            log_file = os.path.join('data_files', 'Results', 'refined_results_' + group + '_' + antibiotic + '_logistic.json')
            if os.path.isfile(log_file):
                with open(log_file) as file:
                    log_dict = json.load(file)

            data['Group'].append(group)
            data['Antibiotic'].append(antibiotic)
            data['Leaf level'].append(leaf_level)
            data['AUC on val'].append(dendro_dict['validation_average'])
            data['AUC on test'].append(dendro_dict["test_average"])
            if os.path.isfile(log_file) and len(list(log_dict.keys)) > 1:
                data['AUC on val (log)'].append(log_dict['validation_average'])
                data['AUC on test (log)'].append(log_dict['test_average'])
            else:
                data['AUC on val (log)'].append('-')
                data['AUC on test (log)'].append('-')

    df = pd.DataFrame(data=data)

    print(df)






