import os
import pandas as pd
import json

if __name__ == '__main__':

    with open(os.path.join('data_files', 'enhancers_seqs.json'), 'r') as jfile:
        enhancer_dict = json.load(jfile)

    enhancers_list = enhancer_dict.keys()
    cell_names = []

    features = ['active', 'repressed', 'bivalent', 'proximal', 'distal', 'CTCF', 'nonCTCF', 'AS', 'nonAS']

    data = {'active': [], 'repressed': [], 'bivalent': [], 'act_rep_biv': [],
            'proximal': [], 'distal': [], 'both_positions': [],
            'CTCF': [], 'nonCTCF': [], 'both_CTCF': [],
            'AS': [], 'nonAS': [], 'both_AS': []}

    for ct_file in os.listdir(os.path.join('data_files', 'CT_enhancer_features_matrices')):
        cell_name = ct_file[0:-29]
        cell_names.append(cell_name)
        print(cell_name)
        cell_df = pd.read_csv(os.path.join('data_files', 'CT_enhancer_features_matrices', ct_file), index_col='cCRE_id',
                              dtype=str)
        cell_df = cell_df.loc[enhancers_list]
        print(cell_df)

        cell_dict = {'active': 0, 'repressed': 0, 'bivalent': 0, 'act_rep_biv': 0,
                     'proximal': 0, 'distal': 0, 'both_positions': 0,
                     'CTCF': 0, 'nonCTCF': 0, 'both_CTCF': 0,
                     'AS': 0, 'nonAS': 0, 'both_AS': 0}

        for row in range(cell_df.shape[0]):
            if row % 10000 == 0:
                print(row)
            for feature in features:
                if cell_df.loc[row, feature] == '1':
                    cell_dict[feature] += 1
            if cell_df.loc[row, 'active'] == '1' and cell_df.loc[row, 'repressed'] == '1' and cell_df.loc[row, 'bivalent'] == '1':
                cell_dict['act_rep_biv'] += 1
            if cell_df.loc[row, 'proximal'] == '1' and cell_df.loc[row, 'distal'] == '1':
                cell_dict['both_positions'] += 1
            if cell_df.loc[row, 'CTCF'] == '1' and cell_df.loc[row, 'nonCTCF'] == '1':
                cell_dict['both_CTCF'] += 1
            if cell_df.loc[row, 'AS'] == '1' and cell_df.loc[row, 'nonAS'] == '1':
                cell_dict['both_AS'] += 1

        for feature in data.keys():
            data[feature].append(cell_dict[feature])

    output_df = pd.DataFrame(data, index=cell_names)

    output_df.to_csv(os.path.join('data_files', 'data_stats.csv'))



