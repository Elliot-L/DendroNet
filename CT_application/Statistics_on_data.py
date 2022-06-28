import os
import pandas as pd
import json

if __name__ == '__main__':

    with open(os.path.join('data_files', 'enhancers_seqs.json'), 'r') as jfile:
        enhancer_dict = json.load(jfile)

    enhancers_list = enhancer_dict.keys()
    cell_names = []

    features = ['active', 'repressed', 'bivalent', 'proximal', 'distal', 'CTCF', 'nonCTCF', 'AS', 'nonAS']

    data = {'tissue': [], 'active': [], 'repressed': [], 'bivalent': [], 'act_rep_biv': [],
            'proximal': [], 'distal': [], 'both_positions': [],
            'CTCF': [], 'nonCTCF': [], 'both_CTCF': [],
            'AS': [], 'nonAS': [], 'both_AS': [], 'no_act_no_rep': [], 'unlabelled': []}

    for ct_file in os.listdir(os.path.join('data_files', 'CT_enhancer_features_matrices')):
        cell_name = ct_file[0:-29]
        data['tissue'].append(cell_name)
        print(cell_name)
        cell_df = pd.read_csv(os.path.join('data_files', 'CT_enhancer_features_matrices', ct_file), dtype=str)
        cell_df_with_pos = cell_df.loc[enhancers_list]
        cell_df_with_pos.set_index('cCRE_id', inplace=True)

        cell_dict = {'active': 0, 'repressed': 0, 'bivalent': 0, 'act_rep_biv': 0,
                     'proximal': 0, 'distal': 0, 'both_positions': 0,
                     'CTCF': 0, 'nonCTCF': 0, 'both_CTCF': 0,
                     'AS': 0, 'nonAS': 0, 'both_AS': 0, 'no_act_no_rep': [], 'unlabelled': []}

        for i, enhancer in enumerate(enhancers_list):
            if i % 10000 == 0:
                print(i)
            for feature in features:
                if cell_df.loc[enhancer, feature] == '1':
                    cell_dict[feature] += 1
            if cell_df.loc[enhancer, 'active'] == '1' and cell_df.loc[enhancer, 'repressed'] == '1' and cell_df.loc[enhancer, 'bivalent'] == '1':
                cell_dict['act_rep_biv'] += 1
            if cell_df.loc[enhancer, 'proximal'] == '1' and cell_df.loc[enhancer, 'distal'] == '1':
                cell_dict['both_positions'] += 1
            if cell_df.loc[enhancer, 'CTCF'] == '1' and cell_df.loc[enhancer, 'nonCTCF'] == '1':
                cell_dict['both_CTCF'] += 1
            if cell_df.loc[enhancer, 'AS'] == '1' and cell_df.loc[enhancer, 'nonAS'] == '1':
                cell_dict['both_AS'] += 1
            if cell_df.loc[enhancer, 'active'] == '0' and cell_df.loc[enhancer, 'repressed'] == '0':
                cell_dict['no_act_no_rep'] += 1
            if cell_df.loc[enhancer, 'active'] == '0' and cell_df.loc[enhancer, 'repressed'] == '0' and cell_df.loc[enhancer, 'bivalent'] == '0' and cell_df.loc[enhancer, 'proximal'] == '0' and cell_df.loc[enhancer, 'distal'] == '0' and cell_df.loc[enhancer, 'CTCF'] == '0' and cell_df.loc[enhancer, 'nonCTCT'] == '0' and cell_df.loc[enhancer, 'AS'] == '0' and cell_df.loc[enhancer, 'nonAS'] == '0':
                cell_dict['unlabelled'] += 1

        for feature in data.keys():
            data[feature].append(cell_dict[feature])

        """
        cell_dict = {'active': 0, 'repressed': 0, 'bivalent': 0, 'act_rep_biv': 0,
                     'proximal': 0, 'distal': 0, 'both_positions': 0,
                     'CTCF': 0, 'nonCTCF': 0, 'both_CTCF': 0,
                     'AS': 0, 'nonAS': 0, 'both_AS': 0}
        
        data['tissue'].append(cell_name + '_total')
        print(cell_name + '_total')

        for row in cell_df:
            if i % 10000 == 0:
                print(i)
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
        """
    output_df = pd.DataFrame(data)

    output_df.to_csv(os.path.join('data_files', 'data_stats.csv'), index=False)



