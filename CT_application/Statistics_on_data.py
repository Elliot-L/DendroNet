import os
import pandas as pd
import json

if __name__ == '__main__':

    with open(os.path.join('data_files', 'enhancers_seqs.json'), 'r') as jfile:
        enhancer_dict = json.load(jfile)

    enhancers_list = list(enhancer_dict.keys())

    features = ['active', 'repressed', 'bivalent', 'proximal', 'distal', 'CTCF', 'nonCTCF', 'AS', 'nonAS']

    data = {'tissue': [], 'active': [], 'repressed': [], 'bivalent': [], 'act_rep_biv': [],
            'proximal': [], 'distal': [], 'both_positions': [],
            'CTCF': [], 'nonCTCF': [], 'both_CTCF': [],
            'AS': [], 'nonAS': [], 'both_AS': [], 'no_act_no_rep': [], 'unlabelled': []}

    for t_file in os.listdir(os.path.join('data_files', 'CT_enhancer_features_matrices')):
        tissue_name = t_file[0:-29]
        data['tissue'].append(tissue_name)
        print(tissue_name)
        tissue_df = pd.read_csv(os.path.join('data_files', 'CT_enhancer_features_matrices', t_file), dtype=str)

        print('All enhancers:')
        tissue_dict = {'active': 0, 'repressed': 0, 'bivalent': 0, 'act_rep_biv': 0,
                       'proximal': 0, 'distal': 0, 'both_positions': 0,
                       'CTCF': 0, 'nonCTCF': 0, 'both_CTCF': 0,
                       'AS': 0, 'nonAS': 0, 'both_AS': 0, 'no_act_no_rep': 0, 'unlabelled': 0}

        data['tissue'].append(tissue_name + '_total')

        for row in range(tissue_df.shape[0]):
            if row % 10000 == 0:
                print(row)
            for feature in features:
                if tissue_df.loc[row, feature] == '1':
                    tissue_dict[feature] += 1
            if tissue_df.loc[row, 'active'] == '1' and tissue_df.loc[row, 'repressed'] == '1' and tissue_df.loc[row, 'bivalent'] == '1':
                tissue_dict['act_rep_biv'] += 1
            if tissue_df.loc[row, 'proximal'] == '1' and tissue_df.loc[row, 'distal'] == '1':
                tissue_dict['both_positions'] += 1
            if tissue_df.loc[row, 'CTCF'] == '1' and tissue_df.loc[row, 'nonCTCF'] == '1':
                tissue_dict['both_CTCF'] += 1
            if tissue_df.loc[row, 'AS'] == '1' and tissue_df.loc[row, 'nonAS'] == '1':
                tissue_dict['both_AS'] += 1
            if tissue_df.loc[row, 'active'] == '0' and tissue_df.loc[row, 'repressed'] == '0':
                tissue_dict['no_act_no_rep'] += 1
            if tissue_df.loc[row, 'active'] == '0' and tissue_df.loc[row, 'repressed'] == '0' and tissue_df.loc[row, 'bivalent'] == '0' and tissue_df.loc[row, 'proximal'] == '0' and tissue_df.loc[row, 'distal'] == '0' and tissue_df.loc[row, 'CTCF'] == '0' and tissue_df.loc[row, 'nonCTCF'] == '0' and tissue_df.loc[row, 'AS'] == '0' and tissue_df.loc[row, 'nonAS'] == '0':
                tissue_dict['unlabelled'] += 1

        for feature in data.keys():
            if feature != 'tissue':
                data[feature].append(tissue_dict[feature])

        print('Enhancers for which we have sequence:')
        tissue_df.set_index('cCRE_id', inplace=True)
        tissue_df = tissue_df.loc[enhancers_list]

        tissue_dict = {'active': 0, 'repressed': 0, 'bivalent': 0, 'act_rep_biv': 0,
                       'proximal': 0, 'distal': 0, 'both_positions': 0,
                       'CTCF': 0, 'nonCTCF': 0, 'both_CTCF': 0,
                       'AS': 0, 'nonAS': 0, 'both_AS': 0, 'no_act_no_rep': 0, 'unlabelled': 0}

        for i, enhancer in enumerate(enhancers_list):
            if i % 10000 == 0:
                print(i)
            for feature in features:
                if tissue_df.loc[enhancer, feature] == '1':
                    tissue_dict[feature] += 1
            if tissue_df.loc[enhancer, 'active'] == '1' and tissue_df.loc[enhancer, 'repressed'] == '1' and tissue_df.loc[enhancer, 'bivalent'] == '1':
                tissue_dict['act_rep_biv'] += 1
            if tissue_df.loc[enhancer, 'proximal'] == '1' and tissue_df.loc[enhancer, 'distal'] == '1':
                tissue_dict['both_positions'] += 1
            if tissue_df.loc[enhancer, 'CTCF'] == '1' and tissue_df.loc[enhancer, 'nonCTCF'] == '1':
                tissue_dict['both_CTCF'] += 1
            if tissue_df.loc[enhancer, 'AS'] == '1' and tissue_df.loc[enhancer, 'nonAS'] == '1':
                tissue_dict['both_AS'] += 1
            if tissue_df.loc[enhancer, 'active'] == '0' and tissue_df.loc[enhancer, 'repressed'] == '0':
                tissue_dict['no_act_no_rep'] += 1
            if tissue_df.loc[enhancer, 'active'] == '0' and tissue_df.loc[enhancer, 'repressed'] == '0' and tissue_df.loc[enhancer, 'bivalent'] == '0' and tissue_df.loc[enhancer, 'proximal'] == '0' and tissue_df.loc[enhancer, 'distal'] == '0' and tissue_df.loc[enhancer, 'CTCF'] == '0' and tissue_df.loc[enhancer, 'nonCTCF'] == '0' and tissue_df.loc[enhancer, 'AS'] == '0' and tissue_df.loc[enhancer, 'nonAS'] == '0':
                tissue_dict['unlabelled'] += 1

        for feature in data.keys():
            if feature != 'tissue':
                data[feature].append(tissue_dict[feature])

    output_df = pd.DataFrame(data)

    output_df.to_csv(os.path.join('data_files', 'data_stats.csv'), index=False)



