import pandas as pd
import os
import json

if __name__ == '__main__':

    activity_df = pd.read_csv(os.path.join('data_files', 'cCRE_decoration.matrix.1.csv'), sep='\t', dtype=str)

    features = set()
    cell_types = set()

    for state in activity_df.columns[1:]:
        sep1 = state.split('-')
        cell_types.add(sep1[1])
        sep2 = sep1[0].split('.')

        for candidate_feature in sep2:
            features.add(candidate_feature)

    cell_types = list(cell_types)
    features = list(features)

    print(cell_types)
    print(len(cell_types))
    print(features)
    print(len(features))

    os.makedirs(os.path.join('data_files', 'CT_enhancer_features_matrices'), exist_ok=True)

    for ct in cell_types:
        print(ct)

        if not os.path.isfile(os.path.join('data_files', 'CT_enhancer_features_matrices',
                                           ct + '_enhancer_features_matrix.csv')):
            data = {'cCRE_id': []}

            for f in features:
                data[f] = []

            print(data)

            for row in range(activity_df.shape[0]):
                if row % 10000 == 0:
                    print(row)
                id = activity_df.loc[row, 'cCRE_id']
                data['cCRE_id'].append(id)
                for feature, activity_list in data.items():
                    if feature != 'cCRE_id':
                        activity_list.append('0')
                        for state in activity_df.columns[1:]:
                            if ct in state:
                                candidate_fs = state.split('-')[0].split('.')
                                if feature in candidate_fs and activity_df.loc[row, state] == '1':
                                    activity_list[-1] = '1'
                                    break

            new_df = pd.DataFrame(data=data)

            new_cols = ['cCRE_id', 'active', 'repressed', 'bivalent', 'proximal', 'distal',
                        'CTCF', 'nonCTCF', 'AS', 'nonAS']

            new_df = new_df[new_cols]

            new_df.to_csv(os.path.join('data_files', 'CT_enhancer_features_matrices', ct +
                                       '_enhancer_features_matrix.csv'), index=False)














