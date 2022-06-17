import os
import pandas as pd

if __name__ == '__main__':

    bad_enhancers = set()

    for ct_file in os.listdir(os.path.join('data_files', 'CT_enhancer_features_matrices')):

        ct_df = pd.read_csv(os.path.join('data_files', 'CT_enhancer_features_matrices', ct_file))
        print(ct_file)
        print(ct_df.shape[0])

        for row in range(ct_df.shape[0]):
            if ct_df.loc[row, 'active'] == 0 and ct_df.loc[row, 'repressed'] == 0 and ct_df.loc[row, 'bivalent'] == 0:
                bad_enhancers.add(ct_df.loc[row, 'cCRE_id'])
                continue
            """
            if ct_df.loc[row, 'proximal'] == 1 and ct_df.loc[row, 'distal'] == 1:
                bad_enhancers.add(ct_df.loc[row, 'cCRE_id'])
                continue
            if ct_df.loc[row, 'CTCF'] == 1 and ct_df.loc[row, 'nonCTCF'] == 1:
                bad_enhancers.add(ct_df.loc[row, 'cCRE_id'])
                continue
            if ct_df.loc[row, 'AS'] == 1 and ct_df.loc[row, 'nonAS'] == 1:
                bad_enhancers.add(ct_df.loc[row, 'cCRE_id'])
            """

    print(len(bad_enhancers))

