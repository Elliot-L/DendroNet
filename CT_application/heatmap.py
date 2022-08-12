import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':

    data_array = []
    for ct_file in os.listdir(os.path.join('data_files', 'CT_enhancer_features_matrices')):
        ct_name = ct_file[0:-29]
        print(ct_name)
        #ct_array = []
        ct_df = pd.read_csv(os.path.join('data_files', 'CT_enhancer_features_matrices', ct_file))
        """
        for col in ct_df.columns[1:]:
            ct_array.extend(list(ct_df.loc[:, col]))
        """
        data_array.append((list(ct_df.loc[:, 'active']))[4000:8000])

    data_array = np.array(data_array)
    print(data_array)
    print(data_array.shape)
    sns.clustermap(data_array, cmap='coolwarm', figsize=(7, 7))
    plt.show()
