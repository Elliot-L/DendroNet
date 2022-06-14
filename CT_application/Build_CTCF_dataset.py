import os
import pandas as pd
import json

if __name__ == '__main__':

    activity_df = pd.read_csv(os.path.join('data_files', 'cCRE_decoration.matrix.1.csv'), dtype=str, sep='\t')
    with open(os.path.join('data_files', 'enhancers_seqs.json'), 'r') as dict_file:
        enhancers_dict = json.load(dict_file)

    print(activity_df)

    CTCF_states = []

    for state in activity_df.columns:
        if '.CTCF' in state:
            print(state)
            CTCF_states.append(state)

    data = {'IDs': [], 'labels': []}

    total = 0
    pos_count = 0

    used_states = set()

    for row in range(activity_df.shape[0]):
        if row % 10000 == 0:
            print(row)
        total += 1
        id = activity_df.loc[row, 'cCRE_id']
        if id in enhancers_dict:
            data['IDs'].append(id)
            data['labels'].append(0)
            first = True
            for state in CTCF_states:
                if activity_df.loc[row, state] == '1':
                    used_states.add(state)
                    if first:
                        data['labels'][-1] = 1
                        pos_count += 1
                        first = False

    print(pos_count)
    print(total)
    print(pos_count/total)
    print(len(list(used_states)))
    df = pd.DataFrame(data)

    df.to_csv(os.path.join('data_files', 'CTCF_samples.csv'), index=False)
