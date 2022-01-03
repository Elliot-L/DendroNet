import os
import argparse
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', type=str, default='Proteobacteria')
    parser.add_argument('--antibiotic', type=str, default='ciprofloxacin')
    parser.add_argument('--threshold', type=str, default='0.05')
    args = parser.parse_args()

    label_file = os.path.join('data_files', 'subproblems', args.group + '_' + args.antibiotic,
                              args.group + '_' + args.antibiotic + '_' + args.threshold + '_samples.csv')
    if os.path.isfile(label_file):
        label_df = pd.read_csv(label_file)

        ids = set()
        ids_count = 0
        features_count = []
        pos_count = 0
        neg_count = 0
        others = 0

        for row in range(label_df.shape[0]):
            ids_count += 1
            ids.append(label_df.iloc[0, row])
            features_count.append(len(label_df.iloc[4, row]))
            phenotype = label_df.iloc[2, row]
            if phenotype == '[1]':
                pos_count += 1
            elif phenotype == '[0]':
                neg_count += 1
            else:
                others += 1

        print(str(ids_count) + ' ids are present in the dataset')
        print(str(len(ids)) + ' unique ones')

        first_count = features_count[0]
        for count in features_count:
            if count != first_count:
                print('Problem: the feature vectors of all the training examples are unequal in size')

        total = pos_count + neg_count + others
        print(str(pos_count/total) + ' of the examples are positive')
        print(str(neg_count/total) + ' of the examples are negative')
        print(str(others/total) + ' of the examples are mislabelled')
    else:
        print('Label file is not created yet. Would you like to create it? (y/n)')
        answer = ''
        while answer != 'y' and answer != 'n':
            answer = input()
        if answer == 'y':

        elif













