import os
import argparse
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', type=str, default='Firmicutes')
    parser.add_argument('--antibiotic', type=str, default='betalactam')
    parser.add_argument('--threshold', type=str, default='0.0')
    args = parser.parse_args()

    basic_file = os.path.join('data_files', 'basic_files', args.group + '_' + args.antibiotic + '_basic.csv')

    if os.path.isfile(basic_file):
        print('Basic file related to this subproblem:')
        basic_df = pd.read_csv(basic_file, dtype=str, sep='\t')

        possible_phenotypes_and_count = {}
        ids_count = 0
        ids = []

        for row in range(basic_df.shape[0]):
            ids_count += 1
            ids.append(basic_df.iloc[row, 0])
            phenotype = basic_df.iloc[row, 1]
            if phenotype not in possible_phenotypes_and_count:
                possible_phenotypes_and_count[phenotype] = 1
            else:
                possible_phenotypes_and_count[phenotype] += 1

        ids = set(ids)
        print(str(ids_count) + ' genome were found in this file.')
        print(str(len(ids)) + ' unique ones.')

        print('All phenotypes that were found and their respective count:')
        for phenotype, count in possible_phenotypes_and_count.items():
            print(phenotype + ': ' + str(count))

    else:
        print('The basic file associated to this problem does not exist. Create it using the ' +
              'create_basic_from_amr_file.py file, or download it from the PATRIC terminal.')

    samples_file = os.path.join('data_files', 'subproblems', args.group + '_' + args.antibiotic,
                              args.group + '_' + args.antibiotic + '_' + args.threshold + '_samples.csv')
    if os.path.isfile(samples_file):
        print('Sample file related to this subproblem:')

        samples_df = pd.read_csv(samples_file, dtype=str)

        ids = []
        ids_count = 0
        features_count = []
        pos_count = 0
        neg_count = 0
        others = 0

        for row in range(samples_df.shape[0]):
            ids_count += 1
            ids.append(samples_df.iloc[row, 0])
            features_count.append(len((samples_df.iloc[row, 2]).split(',')))
            phenotype = samples_df.iloc[row, 1]
            if phenotype == '[1]':
                pos_count += 1
            elif phenotype == '[0]':
                neg_count += 1
            else:
                others += 1

        print(str(ids_count) + ' ids are present in the dataset')
        print(str(len(set(ids))) + ' unique ones')

        first_count = features_count[0]
        no_problem = True
        for count in features_count:
            if count != first_count:
                print('Problem: the feature vectors of some training examples are unequal in size')
                no_problem = False
                break
        if no_problem:
            print('All training examples have feature vectors of size ' + str(first_count))

        total = pos_count + neg_count + others
        print(str((pos_count/total)*100) + ' % of the examples are positive')
        print(str((neg_count/total)*100) + ' % of the examples are negative')
        print(str((others/total)*100) + ' % of the examples are mislabelled')
    else:
        print('The sample file does not exist. Create it using create_samples_file.py.')















