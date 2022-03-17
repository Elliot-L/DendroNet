import os
import argparse
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', type=str, metavar='G', default='Proteobacteria')
    parser.add_argument('--antibiotic', type=str, metavar='A', default='ciprofloxacin')
    parser.add_argument('--threshold', type=str, metavar='T', default='0.0')
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
            ids.append(basic_df.loc[row, 'genome_drug.genome_id'])
            phenotype = basic_df.loc[row, 'genome_drug.resistant_phenotype']
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

    directory = os.path.join('data_files', 'subproblems', args.group + '_' + args.antibiotic + '_' + args.threshold)

    if not os.path.isdir(directory) or len(os.listdir(directory)) == 0:
        print('No samples file was found. Create one using create_samples_file.py.')

    else:
        print('Samples files related to this subproblem:')
        file_name = args.group + '_' + args.antibiotic + '_' + args.threshold + '_samples.csv'
        samples_file = os.path.join(directory, file_name)
        samples_df = pd.read_csv(samples_file, dtype=str)

        ids = []
        ids_count = 0
        features_count = []
        pos_count = 0
        neg_count = 0
        others = 0

        for row in range(samples_df.shape[0]):
            ids_count += 1
            ids.append(samples_df.loc[row, 'ID'])
            features_count.append(len((samples_df.loc[row, 'Features']).split(',')))
            phenotype = samples_df.loc[row, 'Phenotype']
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

        all_binarized = True
        for feature_list in samples_df.loc[:, 'Features']:
            feature_list = feature_list.replace('[', '').replace(']', '').replace('"', '').split(',')
            for feature in feature_list:
                feature = float(feature)
                if feature != 1.0 and feature != 0.0:
                    all_binarized = False
                    break
            if not all_binarized:
                break
        if all_binarized:
            print('All feature in this file are binarized.')
        else:
            print('Not all features in this file are binarized.')

        total = pos_count + neg_count + others
        print(str((pos_count / total) * 100) + ' % of the examples are positive')
        print(str((neg_count / total) * 100) + ' % of the examples are negative')
        print(str((others / total) * 100) + ' % of the examples are mislabelled')
















