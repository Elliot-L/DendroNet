import argparse
import os
import pandas as pd
import subprocess
import json


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', type=str, metavar='G', default='Proteobacteria')
    parser.add_argument('--antibiotic', type=str, metavar='A', default='ciprofloxacin')
    parser.add_argument('--threshold', type=float, help='fraction of genomes that need to have a ' +
                        'certain feature for that feature to be selected')
    parser.add_argument('--force-download', type=str, default='n', help='y/n')
    args = parser.parse_args()

    base_url = 'ftp://ftp.patricbrc.org/genomes/'
    extension = '.PATRIC.spgene.tab'
    base_out = os.path.join('data_files', 'spgenes')
    os.makedirs(base_out, exist_ok=True)
    basic_file = os.path.join('data_files', 'basic_files', args.group + '_' + args.antibiotic + '_basic.csv')

    if not os.path.isfile(basic_file):
        print('The basic file associated to this problem does not exist. Create it using the ' +
              'create_basic_from_amr_file.py file, or download it from the PATRIC terminal.')
        exit()

    basic_df = pd.read_csv(basic_file, sep='\t', dtype=str)
    basic_df = basic_df[(basic_df['genome_drug.resistant_phenotype'].notnull()) &
                        (basic_df['genome_drug.resistant_phenotype'] != 'Not defined')]
    basic_df.set_index(pd.Index(range(basic_df.shape[0])), inplace=True)

    # lists that will be used to build the dataframe at the end
    ids = []
    phenotypes = []
    features = []

    error = set()
    functions = set()
    ids_dict = {}  # keys will be genomes ids and values are dictionaries for which the keys are specialy genes
                   # functions (a string) and the value is the number of time this gene is present in given genome

    for row in range(basic_df.shape[0]):
        genome = basic_df.loc[row, 'genome_drug.genome_id']
        fp = base_url + genome + '/' + genome + extension
        sp_file = os.path.join(base_out, genome + '_spgenes.tab')
        if not os.path.isfile(sp_file) or args.force_download == 'y':
            print("trying download for : " + genome)
            try:
                subprocess.call(['wget', '-O', sp_file, fp])
            except subprocess.CalledProcessError as e:
                error.append(genome)
                print('error when trying to download genome ' + genome)
                continue
        if genome not in error:
            print("getting data for : " + str(genome))
            try:
                sp_df = pd.read_csv(sp_file, sep='\t')
            except pd.errors.EmptyDataError as e:
                error.append(genome)
                print('error when trying to use the data for genome ' + genome)
                continue
            sp_df = sp_df[(sp_df['function'].notnull())]
            feat_dict = {}

            for function in list(sp_df.loc[:, 'function']):
                if function not in feat_dict.keys():
                    feat_dict[function] = 1.0
                else:
                    feat_dict[function] += 1.0

            ids_dict[genome] = feat_dict
            ids.append(genome)
            # 1 describes resistance phenotype and 0 susceptibility
            if basic_df['genome_drug.resistant_phenotype'][row] == 'non_susceptible' or \
                    basic_df['genome_drug.resistant_phenotype'][row] == 'Resistant' or \
                    basic_df['genome_drug.resistant_phenotype'][row] == 'Intermediate' or \
                    basic_df['genome_drug.resistant_phenotype'][row] == 'r' or \
                    basic_df['genome_drug.resistant_phenotype'][row] == 'R' or \
                    basic_df['genome_drug.resistant_phenotype'][row] == 'resistant' or \
                    basic_df['genome_drug.resistant_phenotype'][row] == 'intermediate' or \
                    basic_df['genome_drug.resistant_phenotype'][row] == 'reduced_susceptibility' or \
                    basic_df['genome_drug.resistant_phenotype'][row] == 'IS':
                phenotypes.append([1])
            else:  # In the amr_file, the two other phenotypes used to described non-resistance are susceptible and
                phenotypes.append([0])  # and susceptible-dose dependent.

    for feature_dict in ids_dict.values():
        functions = functions.union(feature_dict.keys())

    functions = list(functions)
    min_number = int((basic_df.shape[0] - len(error))*args.threshold)

    for idx in ids:
        genome_features = []
        for func in functions:
            if func in ids_dict[idx].keys():
                genome_features.append(ids_dict[idx][func])
            else:
                genome_features.append(0.0)
        features.append(genome_features)

    for feature_list in features:
        feature_list.append(1.0)

    useful_functions = functions.copy()
    col = 0
    while col < len(useful_functions):
        c = 0
        for features_list in features:
            if features_list[col] > 0.0:
                c += 1
        if c < min_number:
            del useful_functions[col]
            for features_list in features:
                del features_list[col]
            col -= 1
        col += 1

    """"
    subproblem_infos = {}
    subproblem_infos['number of examples:'] = len(ids)
    subproblem_infos['number of features:'] = len(useful_functions)
    with open(os.path.join('data_files', 'subproblems', args.group + '_' + args.antibiotic, 'subproblem_infos_'
              + str(args.threshold) + '.json'), 'w') as info_file:
        json.dump(subproblem_infos, info_file)
    """

    final_df = pd.DataFrame(data={'ID': ids, 'Phenotype': phenotypes, 'Features': features})
    final_df.to_csv(os.path.join('data_files', 'subproblems', args.group + '_' + args.antibiotic, args.group + '_'
                                 + args.antibiotic + '_' + str(args.threshold) + '_samples' + '.csv'), index=False)



















