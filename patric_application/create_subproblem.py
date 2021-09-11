import argparse
import os
import pandas as pd
import subprocess
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', type=str, default='Proteobacteria', metavar='G')
    parser.add_argument('--antibiotic', type=str, default='ciprofloxacin', metavar='A')
    parser.add_argument('--threshold', type=float, default=0.05, help='fraction of genomes that need to have a ' +
                        'certain feature for that feature to be selected')
    parser.add_argument('--force-download', type=str, default='n', help='y/n')
    args = parser.parse_args()

    base_url = 'ftp://ftp.patricbrc.org/genomes/'
    extension = '.PATRIC.spgene.tab'
    base_out = os.path.join('data_files', 'subproblems', args.group + '_' + args.antibiotic, 'spgenes')
    os.makedirs(base_out, exist_ok=True)
    error = []

    basic_file = os.path.join('data_files', 'basic_files', args.group + '_' + args.antibiotic + '_basic.csv')
    basic_df = pd.read_csv(basic_file, sep='\t')
    basic_df = basic_df[(basic_df['genome_drug.resistant_phenotype'].notnull()) &
                        (basic_df['genome_drug.resistant_phenotype'] != 'IS') &
                        (basic_df['genome_drug.resistant_phenotype'] != 'Not defined')]
    basic_df.set_index(pd.Index(range(basic_df.shape[0])), inplace=True)

    # lists that will be used to build the dataframe at the end
    ids = []
    antibiotics = []
    phenotypes = []
    annotations = []
    features = []

    functions = set()
    # null_functions = set()
    ids_dict = {}

    """
    
    c = 0
    for row in range(amr_df.shape[0]):
        if basic_df['antibiotic'][row] == args.antibiotic and amr_df['genome_id'][row] in genomes_of_interest:
            c += 1
    print(c)

    """

    used_genomes = []
    for row in range(basic_df.shape[0]):
        genome = basic_df['genome_drug.genome_id'][row]
        if genome not in used_genomes and genome not in error:
            fp = base_url + str(genome) + '/' + str(genome) + extension
            sp_file = os.path.join(base_out, str(genome) + '_spgenes.tab')
            print("trying download for : " + str(genome))
            if not os.path.isfile(sp_file) or args.force_download == 'y':
                try:
                    # command = 'wget -P ' + outfile + ' ' + fp
                    # os.system(command)
                    subprocess.call(['wget', '-O', sp_file, fp])
                except:
                    error.append(genome)
                    print('error')
            if genome not in error and os.path.isfile(sp_file):
                try:
                    sp_df = pd.read_csv(sp_file, sep='\t')
                except pd.errors.EmptyDataError as e:
                    error.append(genome)
                    print('error')
                    continue
                used_genomes.append(genome)
                sp_df = sp_df[(sp_df['function'].notnull())]
                feat_dict = {}

                for function in sp_df['function']:
                    if function not in feat_dict.keys():
                        feat_dict[function] = 1.0
                    else:
                        feat_dict[function] += 1.0

                ids_dict[genome] = feat_dict
                ids.append(genome)
                if basic_df['genome_drug.resistant_phenotype'][row] == 'Resistant' or basic_df['genome_drug.resistant_phenotype'][row] == 'Intermediate':
                    phenotypes.append([1])
                elif basic_df['genome_drug.resistant_phenotype'][row] == 'Susceptible':
                    phenotypes.append([0])
                antibiotics.append([basic_df['drug.antibiotic_name'][row]])
                annotations.append([True])

    for id in ids_dict.keys():
        functions = functions.union(ids_dict[id].keys())

    functions = list(functions)
    min_number = int(len(used_genomes)*args.threshold)

    #print("threshold: ", threshold)

    for idx in ids:
        genome_features = []
        for func in functions:
            if func in ids_dict[idx].keys():
                genome_features.append(ids_dict[idx][func])
            else:
                genome_features.append(0.0)
        features.append(genome_features)

    useful_features = functions.copy()
    col = 0
    while col < len(useful_features):
        c = 0
        for feat_list in features:
            if feat_list[col] > 0.0:
                c += 1
        if c < min_number:
            del useful_features[col]
            for feat_list in features:
                del feat_list[col]
            col -= 1
        col += 1
    print(len(functions))
    print(len(useful_features))

    subproblem_infos = {}
    subproblem_infos['number of examples:'] = len(ids)
    subproblem_infos['number of features:'] = len(useful_features)

    final_df = pd.DataFrame(data={'ID': ids, 'Antibiotics': antibiotics, 'Phenotype': phenotypes, 'Annotation': annotations, 'Features': features})
    final_df.to_csv(os.path.join('data_files', 'subproblems', args.group + '_' + args.antibiotic, args.antibiotic + '_' + args.group + '_' + 'samples.csv'), index=False)
    with open(os.path.join('data_files','subproblems', args.group + '_' + args.antibiotic, 'subproblem_infos.json'), 'w') as info_file:
        json.dump(subproblem_infos, info_file)



















