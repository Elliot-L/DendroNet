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
    args = parser.parse_args()

    amr_file = os.path.join('data_files', 'amr_phenotypes.csv')
    amr_df = pd.read_csv(amr_file, delimiter='\t', dtype=str)
    amr_df = amr_df[(amr_df['resistant_phenotype'].notnull()) & (amr_df['genome_id'].notnull())
                    & (amr_df['antibiotic'].notnull())]
    amr_df.drop_duplicates(subset='genome_id', inplace=True)
    amr_file.set_index(pd.Index(range(amr_file.shape[0])))
    genome_file = os.path.join('data_file', 'genome_lineage.csv')
    genome_df = pd.read_csv(genome_file, delimiter='\t', dtype=str)
    genome_df = genome_df[genome_df['kingdom'] == 'Bacteria']
    genome_df = genome_df[
        (genome_df['kingdom'].notnull()) & (genome_df['phylum'].notnull()) & (genome_df['safe_class'].notnull())
        & (genome_df['order'].notnull()) & (genome_df['family'].notnull()) & (genome_df['genus'].notnull())
        & (genome_df['species'].notnull()) & (genome_df['genome_id'].notnull())]
    genome_df.drop_duplicates(subset='genome_id', inplace=True)
    genome_df.set_index(pd.Index(range(genome_df.shape[0])))
    levels = ['kingdom', 'phylum', 'safe_class', 'order', 'family', 'genus', 'species', 'genome_id']
    group_level = ''
    end = False
    for level in levels:
        if end:
            break
        for i in range(genome_df.shape[0]):
            if genome_df[level][i] == args.group:
                group_level = level
                end = True
                break

    ids = []
    for i in range(genome_df.shape[0]):
        if genome_df[group_level][i] == args.group:
            ids.append(genome_df['genome_id'][i])

    data = {}
    data['drug.antibiotic_name'] = []
    data['genome_drug.genome_id'] = []
    data['genome_drug.genome_name'] = []
    data['genome_drug.resistant_phenotype'] = []

    for i in range(amr_df.shape[0]):
        if amr_df['genome_id'][i] in ids and amr_df['antibiotic'][i] == args.antibiotic:
            data['drug.antibiotic_name'].append(args.antibiotic)
            data['genome_drug.genome_id'].append(amr_df['genome_id'][i])
            data['genome_drug.genome_name'].append(amr_df['genome_name'][i])
            data['genome_drug.resistant_phenotype'].append(amr_df['resistant_phenotype'][i])

    samples_df = pd.DataFrame(data=data)
    samples_df.to_csv(os.path.join('data_files', 'basic_files', args.group + '_' + args.antibiotic + '_basic.csv'), index=False)

    """
    
    # lists that will be used to build the dataframe at the end
    ids = []
    antibiotics = []
    phenotypes = []
    annotations = []
    features = []

    functions = set()
    # null_functions = set()
    ids_dict = {}

    used_genomes = []
    for row in range(basic_df.shape[0]):
        genome = basic_df['genome_drug.genome_id'][row]
        if genome not in used_genomes and genome not in error:
            fp = base_url + str(genome) + '/' + str(genome) + extension
            sp_file = os.path.join(base_out, str(genome) + '_spgenes.tab')
            if not os.path.isfile(sp_file) or args.force_download == 'y':
                print("trying download for : " + str(genome))
                try:
                    # command = 'wget -P ' + outfile + ' ' + fp
                    # os.system(command)
                    subprocess.call(['wget', '-O', sp_file, fp])
                except:
                    error.append(genome)
                    print('error')
            if genome not in error and os.path.isfile(sp_file):
                print("getting data for : " + str(genome))
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
                if basic_df['genome_drug.resistant_phenotype'][row] == 'Resistant' or \
                        basic_df['genome_drug.resistant_phenotype'][row] == 'Intermediate' or \
                        basic_df['genome_drug.resistant_phenotype'][row] == 'r':
                    phenotypes.append([1])
                # elif basic_df['genome_drug.resistant_phenotype'][row] == 'Susceptible':
                else:
                    phenotypes.append([0])
                antibiotics.append([basic_df['drug.antibiotic_name'][row]])
                annotations.append([True])

    for id in ids_dict.keys():
        functions = functions.union(ids_dict[id].keys())

    functions = list(functions)
    min_number = int(len(used_genomes) * args.threshold)

    # print("threshold: ", threshold)

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
    subproblem_infos['threshold:'] = args.threshold
    with open(os.path.join('data_files', 'subproblems', args.group + '_' + args.antibiotic,
                           'subproblem_infos_' + str(args.threshold) + '.json'), 'w') as info_file:
        json.dump(subproblem_infos, info_file)

    final_df = pd.DataFrame(
        data={'ID': ids, 'Antibiotics': antibiotics, 'Phenotype': phenotypes, 'Annotation': annotations,
              'Features': features})
    final_df.to_csv(os.path.join('data_files', 'subproblems', args.group + '_' + args.antibiotic,
                                 args.group + '_' + args.antibiotic + '_' + 'samples_' + str(args.threshold) + '.csv'),
                    index=False)


    """
