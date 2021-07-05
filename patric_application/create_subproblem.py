import argparse
import os
import pandas as pd
import subprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', type=str, default='Proteobacteria', metavar='G')
    parser.add_argument('--level', type=str, default='phylum', metavar='L')
    parser.add_argument('--antibiotic', type=str,default='ciprofloxacin', metavar='A')
    args = parser.parse_args()

    genomes_of_interest = []

    lineage_file = os.path.join('data_files', 'genome_lineage.csv')
    lineage_df = pd.read_csv(lineage_file, sep='\t')

    for row in range(lineage_df.shape[0]):
        if lineage_df[args.level][row] == args.group:
            genomes_of_interest.append(lineage_df['genome_id'][row])

    print(len(genomes_of_interest))
    genomes_of_interest = set(genomes_of_interest)
    print(len(genomes_of_interest))

    amr_file = os.path.join('data_files', 'amr_phenotypes.csv')
    amr_df = pd.read_csv(amr_file, sep='\t')
    amr_df = amr_df[(amr_df['resistant_phenotype'].notnull())]
    amr_df.set_index(pd.Index(range(amr_df.shape[0])), inplace=True)

    base_url = 'ftp://ftp.patricbrc.org/genomes/'
    extension = '.PATRIC.spgene.tab'
    base_out = os.path.join('data_files', 'subproblems', args.group + '_' + args.antibiotic, 'spgenes')
    os.makedirs(base_out, exist_ok=True)
    error = []

    # lists that will be used to build the dataframe at the end
    ids = []
    antibiotics = []
    phenotypes = []
    annotations = []
    features = []

    functions = set()
    # null_functions = set()
    ids_dict = {}

    c = 0
    for row in range(amr_df.shape[0]):
        if amr_df['antibiotic'][row] == args.antibiotic and amr_df['genome_id'][row] in genomes_of_interest:
            c += 1
    print(c)
    used_genomes = []
    for row in range(amr_df.shape[0]):
        genome = amr_df['genome_id'][row]
        if genome in genomes_of_interest and amr_df['antibiotic'][row] == args.antibiotic and genome not in used_genomes:
            used_genomes.append(genome)
            fp = base_url + str(genome) + '/' + str(genome) + extension
            sp_file = os.path.join(base_out, str(genome) + '_spgenes.tab')
            print("trying download for :" + str(genome))
            if not os.path.isfile(sp_file):
                try:
                    # command = 'wget -P ' + outfile + ' ' + fp
                    # os.system(command)
                    subprocess.call(['wget', '-O', sp_file, fp])
                except:
                    error.append(genome)
            print(error)
            if genome not in error and os.path.isfile(sp_file):
                try:
                    sp_df = pd.read_csv(sp_file, sep='\t')
                except pd.errors.EmptyDataError as e:
                    break

                sp_df = sp_df[(sp_df['function'].notnull())]
                feat_dict = {}

                for function in sp_df['function']:
                    if function not in feat_dict.keys():
                        feat_dict[function] = 1
                    else:
                        feat_dict[function] += 1

                ids_dict[genome] = feat_dict
                ids.append(genome)
                if amr_df['resistant_phenotype'][row] == 'resistant' or amr_df['resistant_phenotype'][row] == 'intermediate':
                    phenotypes.append([1.0])
                elif amr_df['resistant_phenotype'][row] == 'susceptible':
                    phenotypes.append([0.0])
                antibiotics.append([args.antibiotic])
                annotations.append([True])

    for id in ids_dict.keys():
        functions = functions.intersection(ids_dict[id].keys())

    for id in ids:
        genome_features = []
        for func in functions:
            genome_features.append(ids_dict[id][func])
        features.append(genome_features)

    final_df = pd.DataFrame(data={'ID': ids, 'Antibiotics': antibiotics, 'Phenotype': phenotypes, 'Annotation': annotations, 'Features': features})
    final_df.to_csv(os.path.join('data_files', 'subproblems', args.group + '_' + args.antibiotic, 'dataset.csv'), index=False)


















