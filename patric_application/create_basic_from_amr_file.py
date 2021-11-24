import argparse
import os
import pandas as pd
import subprocess
import json

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--group', type=str, default='Proteobacteria', metavar='G')
    parser.add_argument('--antibiotic', type=str, default='ciprofloxacin', metavar='A')
    args = parser.parse_args()

    amr_file = os.path.join('data_files', 'amr_phenotypes.csv')
    amr_df = pd.read_csv(amr_file, delimiter='\t', dtype=str)
    amr_df = amr_df[(amr_df['resistant_phenotype'].notnull()) & (amr_df['genome_id'].notnull())
                    & (amr_df['antibiotic'].notnull())]
    #amr_df.drop_duplicates(subset='genome_id', inplace=True)
    amr_df.set_index(pd.Index(range(amr_df.shape[0])))
    genome_file = os.path.join('data_files', 'genome_lineage.csv')
    genome_df = pd.read_csv(genome_file, delimiter='\t', dtype=str)
    genome_df = genome_df[genome_df['kingdom'] == 'Bacteria']
    genome_df = genome_df.rename(columns={'class': 'safe_class'}) #class is a keyword in python
    genome_df = genome_df[
        (genome_df['kingdom'].notnull()) & (genome_df['phylum'].notnull()) & (genome_df['safe_class'].notnull())
        & (genome_df['order'].notnull()) & (genome_df['family'].notnull()) & (genome_df['genus'].notnull())
        & (genome_df['species'].notnull()) & (genome_df['genome_id'].notnull())]
    genome_df.drop_duplicates(subset='genome_id', inplace=True)
    genome_df.set_index(pd.Index(range(genome_df.shape[0])), inplace=True)
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
    print(amr_df)
    for i in range(amr_df.shape[0]):
        if amr_df['genome_id'][i] in ids and amr_df['antibiotic'][i] == args.antibiotic:
            data['drug.antibiotic_name'].append(args.antibiotic)
            data['genome_drug.genome_id'].append(amr_df['genome_id'][i])
            data['genome_drug.resistant_phenotype'].append(amr_df['resistant_phenotype'][i])

    samples_df = pd.DataFrame(data=data)
    samples_df.to_csv(os.path.join('data_files', 'basic_files', args.group + '_' + args.antibiotic + '_basic.csv'), index=False)
