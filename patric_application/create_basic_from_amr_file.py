import argparse
import os
import pandas as pd

if __name__ == '__main__':

    """
    As you can see, this code does not require the user to enter the taxonomical level of the group of interest.
    The appropriate level is found inside the available data.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--group', type=str, default='Proteobacteria', metavar='G')
    parser.add_argument('--antibiotic', type=str, default='ciprofloxacin', metavar='A')
    args = parser.parse_args()

    amr_file = os.path.join('data_files', 'amr_phenotypes.csv')
    amr_df = pd.read_csv(amr_file, delimiter='\t', dtype=str)
    antibiotic = args.antibiotic

    while antibiotic not in list(amr_df.loc[:, 'antibiotic']):
        print('The antibiotic ' + antibiotic + ' was not found inside the data.')
        print('Please input another antibiotic name in the console:')
        antibiotic = input()

    amr_df = amr_df[(amr_df['resistant_phenotype'].notnull()) & (amr_df['genome_id'].notnull())
                    & (amr_df['antibiotic'].notnull()) & (amr_df['antibiotic'] == antibiotic)
                    & (amr_df['resistant_phenotype'] != 'Not defined')]
    amr_df.drop_duplicates(subset='genome_id', inplace=True)
    amr_df.set_index(pd.Index(range(amr_df.shape[0])), inplace=True)

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

    group = args.group
    levels = ['kingdom', 'phylum', 'safe_class', 'order', 'family', 'genus', 'species', 'genome_id']
    group_level = ''

    while group_level == '':
        end = False
        for level in levels:
            if end:
                break
            for i in range(genome_df.shape[0]):
                if genome_df.loc[i, level] == group:
                    group_level = level
                    end = True
                    break
        if group_level == '':
            print('The taxonomical group ' + group + ' was not found inside the data.')
            print('Important: group name in the data starts with a capital letter (ex: Proteobacteria or Firmicutes)')
            group = ''
            while group == '':
                print('Please input another group name in the console:')
                group = input()

    ids = []
    for i in range(genome_df.shape[0]):
        if genome_df.loc[i, group_level] == group:
            ids.append(genome_df.loc[i, 'genome_id'])
    print(str(len(ids)) + ' genomes of interest are available in the genome_lineage file.')

    data = {}
    data['genome_drug.genome_id'] = []
    data['genome_drug.resistant_phenotype'] = []

    genomes_in_amr = 0
    for i in range(amr_df.shape[0]):
        if amr_df.loc[i, 'antibiotic'] == antibiotic and amr_df.loc[i, 'genome_id'] in ids:
            genomes_in_amr += 1
            data['genome_drug.genome_id'].append(amr_df.loc[i, 'genome_id'])
            data['genome_drug.resistant_phenotype'].append(amr_df.loc[i, 'resistant_phenotype'])

    print(str(genomes_in_amr) + ' of the genome from genome_lineage are also available in the amr_phenotypes.')

    os.makedirs(os.path.join('data_files', 'basic_files'), exist_ok=True)
    samples_df = pd.DataFrame(data=data)
    samples_df.to_csv(os.path.join('data_files', 'basic_files', group + '_' + args.antibiotic + '_basic.csv'),
                      index=False, sep='\t')

