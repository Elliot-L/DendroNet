import os
import pandas as pd
import re
import json

if __name__ == '__main__':

    ap_df = pd.read_csv(os.path.join('data_files', 'Antibacterial_amps.txt'), sep='\t')

    print(ap_df)

    print(ap_df.columns)

    peptide_dict = {}

    longest = ''
    max_length = 0

    species_count = {'Escherichia coli': 0,
                     'Staphylococcus epidermidis': 0,
                     'Staphylococcus aureus': 0,
                     'Listeria monocytogenes': 0,
                     'Listeria innocua': 0,
                     'Listeria ivanovii': 0,
                     'Clavibacter michiganensis': 0,
                     'Micrococcus luteus': 0,
                     'Klebsiella oxytoca': 0,
                     'Bacillus dysenteriae': 0,
                     'Bacillus subtilis': 0,
                     'Candida tropicalis': 0,
                     'Candida guillermondii': 0,
                     'Leuconostoc mesenteroides': 0,
                     'Bacillus cereus': 0,
                     'Pseudomonas aeruginosa': 0,
                     'Bacillus megateriurn': 0,
                     'Candida albicans': 0,
                     'Streptococcus uberis': 0,
                     'Leuconostoc lactis': 0,
                     'Enterococcus faecalis': 0,
                     'Tricophyton rubrum': 0}

    unwanted = ['(', ')', '-', '2', '4']

    data = {}
    for name in species_count.keys():
        data[name] = {'IDs': [], 'label': []}

    for row in range(ap_df.shape[0]):
        # print(ap_df.loc[row, 'DRAMP_ID'])
        id = ap_df.loc[row, 'DRAMP_ID']
        seq = ap_df.loc[row, 'Sequence']
        valid = True
        for c in unwanted:
            if c in seq:
                valid = False
        if valid:
            peptide_dict[id] = seq
            if len(ap_df.loc[row, 'Sequence']) > max_length:
                longest = ap_df.loc[row, 'DRAMP_ID']
                max_length = len(ap_df.loc[row, 'Sequence'])

            orgs = ap_df.loc[row, 'Target_Organism']
            for org in species_count.keys():
                other_name = org[0] + '. ' + org.split(' ')[1]
                if org in orgs or other_name in orgs:
                    species_count[org] += 1
                    data[org]['IDs'].append(id)
                    data[org]['label'].append(1)
                else:
                    data[org]['IDs'].append(id)
                    data[org]['label'].append(0)

    amino_acids = set()
    for id, seq in peptide_dict.items():
        for aa in seq:
            amino_acids.add(aa.upper())
        if len(seq) < max_length:
            pad_size = max_length - len(seq)
            left = pad_size // 2
            right = pad_size - left
            for i in range(left):
                seq = '0' + seq
            for i in range(right):
                seq = seq + '0'
            peptide_dict[id] = seq

    print(peptide_dict)
    print(amino_acids)
    print(len(amino_acids))
    print(species_count)
    print(max_length)
   
    amino_acids = list(amino_acids)
    with open(os.path.join('data_files', 'amino_acids.json'), 'w') as aa_file:
        json.dump(amino_acids, aa_file)

    with open(os.path.join('data_files', 'peptide_seqs.json'), 'w') as p_file:
        json.dump(peptide_dict, p_file)

    os.makedirs(os.path.join('data_files', 'species_datasets'), exist_ok=True)
    for org in species_count.keys():
        filename = org.split(' ')[0] + '_' + org.split(' ')[1] + '_dataset.csv'
        df = pd.DataFrame(data[org])
        df.to_csv(os.path.join('data_files', 'species_datasets', filename), index=False)

    """   
        for org in orgs.split(';'):
            print(org)


    target_df = ap_df['Target_Organism']

    print(target_df)

    print(target_df.loc[1])
    print(target_df.loc[4157])
    print(target_df.loc[4158])

    all_orgs = set()

    for row in range(target_df.shape[0]):
        orgs = target_df.loc[row].split(';')
        for org in orgs:
            org = org.split(':')
            if len(org) == 2:
                org = org[1]
            else:
                org = org[0]
            org = org.split('(')
            org = org[0]
            all_orgs.add(org)

    print(all_orgs)
    print(len(all_orgs))
    """