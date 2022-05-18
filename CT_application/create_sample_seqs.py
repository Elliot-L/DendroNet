from Bio import SeqIO
import os
import pandas as pd
import json

def change_for_uppercase(s):
    new_s = ''
    for c in s:
        if c == 'A' or c == 'G' or c == 'T' or c == 'C' or c == 'N':
            new_s = new_s + c
        elif c == 'a':
            new_s = new_s + 'A'
        elif c == 'g':
            new_s = new_s + 'G'
        elif c == 'c':
            new_s = new_s + 'C'
        elif c == 't':
            new_s = new_s + 'T'
    return new_s

if __name__ == "__main__":

    if not os.path.isfile(os.path.join('data_files', 'enhancers_seqs.json')):
        chromosomes = {}
        # for chromosomes 1 to 22
        for i in range(1, 23):
            print(i)
            file_name = os.path.join('data_files', 'chr seqs', 'chr' + str(i) + '.fa', 'chr' + str(i) + '.fa')
            if os.path.isdir(os.path.join('data_files', 'chr seqs', 'chr' + str(i) + '.fa')):
                print(file_name)
                chr_rec = SeqIO.parse(file_name, 'fasta')
                for c in chr_rec:
                    chr = c
                print(len(chr.seq))
                chromosomes[chr.id] = chr.seq
        # for X chromosome
        print('X')
        file_name = os.path.join('data_files', 'chr seqs', 'chrX.fa', 'chrX.fa')
        if os.path.isdir(os.path.join('data_files', 'chr seqs', 'chrX.fa')):
            print(file_name)
            chr_rec = SeqIO.parse(file_name, 'fasta')
            for c in chr_rec:
                chr = c
            print(len(chr.seq))
            chromosomes[chr.id] = chr.seq
        # for Y chromosome
        print('Y')
        file_name = os.path.join('data_files', 'chr seqs', 'chrY.fa', 'chrY.fa')
        if os.path.isdir(os.path.join('data_files', 'chr seqs', 'chrY.fa')):
            print(file_name)
            chr_rec = SeqIO.parse(file_name, 'fasta')
            for c in chr_rec:
                chr = c
            print(len(chr.seq))
            chromosomes[chr.id] = chr.seq
        print(chromosomes)

        position_file = os.path.join('data_files', 'GRCh38-cCREs.dELS.bed')
        position_df = pd.read_csv(position_file, names=['Chromosome', 'start', 'end', 'name1', 'name2', 'Unknown'],
                                  dtype=str, sep='\t')
        print(position_df)

        samples = {}

        for row in range(position_df.shape[0]):
            start = int(position_df.loc[row, 'start']) - 250
            end = int(position_df.loc[row, 'end']) + 250
            chr = position_df.loc[row, 'Chromosome']
            name1 = position_df.loc[row, 'name1']
            name2 = position_df.loc[row, 'name2']

            samples[name1] = (chr, change_for_uppercase(str(chromosomes[chr][start - 1: end])))
            samples[name2] = (chr, change_for_uppercase(str(chromosomes[chr][start - 1: end])))

        samples_file = os.path.join('data_files', 'enhancers_seqs.json')

        with open(samples_file, 'w') as out_file:
            json.dump(samples, out_file)

    else:
        print('Didn\'t have to make sequence files!')

    activity_file = os.path.join('data_files', 'cCRE_decoration.matrix.1', 'cCRE_decoration.matrix.1.txt')
    activity_df = pd.read_csv(activity_file, dtype=str, sep='\t')

    pos_neg_ratio_file = os.path.join('data_files', 'pos_neg_ratios.json')

    if not os.path.isfile(pos_neg_ratio_file):

        pos_count = {}
        columns_of_interest = activity_df.columns[1:6]
        for col in columns_of_interest:
            pos_count[col] = 0

        for row in range(activity_df.shape[0]):
            for col in columns_of_interest:
                if int(activity_df.loc[row, col]) == 1:
                    pos_count[col] += 1

        total = activity_df.shape[0]

        for k, v in pos_count.items():
            pos_count[k] = v / total

        with open(pos_neg_ratio_file, 'w') as out_file:
            json.dump(pos_count, out_file)

    # creating samples files for 5 cell types

    id_col = activity_df.columns[0]
    ct_name1 = activity_df.columns[1]
    ct_name2 = activity_df.columns[2]
    ct_name3 = activity_df.columns[3]
    ct_name4 = activity_df.columns[4]
    ct_name5 = activity_df.columns[5]

    with open(os.path.join('data_files', 'enhancers_seqs.json'), 'r') as dict_file:
        enhancer_dict = json.load(dict_file)

    data1 = {'IDs': [], 'sequences': [], 'labels': []}
    data2 = {'IDs': [], 'sequences': [], 'labels': []}
    data3 = {'IDs': [], 'sequences': [], 'labels': []}
    data4 = {'IDs': [], 'sequences': [], 'labels': []}
    data5 = {'IDs': [], 'sequences': [], 'labels': []}

    found = []
    not_found = []

    for row in range(activity_df.shape[0]):
        id = activity_df.loc[row, id_col]
        try:
            seq = enhancer_dict[id][1]
            found.append(id)
            if len(seq) < 501:
                continue
            middle = int(len(seq) / 2)
            seq = seq[(middle - 250):(middle + 251)]

            data1['IDs'].append(id)
            data2['IDs'].append(id)
            data3['IDs'].append(id)
            data4['IDs'].append(id)
            data5['IDs'].append(id)

            data1['sequences'].append(seq)
            data2['sequences'].append(seq)
            data3['sequences'].append(seq)
            data4['sequences'].append(seq)
            data5['sequences'].append(seq)

            data1['labels'].append(activity_df.loc[row, ct_name1])
            data2['labels'].append(activity_df.loc[row, ct_name2])
            data3['labels'].append(activity_df.loc[row, ct_name3])
            data4['labels'].append(activity_df.loc[row, ct_name4])
            data5['labels'].append(activity_df.loc[row, ct_name5])

        except KeyError:
            not_found.append(id)
            continue
    print(found)
    print(len(found))
    print(len(not_found))

    print(data1)
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)
    df3 = pd.DataFrame(data3)
    df4 = pd.DataFrame(data4)
    df5 = pd.DataFrame(data5)

    df1.to_csv(os.path.join('data_files', ct_name1 + '_samples.csv'), index=False)
    df2.to_csv(os.path.join('data_files', ct_name2 + '_samples.csv'), index=False)
    df3.to_csv(os.path.join('data_files', ct_name3 + '_samples.csv'), index=False)
    df4.to_csv(os.path.join('data_files', ct_name4 + '_samples.csv'), index=False)
    df5.to_csv(os.path.join('data_files', ct_name5 + '_samples.csv'), index=False)

