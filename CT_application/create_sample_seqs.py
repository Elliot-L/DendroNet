import Bio
from Bio.SeqIO import SeqRecord, parse
from Bio.Seq import Seq
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
        else:
            new_s = new_s + c
    return new_s

if __name__ == "__main__":

    if not os.path.isfile(os.path.join('data_files', 'enhancers_seqs.json')):
        chromosomes = {}
        # for chromosomes 1 to 22
        for i in range(1, 23):
            print(i)
            file_name = os.path.join('data_files', 'chr seqs', 'chr' + str(i) + '.fa')
            print(file_name)
            chr_rec = parse(file_name, 'fasta')
            for c in chr_rec:
                chr = c
            print(len(chr.seq))
            chromosomes[chr.id] = str(chr.seq)
        # for X chromosome
        print('X')
        file_name = os.path.join('data_files', 'chr seqs', 'chrX.fa')
        print(file_name)
        chr_rec = parse(file_name, 'fasta')
        for c in chr_rec:
            chr = c
        print(len(chr.seq))
        chromosomes[chr.id] = str(chr.seq)
        # for Y chromosome
        print('Y')
        file_name = os.path.join('data_files', 'chr seqs', 'chrY.fa')
        print(file_name)
        chr_rec = parse(file_name, 'fasta')
        for c in chr_rec:
            chr = c
        print(len(chr.seq))
        chromosomes[chr.id] = str(chr.seq)
        # print(chromosomes)

        position_file = os.path.join('data_files', 'GRCh38-cCREs.dELS.bed')
        position_df = pd.read_csv(position_file, names=['Chromosome', 'start', 'end', 'name1', 'name2', 'Class'],
                                  dtype=str, sep='\t')
        enhancers_dict = {}

        activity_file = os.path.join('data_files', 'cCRE_decoration.matrix.1', 'cCRE_decoration.matrix.1.txt')
        activity_df = pd.read_csv(activity_file, dtype=str, sep='\t')

        activity_pos_dict = {}
        id_col_name = activity_df.columns[0]

        for row in range(activity_df.shape[0]):
            activity_pos_dict[activity_df.loc[row, id_col_name]] = row

        print(position_df.shape[0])
        print(activity_df.shape[0])
        print(len(activity_pos_dict))

        for row in range(position_df.shape[0]):
            name = position_df.loc[row, 'name2']
            if name in activity_pos_dict:
                chr = position_df.loc[row, 'Chromosome']
                start = int(position_df.loc[row, 'start'])
                end = int(position_df.loc[row, 'end'])
                middle = int((end + start) / 2)
                enhancers_dict[name] = change_for_uppercase(chromosomes[chr][middle - 250: middle + 251])

        print(len(enhancers_dict.keys()))

        enhancers_file = os.path.join('data_files', 'enhancers_seqs.json')

        with open(enhancers_file, 'w') as outfile:
            json.dump(enhancers_dict, outfile)
        exit()
    else:
        print('Didn\'t have to make sequence files!')

        with open(os.path.join('data_files', 'enhancers_seqs.json'), 'r') as dict_file:
            enhancers_dict = json.load(dict_file)

    activity_file = os.path.join('data_files', 'cCRE_decoration.matrix.1', 'cCRE_decoration.matrix.1.txt')
    activity_df = pd.read_csv(activity_file, dtype=str, sep='\t')

    print(activity_df)

    pos_neg_ratio_file = os.path.join('data_files', 'pos_neg_ratios.json')

    if not os.path.isfile(pos_neg_ratio_file):

        pos_count = {}
        columns_of_interest = activity_df.columns[1:]
        for col in columns_of_interest:
            pos_count[col] = 0

        id_col_name = activity_df.columns[0]

        for row in range(activity_df.shape[0]):
            if activity_df.loc[row, id_col_name] in enhancers_dict:
                print('Found a common enhancer!')
                for col in columns_of_interest:
                    if int(activity_df.loc[row, col]) == 1:
                        pos_count[col] += 1

        total = len(enhancers_dict.keys())

        for k, v in pos_count.items():
            pos_count[k] = v / total

        with open(pos_neg_ratio_file, 'w') as out_file:
            json.dump(pos_count, out_file)

    # creating samples files for x best cell types

    x = 20

    with open(os.path.join('data_files', 'pos_neg_ratios.json'), 'r') as dict_file:
        pos_neg_dict = json.load(dict_file)

    best_ratios = []
    best_cells = []

    for i in range(x):
        best_ratios.append(0)
        best_cells.append('')

    for cell, ratio in pos_neg_dict.items():
        for i in range(x):
            if ratio > best_ratios[i]:
                best_ratios.insert(i, ratio)
                best_ratios = best_ratios[0:x]
                best_cells.insert(i, cell)
                best_cells = best_cells[0:x]
                break

    print(best_cells)
    print(best_ratios)

    id_col_name = activity_df.columns[0]

    for cell in best_cells:
        print(cell)
        data = {'IDs': [], 'labels': []}

        for row in range(activity_df.shape[0]):
            id = activity_df.loc[row, id_col_name]
            if id in enhancers_dict:
                data['IDs'].append(id)
                data['labels'].append(activity_df.loc[row, cell])

        df1 = pd.DataFrame(data)
        file_name = os.path.join('data_files', 'single_cell_datasets', cell + '_samples.csv')
        df1.to_csv(file_name, index=False)

    """
    ct_name1 = best_cells[0]
    ct_name2 = best_cells[1]
    ct_name3 = best_cells[2]
    ct_name4 = best_cells[3]
    ct_name5 = best_cells[4]
    ct_name6 = best_cells[5]
    ct_name7 = best_cells[6]
    ct_name8 = best_cells[7]
    ct_name9 = best_cells[8]
    ct_name10 = best_cells[9]

    data1 = {'IDs': [], 'sequences': [], 'labels': []}
    data2 = {'IDs': [], 'sequences': [], 'labels': []}
    data3 = {'IDs': [], 'sequences': [], 'labels': []}
    data4 = {'IDs': [], 'sequences': [], 'labels': []}
    data5 = {'IDs': [], 'sequences': [], 'labels': []}
    data6 = {'IDs': [], 'sequences': [], 'labels': []}
    data7 = {'IDs': [], 'sequences': [], 'labels': []}
    data8 = {'IDs': [], 'sequences': [], 'labels': []}
    data9 = {'IDs': [], 'sequences': [], 'labels': []}
    data10 = {'IDs': [], 'sequences': [], 'labels': []}

    for row in range(activity_df.shape[0]):
        id = activity_df.loc[row, id_col_name]
        try:
            seq = enhancer_dict[id][1]
            found.append(id)
            if len(seq) < 501:
                print('small lenght????????????????????????')
            middle = int(len(seq) / 2)
            seq = seq[(middle - 250):(middle + 251)]

            data1['IDs'].append(id)
            data2['IDs'].append(id)
            data3['IDs'].append(id)
            data4['IDs'].append(id)
            data5['IDs'].append(id)
            data6['IDs'].append(id)
            data7['IDs'].append(id)
            data8['IDs'].append(id)
            data9['IDs'].append(id)
            data10['IDs'].append(id)

            data1['sequences'].append(seq)
            data2['sequences'].append(seq)
            data3['sequences'].append(seq)
            data4['sequences'].append(seq)
            data5['sequences'].append(seq)
            data6['sequences'].append(seq)
            data7['sequences'].append(seq)
            data8['sequences'].append(seq)
            data9['sequences'].append(seq)
            data10['sequences'].append(seq)

            data1['labels'].append(activity_df.loc[row, ct_name1])
            data2['labels'].append(activity_df.loc[row, ct_name2])
            data3['labels'].append(activity_df.loc[row, ct_name3])
            data4['labels'].append(activity_df.loc[row, ct_name4])
            data5['labels'].append(activity_df.loc[row, ct_name5])
            data6['labels'].append(activity_df.loc[row, ct_name6])
            data7['labels'].append(activity_df.loc[row, ct_name7])
            data8['labels'].append(activity_df.loc[row, ct_name8])
            data9['labels'].append(activity_df.loc[row, ct_name9])
            data10['labels'].append(activity_df.loc[row, ct_name10])

        except KeyError:
            not_found.append(id)
            continue

    print(found)
    print(len(found))
    print(len(not_found))

    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)
    df3 = pd.DataFrame(data3)
    df4 = pd.DataFrame(data4)
    df5 = pd.DataFrame(data5)
    df6 = pd.DataFrame(data6)
    df7 = pd.DataFrame(data7)
    df8 = pd.DataFrame(data8)
    df9 = pd.DataFrame(data9)
    df10 = pd.DataFrame(data10)


    df1.to_csv(os.path.join('data_files', ct_name1 + '_samples.csv'), index=False)
    df2.to_csv(os.path.join('data_files', ct_name2 + '_samples.csv'), index=False)
    df3.to_csv(os.path.join('data_files', ct_name3 + '_samples.csv'), index=False)
    df4.to_csv(os.path.join('data_files', ct_name4 + '_samples.csv'), index=False)
    df5.to_csv(os.path.join('data_files', ct_name5 + '_samples.csv'), index=False)
    df6.to_csv(os.path.join('data_files', ct_name6 + '_samples.csv'), index=False)
    df7.to_csv(os.path.join('data_files', ct_name7 + '_samples.csv'), index=False)
    df8.to_csv(os.path.join('data_files', ct_name8 + '_samples.csv'), index=False)
    df9.to_csv(os.path.join('data_files', ct_name9 + '_samples.csv'), index=False)
    df10.to_csv(os.path.join('data_files', ct_name10 + '_samples.csv'), index=False)

    """
