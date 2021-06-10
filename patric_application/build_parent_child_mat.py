
import pandas as pd
import numpy as np

"""
This function takes as input the file containing the taxonomic classification of all the species of patric and the 
file containing the species of interest for a given antibiotic and build a parent_child matrix for the species of interest alone

"""
def build_pc_mat(genome_file='genome_lineage.csv', label_file='erythromycin_firmicutes_samples.csv'):
    label_df = pd.read_csv(label_file, dtype=str)
    genome_df = pd.read_csv(genome_file, delimiter='\t', dtype=str)
    genome_df = genome_df.rename(columns={'class': 'safe_class'})
    genome_df = genome_df[genome_df.kingdom == 'Bacteria']
    genome_df = genome_df[(genome_df['kingdom'].notnull()) & (genome_df['phylum'].notnull()) & (genome_df['safe_class'].notnull())
                          & (genome_df['order'].notnull()) & (genome_df['family'].notnull()) & (genome_df['genus'].notnull())
                          & (genome_df['species'].notnull())& (genome_df['genome_id'].notnull())]
    ids = list(set(label_df['ID']))
    genome_df = genome_df[genome_df['genome_id'].isin(ids)]
    new_idx = range(len(ids))
    genome_df.set_index(pd.Index(new_idx), inplace=True)
    levels = ['kingdom', 'phylum', 'safe_class', 'order', 'family', 'genus', 'species', 'genome_id']
    nodes = []
    descendents = []
    for i, level in enumerate(levels):
        for j in range(len(ids)):
            print(genome_df[level][j])
            print(level)
            if genome_df[level][j] not in nodes:
                nodes.append(genome_df[level][j])
                descendents.append([])
            if level != 'genome_id':
                pos = nodes.index(genome_df[level][j])
                if genome_df[levels[i+1]][j] not in descendents[pos]:
                    descendents[pos].append(genome_df[levels[i+1]][j])

    parent_child = np.zeros(shape=(len(nodes), len(nodes)))
    for i, node in enumerate(nodes):
        for child in descendents[nodes.index(node)]:
            parent_child[i][nodes.index(child)] = 1
    print(parent_child)
    return parent_child

def build_pc_mat_for_all_species(genome_file='genome_lineage.csv'):
    genome_df = pd.read_csv(genome_file, delimiter='\t', dtype=str)
    genome_df = genome_df.rename(columns={'class': 'safe_class'})
    genome_df = genome_df[genome_df.kingdom == 'Bacteria']
    genome_df = genome_df[
        (genome_df['kingdom'].notnull()) & (genome_df['phylum'].notnull()) & (genome_df['safe_class'].notnull())
        & (genome_df['order'].notnull()) & (genome_df['family'].notnull()) & (genome_df['genus'].notnull())
        & (genome_df['species'].notnull()) & (genome_df['genome_id'].notnull())]
    new_idx = range(genome_df.shape[0])
    genome_df.set_index(pd.Index(new_idx), inplace=True)
    levels = ['kingdom', 'phylum', 'safe_class', 'order', 'family', 'genus', 'species', 'genome_id']
    nodes = []
    descendents = []
    for i, level in enumerate(levels):
        for j in range(len(new_idx)):
            print(genome_df[level][j])
            print(level)
            if genome_df[level][j] not in nodes:
                nodes.append(genome_df[level][j])
                descendents.append([])
            if level != 'genome_id':
                pos = nodes.index(genome_df[level][j])
                if genome_df[levels[i + 1]][j] not in descendents[pos]:
                    descendents[pos].append(genome_df[levels[i + 1]][j])

    print(nodes)
    print(len(nodes))
    parent_child = np.zeros(shape=(len(nodes), len(nodes)))
    for i, node in enumerate(nodes):
        for child in descendents[nodes.index(node)]:
            parent_child[i][nodes.index(child)] = 1
    print(parent_child)
    return parent_child


if __name__ == "__main__":
    #build_pc_mat(genome_file='data_files/genome_lineage.csv', label_file='data_files/erythromycin_firmicutes_samples.csv')
    build_pc_mat_for_all_species(genome_file='data_files/genome_lineage.csv')