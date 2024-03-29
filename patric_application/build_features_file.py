import pandas as pd
import os


def build_feature_matrix(antibiotic='ciprofloxacin', group='Proteobacteria', threshold=0.05):

    folder = os.path.join('data_files', 'subproblems', group + '_' + antibiotic, 'sp_genes')

    #lists for DataFrame
    ids = []
    antibiotics = []
    phenotypes = []
    annotation = []
    features = []

    functions = set()

    ids_dict = {}

    for directory in os.listdir(folder):
        genome = directory.split('_')[0]
        spgenes_file = genome + '.PATRIC.spgene.tab'
        feat_dict = {}
        df = pd.read_csv(os.path.join(folder, directory, spgenes_file), sep='\t')
        for function in df['function']:
            if type(function) is str:
                if function not in feat_dict.keys():
                    feat_dict[function] = 1
                else:
                    feat_dict[function] += 1
        ids_dict[genome] = feat_dict

    for id in ids_dict.keys():
        functions = functions.intersection(set(ids_dict[id].keys()))

    if antibiotic == 'ciprofloxacin':
        pheno_file = os.path.join('data_files', 'proteobacteria_ciprofloxacin.csv')
    elif antibiotic == 'betalactam':
        pheno_file = os.path.join('data_files', 'betalactam_firmicutes_samples.csv')
    elif antibiotic == 'erythromycin':
        pheno_file = os.path.join('data_files', 'erythromycin_firmicutes_samples.csv')

    pheno_df = pd.read_csv(pheno_file)
    if 'Genome ID' in pheno_df.columns:
        pheno_df.rename(columns={'Genome ID': 'ID'}, inplace=True)
    if 'Resistance Phenotype' in pheno_df.columns:
        pheno_df.rename(columns={'Resistance Phenotype': 'Phenotype'}, inplace=True)

    for id in ids_dict.keys():
        ids.append(id)
        antibiotics.append([antibiotic])
        phenotypes.append([pheno_df['Phenotype'][pheno_df['ID'].tolist().index(id)]])
        annotation.append(['True'])
        genome_features = []
        for func in functions:
            genome_features.append(ids_dict[id][func])
        features.append(genome_features)

    file_dict = {'ID': ids, 'Antibiotic': antibiotics, 'Phenotype': phenotypes, 'Annotation': annotation, 'Features': features}
    file_df = pd.DataFrame(file_dict)
    #Creating file from which the X matric (feature matrix), and the y vector (target values vector) can be created
    file_df.to_csv(os.path.join('data_files', group + '_' + antibiotic + '_sample.csv'), index=False)

if __name__ == "__main__":
    build_feature_matrix()
