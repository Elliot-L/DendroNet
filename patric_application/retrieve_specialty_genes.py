import os
#import urllib
#import wget
import time
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--antibiotic', type=str)
parser.add_argument('--group', type=str)
args = parser.parse_args()


# example final url:ftp://ftp.patricbrc.org/genomes/511145.12/511145.12.PATRIC.spgene.tab

# antibiotics = ['moxifloxacin', 'azithromycin', 'clarithromycin', 'clindamycin', 'ceftriaxone']
# antibiotics = ['betalactam']

# antibiotics = ['betalactam', 'ciprofloxacin', 'cloramphenicol', 'cotrimoxazole',
#                'fusidicacid', 'gentamicin', 'rifampin', 'trimethoprim', 'vancomycin']

# antibiotics = ['trimethoprim', 'tetracycline', 'isoniazid', 'ethambutol', 'streptomycin']
#antibiotics = ['chloramphenicol']
antibiotics = ['ciprofloxacin', 'betalactam', 'erythromycin']


# ciprofloxacin
# rifampin
# erythromycin
# gentamicin
# chloramphenicol
# kanamycin
# ofloxacin
# levofloxacin
# cefoxitin
# imipenem

errors = list()
print(os.getcwd())
os.chdir(os.path.join('..', 'patric_application'))
print(os.getcwd())

base_url = 'ftp://ftp.patricbrc.org/genomes/'
extension = '.PATRIC.spgene.tab'

for ab in antibiotics:

    genomes = set()
    """
    below lines were used for retrieving data for the january submission
    """
    # genome_file_dir = 'data_files/genome_ids'
    # base_out = 'data_files/sp_genes/' + ab + '/'
    """
    New lines for the retrieval pattern for the whole tree (april submission)
    """

    base_out = os.path.join('data_files', 'subproblems', args.group + '_' + args.antibiotic, 'sp_genes')

    for file in os.listdir(genome_file_dir):
        if ab in file:
            # df = pd.read_csv(os.path.join(genome_file_dir, file), sep=',', dtype=str)
            # genomes = genomes.union(set(df['Genome ID']))
            df = pd.read_csv(os.path.join(genome_file_dir, file), sep=',', dtype=str)
            if 'Genome ID' in df.columns:
                df.rename(columns={'Genome ID': 'ID'}, inplace=True)
            if args.group in file:
                genomes = genomes.union(set(df['ID']))

            print(df)
    for genome in proteobacteria_genomes:
        try:
            print(genome)
            fp = base_url + genome + '/' + genome + extension
            outfile = base_out + 'proteobacteria/' + genome + '_spgene.tab'
            command = 'wget -P ' + outfile + ' ' + fp
            os.system(command)
            #urllib.urlretrieve(fp, filename=outfile)
            #wget.download(fp, outfile)
        except:
            errors.append(genome)
    for genome in firmicutes_genomes:
        try:
            print(genome)
            fp = base_url + genome + '/' + genome + extension
            outfile = base_out + 'firmicutes/' + genome + '_spgene.tab'
            command = 'wget -P ' + outfile + ' ' + fp
            os.system(command)
            #urllib.urlretrieve(fp, filename=outfile)
            #wget.download(fp, outfile)
        except:
            errors.append(genome)
    for genome in other_genomes:
        try:
            print(genome)
            fp = base_url + genome + '/' + genome + extension
            outfile = base_out + 'other/' + genome + '_spgene.tab'
            command = 'wget -P ' + outfile + ' ' + fp
            os.system(command)
            #urllib.urlretrieve(fp, filename=outfile)
            #wget.download(fp, outfile)
        except:
            errors.append(genome)
    print('done ' + ab)
print('genomes with errors:')
print(errors)

""""
import urllib
urllib.urlretrieve("http://google.com/index.html", filename="local/index.html")
 
os.system.command('wget...')

"""