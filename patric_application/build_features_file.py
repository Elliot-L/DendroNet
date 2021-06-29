import pandas as pd
import os

folder = 'data_files/sp_genes/ciprofloxacin'

for directory in os.listdir(folder):
    print(directory)
    for genome in os.listdir(os.path.join(folder, directory)):
        print(genome)
        df = pd.read_csv(os.path.join(folder, directory, genome), sep='\t')
        print(df)
        print(df['function classification'])
    break