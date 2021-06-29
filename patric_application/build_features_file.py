import pandas as pd
import os

folder = 'data_files/sp_genes/ciprofloxacin'

for dir in os.listdir(folder):
    print(dir)
    for genome in os.listdir(os.path.joing(folder, dir)):
        print(genome)
        df = pd.read_csv(os.path.join(folder, dir, genome))
        print(df)
        print(df['function classification'])
    break