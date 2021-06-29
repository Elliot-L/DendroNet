import pandas as pd
import os

folder = 'data_files/sp_genes/ciprofloxacin'

for dir in os.listdir(folder):
    print(dir)
    for genome in os.listdir(dir):
        df = pd.read_csv(os.path.join(dir, genome))
        print(df)
        print(df['function classification'])
    break