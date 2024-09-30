#%%

# script to read in all csvs, combine them, and save them as a single csv
import os
import pandas as pd

# get the data folder
base_path = os.path.dirname(os.path.abspath(__file__)) # data folder
parent = os.path.dirname(base_path)
grandparent = os.path.dirname(parent)

# base_path + '/data'
base_path = os.path.join(grandparent, 'QA_chunks')

# get all csvs in the data folder that begin with afc_txtFiles_QA_chunk
csvs = [os.path.join(base_path, f) for f in os.listdir(base_path) if f.startswith('afc_txtFiles_QA_chunk')]

# read in all csvs
dfs = [pd.read_csv(f) for f in csvs]

# combine all csvs
df = pd.concat(dfs, ignore_index=True)

# save the combined csv
df.to_csv(os.path.join(base_path, 'afc_txtFiles_QA.csv'), index=False)
