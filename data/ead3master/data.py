#%%
import pandas as pd
import os

def read_data(filepath):
    data = pd.read_csv(filepath)
    return data

def get_collection_types(data):
    collection_types = data['collection'].unique()
    return collection_types


filepath = r'C:\Users\schil\OneDrive\Desktop\School\Capstone\LOC\xml_master3.csv'

data = read_data(filepath)
#collection_types = get_collection_types(data)

# get afc data
data_afc = data[data['collection'] == 'afc']

# save data_afc to csv
data_afc.to_csv(r'C:\Users\schil\OneDrive\Desktop\School\Capstone\LOC\xml_master3_afc.csv', index=False)

# get dirs
base_path = os.getcwd()
data_path = os.path.dirname(base_path)

# save to csv in data folder
data_afc.to_csv(os.path.join(data_path, 'xml_master3_afc.csv'), index=False)
