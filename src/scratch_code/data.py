#%%
import sys
import os
import pandas as pd
from lxml import etree

class DataRetriever:
    '''
    Class to retrieve data from EAD and MARC directories based on
    the search results csv file. It extracts the 'afc1937002' type
    of string from the aka columns of search_results.csv and searches
    for all ead and marc files; typically this will only match ead files.
    It then extracts the value(s) from the item.source_collection column
    and searches for all ead and marc files; typically this will only match
    to marc files. The results are stored in two lists, ead_files and 
    marc_files, and then combined into 'all_files'. 

    Parameters:
    - search_results: pandas DataFrame
    - ead_dir: str
    - marc_dir: str

    Returns:
    - ead_files: list
    - marc_files: list
    '''
    def __init__(self, search_results, ead_dir, marc_dir):
        self.search_results = search_results
        self.ead_dir = ead_dir
        self.marc_dir = marc_dir

    def process_records(self):
        # search EADs
        ead_files = self.search_eads()

        # search MARCs
        marc_files = self.search_marcs()

        return ead_files, marc_files

    def search_eads(self):
        # Extract and compare records from 'aka' fields
        record1 = self.search_results['aka.1'].head(1).str.extract(r'/([^/]+)/?$').values[0][0]
        record2 = self.search_results['aka.2'].head(2).str.extract(r'/([^/]+)/?$').values[0][0]
        
        # Check if they are all the same
        if record1 == record2:
            record = record1
            # Remove everything after a period if present
            if '.' in record:
                record = record.split('.')[0]

            # Search in EAD and MARC directories
            ead_record_files = self.search_in_directory(record, self.ead_dir)
            marc_record_files = self.search_in_directory(record, self.marc_dir)

            # record files found
            return ead_record_files + marc_record_files

        return []

    def search_marcs(self):
        # Extract and compare records from 'item.source_collection' field
        record1 = self.search_results['item.source_collection'].head(1).str.replace(r'\([^)]*\)', '', regex=True).values[0]
        record2 = self.search_results['item.source_collection'].head(2).str.replace(r'\([^)]*\)', '', regex=True).values[1]
        
        # Check if they are all the same
        if record1 == record2:
            record = record1

            # Search in EAD and MARC directories again for the new record
            ead_collection_files = self.search_in_directory(record, self.ead_dir)
            marc_collection_files = self.search_in_directory(record, self.marc_dir)

            # record files found
            return ead_collection_files + marc_collection_files

        return []

    def search_in_directory(self, record, directory):
        matched_files = []
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path):
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    file_content = f.read()
                    if record in file or record in file_content:
                        matched_files.append(file)
        return matched_files


# Add directories to sys path
base_dir = os.path.dirname(os.path.abspath(__file__)) # scratch_code
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # src
gparent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # capstone
sys.path.append(base_dir)
sys.path.append(parent_dir)
sys.path.append(gparent_dir)

# xml directory from capstone/data/xml
xml_dir = os.path.join(gparent_dir, 'data', 'xml')
ead_dir = os.path.join(xml_dir, 'ead')
marc_dir = os.path.join(xml_dir, 'marc')
marc_test = os.path.join(gparent_dir, 'data', 'marc-xl-data')

# read in search results csv
search_results = pd.read_csv(os.path.join(marc_test, 'search_results.csv'))

processor = DataRetriever(search_results, ead_dir, marc_dir)
ead_files, marc_files = processor.process_records()
all_files = ead_files + marc_files

#print(f'ead files: {ead_files}')
#print(f'marc files: {marc_files}')
print(f'all files: {all_files}')


