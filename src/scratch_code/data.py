#%%
import sys
import os
import re
import pandas as pd
from lxml import etree
from typing import Dict
import csv
import xml.etree.ElementTree as ET

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
    def __init__(self, ead_dir, marc_dir, loc_dot_dir): #search_results, 
        #self.search_results = search_results
        self.ead_dir = ead_dir
        self.marc_dir = marc_dir
        self.loc_dot_dir = loc_dot_dir

    def process_records(self, search_results):
        # search EADs
        ead_files = self.search_eads(search_results)

        # search MARCs
        marc_files = self.search_marcs(search_results)

        return ead_files, marc_files

    def search_ead_marc(self, search_results):
        # Extract and process unique 'aka' records
        aka_records = search_results['aka.1'].dropna().unique()
        aka_processed_records = []
        for record in aka_records:
            extracted_record = re.search(r'/([^/]+)/?$', record)
            if extracted_record:
                record = extracted_record.group(1)
            if '.' in record:
                record = record.split('.')[0]
            aka_processed_records.append(record)
        
        # Extract and process unique 'item.source_collection' records
        source_collection_records = search_results['item.source_collection'].dropna().unique()
        processed_source_collection_records = [re.sub(r'\([^)]*\)', '', record) for record in source_collection_records]
        
        # Search in EAD and MARC directories for both 'aka' and 'source_collection' records
        ead_record_files = []
        marc_record_files = []
        
        for record in aka_processed_records:
            ead_record_files += self.search_in_directory(record, self.ead_dir)
            marc_record_files += self.search_in_directory(record, self.marc_dir)
        
        for record in processed_source_collection_records:
            ead_record_files += self.search_in_directory(record, self.ead_dir)
            marc_record_files += self.search_in_directory(record, self.marc_dir)
        
        # Combine all found files from both search criteria
        return ead_record_files, marc_record_files
        
    def search_eads(self, search_results):
        # Extract and compare records from 'aka' fields
        record1 = search_results['aka.1'].head(1).str.extract(r'/([^/]+)/?$').values[0][0]
        #record2 = search_results['aka.2'].head(2).str.extract(r'/([^/]+)/?$').values[0][0]
        
        # Check if they are all the same
        #if record1 == record2:
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

    def search_marcs(self, search_results):
        # Extract and compare records from 'item.source_collection' field
        record1 = search_results['item.source_collection'].head(1).str.replace(r'\([^)]*\)', '', regex=True).values[0]
        record2 = search_results['item.source_collection'].head(2).str.replace(r'\([^)]*\)', '', regex=True).values[1]
        
        # Check if they are all the same
        if record1 == record2:
            record = record1

            # Search in EAD and MARC directories again for the new record
            ead_collection_files = self.search_in_directory(record, self.ead_dir)
            marc_collection_files = self.search_in_directory(record, self.marc_dir)

            # record files found
            return ead_collection_files + marc_collection_files

        return []

    def parse_file_list_csv(self, file_path: str) -> Dict[str, str]: # called in process_metadata below
        filename_to_id = {}
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                source_url = row['source_url']
                filename = os.path.basename(source_url)
                filename_to_id[filename] = row['id']
        return filename_to_id

    def parse_search_results_csv(self, file_path: str) -> Dict[str, Dict]: # called in process_metadata below
        id_to_metadata = {}
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                metadata = {
                    'title': row['title'],
                    'contributors': [row.get(f'contributor.{i}', '') for i in range(3) if row.get(f'contributor.{i}')],
                    'date': row['date'],
                    'subjects': [row.get(f'subject.{i}', '') for i in range(5) if row.get(f'subject.{i}')],
                    'type': row.get('type.0', ''),
                    'language': row.get('language.0', ''),
                    'locations': [row.get(f'location.{i}', '') for i in range(3) if row.get(f'location.{i}')],
                    'original_format': row.get('original_format.0', ''),
                    'online_formats': [row.get(f'online_format.{i}', '') for i in range(2) if
                                    row.get(f'online_format.{i}')],
                    'description': row.get('description', ''),
                    'rights': row.get('rights', ''),
                    'collection': row.get('collection', ''),
                    'timestamp': row.get('timestamp', ''),
                    'created_published': row.get('item.created_published.0', ''),
                    'notes': [row.get(f'item.notes.{i}', '') for i in range(2) if row.get(f'item.notes.{i}')],
                    'url': row.get('url', '')
                }
                id_to_metadata[row['id']] = metadata
        return id_to_metadata

    def parse_ead_xml(self, file_path: str) -> Dict: # called in process_metadata below
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()

            # Define the namespace
            ns = {'ead': 'http://ead3.archivists.org/schema/'}

            collection_title = root.find('.//ead:titleproper', ns)
            collection_date = root.find('.//ead:archdesc/ead:did/ead:unitdate', ns)
            collection_abstract = root.find('.//ead:archdesc/ead:did/ead:abstract', ns)

            return {
                'collection_title': collection_title.text.strip() if collection_title is not None else "Unknown Title",
                'collection_date': collection_date.text.strip() if collection_date is not None else "Unknown Date",
                'collection_abstract': collection_abstract.text.strip() if collection_abstract is not None else "No abstract available"
            }
        
        except Exception as e:
            print(f"Warning: Error parsing EAD XML file at {file_path}: {str(e)}")
            return {
                'collection_title': "Unknown Title",
                'collection_date': "Unknown Date",
                'collection_abstract': "No abstract available"
            }

    def parse_marc_xml(self, file_path: str) -> Dict: # called in process_metadata below
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()

            catalog_title = root.find(".//datafield[@tag='245']/subfield[@code='a']")
            catalog_creator = root.find(".//datafield[@tag='100']/subfield[@code='a']")
            catalog_date = root.find(".//datafield[@tag='260']/subfield[@code='c']")

            return {
                'catalog_title': catalog_title.text if catalog_title is not None else "Unknown Title",
                'catalog_creator': catalog_creator.text if catalog_creator is not None else "Unknown Creator",
                'catalog_date': catalog_date.text if catalog_date is not None else "Unknown Date"
            }
        except Exception as e:
            print(f"Warning: Error parsing MARC XML file at {file_path}: {str(e)}")
            return {
                'catalog_title': "Unknown Title",
                'catalog_creator': "Unknown Creator",
                'catalog_date': "Unknown Date"
            }

    def process_metadata_for_all_subfolders(self) -> Dict[str, Dict]:
        '''
        no 'item.source_collection' in search_results.csv for the below subfolder (req'd to ID MARC files)
        - american-folklife-center-web-archive
        - robert-winslow-gordon-songsters
        - web-cultures-web-archive

        subfolders with multiple file_list csv files
        - songs-of-america
        - traditional-music-and-spoken-word
        - woody-guthrie-correspondence-from-1940-to-1950
        '''
        all_metadata = {}
        
        # Walk through the main directory and its subdirectories
        for subdir, dirs, files in os.walk(self.loc_dot_dir):
            # Check if the necessary files are in the current subdirectory
            if 'file_list.csv' in files and 'search_results.csv' in files:
                search_results_path = os.path.join(subdir, 'search_results.csv')
                try:
                    df = pd.read_csv(search_results_path, low_memory=False)
                    # Check if 'item.source_collection' is in the columns
                    if 'item.source_collection' in df.columns:
                        # Call process_metadata for the current subdirectory
                        print(f"Processing directory: {subdir}")
                        metadata = self.process_metadata(subdir)

                        # Combine the results into a single dictionary
                        all_metadata.update(metadata)
                    else:
                        print(f"Skipping {subdir}, 'item.source_collection' not found in search_results.csv.\n")
                except Exception as e:
                    print(f"Error processing {subdir}: {e}")
            else:
                print(f"Skipping {subdir}, required files not found.\n")

        print(f"\nProcessed metadata from {len(all_metadata)} files across subfolders")
        return all_metadata

    def process_metadata(self, loc_dot_dir_subdir: str) -> Dict[str, Dict]: # called in the main function
        file_list_path = os.path.join(loc_dot_dir_subdir, 'file_list.csv')
        search_results_path = os.path.join(loc_dot_dir_subdir, 'search_results.csv')

        # search for ead and marc files based on search_results.csv
        search_results = pd.read_csv(search_results_path, low_memory=False)
        #ead_files, marc_files = self.process_records(search_results)
        ead_files, marc_files = self.search_ead_marc(search_results)
        all_files = ead_files + marc_files
        print(all_files)
        


        '''
        # if not empty, get the filename for ead and marc
        if ead_files:
            ead_filename = ead_files[0]
            ead_path = os.path.join(self.ead_dir, ead_filename) 
            ead_metadata = self.parse_ead_xml(ead_path)
        else:
            ead_metadata = None
            print('No EAD file found')
        
        if marc_files:
            marc_filename = marc_files[0]
            marc_path = os.path.join(self.marc_dir, marc_filename)
            marc_metadata = self.parse_marc_xml(marc_path)
        else:
            marc_metadata = None
            print('No MARC file found')

        filename_to_id = self.parse_file_list_csv(file_list_path)
        id_to_metadata = self.parse_search_results_csv(search_results_path)

        # iterate through file_list.csv and get metadata from search_results.csv
        filename_to_metadata = {}
        for filename, doc_id in filename_to_id.items():
            if doc_id in id_to_metadata:
                metadata = id_to_metadata[doc_id] # get the row of data in search_results.csv that contains the id/url from the file_list.csv
                metadata.update(ead_metadata) # add the ead metadata to the search_results metadata
                metadata.update(marc_metadata) # add the marc metadata to the search_results metadata
                filename_to_metadata[filename] = metadata # add filename, corresponding metadata to dict
            else:
                print(f"Warning: No metadata found for document ID {doc_id} (filename: {filename})")

        #print(f"Combined metadata for {len(filename_to_metadata)} files")
        '''
        return all_files#filename_to_metadata


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
loc_dot_dir = os.path.join(r'C:\Users\schil\OneDrive\Desktop\School\Capstone\LOC\loc_dot_gov_data2')

# read in search results csv
processor = DataRetriever(ead_dir, marc_dir, loc_dot_dir)
all_files = processor.process_metadata_for_all_subfolders()














































#%%

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
loc_dot_dir = os.path.join(r'C:\Users\schil\OneDrive\Desktop\School\Capstone\LOC\loc_dot_gov_data2')

# read in search results csv
processor = DataRetriever(ead_dir, marc_dir, loc_dot_dir)
all_metadata = processor.process_metadata_for_all_subfolders()







#print(f'ead files: {ead_files}')
#print(f'marc files: {marc_files}')
#print(f'all files: {all_files}')


