import os
import csv
import xml.etree.ElementTree as ET
from typing import Dict

def parse_file_list_csv(file_path: str) -> Dict[str, str]:
    filename_to_id = {}
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            source_url = row['source_url']
            filename = os.path.basename(source_url)
            filename_to_id[filename] = row['id']
    return filename_to_id

def parse_search_results_csv(file_path: str) -> Dict[str, Dict]:
    id_to_metadata = {}
    with open(file_path, 'r') as csvfile:
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
                'online_formats': [row.get(f'online_format.{i}', '') for i in range(2) if row.get(f'online_format.{i}')],
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

def parse_ead_xml(file_path: str) -> Dict:
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

def parse_marc_xml(file_path: str) -> Dict:
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

def process_metadata(data_dir: str) -> Dict[str, Dict]:
    file_list_path = os.path.join(data_dir, 'file_list.csv')
    search_results_path = os.path.join(data_dir, 'search_results.csv')
    ead_path = os.path.join(data_dir, 'af012006.xml')
    marc_path = os.path.join(data_dir, 'af012006_marc.xml')

    print(f"Processing metadata from:")
    print(f"  File list: {file_list_path}")
    print(f"  Search results: {search_results_path}")
    print(f"  EAD file: {ead_path}")
    print(f"  MARC file: {marc_path}")

    filename_to_id = parse_file_list_csv(file_list_path)
    print(f"Parsed {len(filename_to_id)} entries from file_list.csv")

    id_to_metadata = parse