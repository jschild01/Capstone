import os
import sys
import time
import torch
import logging
import datetime
import gc
import pandas as pd
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings
import shutil
import re
import csv
from typing import Dict, List, Any
#from langchain.vectorstores import DeepLake
from langchain_community.vectorstores import DeepLake # last one was depracated
from dotenv import load_dotenv
import boto3
from botocore.exceptions import ClientError
import json
import xml.etree.ElementTree as ET
from configparser import ConfigParser, ExtendedInterpolation
from langchain_aws import BedrockEmbeddings
from pymarc import parse_xml_to_array



def extract_call_number(text: str) -> str:
    # Modify the regex to capture 'AFC 1937/002' even if followed by additional characters
    pattern = r'\bAFC\s*\d{4}/\d{3}\b'
    match = re.search(pattern, text)
    return match.group(0).strip() if match else None

def parse_file_list_csv(file_path: str) -> Dict[str, str]:
    filename_to_id = {}
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            source_url = row['source_url']
            filename = os.path.basename(source_url)
            filename_to_id[filename] = row['id']
    return filename_to_id


def parse_search_results_csv_sample(file_path: str) -> Dict[str, Dict]:
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
                'online_formats': [row.get(f'online_format.{i}', '') for i in range(2) if row.get(f'online_format.{i}')],
                'description': row.get('description', ''),
                'rights': row.get('rights', ''),
                'collection': row.get('collection', ''),
                'timestamp': row.get('timestamp', ''),
                'created_published': row.get('item.created_published.0', ''),
                'notes': [row.get(f'item.notes.{i}', '') for i in range(2) if row.get(f'item.notes.{i}')],
                'url': row.get('url', ''),
                'call_number': row.get('item.call_number.0', '')  # Extract call_number here
            }
            id_to_metadata[row['id']] = metadata
    return id_to_metadata

def parse_search_results_csv(file_path: str) -> Dict[str, Dict]:
    id_to_metadata = {}
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            full_call_number = row.get('item.call_number.0', '').strip()
            call_number = extract_call_number(full_call_number)

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
                'url': row.get('url', ''),
                'call_number': call_number,
                'full_call_number': full_call_number
            }
            id_to_metadata[row['id']] = metadata

    return id_to_metadata


def parse_ead_xml(file_path: str) -> Dict[str, Dict]:
    ead_metadata = {}
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        ns = {'ead': 'http://ead3.archivists.org/schema/'}

        collection_title_element = root.find('.//ead:titleproper', ns)
        collection_title = collection_title_element.text.strip() if collection_title_element is not None else 'N/A'

        collection_date_element = root.find('.//ead:archdesc/ead:did/ead:unitdate', ns)
        collection_date = collection_date_element.text.strip() if collection_date_element is not None else 'N/A'

        collection_abstract_element = root.find('.//ead:archdesc/ead:did/ead:abstract', ns)
        collection_abstract = collection_abstract_element.text.strip() if collection_abstract_element is not None else 'N/A'

        for unitid in root.findall(
                './/ead:unitid[@label="Call No."][@encodinganalog="050"][@countrycode="US"][@repositorycode="US-DLC"]',
                ns):
            call_number = extract_call_number(unitid.text)
            if call_number:
                series_title_element = unitid.find('../ead:unittitle', ns)
                series_title = series_title_element.text.strip() if series_title_element is not None else 'N/A'

                ead_metadata[call_number] = {
                    'collection_title': collection_title,
                    'collection_date': collection_date,
                    'collection_abstract': collection_abstract,
                    'series_title': series_title
                }
                
    except Exception as e:
        print(f"Error parsing EAD XML file at {file_path}: {str(e)}")
    return ead_metadata

def parse_ead_xml_sample(file_path: str) -> Dict[str, Dict]:
    ead_metadata = {}
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        ns = {'ead': 'http://ead3.archivists.org/schema/'}

        collection_title_element = root.find('.//ead:titleproper', ns)
        collection_title = collection_title_element.text.strip() if collection_title_element is not None else 'N/A'

        collection_date_element = root.find('.//ead:archdesc/ead:did/ead:unitdate', ns)
        collection_date = collection_date_element.text.strip() if collection_date_element is not None else 'N/A'

        collection_abstract_element = root.find('.//ead:archdesc/ead:did/ead:abstract', ns)
        collection_abstract = collection_abstract_element.text.strip() if collection_abstract_element is not None else 'N/A'

        for unitid in root.findall(
                './/ead:unitid[@label="Call No."][@encodinganalog="050"][@countrycode="US"][@repositorycode="US-DLC"]',
                ns):
            call_number = extract_call_number(unitid.text)
            if call_number:
                series_title_element = unitid.find('../ead:unittitle', ns)
                series_title = series_title_element.text.strip() if series_title_element is not None else 'N/A'

                ead_metadata[call_number] = {
                    'collection_title': collection_title,
                    'collection_date': collection_date,
                    'collection_abstract': collection_abstract,
                    'series_title': series_title
                }


    except Exception as e:
        print(f"Warning: Error parsing EAD XML file at {file_path}: {str(e)}")
    return ead_metadata


def parse_marc_xml(file_path: str) -> Dict[str, Dict]:
    marc_metadata = {}
    try:
        # Use pymarc to parse the XML file
        records = parse_xml_to_array(file_path)

        for record in records:
            # Extract call numbers from 090 field
            call_numbers = []
            for field in record.get_fields('090'):
                for subfield in field.get_subfields('a'):
                    if subfield:
                        # Split multiple call numbers separated by ';' or ','
                        possible_call_numbers = re.split(r';|,', subfield)
                        for cn in possible_call_numbers:
                            cn = cn.strip()
                            call_number = extract_call_number(cn)
                            if call_number:
                                call_numbers.append(call_number)

            # Process each valid call number found
            for call_number in call_numbers:
                metadata = {
                    'catalog_title': (record.get_fields('245')[0].get_subfields('a')[0].strip()
                                      if record.get_fields('245') and record.get_fields('245')[0].get_subfields('a')
                                      else 'N/A'),
                    'catalog_creator': (record.get_fields('100')[0].get_subfields('a')[0].strip()
                                        if record.get_fields('100') and record.get_fields('100')[0].get_subfields('a')
                                        else 'N/A'),
                    'catalog_date': (record.get_fields('260')[0].get_subfields('c')[0].strip()
                                     if record.get_fields('260') and record.get_fields('260')[0].get_subfields('c')
                                     else 'N/A'),
                    'catalog_description': (record.get_fields('520')[0].get_subfields('a')[0].strip()
                                            if record.get_fields('520') and record.get_fields('520')[0].get_subfields('a')
                                            else 'N/A'),
                    'catalog_subjects': [field.get_subfields('a')[0].strip() for field in record.get_fields('650')
                                         if field.get_subfields('a')],
                    'catalog_notes': [field.get_subfields('a')[0].strip() for field in record.get_fields('500')
                                      if field.get_subfields('a')],
                    'catalog_language': (record.get_fields('041')[0].get_subfields('a')[0].strip()
                                         if record.get_fields('041') and record.get_fields('041')[0].get_subfields('a')
                                         else 'N/A'),
                    'catalog_genre': [field.get_subfields('a')[0].strip() for field in record.get_fields('655')
                                      if field.get_subfields('a')],
                    'catalog_contributors': [field.get_subfields('a')[0].strip() for field in record.get_fields('700')
                                             if field.get_subfields('a')],
                    'catalog_repository': (record.get_fields('852')[0].get_subfields('a')[0].strip()
                                           if record.get_fields('852') and record.get_fields('852')[0].get_subfields('a')
                                           else 'N/A'),
                    'catalog_collection_id': (record.get_fields('001')[0].data
                                              if record.get_fields('001')
                                              else 'N/A')
                }
                marc_metadata[call_number] = metadata

    except Exception as e:
        print(f"Error parsing MARC XML file at {file_path}: {str(e)}")
    return marc_metadata

def parse_marc_xml_sample(file_path: str) -> Dict[str, Dict]:
    marc_metadata = {}
    try:
        # Use pymarc to parse the XML file
        records = parse_xml_to_array(file_path)

        for record in records:
            # Extract call numbers from 090 field
            call_numbers = []
            for field in record.get_fields('090'):
                for subfield in field.get_subfields('a'):
                    if subfield:
                        # Split multiple call numbers separated by ';' or ','
                        possible_call_numbers = re.split(r';|,', subfield)
                        for cn in possible_call_numbers:
                            cn = cn.strip()
                            call_number = extract_call_number(cn)
                            if call_number:
                                call_numbers.append(call_number)

            # Process each valid call number found
            for call_number in call_numbers:
                metadata = {
                    'catalog_title': record.get_fields('245')[0].get_subfields('a')[0].strip() if record.get_fields(
                        '245') else 'N/A',
                    'catalog_creator': record.get_fields('100')[0].get_subfields('a')[0].strip() if record.get_fields(
                        '100') else 'N/A',
                    'catalog_date': record.get_fields('260')[0].get_subfields('c')[0].strip() if record.get_fields(
                        '260') else 'N/A',
                    'catalog_description': record.get_fields('520')[0].get_subfields('a')[
                        0].strip() if record.get_fields('520') else 'N/A',
                    'catalog_subjects': [field.get_subfields('a')[0].strip() for field in record.get_fields('650') if
                                         field.get_subfields('a')],
                    'catalog_notes': [field.get_subfields('a')[0].strip() for field in record.get_fields('500') if
                                      field.get_subfields('a')],
                    'catalog_language': record.get_fields('041')[0].get_subfields('a')[0].strip() if record.get_fields(
                        '041') else 'N/A',
                    'catalog_genre': [field.get_subfields('a')[0].strip() for field in record.get_fields('655') if
                                      field.get_subfields('a')],
                    'catalog_contributors': [field.get_subfields('a')[0].strip() for field in record.get_fields('700')
                                             if field.get_subfields('a')],
                    'catalog_repository': record.get_fields('852')[0].get_subfields('a', 'b')[
                        0].strip() if record.get_fields('852') else 'N/A',
                    'catalog_collection_id': record.get_fields('001')[0].data if record.get_fields('001') else 'N/A'
                }
                marc_metadata[call_number] = metadata
                
    except Exception as e:
        print(f"Error parsing MARC XML file at {file_path}: {str(e)}")
    return marc_metadata

def parse_marc_xml_older(file_path: str) -> Dict[str, Dict]:
    marc_metadata = {}
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        for record in root.findall('.//record'):
            # Find all '090' datafields in the record
            datafields_090 = record.findall('.//datafield[@tag="090"]/subfield[@code="a"]')

            # Extract call numbers from each '090' datafield
            for subfield_a in datafields_090:
                call_number_text = subfield_a.text
                if call_number_text:
                    # Split multiple call numbers separated by ';' or ','
                    possible_call_numbers = re.split(r';|,', call_number_text)
                    for cn in possible_call_numbers:
                        cn = cn.strip()
                        if cn:
                            # Extract the primary call number using the existing function
                            call_number = extract_call_number(cn)
                            if call_number:
                                # Extract other relevant fields once per record
                                catalog_title_field = record.find('.//datafield[@tag="245"]/subfield[@code="a"]')
                                catalog_title = catalog_title_field.text.strip() if catalog_title_field is not None else 'N/A'

                                catalog_creator_field = record.find('.//datafield[@tag="100"]/subfield[@code="a"]')
                                catalog_creator = catalog_creator_field.text.strip() if catalog_creator_field is not None else 'N/A'

                                catalog_date_field = record.find('.//datafield[@tag="260"]/subfield[@code="c"]')
                                catalog_date = catalog_date_field.text.strip() if catalog_date_field is not None else 'N/A'

                                # Map the call number to the metadata
                                marc_metadata[call_number] = {
                                    'catalog_title': catalog_title,
                                    'catalog_creator': catalog_creator,
                                    'catalog_date': catalog_date
                                }

                                # Debug statement
                                #print(f"Extracted MARC Call Number: {call_number}")
    except Exception as e:
        print(f"Warning: Error parsing MARC XML file at {file_path}: {str(e)}")
    # Debug statement to print all keys
    #print("MARC Metadata Keys:", marc_metadata.keys())
    return marc_metadata


def find_metadata_in_xml_files(call_number: str, xml_dir: str, parser_function) -> Dict:
    for filename in os.listdir(xml_dir):
        if filename.endswith('.xml'):
            file_path = os.path.join(xml_dir, filename)
            metadata = parser_function(file_path)
            if call_number in metadata:
                return metadata[call_number]
    return {}


def process_metadata_sample(data_dir: str) -> Dict[str, Dict]:
    # file paths
    file_list_path = os.path.join(data_dir, 'file_list.csv')
    search_results_path = os.path.join(data_dir, 'search_results.csv')
    ead_path = os.path.join(data_dir, 'af012006.xml')
    marc_path = os.path.join(data_dir, 'af012006_marc.xml')

    filename_to_id = parse_file_list_csv(file_list_path)
    id_to_metadata = parse_search_results_csv_sample(search_results_path)
    ead_metadata = parse_ead_xml_sample(ead_path)
    marc_metadata = parse_marc_xml_sample(marc_path) # not 'older'

    filename_to_metadata = {}
    txt_dir = os.path.join(data_dir, 'txt')
    for filename in os.listdir(txt_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(txt_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

            base_filename = re.sub(r'_(en|en_translation)\.txt$', '.mp3', filename)

            if base_filename in filename_to_id:
                doc_id = filename_to_id[base_filename]
                if doc_id in id_to_metadata:
                    metadata = id_to_metadata[doc_id].copy()
                    metadata['original_filename'] = filename

                    # Extract call number from the metadata
                    call_number_full = metadata.get('call_number', '').strip()

                    # Use extract_call_number to get only the primary call number
                    call_number = extract_call_number(call_number_full)

                    if not call_number:
                        # Fallback: Extract call number from the content if not present in metadata
                        call_number = extract_call_number(content) or 'N/A'

                    metadata['call_number'] = call_number

                    # Integrate EAD metadata
                    if call_number in ead_metadata:
                        metadata.update(ead_metadata[call_number])
                    else:
                        print(f"\nWarning: EAD metadata not found for call number '{call_number}' (file: {filename})\n")

                    # Integrate MARC metadata
                    if call_number in marc_metadata:
                        metadata.update(marc_metadata[call_number])
                    else:
                        print(f"\nWarning: MARC metadata not found for call number '{call_number}' (file: {filename})\n")

                    # Ensure all metadata fields are strings and non-empty
                    metadata = {k: str(v) if v else 'N/A' for k, v in metadata.items()}
                    filename_to_metadata[filename] = metadata
                else:
                    print(f"\nWarning: No metadata found for document ID {doc_id} (filename: {filename})\n")
            else:
                print(f"\nWarning: No matching entry found in file_list.csv for {filename} (base: {base_filename})\n")

    print(f"Processed metadata for {len(filename_to_metadata)} files.")
    return filename_to_metadata

def process_metadata(data_dir: str) -> Dict[str, Dict]:
    # Initialize paths
    loc_data_dir = os.path.join(data_dir, 'loc_dot_gov_data')
    ead_dir = os.path.join(data_dir, 'xml', 'ead')
    marc_dir = os.path.join(data_dir, 'xml', 'marc')

    all_metadata = {}
    error_log = []

    # Process each collection
    for collection_name in os.listdir(loc_data_dir):
        collection_dir = os.path.join(loc_data_dir, collection_name)
        if os.path.isdir(collection_dir):

            # Process CSVs
            file_list_path = os.path.join(collection_dir, 'file_list.csv')
            search_results_path = os.path.join(collection_dir, 'search_results.csv')

            filename_to_id = parse_file_list_csv(file_list_path)
            id_to_metadata = parse_search_results_csv(search_results_path)

            # Process all text files
            txt_dir = os.path.join(data_dir, 'txt')
            transcripts_dir = os.path.join(data_dir, 'transcripts')
            ocr_dir = os.path.join(data_dir, 'pdf', 'txtConversion')

            for directory in [txt_dir, transcripts_dir, ocr_dir]:
                for filename in os.listdir(directory):
                    if filename.endswith('.txt'):
                        file_path = os.path.join(directory, filename)

                        # Handle different file types
                        if directory == transcripts_dir:
                            # Only transcripts need the mp3 conversion
                            base_filename = re.sub(r'_(en|en_translation)\.txt$', '.mp3', filename)
                            file_type = 'transcript'
                        elif directory == ocr_dir:
                            base_filename = re.sub(r'\.txt$', '.pdf', filename)
                            file_type = 'pdf_ocr'
                        else:  # txt_dir
                            # Regular text files don't need conversion
                            base_filename = filename
                            file_type = 'text'

                        if base_filename in filename_to_id:
                            doc_id = filename_to_id[base_filename]
                            
                            if doc_id in id_to_metadata:
                                metadata = id_to_metadata[doc_id].copy()
                                
                                # Add file metadata
                                metadata['original_filename'] = filename
                                metadata['file_type'] = file_type

                                # Get call number
                                call_number = metadata.get('call_number')
                                if call_number:
                                    # Add EAD metadata
                                    ead_metadata = find_metadata_in_xml_files(call_number, ead_dir, parse_ead_xml)
                                    if ead_metadata:
                                        metadata.update(ead_metadata)
                                        
                                    # Add MARC metadata
                                    marc_metadata = find_metadata_in_xml_files(call_number, marc_dir, parse_marc_xml)
                                    if marc_metadata:
                                        metadata.update(marc_metadata)
                                        
                                # Clean metadata
                                metadata = {k: str(v) if v is not None else 'N/A' for k, v in metadata.items()}
                                all_metadata[filename] = metadata
                            else:
                                error_msg = f"No metadata found in search_results.csv for ID {doc_id} (file: {filename})"
                                error_log.append(error_msg)
                        else:
                            error_msg = f"No entry found in file_list.csv for {base_filename} (original: {filename})"
                            error_log.append(error_msg)

    # Log errors
    if error_log:
        error_log_path = os.path.join(data_dir, 'metadata_processing_errors.log')
        print(f"\nWriting {len(error_log)} errors to: {error_log_path}")
        with open(error_log_path, 'w') as f:
            for error in error_log:
                f.write(f"{error}\n")
    else:
        print("\nNo errors encountered during metadata processing")

    return all_metadata

def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def chunk_documents(documents, chunk_size):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_size * 15 // 100,
        length_function=len,
    )

    chunked_documents = []
    for doc in documents:
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            chunked_doc = Document(
                page_content=chunk,
                metadata={**doc.metadata, 'chunk_id': i}
            )
            chunked_documents.append(chunked_doc)
    return chunked_documents

def find_correct_chunk(documents: List[Document], answer: str, chunk_size: int) -> int:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_size * 15 // 100,  # Assuming 15% overlap as before
        length_function=len
    )
    for doc in documents:
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            if answer in chunk:
                return i
    return -1  # Return -1 if no chunk contains the answer

def get_chunk_text(document: Document, chunk_id: int, chunk_size: int) -> str:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_size * 15 // 100,  # Assuming 15% overlap as before
        length_function=len
    )
    chunks = text_splitter.split_text(document.page_content)
    if chunk_id < len(chunks):
        return chunks[chunk_id]
    return "Chunk ID out of range" 

def load_data_sample(data_dir, metadata):
    documents = []
    txt_dir = os.path.join(data_dir, 'txt')

    for filename in os.listdir(txt_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(txt_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

            if filename in metadata:
                doc_metadata = metadata[filename]
                doc_metadata['original_filename'] = filename
                # Ensure all metadata fields are strings and non-empty
                doc_metadata = {k: str(v) if v is not None and v != '' else 'N/A' for k, v in doc_metadata.items()}
            else:
                #print(f"\nWarning: No metadata found for {filename}\n")
                continue

            doc = Document(page_content=content, metadata=doc_metadata)
            documents.append(doc)
    return documents

def setup_logging(log_dir: str = 'logs') -> logging.Logger:
    """
    Set up logging configuration that writes to both file and console.
    
    Args:
        log_dir: Directory where log files will be stored
    
    Returns:
        Logger object configured to write to both file and console
    """
    # Create logs directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create a unique log file name with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'rag_pipeline_{timestamp}.log')
    
    # Create logger
    logger = logging.getLogger('RAGPipeline')
    logger.setLevel(logging.DEBUG)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatters and add them to the handlers
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(message)s')
    
    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)
    
    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger

def load_data(data_dir, metadata, logger=None):
    logger = logger or setup_logging()
    logger.info("\nStarting document loading...")
    logger.info(f"Number of metadata entries: {len(metadata)}")
    if metadata:
        logger.debug("Sample metadata keys: %s", list(next(iter(metadata.values())).keys()))

    documents = []
    txt_dir = os.path.join(data_dir, 'txt')

    for filename in os.listdir(txt_dir):
        if filename.endswith('.txt'):
            logger.debug(f"\nProcessing: {filename}")
            file_path = os.path.join(txt_dir, filename)

            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()

                if filename in metadata:
                    doc_metadata = metadata[filename].copy()
                    logger.debug(f"Found metadata with {len(doc_metadata)} fields")
                    logger.debug(f"Metadata keys: {list(doc_metadata.keys())}")

                    doc_metadata['original_filename'] = filename
                    doc_metadata = {k: str(v) if v is not None and v != '' else 'N/A'
                                    for k, v in doc_metadata.items()}

                    doc = Document(page_content=content, metadata=doc_metadata)
                    documents.append(doc)
                else:
                    logger.warning(f"No metadata found for {filename}")
            except Exception as e:
                logger.error(f"Error processing file {filename}: {e}")

    logger.info(f"\nLoaded {len(documents)} documents")
    if documents:
        logger.info("\nFirst document metadata:")
        for k, v in documents[0].metadata.items():
            logger.debug(f"  {k}: {v}")

    return logger, documents

def is_empty(vectorstore):
    try:
        # Try to peek at the first item in the dataset
        vectorstore.peek(1)
        return False
    
    except IndexError:
        # If an IndexError is raised, the dataset is empty
        return True
    
    except Exception as e:
        #print(f"\nError checking if vectorstore is empty: {e}\n")
        return True  # Assume empty if there's an error

def load_configuration():
    # Set the current working directory to the project root
    src_dir = os.path.dirname(__file__)
    root_dir = os.path.abspath(os.path.join(src_dir, os.pardir))
    config_dir = os.path.join(root_dir, 'config')

    load_dotenv(dotenv_path=os.path.join(config_dir, '.env'))
    config_file = os.environ['CONFIG_FILE']
    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read(f"{config_dir}/{config_file}")
    return config

def create_bedrock_client(config):
    session = boto3.Session(
        aws_access_key_id=config['BedRock_LLM_API']['aws_access_key_id'],
        aws_secret_access_key=config['BedRock_LLM_API']['aws_secret_access_key'],
        aws_session_token=config['BedRock_LLM_API']['aws_session_token']
    )
    return session.client("bedrock-runtime", region_name="us-east-1")

def get_embedding_vectors(text, embeddings):
    response = embeddings.invoke_model(
        modelId='amazon.titan-embed-text-v2:0',
        contentType="application/json",
        accept="application/json",
        body=json.dumps({"inputText": text})
    )
    response_body = json.loads(response['body'].read())
    return response_body['embedding']

def set_model(model_name='instructor'):
    if model_name == 'instructor':
        embeddor = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl')
        return embeddor
    
    elif model_name=='mini':
        embeddor = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
        return embeddor

    elif model_name == 'titan':
        config = load_configuration()
        bedrock_client = create_bedrock_client(config)
        embeddor = BedrockEmbeddings(
            client=bedrock_client,
            region_name="us-east-1",
            model_id="amazon.titan-embed-text-v2:0")
        return embeddor

    else:
        #print(f"\nModel name not recognized. Implementing default HuggingFace Embedding model\n")
        embeddor = HuggingFaceEmbeddings()
        return model_name, embeddor

def generate_embeddings(logger, dataset_path, chunked_documents, embeddor):
    logger.info("\nGenerating embeddings...")
    logger.info(f"Processing {len(chunked_documents)} documents")

    if chunked_documents:
        logger.debug("\nFirst document metadata before embedding:")
        for k, v in chunked_documents[0].metadata.items():
            logger.debug(f"  {k}: {v}")

    # Create vectorstore
    vectorstore = DeepLake(dataset_path=dataset_path, 
                        embedding_function=embeddor,
                        read_only=False)

    # get data
    texts = [doc.page_content for doc in chunked_documents]
    metadatas = [doc.metadata for doc in chunked_documents]

    # Embedd and add to vectorstore
    vectorstore.add_texts(texts, metadatas=metadatas)
    logger.info("\nEmbeddings generated and added to vectorstore")

    # Verify storage
    logger.info("\nVerifying stored metadata...")
    try:
        sample = vectorstore.get(ids=[vectorstore.get_ids()[0]])
        logger.info("First stored document metadata:")
        for k, v in sample[0].metadata.items():
            logger.debug(f"  {k}: {v}")
    except Exception as e:
        logger.error(f"Error verifying stored metadata: {e}")

    return logger, vectorstore

def generate_embeddings_sample(dataset_path, chunked_documents, embeddor):

    # Delete all contents at the dataset_path
    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)
    
    # Create vectorstore
    vectorstore = DeepLake(dataset_path=dataset_path, 
                           embedding_function=embeddor,
                           read_only=False)

    # Get data
    texts = [doc.page_content for doc in chunked_documents]
    metadatas = [doc.metadata for doc in chunked_documents]

    # Embedd and add to vectorstore
    vectorstore.add_texts(texts, metadatas=metadatas)

    return vectorstore

def search_vector_store_sample(query, vectorstore, top_k, filter=None):
    try:
        results = vectorstore.similarity_search(query, k=top_k)
        return results
    
    except Exception as e:
        #print(f"\nError during similarity search: {e}\n")
        return []

def search_vector_store(query, logger, vectorstore, top_k, filter=None):
    try:
        results = vectorstore.similarity_search(query, k=top_k)
        logger.info("\n--- Search Results ---")
        logger.info(f"Query: {query}")
        logger.info(f"Filter: {filter}")
        logger.info(f"Number of results: {len(results)}")
        
        if results:
            logger.info("Metadata fields in search results:")
            for key in results[0].metadata.keys():
                logger.info(f"  {key}")
        logger.info("----------------------\n")
        return logger, results
    
    except Exception as e:
        logger.error(f"Error during similarity search: {e}")
        return logger, []

def test_document_retrieval(query, logger, vectorstore, top_k):
    # Perform the search
    logger, results = search_vector_store(query=query, logger=logger, vectorstore=vectorstore, top_k=top_k)
    if not results:
        print(f"\nNo results found for the query.\n")
        return query, logger, [], 0, None, None, None, [], []
    
    # Assuming the first result is the most relevant
    num_matches = len(results)

    best_match = results[0]
    best_match_content = best_match.page_content
    best_match_filename = best_match.metadata.get('original_filename', 'Unknown')
    best_match_chunkid = best_match.metadata.get('chunk_id', -1)  # Assuming chunk IDs are stored in metadata

    # Get overall data
    matches_info = []
    for match in results:
        match_content = match.page_content
        match_filename = match.metadata.get('original_filename', 'Unknown')
        match_chunkid = match.metadata.get('chunk_id', -1)
        
        # Collect relevant information for each match
        matches_info.append({
            'content': match_content,
            'filename': match_filename,
            'chunk_id': match_chunkid
        })

    all_match_filenames = list({match['filename'] for match in matches_info})
    all_match_chunkids = list({match['chunk_id'] for match in matches_info})
   
    return query, logger, results, num_matches, best_match_content, best_match_filename, best_match_chunkid, all_match_filenames, all_match_chunkids


def test_document_retrieval_sample(query, vectorstore, top_k):
    # Perform the search
    results = search_vector_store_sample(query=query, vectorstore=vectorstore, top_k=top_k)
    if not results:
        print(f"\nNo results found for the query.\n")
        return query, [], 0, None, None, None, [], []
    
    # Assuming the first result is the most relevant
    num_matches = len(results)

    best_match = results[0]
    best_match_content = best_match.page_content
    best_match_filename = best_match.metadata.get('original_filename', 'Unknown')
    best_match_chunkid = best_match.metadata.get('chunk_id', -1)  # Assuming chunk IDs are stored in metadata

    # Get overall data
    matches_info = []
    for match in results:
        match_content = match.page_content
        match_filename = match.metadata.get('original_filename', 'Unknown')
        match_chunkid = match.metadata.get('chunk_id', -1)
        
        # Collect relevant information for each match
        matches_info.append({
            'content': match_content,
            'filename': match_filename,
            'chunk_id': match_chunkid
        })

    all_match_filenames = list({match['filename'] for match in matches_info})
    all_match_chunkids = list({match['chunk_id'] for match in matches_info})
   
    return query, results, num_matches, best_match_content, best_match_filename, best_match_chunkid, all_match_filenames, all_match_chunkids


def retriever_eval_allData():
    set_seed(42)

    # Add the src directory to the Python path
    project_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    src_dir = os.path.join(project_root, 'src')
    data_dir = os.path.join(project_root, 'data')
    eval_dir = os.path.join(src_dir, 'retrieval_eval')
    
    #data_dir = os.path.join(project_root, 'data', 'marc-xl-data')
    vstore_dir = os.path.join(project_root, 'data', 'vectorstore_pauls')

     # Ensure the retrieval_eval folder exists
    eval_dir = os.path.join(src_dir, 'retrieval_eval')
    os.makedirs(eval_dir, exist_ok=True)

    # Setup
    embeddor = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl')
    logger = setup_logging()
    top_ks = [1, 10, 25, 50]
    vectorstore = DeepLake(dataset_path=vstore_dir, embedding_function=embeddor, read_only=False)
    
    # empty dataframe to hold results
    df_results = pd.DataFrame(columns=["Model",
                                    "Top_k",
                                    #"Chunk Size",
                                    "Query",
                                    #"Num doc Matches",
                                    "Expected Answer",
                                    "Expected Doc",
                                    #"Best Retrieved Doc",
                                    #"Doc Match",
                                    "All Retrieved Docs",
                                    "Expected Doc Found In All Retrieved Docs",
                                    #"Expected Chunk ID",
                                    #"Expected Chunk Text",
                                    #"Best Retrieved Chunk",
                                    #"Chunk Match",
                                    #"All Retrieved Chunks",
                                    #"Expected Chunk Found In All Retrieved Chunks",
                                    "Best Retrieved Content",
                                    "All Results"])
    # test Q/S/A
    queries_answers = [
                        ("Complete this sentence: 'The mules are not hungry. They're lively and'", "sr22a_en.txt", "gay"),
                        ("Complete this sentence: 'Take a trip on the canal if you want to have'", "sr28a_en.txt or sr13a_en.txt", "fun"),
                        ("What is the name of the female character mentioned in the song that begins 'In Scarlett town where I was born'?", "sr02b_en.txt", "Barbrae Allen"), 
                        ("According to the transcript, what is Captain Pearl R. Nye's favorite ballad?", "sr28a_en.txt", "Barbara Allen"),
                        ("Complete this phrase from the gospel train song: 'The gospel train is'", "sr26a_en.txt", "night"), 
                        ("In the song 'Barbara Allen,' where was Barbara Allen from?", "sr02b_en.txt", "Scarlett town"),
                        ("In the song 'Lord Lovele,' how long was Lord Lovele gone before returning?", "sr08a_en.txt", "A year or two or three at most"),
                        ("What instrument does Captain Nye mention loving?", "sr22a_en.txt", "old fiddled mouth organ banjo"), 
                        ("In the song about pumping out Lake Erie, what will be on the moon when they're done?", "sr27b_en.txt", "whiskers"),
                        ("Complete this line from a song: 'We land this war down by the'", "sr05a_en.txt", "river"),
                        ("What does the singer say they won't do in the song 'I Won't Marry At All'?", "sr01b_en.txt", "Marry/Mary at all"),
                        ("What does the song say will 'outshine the sun'?", "sr17b_en.txt", "We'll/not"),
                        ("In the 'Dying Cowboy' song, where was the cowboy born?", "sr20b_en.txt", "Boston")
                    ]
    
    # loop for testing
    for top_k in top_ks:
        for query, doc_filenames, answer in queries_answers:
            # in case there are multiple files that contain the answer
            possible_filenames = [filename.strip() for filename in doc_filenames.split('or')]

            # query vectorstore
            query, logger, results, num_matches, best_match_content, best_match_filename, best_match_chunkid, all_match_filenames, all_match_chunkids = test_document_retrieval(query, logger, vectorstore, top_k)
                                
            # add results to dataframe
            doc_match = best_match_filename in possible_filenames
            #chunk_match = best_match_chunkid == expected_chunk_id
            new_row = {
                                    "Model": 'instructor',
                                    "Top_k": top_k,
                                    "Chunk Size": 100,
                                    "Query": query,
                                    #"Num doc Matches": num_matches,
                                    "Expected Answer": answer,
                                    "Expected Doc": doc_filenames,
                                    #"Best Retrieved Doc": best_match_filename,
                                    #"Doc Match": doc_match,
                                    "All Retrieved Docs": all_match_filenames,
                                    "Expected Doc Found In All Retrieved Docs": any(filename in all_match_filenames for filename in possible_filenames),
                                    #"Expected Chunk ID": expected_chunk_id,
                                    #"Expected Chunk Text": expected_chunk_text,
                                    #"Best Retrieved Chunk": best_match_chunkid,
                                    #"Chunk Match": chunk_match,
                                    #"All Retrieved Chunks":  all_match_chunkids,
                                    #"Expected Chunk Found In All Retrieved Chunks": expected_chunk_id in all_match_chunkids,
                                    "Best Retrieved Content": best_match_content,
                                    "All Results": results
                                }
            df_results = pd.concat([df_results, pd.DataFrame([new_row])], ignore_index=True)

    # Create df that compares all models by top_k and chunk size
    compare_df = df_results.groupby(['Model', 'Top_k', 'Chunk Size']).agg(
        Accuracy=('Expected Doc Found In All Retrieved Docs', 'sum')
    ).reset_index()
    compare_df[f'Accuracy'] = (compare_df[f'Accuracy'] / len(queries_answers))
    compare_df.rename(columns={'Accuracy': f'Accuracy (% Docs Correct Out of {len(queries_answers)} Q/As)'}, inplace=True)
    
    # Save the detailed results and the comparable df to a CSV
    df_results_path = os.path.join(eval_dir, 'test_on_PaulsAll.csv')
    df_results.to_csv(df_results_path, index=False)

    df_compare_path = os.path.join(eval_dir, 'test_on_PaulsAll_summary.csv')
    compare_df.to_csv(df_compare_path, index=False)

def retriever_eval_sample():
    set_seed(42)
    
    # Add the src directory to the Python path
    project_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    src_dir = os.path.join(project_root, 'src')
    data_dir = os.path.join(project_root, 'data', 'marc-xl-data')
    eval_dir = os.path.join(src_dir, 'retrieval_eval')
    os.makedirs(eval_dir, exist_ok=True)

    # Setup
    model_names = ['instructor', 'mini']#, 'instructor', 'mini'] # mini, instructor, titan
    top_ks = [1, 2]
    chunk_sizes = [250, 500]
    
    # empty dataframe to hold results
    df_results = pd.DataFrame(columns=["Model",
                                    "Top_k",
                                    "Chunk Size",
                                    "Query",
                                    "Num doc Matches",
                                    "Expected Answer",
                                    "Expected Doc",
                                    "Best Retrieved Doc",
                                    "Doc Match",
                                    "All Retrieved Docs",
                                    "Expected Doc Found In All Retrieved Docs",
                                    "Expected Chunk ID",
                                    "Expected Chunk Text",
                                    "Best Retrieved Chunk",
                                    "Chunk Match",
                                    "All Retrieved Chunks",
                                    "Expected Chunk Found In All Retrieved Chunks",
                                    "Best Retrieved Content",
                                    "All Results"])
    
    # iterate through topks and chunksizes for each question
    for model_name in model_names:
        for top_k in top_ks:
            for chunk_size in chunk_sizes:
                metadata = process_metadata_sample(data_dir)
                dataset_path = os.path.join(data_dir, f'deeplake_dataset_chunk_{chunk_size}')

                # Load data and chunk
                documents = load_data_sample(data_dir, metadata)
                chunked_documents = chunk_documents(documents, chunk_size)

                # Generate embeddings for documents/chunks
                embeddor = set_model(model_name=model_name)
                print(f"\nGenerating embeddings for {len(documents)} documents, in {len(chunked_documents)} chunks of {chunk_size} using {model_name} for top_k {top_k}\n")
                vectorstore = generate_embeddings_sample(dataset_path, chunked_documents, embeddor)

                # load test questions (questions/source doc/answer)
                queries_answers = [
                        ("Complete this sentence: 'The mules are not hungry. They're lively and'", "sr22a_en.txt", "gay"),
                        ("Complete this sentence: 'Take a trip on the canal if you want to have'", "sr28a_en.txt or sr13a_en.txt", "fun"),
                        ("What is the name of the female character mentioned in the song that begins 'In Scarlett town where I was born'?", "sr02b_en.txt", "Barbrae Allen"), 
                        ("According to the transcript, what is Captain Pearl R. Nye's favorite ballad?", "sr28a_en.txt", "Barbara Allen"),
                        ("Complete this phrase from the gospel train song: 'The gospel train is'", "sr26a_en.txt", "night"), 
                        ("In the song 'Barbara Allen,' where was Barbara Allen from?", "sr02b_en.txt", "Scarlett town"),
                        ("In the song 'Lord Lovele,' how long was Lord Lovele gone before returning?", "sr08a_en.txt", "A year or two or three at most"),
                        ("What instrument does Captain Nye mention loving?", "sr22a_en.txt", "old fiddled mouth organ banjo"), 
                        ("In the song about pumping out Lake Erie, what will be on the moon when they're done?", "sr27b_en.txt", "whiskers"),
                        ("Complete this line from a song: 'We land this war down by the'", "sr05a_en.txt", "river"),
                        ("What does the singer say they won't do in the song 'I Won't Marry At All'?", "sr01b_en.txt", "Marry/Mary at all"),
                        ("What does the song say will 'outshine the sun'?", "sr17b_en.txt", "We'll/not"),
                        ("In the 'Dying Cowboy' song, where was the cowboy born?", "sr20b_en.txt", "Boston")
                    ]

                # iterate through test questions
                for query, doc_filenames, answer in queries_answers:
                    # handle instances where multiple docs contain the answer
                    possible_filenames = [filename.strip() for filename in doc_filenames.split('or')]

                    # iterate through unchunked docs and get data for comparing to retriever data
                    for doc in documents:
                        if doc.metadata['original_filename'] in possible_filenames:
                            expected_chunk_id = find_correct_chunk([doc], answer, chunk_size)
                            expected_chunk_text = get_chunk_text(doc, expected_chunk_id, chunk_size)


                            query, results, num_matches, best_match_content, best_match_filename, best_match_chunkid, all_match_filenames, all_match_chunkids = test_document_retrieval_sample(query, vectorstore, top_k)
                            doc_match = best_match_filename in possible_filenames
                            chunk_match = best_match_chunkid == expected_chunk_id
                            new_row = {
                                "Model": model_name,
                                "Top_k": top_k,
                                "Chunk Size": chunk_size,
                                "Query": query,
                                "Num doc Matches": num_matches,
                                "Expected Answer": answer,
                                "Expected Doc": doc_filenames,
                                "Best Retrieved Doc": best_match_filename,
                                "Doc Match": doc_match,
                                "All Retrieved Docs": all_match_filenames,
                                "Expected Doc Found In All Retrieved Docs": any(filename in all_match_filenames for filename in possible_filenames),
                                "Expected Chunk ID": expected_chunk_id,
                                "Expected Chunk Text": expected_chunk_text,
                                "Best Retrieved Chunk": best_match_chunkid,
                                "Chunk Match": chunk_match,
                                "All Retrieved Chunks":  all_match_chunkids,
                                "Expected Chunk Found In All Retrieved Chunks": expected_chunk_id in all_match_chunkids,
                                "Best Retrieved Content": best_match_content,
                                "All Results": results
                            }
                            df_results = pd.concat([df_results, pd.DataFrame([new_row])], ignore_index=True)

                # Handling duplicative queries caused by instances where the answer is found in multiple docs
                scores = []
                for index, row in df_results.iterrows():
                    score = 0
                    if row['Doc Match'] and row['Chunk Match']:
                        score = 3  # Highest priority for TRUE, TRUE
                    elif row['Doc Match']:
                        score = 2  # Second priority for TRUE, FALSE
                    elif row['Chunk Match']:
                        score = 1  # Third priority for FALSE, TRUE
                    scores.append(score)
                df_results['Score'] = scores

                df_results.sort_values(by='Score', ascending=False, inplace=True) # Sort by Score to prioritize higher scores
                df_results = df_results.drop_duplicates(subset=['Model', 'Top_k', 'Chunk Size', 'Query', ], keep='first') # Drop duplicate queries, keeping the first, highest score

                # reset
                del documents, chunked_documents, metadata, scores, possible_filenames, embeddor, vectorstore
                gc.collect()

    # Create df that compares all models by top_k and chunk size
    compare_df = df_results.groupby(['Model', 'Top_k', 'Chunk Size']).agg(
        Accuracy=('Expected Doc Found In All Retrieved Docs', 'sum')
    ).reset_index()
    compare_df[f'Accuracy'] = (compare_df[f'Accuracy'] / len(queries_answers))
    compare_df.rename(columns={'Accuracy': f'Accuracy (% Docs Correct Out of {len(queries_answers)} Q/As)'}, inplace=True)
    

    # Save the detailed results and the comparable df to a CSV
    df_results_path = os.path.join(eval_dir, 'test_on_sample.csv')
    compare_df_path = os.path.join(eval_dir, 'test_on_sample_summary.csv')
    df_results.to_csv(df_results_path, index=False)
    compare_df.to_csv(compare_df_path, index=False)
    print(f"\nOverall accuracy summary has been saved to '{eval_dir}'.\n")


if __name__ == "__main__":
    #retriever_eval_allData()
    retriever_eval_sample()


