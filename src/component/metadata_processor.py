import os
import re
import csv
import xml.etree.ElementTree as ET
from typing import Dict, List
from pymarc import parse_xml_to_array


def clean_metadata_value(value):
    """Clean and convert metadata values to appropriate types."""
    if not value or value == '':
        return 'N/A'

    # If the value is already a list, return it
    if isinstance(value, list):
        return value

    # If value is a string representation of a list, try to convert it
    if isinstance(value, str):
        # Remove any extra quotes and spaces
        value = value.strip()
        if value.startswith('[') and value.endswith(']'):
            try:
                # Safely evaluate string representation of list
                cleaned_value = eval(value)
                if isinstance(cleaned_value, list):
                    return cleaned_value
            except:
                pass

        # Handle 'N/A' and empty strings
        if value.lower() in ['n/a', 'none', '']:
            return 'N/A'

    # Convert non-string values to strings
    return str(value)


def clean_metadata_dict(metadata: dict) -> dict:
    """Clean all values in a metadata dictionary."""
    cleaned = {}
    try:
        for key, value in metadata.items():
            print(f"Processing metadata key: {key}: {value}")  # Debug
            cleaned[key] = clean_metadata_value(value)
    except Exception as e:
        print(f"Error in clean_metadata_dict: {e}")
        print(f"Problematic metadata: {metadata}")
        raise
    return cleaned


def extract_call_number(text: str) -> str:
    if not text:
        return None
    pattern = r'\bAFC\s*\d{4}/\d{3}\b'
    match = re.search(pattern, text)
    return match.group(0).strip() if match else None


def parse_file_list_csv(file_path: str) -> Dict[str, str]:
    """Parse file_list.csv with enhanced filename matching."""
    print(f"\nProcessing file_list.csv: {file_path}")
    if not os.path.exists(file_path):
        print(f"Warning: file_list.csv not found at {file_path}")
        return {}

    filename_to_id = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                source_url = row['source_url']
                filename = os.path.basename(source_url.strip().lower())

                # Store multiple variations for flexible matching
                filename_to_id[filename] = row['id']
                filename_to_id[filename.upper()] = row['id']
                base_name = os.path.splitext(filename)[0]
                filename_to_id[base_name] = row['id']

                # Store variations with different extensions
                for ext in ['.mp3', '.pdf', '.PDF', '.txt', '.TXT']:
                    alt_filename = base_name + ext
                    filename_to_id[alt_filename] = row['id']
                    filename_to_id[alt_filename.upper()] = row['id']
                    filename_to_id[alt_filename.lower()] = row['id']

        print(f"Processed {len(filename_to_id)} entries from file_list.csv")
        return filename_to_id
    except Exception as e:
        print(f"Error processing file_list.csv: {e}")
        return {}


def parse_search_results_csv(file_path: str) -> Dict[str, Dict]:
    """Parse search_results.csv with enhanced metadata cleaning."""
    print(f"\nProcessing search_results.csv: {file_path}")
    if not os.path.exists(file_path):
        print(f"Warning: search_results.csv not found at {file_path}")
        return {}

    id_to_metadata = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                full_call_number = row.get('item.call_number.0', '').strip()
                call_number = extract_call_number(full_call_number)

                metadata = {
                    'title': row.get('title', 'N/A'),
                    'contributors': [c for c in [row.get(f'contributor.{i}', '') for i in range(3)] if c],
                    'date': row.get('date', 'N/A'),
                    'subjects': [s for s in [row.get(f'subject.{i}', '') for i in range(5)] if s],
                    'type': row.get('type.0', 'N/A'),
                    'language': row.get('language.0', 'N/A'),
                    'locations': [l for l in [row.get(f'location.{i}', '') for i in range(3)] if l],
                    'original_format': row.get('original_format.0', 'N/A'),
                    'online_formats': [f for f in [row.get(f'online_format.{i}', '') for i in range(2)] if f],
                    'description': row.get('description', 'N/A'),
                    'rights': row.get('rights', 'N/A'),
                    'collection': row.get('collection', 'N/A'),
                    'created_published': row.get('item.created_published.0', 'N/A'),
                    'notes': [n for n in [row.get(f'item.notes.{i}', '') for i in range(2)] if n],
                    'url': row.get('url', 'N/A'),
                    'call_number': call_number or 'N/A'
                }

                # Clean metadata
                metadata = clean_metadata_dict(metadata)
                id_to_metadata[row['id']] = metadata

        print(f"Processed {len(id_to_metadata)} entries from search_results.csv")
        return id_to_metadata
    except Exception as e:
        print(f"Error processing search_results.csv: {e}")
        return {}


def process_metadata(data_dir: str) -> Dict[str, Dict]:
    """Process metadata from all sources with enhanced error handling and cleaning."""
    print("\n=== Starting Metadata Processing ===")
    print(f"Processing directory: {data_dir}")

    # Initialize paths
    loc_data_dir = os.path.join(data_dir, 'loc_dot_gov_data')
    txt_dir = os.path.join(data_dir, 'txt')
    transcripts_dir = os.path.join(data_dir, 'transcripts')
    ocr_dir = os.path.join(data_dir, 'pdf', 'txtConversion')

    all_metadata = {}
    error_log = []

    # Process each collection
    for collection_name in os.listdir(loc_data_dir):
        collection_dir = os.path.join(loc_data_dir, collection_name)
        if os.path.isdir(collection_dir):
            print(f"\nProcessing collection: {collection_name}")

            # Process CSVs
            file_list_path = os.path.join(collection_dir, 'file_list.csv')
            search_results_path = os.path.join(collection_dir, 'search_results.csv')

            filename_to_id = parse_file_list_csv(file_list_path)
            id_to_metadata = parse_search_results_csv(search_results_path)

            # Process text files from all directories
            for directory in [txt_dir, transcripts_dir, ocr_dir]:
                if not os.path.exists(directory):
                    print(f"Warning: Directory not found: {directory}")
                    continue

                for filename in os.listdir(directory):
                    if filename.endswith('.txt'):
                        try:
                            print(f"\nProcessing: {filename} from {os.path.basename(directory)}")
                            file_path = os.path.join(directory, filename)

                            # Determine file type and base filename
                            if directory == transcripts_dir:
                                file_type = 'transcript'
                                base_filename = re.sub(r'_(en|en_translation)\.txt$', '.mp3', filename)
                            elif directory == ocr_dir:
                                file_type = 'pdf_ocr'
                                base_filename = re.sub(r'\.txt$', '.pdf', filename)
                            else:
                                file_type = 'text'
                                base_filename = filename

                            # Find matching ID
                            doc_id = None
                            for name_variant in [base_filename, base_filename.lower(), base_filename.upper()]:
                                if name_variant in filename_to_id:
                                    doc_id = filename_to_id[name_variant]
                                    break

                            if doc_id and doc_id in id_to_metadata:
                                metadata = id_to_metadata[doc_id].copy()
                                metadata['original_filename'] = filename
                                metadata['file_type'] = file_type

                                # Clean metadata
                                metadata = clean_metadata_dict(metadata)
                                all_metadata[filename] = metadata
                                print(f"Successfully processed metadata for {filename}")
                            else:
                                error_msg = f"No metadata found for {filename} (base: {base_filename})"
                                print(f"Warning: {error_msg}")
                                error_log.append(error_msg)

                        except Exception as e:
                            error_msg = f"Error processing file {filename}: {str(e)}"
                            print(f"Error: {error_msg}")
                            error_log.append(error_msg)

    # Log errors
    if error_log:
        error_log_path = os.path.join(data_dir, 'metadata_processing_errors.log')
        print(f"\nWriting {len(error_log)} errors to: {error_log_path}")
        with open(error_log_path, 'w') as f:
            for error in error_log:
                f.write(f"{error}\n")

    print(f"\nProcessed metadata for {len(all_metadata)} files")
    return all_metadata