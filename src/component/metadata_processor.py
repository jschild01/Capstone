import os
import re
import csv
import xml.etree.ElementTree as ET
from typing import Dict, List
from pymarc import parse_xml_to_array


def extract_call_number(text: str) -> str:
    if not text:
        return None
    pattern = r'\bAFC\s*\d{4}/\d{3}\b'
    match = re.search(pattern, text)
    return match.group(0).strip() if match else None


def parse_file_list_csv(file_path: str) -> Dict[str, str]:
    print(f"\nProcessing file_list.csv: {file_path}")
    if not os.path.exists(file_path):
        print(f"Warning: file_list.csv not found at {file_path}")
        return {}

    filename_to_id = {}
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            source_url = row['source_url']
            filename = os.path.basename(source_url)
            filename_to_id[filename] = row['id']

    print(f"Parsed {len(filename_to_id)} entries from file_list.csv")
    sample_entries = list(filename_to_id.items())[:3]
    print("Sample filename to ID mappings:")
    for filename, id_ in sample_entries:
        print(f"  {filename} -> {id_}")
    return filename_to_id


def parse_search_results_csv(file_path: str) -> Dict[str, Dict]:
    print(f"\nProcessing search_results.csv: {file_path}")
    if not os.path.exists(file_path):
        print(f"Warning: search_results.csv not found at {file_path}")
        return {}

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

    print(f"Parsed {len(id_to_metadata)} entries from search_results.csv")
    if id_to_metadata:
        first_id = next(iter(id_to_metadata))
        print("\nSample metadata structure:")
        for key, value in id_to_metadata[first_id].items():
            print(f"  {key}: {value}")
    return id_to_metadata


def parse_ead_xml(file_path: str) -> Dict[str, Dict]:
    print(f"\nProcessing EAD XML: {file_path}")
    if not os.path.exists(file_path):
        print(f"Warning: EAD XML file not found at {file_path}")
        return {}

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
                print(f"Found EAD metadata for call number: {call_number}")

    except Exception as e:
        print(f"Error parsing EAD XML file at {file_path}: {str(e)}")

    if ead_metadata:
        print("\nSample EAD metadata structure:")
        sample_call_number = next(iter(ead_metadata))
        for key, value in ead_metadata[sample_call_number].items():
            print(f"  {key}: {value}")
    return ead_metadata


def parse_marc_xml(file_path: str) -> Dict[str, Dict]:
    print(f"\nProcessing MARC XML: {file_path}")
    if not os.path.exists(file_path):
        print(f"Warning: MARC XML file not found at {file_path}")
        return {}

    marc_metadata = {}
    try:
        # Use pymarc to parse the XML file
        records = parse_xml_to_array(file_path)
        print(f"Found {len(records)} MARC records")

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
                print(f"Found MARC metadata for call number: {call_number}")

    except Exception as e:
        print(f"Error parsing MARC XML file at {file_path}: {str(e)}")

    return marc_metadata


def find_metadata_in_xml_files(call_number: str, xml_dir: str, parser_function) -> Dict:
    if not os.path.exists(xml_dir):
        print(f"Warning: XML directory not found: {xml_dir}")
        return {}

    for filename in os.listdir(xml_dir):
        if filename.endswith('.xml'):
            file_path = os.path.join(xml_dir, filename)
            metadata = parser_function(file_path)
            if call_number in metadata:
                print(f"Found metadata for {call_number} in {filename}")
                return metadata[call_number]
    print(f"No metadata found for call number: {call_number} in {xml_dir}")
    return {}

def process_metadata(data_dir: str) -> Dict[str, Dict]:
    """Process metadata from all sources and combine them."""
    print("\n=== Starting Metadata Processing ===")
    print(f"Processing directory: {data_dir}")

    # Initialize paths
    loc_data_dir = os.path.join(data_dir, 'loc_dot_gov_data')
    ead_dir = os.path.join(data_dir, 'xml', 'ead')
    marc_dir = os.path.join(data_dir, 'xml', 'marc')

    print("\nVerifying directories:")
    print(f"LOC data directory: {loc_data_dir} (exists: {os.path.exists(loc_data_dir)})")
    print(f"EAD directory: {ead_dir} (exists: {os.path.exists(ead_dir)})")
    print(f"MARC directory: {marc_dir} (exists: {os.path.exists(marc_dir)})")

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

            # Process all text files
            txt_dir = os.path.join(data_dir, 'txt')
            transcripts_dir = os.path.join(data_dir, 'transcripts')
            ocr_dir = os.path.join(data_dir, 'pdf', 'txtConversion')

            print(f"\nProcessing files from:")
            print(f"- Text files: {txt_dir}")
            print(f"- Transcripts: {transcripts_dir}")
            print(f"- OCR files: {ocr_dir}")

            for directory in [txt_dir, transcripts_dir, ocr_dir]:
                if not os.path.exists(directory):
                    print(f"Warning: Directory not found: {directory}")
                    continue

                for filename in os.listdir(directory):
                    if filename.endswith('.txt'):
                        print(f"\nProcessing: {filename} from {os.path.basename(directory)}")
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

                        print(f"Base filename: {base_filename}")

                        if base_filename in filename_to_id:
                            doc_id = filename_to_id[base_filename]
                            print(f"Found ID: {doc_id}")

                            if doc_id in id_to_metadata:
                                metadata = id_to_metadata[doc_id].copy()
                                print(f"Found base metadata with fields: {list(metadata.keys())}")

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
                                        print(f"Added EAD metadata fields: {list(ead_metadata.keys())}")

                                    # Add MARC metadata
                                    marc_metadata = find_metadata_in_xml_files(call_number, marc_dir, parse_marc_xml)
                                    if marc_metadata:
                                        metadata.update(marc_metadata)
                                        print(f"Added MARC metadata fields: {list(marc_metadata.keys())}")

                                # Clean metadata
                                metadata = {k: str(v) if v is not None else 'N/A' for k, v in metadata.items()}
                                all_metadata[filename] = metadata
                            else:
                                error_msg = f"No metadata found in search_results.csv for ID {doc_id} (file: {filename})"
                                print(f"Warning: {error_msg}")
                                error_log.append(error_msg)
                        else:
                            error_msg = f"No entry found in file_list.csv for {base_filename} (original: {filename})"
                            print(f"Warning: {error_msg}")
                            error_log.append(error_msg)

    # Verify metadata richness
    if all_metadata:
        sample_file = next(iter(all_metadata))
        sample_metadata = all_metadata[sample_file]
        if len(sample_metadata.keys()) <= 3:
            print("\nWARNING: Metadata appears to be stripped - only basic fields present!")
            print("Present fields:", list(sample_metadata.keys()))
        else:
            print("\nMetadata preserved successfully")
            print(f"Number of metadata fields: {len(sample_metadata.keys())}")
            print("Fields present:", list(sample_metadata.keys()))

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


if __name__ == "__main__":
    data_directory = "/path/to/your/data"  # Update this path
    result = process_metadata(data_directory)
    print(f"\nProcessed metadata for {len(result)} files.")

    if result:
        sample_file = next(iter(result))
        print(f"\nSample metadata for file '{sample_file}':")
        for key, value in result[sample_file].items():
            print(f"{key}: {value}")
    else:
        print("No metadata processed.")

    print("Script execution completed.")