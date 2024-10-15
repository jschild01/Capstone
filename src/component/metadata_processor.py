import os
import re
import csv
import xml.etree.ElementTree as ET
from typing import Dict, List


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


def parse_search_results_csv(file_path: str) -> Dict[str, Dict]:
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

                print(f"Extracted EAD Call Number: {call_number}")  # Debug statement

    except Exception as e:
        print(f"Warning: Error parsing EAD XML file at {file_path}: {str(e)}")
    return ead_metadata


def parse_marc_xml(file_path: str) -> Dict[str, Dict]:
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
                                print(f"Extracted MARC Call Number: {call_number}")
    except Exception as e:
        print(f"Warning: Error parsing MARC XML file at {file_path}: {str(e)}")
    # Debug statement to print all keys
    print("MARC Metadata Keys:", marc_metadata.keys())
    return marc_metadata

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
    id_to_metadata = parse_search_results_csv(search_results_path)
    ead_metadata = parse_ead_xml(ead_path)
    marc_metadata = parse_marc_xml(marc_path)

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
                        print(f"Warning: EAD metadata not found for call number '{call_number}' (file: {filename})")

                    # Integrate MARC metadata
                    if call_number in marc_metadata:
                        metadata.update(marc_metadata[call_number])
                    else:
                        print(f"Warning: MARC metadata not found for call number '{call_number}' (file: {filename})")

                    # Ensure all metadata fields are strings and non-empty
                    metadata = {k: str(v) if v else 'N/A' for k, v in metadata.items()}
                    filename_to_metadata[filename] = metadata
                else:
                    print(f"Warning: No metadata found for document ID {doc_id} (filename: {filename})")
            else:
                print(f"Warning: No matching entry found in file_list.csv for {filename} (base: {base_filename})")

    print(f"Processed metadata for {len(filename_to_metadata)} files.")
    if filename_to_metadata:
        sample_file = next(iter(filename_to_metadata))
        print(f"\nSample metadata for file '{sample_file}':")
        for key, value in filename_to_metadata[sample_file].items():
            print(f"  {key}: {value}")
    else:
        print("No files processed.")

    return filename_to_metadata

if __name__ == "__main__":
    # Example usage
    data_directory = "path/to/your/data/directory"  # Update this path
    result = process_metadata(data_directory)
    print(f"\nProcessed metadata for {len(result)} files.")

    if result:
        # Optionally, print a sample of the processed metadata
        sample_file = next(iter(result))
        print(f"\nSample metadata for file '{sample_file}':")
        for key, value in result[sample_file].items():
            print(f"{key}: {value}")
