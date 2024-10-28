import os
import sys
import time
import torch
import gc, re, csv, json
import pandas as pd
import ast
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import xml.etree.ElementTree as ET
from pymarc import parse_xml_to_array

# Add the src directory to the Python path
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, src_dir)

from component.rag_retriever_deeplake import RAGRetriever
from component.rag_generator_deeplake import RAGGenerator
from component.rag_pipeline_deeplake import RAGPipeline


def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



def extract_call_number(text):
    if not text:
        return None
    pattern = r'\bAFC\s*\d{4}/\d{3}\b'
    match = re.search(pattern, text)
    return match.group(0).strip() if match else None

def parse_file_list_csv(file_path):
    filename_to_id = {}
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            source_url = row['source_url']
            filename = os.path.basename(source_url)
            filename_to_id[filename] = row['id']

    sample_entries = list(filename_to_id.items())[:3]
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

def parse_search_results_csv_new(file_path):
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

def parse_ead_xml_new(file_path):
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
        print(f"Warning: Error parsing EAD XML file at {file_path}: {str(e)}")
    return ead_metadata

def parse_marc_xml_new(file_path):
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
                                #print(f"Extracted MARC Call Number: {call_number}")
    except Exception as e:
        print(f"Warning: Error parsing MARC XML file at {file_path}: {str(e)}")
    # Debug statement to print all keys
    #print("MARC Metadata Keys:", marc_metadata.keys())
    return marc_metadata

def find_metadata_in_xml_files(call_number, xml_dir, parser_function):
    for filename in os.listdir(xml_dir):
        if filename.endswith('.xml'):
            file_path = os.path.join(xml_dir, filename)
            metadata = parser_function(file_path)
            if call_number in metadata:
                return metadata[call_number]
    return {}

def process_metadata_new(data_dir):
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

def process_metadata(data_dir):
    # file paths
    file_list_path = os.path.join(data_dir, 'file_list.csv')
    search_results_path = os.path.join(data_dir, 'search_results.csv')
    ead_path = os.path.join(data_dir, 'af012006.xml')
    marc_path = os.path.join(data_dir, 'af012006_marc.xml')

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

def chunk_documents(documents: List[Document], chunk_size: int) -> List[Document]:
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

def get_embeddings(text, qa_generator):
    # Convert text to tokens and prepare input IDs
    inputs = qa_generator.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    # Generate model output
    with torch.no_grad():
        outputs = qa_generator.model(**inputs, output_hidden_states=True)
    # Extract hidden states
    hidden_states = outputs.hidden_states
    # Use the last layer's hidden state or apply pooling (e.g., mean pooling)
    last_layer = hidden_states[-1]
    embeddings = last_layer.mean(dim=1)  # Mean pooling over the sequence dimension
    return embeddings.squeeze()

def compute_cosine_similarity(embed1, embed2):
    # Normalize the embeddings to unit length
    embed1_norm = embed1 / embed1.norm(p=2)
    embed2_norm = embed2 / embed2.norm(p=2)
    # Compute cosine similarity
    cosine_sim = torch.dot(embed1_norm, embed2_norm)
    return cosine_sim.item()

def clean_response(response):
    try:
        # Parse JSON structure from the response text
        response_json = json.loads(response)
        
        # Extract main text content from the 'content' field
        main_text = response_json.get("content", [{}])[0].get("text", "")
        
        # Remove unnecessary escape characters and line breaks
        clean_text = main_text.replace("\\n", " ").replace('\\"', '"').replace("\\'", "'").strip()
        
        # Remove extra line breaks and spaces
        clean_text = re.sub(r'\s+', ' ', clean_text)
        
        return clean_text
    except Exception:
        return response



def main():
    set_seed(42)
    data_dir = os.path.join(project_root, 'data', 'marc-xl-data')
    chunk_size = 250  # Fixed chunk size of 100
    model_names = ['claude', 'llama', 't5'] # claude in the future when functional

    # test q&a
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

    combined_results = []
    for model_name in model_names:
        # Initialize components
        metadata = process_metadata(data_dir)
        dataset_path = os.path.join(data_dir, f'deeplake_dataset_chunk_{chunk_size}')
        text_retriever = RAGRetriever(dataset_path=dataset_path, model_name='instructor')
        text_retriever.delete_dataset() # delete old/previous data
        
        # Load and prepare documents
        documents = text_retriever.load_data(data_dir, metadata)
        if len(documents) == 0:
            print("No documents loaded. Check the load_data method in RAGRetriever.")
            return

        # Chunk documents
        chunked_documents = chunk_documents(documents, chunk_size)
        num_chunks = len(chunked_documents)
        print(f"Prepared {num_chunks} chunks with size {chunk_size}")

        # Generate embeddings if the dataset is empty (should be empty)
        if text_retriever.is_empty():
            print("Generating embeddings for chunked documents...")
            text_retriever.generate_embeddings(chunked_documents)
            print("Embeddings generated and saved.")
        else:
            print("Using existing embeddings.")

        # Initialize RAG components
        qa_generator = RAGGenerator(model_name=model_name)
        rag_pipeline = RAGPipeline(text_retriever, qa_generator)

        # List to store dataframes for concatenation
        results_list = []

        # iterate through q&a; # apply rag
        for query, file, answer in queries_answers:
            if model_name == 'claude':
                retrieved_docs, most_relevant_passage, raw_response, validated_response, structured_response, final_response = rag_pipeline.run_claude(query=query, top_k=3)
            else:
                retrieved_docs, most_relevant_passage, raw_response, validated_response, structured_response, final_response = rag_pipeline.run(query=query, top_k=3)
                
            # extract just the generated response
            rag_response = rag_pipeline.extract_answer(raw_response)

            # matched documents filenames
            match_filenames = []
            for doc in retrieved_docs:
                match_filename = doc.metadata.get('original_filename', 'Unknown')
                match_filenames.append(match_filename)
            match_filenames = set(match_filenames)

            # build df
            temp_df = pd.DataFrame([{
                "Model": model_name,
                "Query": query,
                "Expected Docs": file,
                "Expected Answer": answer,
                "RAG Sole Response": rag_response,
                "RAG Raw Response": raw_response,
                "RAG Validated Response": validated_response,
                "RAG Structured Response": structured_response,
                "RAG Final Response": final_response,
                "RAG Relevant Passage": most_relevant_passage,
                "RAG Docs Retrieved": match_filenames
            }])
            results_list.append(temp_df)

        # Concatenate all result dataframes
        df_results = pd.concat(results_list, ignore_index=True)

        # Append to main comparison list
        combined_results.append(df_results)

        # save dataframe csv; ensure the retrieval_eval folder exists
        eval_dir = os.path.join(src_dir, 'generator_eval')
        os.makedirs(eval_dir, exist_ok=True)
        csv_path = os.path.join(eval_dir, f'generator_results_{model_name}.csv')
        df_results.to_csv(csv_path, index=False)

        # Cleanup
        del text_retriever, qa_generator, rag_pipeline
        gc.collect()
        torch.cuda.empty_cache()

    # save overall comparing dataframe
    final_df = pd.concat(combined_results, ignore_index=True)

    # cosine similarity to compare generated output with correct answer
    #qa_generator = RAGGenerator(model_name='llama')
    #final_df['Cosine Similarity'] = final_df.apply(lambda row: compute_cosine_similarity(
    #        get_embeddings(row['Expected Answer'], qa_generator = qa_generator),
    #        get_embeddings(row['RAG Sole Response'], qa_generator = qa_generator)
    #    ), axis=1)
    
    # clean up resposne answer
    final_df['RAG Sole Response'] = final_df['RAG Sole Response'].apply(clean_response)
    final_df['RAG Sole Response'] = final_df['RAG Sole Response'].str.replace(r'\n+', ' ', regex=True)  # Replaces multiple newlines with a single space
    final_df['RAG Sole Response'] = final_df['RAG Sole Response'].str.replace(r'\s+', ' ', regex=True)  # Replaces multiple spaces with a single space
    final_df['RAG Sole Response'] = final_df['RAG Sole Response'].str.strip() 

    # save csv
    final_df = final_df.sort_values(by="Query")  # Sort by Query column to put like-queries together
    final_csv_path = os.path.join(eval_dir, 'generator_eval_all.csv')
    final_df.to_csv(final_csv_path, index=False)

    final_df_clean = final_df[['Model', 'Query', 'Expected Answer', 'RAG Sole Response', 'RAG Docs Retrieved', 'RAG Raw Response']]
    final_csv_path_clean = os.path.join(eval_dir, 'generator_eval_summary.csv')
    final_df_clean.to_csv(final_csv_path_clean, index=False)
    print(f"\nThe final combined DataFrames has been saved to '{eval_dir}'.\n")    

if __name__ == "__main__":
    main()
    
    