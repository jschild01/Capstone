import os
import sys
import time
import torch
import gc
import csv
import xml.etree.ElementTree as ET
import shutil
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.vectorstores import DeepLake
from typing import Dict, List, Any, Tuple
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from langchain.schema import Document

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings
except ImportError:
    from langchain.embeddings import HuggingFaceEmbeddings


# rag utils file -------------------------------------------------------------------
def generate_prompt(query: str, context: str, metadata: dict) -> str: # called in RAGpipeline
    metadata_str = "\n".join([f"{k}: {v}" for k, v in metadata.items()])
    return f"""Question: {query}
Context: {context}
Metadata: {metadata_str}

Instructions: 
1. Answer the question using ONLY the information provided in the Context and Metadata above.
2. Do NOT include any information that is not explicitly stated in the Context or Metadata.
3. If the information provided is not sufficient to answer the question fully, state this clearly.
4. Begin your answer with a direct response to the question asked.
5. Include relevant details from the Context and Metadata to support your answer.
6. Pay special attention to the recording date, contributors, and locations provided in the metadata.

Answer:"""

def validate_response(response: str, metadata: dict) -> str: # called in RAGpipeline
    validated_response = response
    corrections = []

    # Check for date consistency
    if metadata['date'] not in response:
        corrections.append(f"The correct recording date is {metadata['date']}.")

    # Check for contributor consistency
    contributors = ", ".join(metadata['contributors'])
    if not any(contrib.lower() in response.lower() for contrib in metadata['contributors']):
        corrections.append(f"The contributors to this recording are {contributors}.")

    # Check for location consistency (if available in metadata)
    if 'locations' in metadata and metadata['locations']:
        location = metadata['locations'][0]
        if location.lower() not in response.lower():
            corrections.append(f"The recording location is {location}.")

    # Check for title consistency
    if metadata['title'].lower() not in response.lower():
        corrections.append(f"The correct title of the recording is '{metadata['title']}'.")

    if corrections:
        validated_response += "\n\nCorrections:"
        for correction in corrections:
            validated_response += f"\n• {correction}"

    return validated_response

def structure_response(response: str) -> str: # called in RAGpipeline
    parts = response.split("\n\nCorrections:")
    main_response = parts[0]
    corrections = parts[1] if len(parts) > 1 else ""

    sentences = main_response.split('. ')
    structured_response = "RAG Response:\n\n"
    for sentence in sentences:
        structured_response += f"• {sentence.strip()}.\n"

    if corrections:
        structured_response += f"\nCorrections:{corrections}"

    return structured_response

def integrate_metadata(response: str, metadata: dict) -> str: # called in RAGpipeline
    relevant_fields = ['title', 'date', 'contributors', 'subjects', 'type', 'url']
    metadata_section = "Relevant Metadata:\n"

    for field in relevant_fields:
        if field in metadata and metadata[field]:
            value = metadata[field] if isinstance(metadata[field], str) else ', '.join(metadata[field])
            metadata_section += f"• {field.capitalize()}: {value}\n"

    return f"{metadata_section}\n{response}"

 
# metadata processor file -------------------------------------------------------------------
def parse_file_list_csv(file_path: str) -> Dict[str, str]: # called in process_metadata below
    filename_to_id = {}
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            source_url = row['source_url']
            filename = os.path.basename(source_url)
            filename_to_id[filename] = row['id']
    return filename_to_id

def parse_search_results_csv(file_path: str) -> Dict[str, Dict]: # called in process_metadata below
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

def parse_ead_xml(file_path: str) -> Dict: # called in process_metadata below
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

def parse_marc_xml(file_path: str) -> Dict: # called in process_metadata below
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

def process_metadata(data_dir: str) -> Dict[str, Dict]: # called in the main function
    loc_dot_gov_data = r'C:\Users\schil\OneDrive\Desktop\School\Capstone\LOC\loc_dot_gov_data2'

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

    id_to_metadata = parse_search_results_csv(search_results_path)
    print(f"Parsed {len(id_to_metadata)} entries from search_results.csv")

    ead_metadata = parse_ead_xml(ead_path)
    print(f"Parsed EAD metadata: {ead_metadata}")

    marc_metadata = parse_marc_xml(marc_path)
    print(f"Parsed MARC metadata: {marc_metadata}")

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

    print(f"Combined metadata for {len(filename_to_metadata)} files")

    return filename_to_metadata


# retriever deeplake file -------------------------------------------------------------------
class RAGRetriever:
    def __init__(self, dataset_path: str = './my_deeplake', model_name: str = 'instructor'):
        if model_name == 'instructor':
            self.embeddings = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl')
        elif model_name=='mini':
            self.embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
        else:
            print('Model name not recognized. Implementing default HuggingFace Embedding model')
            self.embeddings = HuggingFaceEmbeddings()

        self.dataset_path = dataset_path
        self.vectorstore = self.load_vectorstore()
        self.documents = []  # Store loaded documents

    def load_vectorstore(self): # called in RAGRetriever's init
        if os.path.exists(self.dataset_path):
            print("Loading existing DeepLake dataset...")
            try:
                vectorstore = DeepLake(dataset_path=self.dataset_path, embedding_function=self.embeddings,
                                       read_only=False)
                self.print_dataset_info(vectorstore)
                return vectorstore
            except Exception as e:
                print(f"Error loading existing dataset: {e}")
                print("Creating a new dataset...")
                return self.create_new_vectorstore()
        else:
            return self.create_new_vectorstore()

    def create_new_vectorstore(self): # called in RAGRetriever's load_vectorstore and delete_dataset
        print("Creating new DeepLake dataset...")
        return DeepLake(dataset_path=self.dataset_path, embedding_function=self.embeddings)

    def print_dataset_info(self, vectorstore): # called in RAGRetriever's load_vectorstore and generate_embeddings
        print("\n--- Dataset Information ---")
        try:
            print(f"Number of elements: {len(vectorstore)}")
            if len(vectorstore) > 0:
                print("Available metadata fields:")
                sample = vectorstore.get(ids=[vectorstore.get_ids()[0]])
                for key in sample[0].metadata.keys():
                    print(f"  {key}")
                print("\nSample metadata values:")
                for key, value in sample[0].metadata.items():
                    print(f"  {key}: {value}")
        except Exception as e:
            print(f"Error printing dataset info: {e}")
        print("----------------------------\n")

    def load_data(self, data_dir: str, metadata: Dict[str, Dict]) -> List[Document]: # called in the main function
        self.documents = []  # Reset documents
        txt_dir = os.path.join(data_dir, 'txt')
        for filename in os.listdir(txt_dir):
            if filename.endswith('.txt'):
                file_path = os.path.join(txt_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()

                base_filename = re.sub(r'_(en|nn_en_translation)\.txt$', '.mp3', filename)

                if base_filename in metadata:
                    doc_metadata = metadata[base_filename]
                    doc_metadata['original_filename'] = filename
                    # Ensure all metadata fields are strings and non-empty
                    doc_metadata = {k: str(v) if v is not None and v != '' else 'N/A' for k, v in doc_metadata.items()}
                else:
                    print(f"Warning: No metadata found for {filename} (base: {base_filename})")
                    continue

                doc = Document(page_content=content, metadata=doc_metadata)
                self.documents.append(doc)

        print(f"Loaded {len(self.documents)} documents with metadata")
        print("\n--- Sample Document Metadata ---")
        if self.documents:
            sample_doc = self.documents[0]
            print(f"Metadata fields for document: {sample_doc.metadata.get('original_filename', 'Unknown')}")
            for key, value in sample_doc.metadata.items():
                print(f"  {key}: {value}")
        print("--------------------------------\n")
        return self.documents

    def generate_embeddings(self, documents: List[Document]): # called in the main function
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        self.vectorstore.add_texts(texts, metadatas=metadatas)
        print(f"Added {len(documents)} documents to DeepLake dataset")
        self.print_dataset_info(self.vectorstore)

    def search_vector_store(self, query: str, filter: Dict = None, top_k: int = 3): # called in the RAGpipeline
        try:
            results = self.vectorstore.similarity_search(query, filter=filter, k=top_k)
            print("\n--- Search Results ---")
            print(f"Query: {query}")
            print(f"Filter: {filter}")
            print(f"Number of results: {len(results)}")
            if results:
                print("Metadata fields in search results:")
                for key in results[0].metadata.keys():
                    print(f"  {key}")
            print("----------------------\n")
            return results
        except Exception as e:
            print(f"Error during similarity search: {e}")
            return []

    def query_metadata(self, filter: Dict = None, limit: int = 10): # not being called anywhere
        if not self.documents:
            print("No documents loaded. Please load documents first.")
            return []

        filtered_docs = self.documents
        if filter and 'date' in filter:
            year = filter['date']['$regex'].strip('.*')
            filtered_docs = [doc for doc in self.documents if year in doc.metadata.get('date', '')]

        print("\n--- Metadata Query Results ---")
        print(f"Filter: {filter}")
        print(f"Number of matching documents: {len(filtered_docs)}")
        if filtered_docs:
            print("Metadata fields in query results:")
            for key in filtered_docs[0].metadata.keys():
                print(f"  {key}")
        print("-------------------------------\n")

        return filtered_docs[:limit]

    def print_sample_documents(self, num_samples=5): # not being called anywhere
        samples = self.documents[:num_samples] if self.documents else []
        print(f"\nPrinting {len(samples)} sample documents:")
        for i, doc in enumerate(samples, 1):
            print(f"\nDocument {i}:")
            print(f"Content: {doc.page_content[:100]}...")
            print("Metadata:")
            for key, value in doc.metadata.items():
                print(f"  {key}: {value}")
            print("-" * 50)

    def is_empty(self): # called in the main function
        try:
            # Try to peek at the first item in the dataset
            self.vectorstore.peek(1)
            return False
        except IndexError:
            # If an IndexError is raised, the dataset is empty
            return True
        except Exception as e:
            print(f"Error checking if vectorstore is empty: {e}")
            return True  # Assume empty if there's an error

    def delete_dataset(self): # called in the main function
        if os.path.exists(self.dataset_path):
            print(f"Deleting existing DeepLake dataset at {self.dataset_path}")
            shutil.rmtree(self.dataset_path)
            print("Dataset deleted successfully.")
            self.vectorstore = self.create_new_vectorstore()
        else:
            print("No existing dataset found.")

# generator deeplake file -------------------------------------------------------------------
class RAGGenerator:
    def __init__(self, model_name):
        if model_name == 'llama3':
            model_name = 'meta-llama/Llama-3.2-3B-Instruct'
            hf_token = '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'
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_token)
        elif model_name == 't5':
            model_name = 'google/flan-t5-xl'
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        else:
            raise ValueError("Invalid model name provided. Input either 'llama' or 't5' as model name.")

    def generate_response(self, prompt: str, max_length: int = 300) -> str: # called in the RAGpipeline
        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024) # 512, 1024, 2048

        # Set attention mask
        attention_mask = inputs['attention_mask']

        # Set pad token ID
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        outputs = self.model.generate(
            inputs['input_ids'],
            attention_mask=attention_mask,
            max_new_tokens=max_length,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.8,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.strip()

# pipeline deeplake file -------------------------------------------------------------------
class RAGPipeline:
    def __init__(self, text_retriever: RAGRetriever, qa_generator: RAGGenerator):
        self.text_retriever = text_retriever
        self.qa_generator = qa_generator

    def run(self, query: str, top_k: int = 3) -> Tuple[List[Document], str, str]: # called in the main function
        retrieved_docs = self.text_retriever.search_vector_store(query, top_k=top_k)

        if not retrieved_docs:
            return [], "", "No relevant documents found."

        combined_context = " ".join([doc.page_content for doc in retrieved_docs])
        most_relevant_doc = retrieved_docs[0]

        prompt = generate_prompt(query, combined_context, most_relevant_doc.metadata)
        raw_response = self.qa_generator.generate_response(prompt)
        validated_response = validate_response(raw_response, most_relevant_doc.metadata)
        structured_response = structure_response(validated_response)
        final_response = integrate_metadata(structured_response, most_relevant_doc.metadata)

        return retrieved_docs, most_relevant_doc.page_content, final_response

# main deeplake file -------------------------------------------------------------------
def set_seed(seed=42): # called in main
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def sentence_splitter(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    sentences = re.split(r'(?<=[.!?]) +', text)  # Split based on sentence-ending punctuation
    chunks = []
    current_chunk = []

    for sentence in sentences:
        current_chunk.append(sentence)
        if len(' '.join(current_chunk)) >= chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = current_chunk[-(chunk_overlap//len(current_chunk)):]  # Apply overlap
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def chunk_documents(chunk_method: str, documents: List[Document], chunk_size: int) -> List[Document]:
    chunked_documents = []

    if chunk_method == 'sentence':
        chunk_overlap = chunk_size * 15 // 100  # 15% overlap for sentence chunking
        
        for doc in documents:
            chunk_size = chunk_size // 30
            chunks = sentence_splitter(doc.page_content, chunk_size, chunk_overlap)
            for i, chunk in enumerate(chunks):
                chunked_doc = Document(
                    page_content=chunk,
                    metadata={**doc.metadata, 'chunk_id': i}
                )
                chunked_documents.append(chunked_doc)

    else:
        char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_size * 15 // 100,  # 15% overlap for character chunking
            length_function=len,
        )

        for doc in documents:
            chunks = char_splitter.split_text(doc.page_content)
            for i, chunk in enumerate(chunks):
                chunked_doc = Document(
                    page_content=chunk,
                    metadata={**doc.metadata, 'chunk_id': i}
                )
                chunked_documents.append(chunked_doc)
                
    return chunked_documents

def chunk_documents_old(documents: List[Document], chunk_size: int) -> List[Document]: # called in main
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        #chunk_overlap=chunk_size // 10, # 10%
        chunk_overlap = chunk_size * 15 // 100, # 15%
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

def main():
    
    # Add the src directory to the Python path
    project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) # capstone directory
    src_dir = os.path.join(project_root, 'src')
    sys.path.insert(0, src_dir)

    # set seed for reproducibility
    set_seed(42)

    # set directory where data is housed
    data_dir = os.path.join(project_root, 'data', 'marc-xl-data')

    # Ask if user wants to delete existing data (y to start fresh)
    delete_existing = input("Do you want to delete the existing dataset? (y/n): ").lower() == 'y'

    # Initialize RAG components; NOT SURE what the deeplake_dataset is/does/is from
    chunk_size = 100 # in characters
    dataset_path = os.path.join(data_dir, f'deeplake_dataset_chunk_{chunk_size}')
    text_retriever = RAGRetriever(dataset_path=dataset_path, model_name='instructor') # use 'instructor' (default) or 'mini'
    qa_generator = RAGGenerator(model_name='llama3') # use 'llama3' (default) or 't5'
    rag_pipeline = RAGPipeline(text_retriever, qa_generator)
    
    # process metadata+whisper data
    metadata = process_metadata(data_dir) # get all metadata with its corresponding filename, into a dict
    documents = text_retriever.load_data(data_dir, metadata) # get txt whisper data matched to its related metadata

    # chunk our documents into segments
    chunked_documents = chunk_documents(chunk_method='character', documents=documents, chunk_size=chunk_size)
    #chunked_documents = chunk_documents_old(documents=documents, chunk_size=chunk_size)
    num_chunks = len(chunked_documents)
    print(f"Prepared {num_chunks} in sizes of {chunk_size} per chunk")

    # generate embeddings, depending on delete_existing, and add to/generate vectorstore (deeplake)
    if delete_existing:
        text_retriever.delete_dataset()
        print("Existing dataset deleted. Generating new embeddings for chunked documents...")
        text_retriever.generate_embeddings(chunked_documents)
        print("Embeddings generated and saved.")
    elif text_retriever.is_empty():
        print("Dataset is empty. Generating embeddings for chunked documents...")
        text_retriever.generate_embeddings(chunked_documents)
        print("Embeddings generated and saved.")
    else:
        print("Using existing embeddings.")

    # Get user question
    while True:
        query = input("Enter your question (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break

        start_time = time.time()

        # feed query to pipeline to search vectorstore for top k (3x) documents
        # generate prompt and get response
        # add fields from original metadata for most relevant doc to the response
        retrieved_docs, most_relevant_passage, response = rag_pipeline.run(query)

        end_time = time.time()

        print("\n--- Results ---")
        print(f"Processing time: {end_time - start_time:.2f} seconds")
        print(f"Number of retrieved documents: {len(retrieved_docs)}")
        print("Response:")
        print(response)
        print("-------------------\n")

    # Cleanup
    del text_retriever, qa_generator, rag_pipeline
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

