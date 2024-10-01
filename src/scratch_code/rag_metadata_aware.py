import os
import sys
import time
import torch
import gc
import csv
import xml.etree.ElementTree as ET
import re
from typing import Dict, List, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.schema import Document
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Metadata processing functions
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

    id_to_metadata = parse_search_results_csv(search_results_path)
    print(f"Parsed {len(id_to_metadata)} entries from search_results.csv")

    ead_metadata = parse_ead_xml(ead_path)
    print(f"Parsed EAD metadata: {ead_metadata}")

    marc_metadata = parse_marc_xml(marc_path)
    print(f"Parsed MARC metadata: {marc_metadata}")

    filename_to_metadata = {}
    for filename, doc_id in filename_to_id.items():
        if doc_id in id_to_metadata:
            metadata = id_to_metadata[doc_id]
            metadata.update(ead_metadata)
            metadata.update(marc_metadata)
            filename_to_metadata[filename] = metadata
        else:
            print(f"Warning: No metadata found for document ID {doc_id} (filename: {filename})")

    print(f"Combined metadata for {len(filename_to_metadata)} files")

    return filename_to_metadata


class RAGRetriever:
    def __init__(self, model_name: str = 'hkunlp/instructor-xl', chunk_size: int = 1000, chunk_overlap: int = 200):
        self.embeddings = HuggingFaceInstructEmbeddings(model_name=model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.vectorstore = None

    def load_data(self, data_dir: str, metadata: Dict[str, Dict]) -> List[Document]:
        documents = []
        txt_dir = os.path.join(data_dir, 'txt')
        for filename in os.listdir(txt_dir):
            if filename.endswith('.txt'):
                file_path = os.path.join(txt_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()

                # Check if the file is a transcript
                is_transcript = re.search(r'_(en|nn_en_translation)\.txt$', filename)
                if is_transcript:
                    # Convert transcript filename to corresponding .mp3 filename
                    mp3_filename = re.sub(r'_(en|nn_en_translation)\.txt$', '.mp3', filename)
                    if mp3_filename in metadata:
                        doc_metadata = metadata[mp3_filename]
                        doc_metadata['transcript_file'] = filename  # Add transcript filename to metadata
                    else:
                        print(f"Warning: No metadata found for {mp3_filename} (transcript: {filename})")
                        continue
                elif filename in metadata:
                    doc_metadata = metadata[filename]
                else:
                    print(f"Warning: No metadata found for {filename}")
                    continue

                doc = Document(page_content=content, metadata=doc_metadata)
                documents.append(doc)

        return documents

    def generate_embeddings(self, documents: List[Document]):
        texts = self.text_splitter.split_documents(documents)
        self.vectorstore = FAISS.from_documents(texts, self.embeddings)

    def search_vector_store(self, query: str, top_k: int = 3) -> List[Tuple[Document, float]]:
        if not self.vectorstore:
            raise ValueError("Vector store has not been initialized. Call generate_embeddings() first.")

        results = self.vectorstore.similarity_search_with_score(query, k=top_k)
        return results


class RAGGenerator:
    def __init__(self, model_name='google/flan-t5-base'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def generate_response(self, query: str, context: str, max_length: int = 150) -> str:
        input_text = f"Question: {query}\nContext: {context}\nAnswer:"
        input_ids = self.tokenizer(input_text, return_tensors='pt', truncation=True, max_length=512).input_ids

        outputs = self.model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.strip()


class RAGPipeline:
    def __init__(self, text_retriever, qa_generator):
        self.text_retriever = text_retriever
        self.qa_generator = qa_generator

    def run(self, query: str, top_k: int = 3) -> Tuple[List[Tuple[Document, float]], str, str]:
        # Retrieve relevant passages
        retrieved_docs = self.text_retriever.search_vector_store(query, top_k=top_k)

        # Find most relevant passage
        most_relevant_doc, _ = min(retrieved_docs, key=lambda x: x[1])

        # Generate response
        response = self.qa_generator.generate_response(query, most_relevant_doc.page_content)

        return retrieved_docs, most_relevant_doc.page_content, response


def test_rag_system(data_dir: str, query: str):
    try:
        print("Starting metadata processing...")
        metadata = process_metadata(data_dir)
        print("Metadata processing completed.")

        print("Initializing RAG Retriever...")
        text_retriever = RAGRetriever(model_name='hkunlp/instructor-xl', chunk_size=1000, chunk_overlap=200)
        print("Loading documents...")
        documents = text_retriever.load_data(data_dir, metadata)
        if not documents:
            print("No documents were loaded. Please check your data directory and file names.")
            return
        print(f"Loaded {len(documents)} documents.")

        print("Generating embeddings...")
        text_retriever.generate_embeddings(documents)
        print("Embeddings generated.")

        print("Initializing RAG Generator...")
        qa_generator = RAGGenerator(model_name='google/flan-t5-base')
        rag_pipeline = RAGPipeline(text_retriever, qa_generator)

        print(f"Processing query: {query}")
        start_time = time.time()
        retrieved_docs, most_relevant_passage, response = rag_pipeline.run(query, top_k=3)
        end_time = time.time()
        total_time = end_time - start_time

        print(f"\nRetrieved Passages (3x):")
        for doc, score in retrieved_docs:
            print(f"Passage: {doc.page_content[:100]}...")
            print(f"Metadata:")
            for key, value in doc.metadata.items():
                print(f"  {key}: {value}")
            print(f"Score: {score}\n")

        print(f"\nMost Relevant Passage Used for Response:")
        print(most_relevant_passage)

        most_relevant_metadata = retrieved_docs[0][0].metadata
        print(f"\nMetadata for Most Relevant Passage:")
        for key, value in most_relevant_metadata.items():
            print(f"  {key}: {value}")

        print(f"\nRAG Response:")
        print(response)

        print(f"\nTotal processing time: {total_time:.2f} seconds")
        print("-" * 50)

        del text_retriever, qa_generator, rag_pipeline
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        print("Please check your data files and directory structure.")


if __name__ == "__main__":
    set_seed(42)

    base_path = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_path, 'data')

    query = input("\nEnter your question: ")
    test_rag_system(data_dir, query)