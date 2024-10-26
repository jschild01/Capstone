import os
import shutil
import re
from typing import Dict, List, Any
from langchain_community.vectorstores import DeepLake
from langchain.schema import Document
from dotenv import load_dotenv
import boto3
from botocore.exceptions import ClientError
import json
from configparser import ConfigParser, ExtendedInterpolation
from langchain_aws import BedrockEmbeddings
from component.logging_config import setup_logging

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings
except ImportError:
    from langchain.embeddings import HuggingFaceEmbeddings


class RAGRetriever:
    def __init__(self, dataset_path: str = './my_deeplake', model_name: str = 'instructor', logger=None):
        self.logger = logger or setup_logging()

        if model_name == 'instructor':
            self.embeddings = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl')
        elif model_name == 'mini':
            self.embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
        elif model_name == 'titan':
            self.config = self.load_configuration()
            self.bedrock_client = self.create_bedrock_client(self.config)
            self.embeddings = BedrockEmbeddings(
                client=self.bedrock_client,
                region_name="us-east-1",
                model_id="amazon.titan-embed-text-v2:0")
        else:
            self.logger.info('Model name not recognized. Implementing default HuggingFace Embedding model')
            self.embeddings = HuggingFaceEmbeddings()

        self.dataset_path = dataset_path
        self.vectorstore = self.load_vectorstore()
        self.documents = []

    def load_vectorstore(self):
        if os.path.exists(self.dataset_path):
            self.logger.info("Loading existing DeepLake dataset...")
            try:
                vectorstore = DeepLake(dataset_path=self.dataset_path, embedding_function=self.embeddings,
                                       read_only=False)
                self.print_dataset_info(vectorstore)
                return vectorstore
            except Exception as e:
                self.logger.error(f"Error loading existing dataset: {e}")
                self.logger.info("Creating a new dataset...")
                return self.create_new_vectorstore()
        else:
            return self.create_new_vectorstore()

    def create_new_vectorstore(self):
        self.logger.info("Creating new DeepLake dataset...")
        return DeepLake(dataset_path=self.dataset_path, embedding_function=self.embeddings)

    def print_dataset_info(self, vectorstore):
        self.logger.info("\n--- Dataset Information ---")
        try:
            self.logger.info(f"Number of elements: {len(vectorstore)}")
            if len(vectorstore) > 0:
                self.logger.info("Available metadata fields:")
                sample = vectorstore.get(ids=[vectorstore.get_ids()[0]])
                for key in sample[0].metadata.keys():
                    self.logger.info(f"  {key}")
                self.logger.info("\nSample metadata values:")
                for key, value in sample[0].metadata.items():
                    self.logger.info(f"  {key}: {value}")
        except Exception as e:
            self.logger.error(f"Error printing dataset info: {e}")
        self.logger.info("----------------------------\n")

    def load_data(self, data_dir: str, metadata: Dict[str, Dict]) -> List[Document]:
        self.logger.info("\nStarting document loading...")
        self.logger.info(f"Number of metadata entries: {len(metadata)}")
        if metadata:
            self.logger.debug("Sample metadata keys: %s", list(next(iter(metadata.values())).keys()))

        self.documents = []
        txt_dir = os.path.join(data_dir, 'txt')

        for filename in os.listdir(txt_dir):
            if filename.endswith('.txt'):
                self.logger.debug(f"\nProcessing: {filename}")
                file_path = os.path.join(txt_dir, filename)

                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()

                    if filename in metadata:
                        doc_metadata = metadata[filename].copy()
                        self.logger.debug(f"Found metadata with {len(doc_metadata)} fields")
                        self.logger.debug(f"Metadata keys: {list(doc_metadata.keys())}")

                        doc_metadata['original_filename'] = filename
                        doc_metadata = {k: str(v) if v is not None and v != '' else 'N/A'
                                        for k, v in doc_metadata.items()}

                        doc = Document(page_content=content, metadata=doc_metadata)
                        self.documents.append(doc)
                    else:
                        self.logger.warning(f"No metadata found for {filename}")
                except Exception as e:
                    self.logger.error(f"Error processing file {filename}: {e}")

        self.logger.info(f"\nLoaded {len(self.documents)} documents")
        if self.documents:
            self.logger.info("\nFirst document metadata:")
            for k, v in self.documents[0].metadata.items():
                self.logger.debug(f"  {k}: {v}")

        return self.documents

    def generate_embeddings(self, documents: List[Document]):
        self.logger.info("\nGenerating embeddings...")
        self.logger.info(f"Processing {len(documents)} documents")

        if documents:
            self.logger.debug("\nFirst document metadata before embedding:")
            for k, v in documents[0].metadata.items():
                self.logger.debug(f"  {k}: {v}")

        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        self.vectorstore.add_texts(texts, metadatas=metadatas)
        self.logger.info("\nEmbeddings generated and added to vectorstore")

        # Verify storage
        self.logger.info("\nVerifying stored metadata...")
        try:
            sample = self.vectorstore.get(ids=[self.vectorstore.get_ids()[0]])
            self.logger.info("First stored document metadata:")
            for k, v in sample[0].metadata.items():
                self.logger.debug(f"  {k}: {v}")
        except Exception as e:
            self.logger.error(f"Error verifying stored metadata: {e}")

    def search_vector_store(self, query: str, filter: Dict = None, top_k: int = 3):
        try:
            results = self.vectorstore.similarity_search(query, filter=filter, k=top_k)
            self.logger.info("\n--- Search Results ---")
            self.logger.info(f"Query: {query}")
            self.logger.info(f"Filter: {filter}")
            self.logger.info(f"Number of results: {len(results)}")
            if results:
                self.logger.info("Metadata fields in search results:")
                for key in results[0].metadata.keys():
                    self.logger.info(f"  {key}")
            self.logger.info("----------------------\n")
            return results
        except Exception as e:
            self.logger.error(f"Error during similarity search: {e}")
            return []

    def query_metadata(self, filter: Dict = None, limit: int = 10):
        if not self.documents:
            self.logger.info("No documents loaded. Please load documents first.")
            return []

        filtered_docs = self.documents
        if filter and 'date' in filter:
            year = filter['date']['$regex'].strip('.*')
            filtered_docs = [doc for doc in self.documents if year in doc.metadata.get('date', '')]

        self.logger.info("\n--- Metadata Query Results ---")
        self.logger.info(f"Filter: {filter}")
        self.logger.info(f"Number of matching documents: {len(filtered_docs)}")
        if filtered_docs:
            self.logger.info("Metadata fields in query results:")
            for key in filtered_docs[0].metadata.keys():
                self.logger.info(f"  {key}")
        self.logger.info("-------------------------------\n")

        return filtered_docs[:limit]

    def print_sample_documents(self, num_samples=5):
        samples = self.documents[:num_samples] if self.documents else []
        self.logger.info(f"\nPrinting {len(samples)} sample documents:")
        for i, doc in enumerate(samples, 1):
            self.logger.info(f"\nDocument {i}:")
            self.logger.info(f"Content: {doc.page_content[:100]}...")
            self.logger.info("Metadata:")
            for key, value in doc.metadata.items():
                self.logger.info(f"  {key}: {value}")
            self.logger.info("-" * 50)

    def is_empty(self):
        try:
            # Try to peek at the first item in the dataset
            self.vectorstore.peek(1)
            return False
        except IndexError:
            # If an IndexError is raised, the dataset is empty
            return True
        except Exception as e:
            self.logger.error(f"Error checking if vectorstore is empty: {e}")
            return True  # Assume empty if there's an error

    def delete_dataset(self):
        if os.path.exists(self.dataset_path):
            self.logger.info(f"Deleting existing DeepLake dataset at {self.dataset_path}")
            shutil.rmtree(self.dataset_path)
            self.logger.info("Dataset deleted successfully.")
            self.vectorstore = self.create_new_vectorstore()
        else:
            self.logger.info("No existing dataset found.")

    def load_configuration(self):
        # Set the current working directory to the project root
        components_dir = os.path.dirname(__file__)
        src_dir = os.path.abspath(os.path.join(components_dir, os.pardir))
        root_dir = os.path.abspath(os.path.join(src_dir, os.pardir))
        config_dir = os.path.join(root_dir, 'config')

        load_dotenv(dotenv_path=os.path.join(config_dir, '.env'))
        config_file = os.environ['CONFIG_FILE']
        config = ConfigParser(interpolation=ExtendedInterpolation())
        config.read(f"{config_dir}/{config_file}")
        return config

    def create_bedrock_client(self, config):
        session = boto3.Session(
            aws_access_key_id=config['BedRock_LLM_API']['aws_access_key_id'],
            aws_secret_access_key=config['BedRock_LLM_API']['aws_secret_access_key'],
            aws_session_token=config['BedRock_LLM_API']['aws_session_token']
        )
        return session.client("bedrock-runtime", region_name="us-east-1")