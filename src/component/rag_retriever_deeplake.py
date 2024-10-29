from typing import Dict, List, Any
from langchain_community.vectorstores import DeepLake
from langchain.schema import Document
from dotenv import load_dotenv
import boto3
from botocore.exceptions import ClientError
import json
import os
import shutil
import re
from configparser import ConfigParser, ExtendedInterpolation
from langchain_aws import BedrockEmbeddings
from component.logging_config import setup_logging

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings
except ImportError:
    from langchain.embeddings import HuggingFaceEmbeddings


class RAGRetriever:
    def __init__(self, dataset_path: str = './my_deeplake', model_name: str = 'instructor', logger=None):
        """Initialize RAGRetriever with tensor parameters."""
        self.logger = logger or setup_logging()
        self.dataset_path = dataset_path

        # Define tensor parameters as class attribute
        self.tensor_params = [
            # Core content tensors
            {"name": "text", "dtype": "str"},
            {"name": "embedding", "dtype": "float32"},

            # Document structure
            {"name": "original_filename", "dtype": "str"},
            {"name": "file_type", "dtype": "str"},
            {"name": "chunk_id", "dtype": "int64"},
            {"name": "total_chunks", "dtype": "int64"},

            # Basic identification
            {"name": "call_number", "dtype": "str"},
            {"name": "title", "dtype": "str"},
            {"name": "date", "dtype": "str"},
            {"name": "created_published", "dtype": "str"},
            {"name": "language", "dtype": "str"},
            {"name": "type", "dtype": "str"},

            # Contributors and sources
            {"name": "contributors", "dtype": "str"},  # JSON array as string
            {"name": "creator", "dtype": "str"},
            {"name": "repository", "dtype": "str"},
            {"name": "collection", "dtype": "str"},
            {"name": "source_collection", "dtype": "str"},

            # Content details
            {"name": "description", "dtype": "str"},
            {"name": "notes", "dtype": "str"},  # JSON array as string
            {"name": "subjects", "dtype": "str"},  # JSON array as string
            {"name": "original_format", "dtype": "str"},
            {"name": "online_formats", "dtype": "str"},  # JSON array as string

            # Rights and access
            {"name": "rights", "dtype": "str"},
            {"name": "access_restricted", "dtype": "str"},

            # Locations
            {"name": "locations", "dtype": "str"},  # JSON array as string

            # Resource location
            {"name": "url", "dtype": "str"},

            # EAD specific fields
            {"name": "collection_title", "dtype": "str"},
            {"name": "collection_date", "dtype": "str"},
            {"name": "collection_abstract", "dtype": "str"},
            {"name": "series_title", "dtype": "str"},

            # MARC specific fields
            {"name": "catalog_title", "dtype": "str"},
            {"name": "catalog_creator", "dtype": "str"},
            {"name": "catalog_date", "dtype": "str"},
            {"name": "catalog_description", "dtype": "str"},
            {"name": "catalog_subjects", "dtype": "str"},  # JSON array as string
            {"name": "catalog_notes", "dtype": "str"},  # JSON array as string
            {"name": "catalog_language", "dtype": "str"},
            {"name": "catalog_genre", "dtype": "str"},  # JSON array as string
            {"name": "catalog_contributors", "dtype": "str"},  # JSON array as string
            {"name": "catalog_repository", "dtype": "str"},
            {"name": "catalog_collection_id", "dtype": "str"}
        ]

        # Initialize embedding model based on selection
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

        self.vectorstore = None
        self.documents = []

    def load_configuration(self):
        """Load configuration from config files."""
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
        """Create AWS Bedrock client."""
        session = boto3.Session(
            aws_access_key_id=config['BedRock_LLM_API']['aws_access_key_id'],
            aws_secret_access_key=config['BedRock_LLM_API']['aws_secret_access_key'],
            aws_session_token=config['BedRock_LLM_API']['aws_session_token']
        )
        return session.client("bedrock-runtime", region_name="us-east-1")

    def initialize_vectorstore(self, delete_existing: bool = False) -> None:
        """Initialize vectorstore with all required tensors."""
        try:
            if self.vectorstore is not None:
                self.logger.info("Cleaning up existing vectorstore reference")
                self.vectorstore = None

            if delete_existing and os.path.exists(self.dataset_path):
                self.logger.info(f"Deleting existing DeepLake dataset at {self.dataset_path}")
                try:
                    shutil.rmtree(self.dataset_path)
                    self.logger.info("Dataset deleted successfully")
                except Exception as e:
                    self.logger.error(f"Error deleting dataset: {e}")
                    raise

            # Create fresh dataset with all required tensors
            import deeplake
            self.logger.info("Creating new dataset with all tensors")

            ds = deeplake.empty(self.dataset_path, overwrite=True)

            # Create all tensors upfront
            ds.create_tensor('text', dtype='str', sample_compression=None)
            ds.create_tensor('metadata', dtype='str', sample_compression=None)  # For compatibility
            ds.create_tensor('embedding', dtype='float32', sample_compression=None)
            ds.create_tensor('id', dtype='str', sample_compression=None)

            # Create metadata tensors
            metadata_fields = [
                'original_filename', 'file_type', 'call_number', 'title', 'date',
                'created_published', 'language', 'type', 'creator', 'repository',
                'collection', 'source_collection', 'description', 'subjects',
                'original_format', 'online_formats', 'rights', 'access_restricted',
                'locations', 'url', 'collection_title', 'collection_date',
                'collection_abstract', 'series_title', 'catalog_title', 'catalog_creator',
                'catalog_date', 'catalog_description', 'catalog_language',
                'catalog_repository', 'catalog_collection_id'
            ]

            # Create tensor for each metadata field
            for field in metadata_fields:
                ds.create_tensor(field, dtype='str', sample_compression=None)

            # Create tensors for array fields (stored as JSON strings)
            array_fields = [
                'contributors', 'notes', 'catalog_subjects', 'catalog_notes',
                'catalog_genre', 'catalog_contributors'
            ]
            for field in array_fields:
                ds.create_tensor(field, dtype='str', sample_compression=None)

            # Create numeric tensors
            ds.create_tensor('chunk_id', dtype='int64', sample_compression=None)
            ds.create_tensor('total_chunks', dtype='int64', sample_compression=None)

            self.logger.info(f"Created dataset with tensors: {list(ds.tensors.keys())}")

            # Initialize vectorstore with the prepared dataset
            self.vectorstore = DeepLake(
                dataset_path=self.dataset_path,
                embedding_function=self.embeddings,
                read_only=False
            )

            self.logger.info("Vectorstore initialized successfully")

        except Exception as e:
            self.logger.error(f"Error during vectorstore initialization: {e}")
            raise

    def load_data(self, data_dir: str, metadata: Dict[str, Dict]) -> List[Document]:
        """Load documents from the data directory."""
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

                        # Set file_type based on filename pattern
                        if '_en.txt' in filename or '_en_translation.txt' in filename:
                            doc_metadata['file_type'] = 'transcript'
                        else:
                            doc_metadata['file_type'] = 'text'

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

    def generate_embeddings(self, documents: List[Document]) -> None:
        """Generate embeddings for documents."""
        if self.vectorstore is None:
            self.logger.error("Vectorstore not initialized. Call initialize_vectorstore first.")
            raise ValueError("Vectorstore not initialized")

        try:
            self.logger.info(f"Processing {len(documents)} documents")
            ds = self.vectorstore.vectorstore.dataset

            for doc in documents:
                text = doc.page_content
                metadata = doc.metadata.copy()

                # Create embeddings
                embedding = self.embeddings.embed_query(text)

                # Add data to each tensor
                with ds:
                    ds.text.append(text)
                    ds.embedding.append(embedding)

                    # Handle metadata
                    ds.metadata.append(json.dumps(metadata))  # Store full metadata as JSON

                    # Add values for each metadata field
                    for field in ds.tensors:
                        if field not in ['text', 'embedding', 'metadata', 'id']:
                            if field in ['chunk_id', 'total_chunks']:
                                try:
                                    value = int(metadata.get(field, 0))
                                except (ValueError, TypeError):
                                    value = 0
                                getattr(ds, field).append(value)
                            else:
                                value = metadata.get(field, '')
                                if isinstance(value, (list, dict)):
                                    value = json.dumps(value)
                                else:
                                    value = str(value)
                                getattr(ds, field).append(value)

                    # Generate ID
                    ds.id.append(str(len(ds)))

            self.logger.info("Embeddings and metadata generated successfully")

        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            raise

    def search_vector_store(self, query: str, filter: Dict = None, top_k: int = 3) -> List[Document]:
        """Enhanced search with complete metadata filtering capabilities."""
        if self.vectorstore is None:
            self.logger.error("Vectorstore not initialized")
            return []

        try:
            # Log search parameters
            self.logger.info("\n--- Search Parameters ---")
            self.logger.info(f"Query: {query}")
            self.logger.info(f"Filter: {filter}")
            self.logger.info(f"Top K: {top_k}")

            results = self.vectorstore.similarity_search(
                query,
                filter=filter,
                k=top_k
            )

            self.logger.info("\n--- Search Results ---")
            self.logger.info(f"Number of results: {len(results)}")

            if results:
                self.logger.debug("First result metadata fields:")
                for key in sorted(results[0].metadata.keys()):
                    self.logger.debug(f"  {key}")

            return results

        except Exception as e:
            self.logger.error(f"Error during similarity search: {e}")
            return []

    def is_empty(self) -> bool:
        """Check if the vectorstore is empty."""
        if self.vectorstore is None:
            return True

        try:
            return len(self.vectorstore.get_ids()) == 0
        except Exception as e:
            self.logger.error(f"Error checking vectorstore: {e}")
            return True

    def print_dataset_info(self) -> None:
        """Print information about the current state of the dataset."""
        if self.vectorstore is None:
            self.logger.info("No vectorstore initialized")
            return

        try:
            num_elements = len(self.vectorstore.get_ids())
            self.logger.info(f"Number of elements in vectorstore: {num_elements}")

            if num_elements > 0:
                sample = self.vectorstore.get(ids=[self.vectorstore.get_ids()[0]])
                self.logger.info("\nMetadata fields in first element:")
                for key in sorted(sample[0].metadata.keys()):
                    self.logger.info(f"  {key}")
        except Exception as e:
            self.logger.error(f"Error printing dataset info: {e}")