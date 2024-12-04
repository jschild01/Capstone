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
import numpy as np
from datetime import datetime, timezone
import time

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings
except ImportError:
    from langchain.embeddings import HuggingFaceEmbeddings


class AWSCredentialManager:
    def __init__(self, logger=None):
        self.logger = logger
        self._client = None

        # Ask user to check credentials at startup
        input("Please refresh your Bedrock credentials in config.ini and press Enter to continue...")

        if self.logger:
            self.logger.info("Initializing AWS credential manager")

        # Load initial configuration
        self.config = self.load_configuration(initial=True)

    def load_configuration(self, initial=False):
        """Load configuration from config files."""
        try:
            components_dir = os.path.dirname(os.path.dirname(__file__))  # This gets us to src/
            project_root = os.path.dirname(components_dir)  # This gets us to Capstone/
            config_dir = os.path.join(project_root, 'config')
            config_file = os.path.join(config_dir, 'config.ini')

            if initial:  # Only print during initialization
                print(f"Loading config from: {config_file}")

            config = ConfigParser(interpolation=ExtendedInterpolation())
            config.read(config_file)
            return config
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error loading configuration: {e}")
            print(f"Error loading configuration: {e}")
            raise

    def get_bedrock_client(self):
        """Get a Bedrock client with current credentials."""
        try:
            config = self.load_configuration()
            session = boto3.Session(
                aws_access_key_id=config['BedRock_LLM_API']['aws_access_key_id'],
                aws_secret_access_key=config['BedRock_LLM_API']['aws_secret_access_key'],
                aws_session_token=config['BedRock_LLM_API']['aws_session_token']
            )
            #print("Loaded credentials from config file")
            self._client = session.client("bedrock-runtime", region_name="us-east-1")
            return self._client

        except ClientError as e:
            if 'ExpiredToken' in str(e) or 'ExpiredTokenException' in str(e):
                if self.logger:
                    self.logger.info("Credentials expired, waiting for user to update them...")
                print("\nBedrock credentials expired. Please update config.ini and press Enter to continue...")
                input()
                # Retry with new credentials
                return self.get_bedrock_client()
            print(f"Failed to create Bedrock client: {str(e)}")
            raise

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

        print(f"\nInitializing RAG Retriever with {model_name} model")

        # Initialize embedding model based on selection
        if model_name == 'instructor':
            self.embeddings = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl')
        elif model_name == 'mini':
            self.embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
        elif model_name == 'titan':
            self.credential_manager = AWSCredentialManager(logger=self.logger)
            self.embeddings = BedrockEmbeddings(
                client=self.credential_manager.get_bedrock_client(),
                region_name="us-east-1",
                model_id="amazon.titan-embed-text-v2:0")
        else:
            print('Model name not recognized. Implementing default HuggingFace Embedding model')
            self.embeddings = HuggingFaceEmbeddings()

        self.vectorstore = None
        self.documents = []

    def initialize_vectorstore(self, delete_existing: bool = False) -> None:
        """Initialize vectorstore with all required tensors."""
        try:
            if self.vectorstore is not None:
                print("Cleaning up existing vectorstore reference")
                self.vectorstore = None

            if delete_existing and os.path.exists(self.dataset_path):
                print(f"Deleting existing DeepLake dataset at {self.dataset_path}")
                try:
                    shutil.rmtree(self.dataset_path)
                    print("Dataset deleted successfully")
                except Exception as e:
                    print(f"Error deleting dataset: {e}")
                    raise

            # Create fresh dataset with all required tensors
            import deeplake
            print("Creating new dataset with all tensors")

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

            print("Creating metadata tensors...")
            # Create tensor for each metadata field
            for field in metadata_fields:
                ds.create_tensor(field, dtype='str', sample_compression=None)

            # Create tensors for array fields (stored as JSON strings)
            array_fields = [
                'contributors', 'notes', 'catalog_subjects', 'catalog_notes',
                'catalog_genre', 'catalog_contributors'
            ]
            print("Creating array field tensors...")
            for field in array_fields:
                ds.create_tensor(field, dtype='str', sample_compression=None)

            # Create numeric tensors
            ds.create_tensor('chunk_id', dtype='int64', sample_compression=None)
            ds.create_tensor('total_chunks', dtype='int64', sample_compression=None)

            print(f"Created dataset with tensors: {list(ds.tensors.keys())}")

            # Initialize vectorstore with the prepared dataset
            self.vectorstore = DeepLake(
                dataset_path=self.dataset_path,
                embedding_function=self.embeddings,
                read_only=False
            )

            print("Vectorstore initialized successfully")

        except Exception as e:
            print(f"Error during vectorstore initialization: {e}")
            raise

    def normalize_transcript_filename(self, filename: str) -> str:
        """Normalize English transcript filenames by converting to mp3."""
        if '_en' not in filename:
            return None
        base = re.sub(r'_(en|en_translation)\.txt$', '', filename)
        return f"{base}.mp3"

    def process_with_checkpoints(self, documents: List[Document], batch_size: int = 1000,
                                 checkpoint_dir: str = 'checkpoints') -> None:
        """Process documents with checkpointing for recovery from failures."""
        import os
        import json
        from datetime import datetime

        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_file = os.path.join(checkpoint_dir,
                                       f'embedding_checkpoint_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')

        print(f"\nUsing checkpoint file: {checkpoint_file}")

        # Load last checkpoint if exists
        start_idx = 0
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'r') as f:
                    checkpoint_data = json.load(f)
                    start_idx = checkpoint_data.get('last_processed_index', 0)
                    print(f"Resuming from checkpoint at index {start_idx}")
            except Exception as e:
                print(f"Error loading checkpoint: {e}")

        total_docs = len(documents)

        try:
            for batch_start in range(start_idx, total_docs, batch_size):
                batch_end = min(batch_start + batch_size, total_docs)
                batch = documents[batch_start:batch_end]

                try:
                    # Process batch
                    print(f"\nProcessing batch {batch_start}-{batch_end} of {total_docs}")
                    self.generate_embeddings(batch)

                    # Save checkpoint
                    checkpoint_data = {
                        'last_processed_index': batch_end,
                        'total_documents': total_docs,
                        'timestamp': datetime.now().isoformat()
                    }
                    with open(checkpoint_file, 'w') as f:
                        json.dump(checkpoint_data, f)

                    print(f"Checkpoint saved at document {batch_end}")

                except Exception as e:
                    print(f"Error processing batch {batch_start}-{batch_end}: {e}")
                    # Save checkpoint at failure point
                    checkpoint_data = {
                        'last_processed_index': batch_start,
                        'total_documents': total_docs,
                        'timestamp': datetime.now().isoformat(),
                        'error': str(e)
                    }
                    with open(checkpoint_file, 'w') as f:
                        json.dump(checkpoint_data, f)
                    raise

        except Exception as e:
            print(f"Processing failed at document {batch_start}: {e}")
            raise

    def load_data(self, data_dir: str, metadata: Dict[str, Dict]) -> List[Document]:
        """Load all documents from the data directory, with or without metadata."""
        print("\nStarting document loading...")
        print(f"Number of metadata entries: {len(metadata)}")
        if metadata:
            print("Sample metadata keys:", list(next(iter(metadata.values())).keys()))

        self.documents = []
        txt_dir = os.path.join(data_dir, 'txt')
        transcripts_dir = os.path.join(data_dir, 'transcripts')
        ocr_dir = os.path.join(data_dir, 'pdf', 'txtConversion')

        docs_with_metadata = 0
        docs_without_metadata = 0

        for directory in [txt_dir, transcripts_dir, ocr_dir]:
            if not os.path.exists(directory):
                print(f"Directory not found: {directory}")
                continue

            print(f"\nProcessing directory: {directory}")
            for filename in os.listdir(directory):
                if not filename.endswith('.txt'):
                    continue

                print(f"\nProcessing: {filename} from {os.path.basename(directory)}")
                file_path = os.path.join(directory, filename)

                # Handle different file types
                if directory == transcripts_dir:
                    base_filename = self.normalize_transcript_filename(filename)
                    if base_filename is None:
                        print(f"Skipping non-English transcript: {filename}")
                        continue
                    file_type = 'transcript'
                elif directory == ocr_dir:
                    base_filename = re.sub(r'\.txt$', '.pdf', filename)
                    file_type = 'pdf_ocr'
                else:  # txt_dir
                    base_filename = filename
                    file_type = 'text'

                print(f"Base filename: {base_filename}")

                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()

                    if base_filename in metadata:
                        doc_metadata = metadata[base_filename].copy()
                        print(f"Found metadata with {len(doc_metadata)} fields")
                        print(f"Metadata keys: {list(doc_metadata.keys())}")

                        doc_metadata['original_filename'] = filename
                        doc_metadata['file_type'] = file_type
                        doc_metadata['metadata_status'] = 'complete'

                        doc_metadata = {k: str(v) if v is not None and v != '' else 'N/A'
                                        for k, v in doc_metadata.items()}

                        docs_with_metadata += 1
                    else:
                        print(f"No metadata found for {base_filename}")
                        doc_metadata = {
                            'original_filename': filename,
                            'base_filename': base_filename,
                            'file_type': file_type,
                            'source_directory': os.path.basename(directory),
                            'metadata_status': 'missing',
                            'title': 'N/A',
                            'date': 'N/A',
                            'contributors': 'N/A'
                        }
                        docs_without_metadata += 1

                    doc = Document(page_content=content, metadata=doc_metadata)
                    self.documents.append(doc)

                except Exception as e:
                    print(f"Error processing file {filename}: {e}")

        print(f"\nLoading complete:")
        print(f"Documents with metadata: {docs_with_metadata}")
        print(f"Documents without metadata: {docs_without_metadata}")
        print(f"Total documents loaded: {len(self.documents)}")

        if self.documents:
            print("\nFirst document metadata:")
            for k, v in self.documents[0].metadata.items():
                print(f"  {k}: {v}")

        return self.documents

    def generate_embeddings(self, documents: List[Document], batch_size: int = 1000) -> None:
        """Generate embeddings with user-prompted credential refresh."""
        if self.vectorstore is None:
            print("Vectorstore not initialized. Call initialize_vectorstore first.")
            raise ValueError("Vectorstore not initialized")

        try:
            total_docs = len(documents)
            print(f"\nProcessing {total_docs} documents in batches of {batch_size}")
            ds = self.vectorstore.vectorstore.dataset

            # Process documents in batches
            for batch_start in range(0, total_docs, batch_size):
                batch_end = min(batch_start + batch_size, total_docs)
                batch = documents[batch_start:batch_end]

                try:
                    # Initialize lists for batch data
                    texts = []
                    embeddings = []
                    metadata_list = []
                    tensor_data = {field: [] for field in ds.tensors.keys() if
                                   field not in ['text', 'embedding', 'metadata', 'id']}
                    ids = []

                    # Process documents in current batch
                    i = 0
                    while i < len(batch):
                        doc = batch[i]
                        try:
                            # Update client before embedding
                            if hasattr(self, 'credential_manager'):
                                try:
                                    self.embeddings.client = self.credential_manager.get_bedrock_client()
                                except ClientError as e:
                                    if 'ExpiredToken' in str(e) or 'ExpiredTokenException' in str(e):
                                        print(
                                            "\nBedrock credentials expired. Please update config.ini and press Enter to continue...")
                                        input()
                                        continue  # Retry getting client with new credentials
                                    raise

                            print(f"Processing document {batch_start + i + 1}/{total_docs}")

                            try:
                                embedding = self.embeddings.embed_query(doc.page_content)
                            except ClientError as e:
                                if 'ExpiredToken' in str(e) or 'ExpiredTokenException' in str(e):
                                    print(
                                        "\nBedrock credentials expired during embedding. Please update config.ini and press Enter to continue...")
                                    input()
                                    continue  # Retry this document
                                raise

                            # If we get here, embedding was successful
                            texts.append(doc.page_content)
                            embeddings.append(embedding)
                            metadata_list.append(json.dumps(doc.metadata))

                            # Process tensor fields
                            for field in tensor_data:
                                if field in ['chunk_id', 'total_chunks']:
                                    value = int(doc.metadata.get(field, 0))
                                else:
                                    value = doc.metadata.get(field, '')
                                    if isinstance(value, (list, dict)):
                                        value = json.dumps(value)
                                    value = str(value) if value is not None else ''
                                tensor_data[field].append(value)

                            ids.append(str(len(ds) + batch_start + i))
                            i += 1  # Only increment if successful

                        except Exception as e:
                            if not isinstance(e, ClientError) or \
                                    ('ExpiredToken' not in str(e) and 'ExpiredTokenException' not in str(e)):
                                print(f"Error processing document {batch_start + i + 1}: {e}")
                                i += 1  # Skip this document on other errors
                                continue

                    # Only try to add batch if we have processed documents
                    if texts:
                        try:
                            with ds:
                                ds.text.extend(texts)
                                ds.embedding.extend(embeddings)
                                ds.metadata.extend(metadata_list)
                                ds.id.extend(ids)

                                # Extend all other tensors
                                for field, values in tensor_data.items():
                                    tensor = getattr(ds, field)
                                    tensor.extend(values)

                            print(f"Successfully added batch of {len(texts)} documents")

                            # Clear memory
                            import gc
                            import torch
                            del texts, embeddings, metadata_list, tensor_data, ids
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                        except Exception as e:
                            print(f"Error adding batch to dataset: {e}")
                            raise

                except Exception as e:
                    print(f"Error processing batch: {str(e)}")
                    raise

            print(f"\nCompleted processing {total_docs} documents")
            print("Final tensor lengths:")
            final_lengths = {name: len(tensor) for name, tensor in ds.tensors.items()}
            print(str(final_lengths))

        except Exception as e:
            print(f"Error in batch processing: {e}")
            raise

    def search_vector_store(self, query: str, filter: Dict = None, top_k: int = 3, txt_boost: float = 1.2) -> List[
        Document]:
        """Search with efficient boosting for txt files."""
        if self.vectorstore is None:
            print("Vectorstore not initialized")
            return []

        try:
            # Update client before search if using AWS
            if hasattr(self, 'credential_manager'):
                self.embeddings.client = self.credential_manager.get_bedrock_client()

            print("\n--- Search Parameters ---")
            print(f"Query: {query}")
            print(f"Filter: {filter}")
            print(f"Top K: {top_k}")
            print(f"TXT Boost Factor: {txt_boost}")

            ds = self.vectorstore.vectorstore.dataset
            dataset_size = len(ds.embedding)
            print(f"Dataset size: {dataset_size}")

            if dataset_size == 0:
                print("Empty dataset")
                return []

            query_embedding = self.embeddings.embed_query(query)

            try:
                embeddings = ds.embedding.numpy()
                if len(embeddings.shape) == 3:
                    embeddings = embeddings.squeeze(1)
            except Exception as e:
                print(f"Error accessing embeddings: {e}")
                return []

            # Calculate base similarities
            similarities = np.dot(embeddings, query_embedding) / (
                    np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
            )

            # Get initial top candidates (3x the requested amount to account for potential boosting)
            candidate_count = min(top_k * 3, dataset_size)
            candidate_indices = np.argsort(similarities)[-candidate_count:][::-1]

            # Process only the candidates
            scored_indices = []
            for idx in candidate_indices:
                try:
                    metadata_array = ds.metadata[int(idx)].numpy()
                    metadata_dict = self._parse_metadata(metadata_array)

                    # Apply boost to txt files
                    adjusted_similarity = similarities[idx]
                    if metadata_dict.get('file_type') == 'text':
                        adjusted_similarity *= txt_boost

                    scored_indices.append((idx, adjusted_similarity))
                except Exception as e:
                    print(f"Error processing metadata at index {idx}: {e}")
                    continue

            # Sort candidates by adjusted similarity and get top k
            scored_indices.sort(key=lambda x: x[1], reverse=True)
            top_k = min(top_k, len(scored_indices))
            top_indices = [idx for idx, _ in scored_indices[:top_k]]

            processed_results = []
            for idx in top_indices:
                try:
                    text_array = ds.text[int(idx)].numpy()
                    metadata_array = ds.metadata[int(idx)].numpy()

                    text_str = self._convert_to_string(text_array)
                    if not text_str:
                        continue

                    metadata_dict = self._parse_metadata(metadata_array)
                    metadata_dict['similarity_score'] = float(similarities[idx])
                    metadata_dict['adjusted_similarity_score'] = float(scored_indices[top_indices.index(idx)][1])
                    metadata_dict['dataset_index'] = int(idx)

                    doc = Document(
                        page_content=str(text_str),
                        metadata=metadata_dict
                    )
                    processed_results.append(doc)

                except IndexError as e:
                    print(f"Index {idx} out of bounds: {e}")
                    continue
                except Exception as e:
                    print(f"Error processing result at index {idx}: {e}")
                    continue

            # Log the types of documents retrieved
            file_types = [
                f"{doc.metadata.get('file_type', 'unknown')}"
                f"{'*' if doc.metadata.get('file_type') == 'text' else ''}"
                for doc in processed_results
            ]
            print(f"\nSuccessfully processed {len(processed_results)} results")
            print(f"Retrieved documents: {file_types} (* = boosted)")

            return processed_results

        except Exception as e:
            print(f"Error during similarity search: {e}")
            return []

    def _convert_to_string(self, value) -> str:
        """Convert various types to string, with special handling for numpy arrays."""
        try:
            if isinstance(value, np.ndarray):
                if value.dtype.kind in ['U', 'S']:
                    if value.size == 1:
                        return value.item()
                    elif value.size > 0:
                        return value.flatten()[0]
                    else:
                        return ""
                return str(value.tolist())
            elif isinstance(value, bytes):
                return value.decode('utf-8')
            elif isinstance(value, (list, dict)):
                return json.dumps(value)
            else:
                return str(value)
        except Exception as e:
            print(f"Error converting value to string: {e}")
            print(f"Value type: {type(value)}")
            print(f"Value: {value}")
            return ""

    def _parse_metadata(self, metadata) -> dict:
        """Parse metadata into dictionary with improved numpy handling."""
        try:
            if isinstance(metadata, np.ndarray):
                if metadata.size == 1:
                    metadata = metadata.item()
                elif metadata.size > 0:
                    metadata = metadata.flatten()[0]
                else:
                    return {}

            if isinstance(metadata, (bytes, np.bytes_)):
                metadata = metadata.decode('utf-8')

            if isinstance(metadata, str):
                try:
                    return json.loads(metadata)
                except json.JSONDecodeError:
                    print("Failed to parse metadata JSON string")
                    return {}

            if isinstance(metadata, dict):
                return {k: self._convert_to_string(v) for k, v in metadata.items()}

            print(f"Unexpected metadata type: {type(metadata)}")
            return {}

        except Exception as e:
            print(f"Error processing metadata: {e}")
            return {}

    def is_empty(self) -> bool:
        """Check if the vectorstore is empty."""
        if self.vectorstore is None:
            return True

        try:
            ds = self.vectorstore.vectorstore.dataset
            return len(ds.embedding) == 0
        except Exception as e:
            print(f"Error checking vectorstore: {e}")
            return True