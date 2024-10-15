import os
import shutil
import re
from typing import Dict, List, Any
#from langchain.vectorstores import DeepLake
from langchain_community.vectorstores import DeepLake # last one was depracated
from langchain.schema import Document


try:
    from langchain_community.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings
except ImportError:
    from langchain.embeddings import HuggingFaceEmbeddings


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

    def load_vectorstore(self):
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

    def create_new_vectorstore(self):
        print("Creating new DeepLake dataset...")
        return DeepLake(dataset_path=self.dataset_path, embedding_function=self.embeddings)

    def print_dataset_info(self, vectorstore):
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

    def is_empty(self):
        try:
            return len(self.vectorstore) == 0
        except Exception as e:
            print(f"Error checking if vectorstore is empty: {e}")
            return True  # Assume empty if there's an error

    def load_data(self, data_dir: str, metadata: Dict[str, Dict]) -> List[Document]:
        self.documents = []  # Reset documents
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
                    print(f"Warning: No metadata found for {filename}")
                    continue

                doc = Document(page_content=content, metadata=doc_metadata)
                self.documents.append(doc)

        print(f"Loaded {len(self.documents)} documents with metadata")
        if self.documents:
            sample_doc = self.documents[0]
            print(f"\nSample document metadata for file: {sample_doc.metadata.get('original_filename', 'Unknown')}")
            for key, value in sample_doc.metadata.items():
                print(f"  {key}: {value}")
        return self.documents

    def generate_embeddings(self, documents: List[Document]):
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        print("\n--- Metadata being added to vectorstore ---")
        print(f"Number of documents: {len(documents)}")
        print("Metadata fields being added:")
        if metadatas:
            for key in metadatas[0].keys():
                print(f"  {key}")
        print("------------------------------------------\n")

        self.vectorstore.add_texts(texts, metadatas=metadatas)
        print(f"Added {len(documents)} documents to DeepLake dataset")
        self.print_dataset_info(self.vectorstore)

    def search_vector_store(self, query: str, filter: Dict = None, top_k: int = 3):
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

    def query_metadata(self, filter: Dict = None, limit: int = 10):
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

    def print_sample_documents(self, num_samples=5):
        samples = self.documents[:num_samples] if self.documents else []
        print(f"\nPrinting {len(samples)} sample documents:")
        for i, doc in enumerate(samples, 1):
            print(f"\nDocument {i}:")
            print(f"Content: {doc.page_content[:100]}...")
            print("Metadata:")
            for key, value in doc.metadata.items():
                print(f"  {key}: {value}")
            print("-" * 50)

    def is_empty(self):
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

    def delete_dataset(self):
        if os.path.exists(self.dataset_path):
            print(f"Deleting existing DeepLake dataset at {self.dataset_path}")
            shutil.rmtree(self.dataset_path)
            print("Dataset deleted successfully.")
            self.vectorstore = self.create_new_vectorstore()
        else:
            print("No existing dataset found.")