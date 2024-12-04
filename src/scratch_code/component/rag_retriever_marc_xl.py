import os
import re
import random  # Added import for random sampling
from typing import Dict, List, Tuple, Callable, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.schema import Document


class RAGRetriever:
    def __init__(self, model_name: str = 'hkunlp/instructor-xl', chunk_size: int = 1000, chunk_overlap: int = 200,
                 vectorstore_path: str = 'vectorstore', allow_deserialization: bool = False):
        self.embeddings = HuggingFaceInstructEmbeddings(model_name=model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.vectorstore_path = vectorstore_path
        self.allow_deserialization = allow_deserialization
        self.vectorstore = self.load_vectorstore()

    def print_sample_documents(self, num_samples=5):
        if self.vectorstore is None:
            print("Vectorstore is not initialized.")
            return

        print(f"\nPrinting {num_samples} sample documents from the vectorstore:")

        # Get all the documents from the vectorstore
        all_docs = list(self.vectorstore.docstore._dict.values())

        # Randomly sample documents if there are more than num_samples
        sample_docs = random.sample(all_docs, min(num_samples, len(all_docs)))

        for i, doc in enumerate(sample_docs, 1):
            print(f"\nDocument {i}:")
            print(f"Content: {doc.page_content[:100]}...")  # Print first 100 characters of content
            print("Metadata:")
            for key, value in doc.metadata.items():
                print(f"  {key}: {value}")
            print("-" * 50)

    def load_vectorstore(self):
        if os.path.exists(self.vectorstore_path):
            print("Loading existing vectorstore...")
            try:
                return FAISS.load_local(
                    self.vectorstore_path,
                    self.embeddings,
                    allow_dangerous_deserialization=self.allow_deserialization  # Allow deserialization based on parameter
                )
            except ValueError as ve:
                print(f"Failed to load vectorstore: {ve}")
                return None
        return None

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
        if self.vectorstore is None:
            texts = self.text_splitter.split_documents(documents)
            self.vectorstore = FAISS.from_documents(texts, self.embeddings)
            self.vectorstore.save_local(self.vectorstore_path)
            print(f"Vectorstore saved to {self.vectorstore_path}")
        else:
            print("Using existing vectorstore.")

    def search_vector_store(self, query: str, top_k: int = 3) -> List[Tuple[Document, float]]:
        if not self.vectorstore:
            raise ValueError("Vector store has not been initialized. Call generate_embeddings() first.")

        results = self.vectorstore.similarity_search_with_score(query, k=top_k)
        return results

    def query_metadata(self,
                       filter_func: Callable[[Dict[str, Any]], bool] = None,
                       sort_key: Callable[[Dict[str, Any]], Any] = None,
                       reverse: bool = False,
                       limit: int = None) -> List[Dict[str, Any]]:
        if self.vectorstore is None:
            print("Vectorstore is not initialized.")
            return []

        # Extract metadata from all documents
        all_metadata = [doc.metadata for doc in self.vectorstore.docstore._dict.values()]

        # Apply filter if provided
        if filter_func:
            all_metadata = [meta for meta in all_metadata if filter_func(meta)]

        # Sort if sort_key is provided
        if sort_key:
            all_metadata.sort(key=sort_key, reverse=reverse)

        # Apply limit if provided
        if limit:
            all_metadata = all_metadata[:limit]

        return all_metadata

    def print_sorted_documents(self, sort_key: str, reverse: bool = False, limit: int = 10):
        sorted_metadata = self.query_metadata(
            sort_key=lambda x: x.get(sort_key, ''),
            reverse=reverse,
            limit=limit
        )

        print(f"\nTop {limit} documents sorted by '{sort_key}' ({'descending' if reverse else 'ascending'}):")
        for i, meta in enumerate(sorted_metadata, 1):
            print(f"\nDocument {i}:")
            print(f"  {sort_key}: {meta.get(sort_key, 'N/A')}")
            for key, value in meta.items():
                if key != sort_key:
                    print(f"  {key}: {value}")
            print("-" * 50)