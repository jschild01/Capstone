import os
import sys
import time
import torch
import gc
import pandas as pd
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Add the src directory to the Python path
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, src_dir)

from component.rag_retriever_deeplake import RAGRetriever
from component.rag_generator_deeplake import RAGGenerator
from component.rag_pipeline_deeplake import RAGPipeline
from component.metadata_processor import process_metadata


def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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

def main():
    set_seed(42)

    data_dir = os.path.join(project_root, 'data', 'marc-xl-data')
    chunk_size = 100  # Fixed chunk size of 100

    print(f"Data directory: {data_dir}")
    print(f"Chunk size: {chunk_size}")

    # Ask if user wants to delete existing data
    delete_existing = input("Do you want to delete the existing dataset? (y/n): ").lower() == 'y'

    # Initialize components
    metadata = process_metadata(data_dir)
    print(f"Number of documents with metadata: {len(metadata)}")

    dataset_path = os.path.join(data_dir, f'deeplake_dataset_chunk_{chunk_size}')
    print(f"Dataset path: {dataset_path}")

    text_retriever = RAGRetriever(dataset_path=dataset_path, model_name='titan') # main, instructor, titan
    print("RAGRetriever initialized")

    if delete_existing:
        text_retriever.delete_dataset()
        print("Existing dataset deleted.")

    # Load and prepare documents
    documents = text_retriever.load_data(data_dir, metadata)
    print(f"Number of documents loaded: {len(documents)}")

    if len(documents) == 0:
        print("No documents loaded. Check the load_data method in RAGRetriever.")
        return

    print("Sample of loaded documents:")
    for i, doc in enumerate(documents[:3]):  # Print details of first 3 documents
        print(f"Document {i+1}:")
        print(f"Content preview: {doc.page_content[:100]}...")
        print(f"Metadata: {doc.metadata}")
        print("---")

    chunked_documents = chunk_documents(documents, chunk_size)
    num_chunks = len(chunked_documents)
    print(f"Prepared {num_chunks} chunks with size {chunk_size}")

    if num_chunks == 0:
        print("No chunks created. Check the chunking process.")
        return

    print("Sample of chunked documents:")
    for i, chunk in enumerate(chunked_documents[:3]):  # Print details of first 3 chunks
        print(f"Chunk {i+1}:")
        print(f"Content preview: {chunk.page_content[:100]}...")
        print(f"Metadata: {chunk.metadata}")
        print("---")

    # Generate embeddings if the dataset is empty
    if text_retriever.is_empty():
        print("Generating embeddings for chunked documents...")
        text_retriever.generate_embeddings(chunked_documents)
        print("Embeddings generated and saved.")
    else:
        print("Using existing embeddings.")

    # Initialize RAG components
    model_name='claude'
    qa_generator = RAGGenerator(model_name=model_name) # llama, t5, claude
    rag_pipeline = RAGPipeline(text_retriever, qa_generator)

    while True:
        query = input("Enter your question (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break

        start_time = time.time()
        top_k=3
        if model_name == 'claude':
            retrieved_docs, most_relevant_passage, raw_response, validated_response, structured_response, final_response = rag_pipeline.run_claude(query=query, top_k=top_k)
        else:
            retrieved_docs, most_relevant_passage, raw_response, validated_response, structured_response, final_response = rag_pipeline.run(query=query, top_k=top_k)
        #retrieved_docs, most_relevant_passage, response = rag_pipeline.run(query)
        end_time = time.time()

        print("\n--- Results ---")
        print(f"Processing time: {end_time - start_time:.2f} seconds")
        print(f"Number of retrieved documents: {len(retrieved_docs)}")
        print("Response:")
        print(final_response)
        print("-------------------\n")

    # Cleanup
    del text_retriever, qa_generator, rag_pipeline
    gc.collect()
    torch.cuda.empty_cache()
    

if __name__ == "__main__":
    main()
    
    