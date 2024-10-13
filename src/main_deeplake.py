import os
import sys
import time
import torch
import gc
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
        #chunk_overlap=chunk_size // 10,
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
    set_seed(42)

    data_dir = os.path.join(project_root, 'data', 'marc-xl-data')
    chunk_size = 100  # Fixed chunk size of 100

    # Ask if user wants to delete existing data
    delete_existing = input("Do you want to delete the existing dataset? (y/n): ").lower() == 'y'

    # Initialize components
    metadata = process_metadata(data_dir)
    dataset_path = os.path.join(data_dir, f'deeplake_dataset_chunk_{chunk_size}')
    text_retriever = RAGRetriever(dataset_path=dataset_path, model_name='instructor') # use 'instructor' (default) or 'mini'

    if delete_existing:
        text_retriever.delete_dataset()
        print("Existing dataset deleted.")

    # Load and prepare documents
    documents = text_retriever.load_data(data_dir, metadata)
    chunked_documents = chunk_documents(documents, chunk_size)
    num_chunks = len(chunked_documents)
    print(f"Prepared {num_chunks} chunks with size {chunk_size}")

    # Generate embeddings if the dataset is empty
    if text_retriever.is_empty():
        print("Generating embeddings for chunked documents...")
        text_retriever.generate_embeddings(chunked_documents)
        print("Embeddings generated and saved.")
    else:
        print("Using existing embeddings.")

    # Initialize RAG components
    qa_generator = RAGGenerator(model_name='llama3')
    rag_pipeline = RAGPipeline(text_retriever, qa_generator)

    while True:
        query = input("Enter your question (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break

        start_time = time.time()
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