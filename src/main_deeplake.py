import os
import sys
import time
import torch
import gc
import pandas as pd
from typing import List, Dict, Any
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
from component.logging_config import setup_logging


def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def chunk_documents(documents: List[Document], chunk_size: int, logger) -> List[Document]:
    logger.info("\nStarting document chunking process...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_size * 15 // 100,
        length_function=len,
    )

    chunked_documents = []
    for doc in documents:
        logger.info(f"\nChunking document: {doc.metadata.get('original_filename', 'unknown')}")
        logger.debug(f"Original metadata keys: {list(doc.metadata.keys())}")

        chunks = text_splitter.split_text(doc.page_content)
        logger.info(f"Creating {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            chunked_doc = Document(
                page_content=chunk,
                metadata={
                    **doc.metadata,
                    'chunk_id': i,
                    'total_chunks': len(chunks),
                    'chunk_size': chunk_size,
                    'chunk_length': len(chunk)
                }
            )
            chunked_documents.append(chunked_doc)

            # Log first chunk's metadata for verification
            if i == 0:
                logger.debug(f"First chunk metadata keys: {list(chunked_doc.metadata.keys())}")

    logger.info(f"\nTotal chunks created: {len(chunked_documents)}")
    return chunked_documents


def main():
    # Setup logging
    logger = setup_logging()

    set_seed(42)
    data_dir = os.path.join(project_root, 'data')
    chunk_size = 100
    logger.info(f"\nInitializing with data_dir: {data_dir}")

    # Process metadata
    logger.info("\n=== STAGE 1: METADATA PROCESSING ===")
    metadata = process_metadata(data_dir)
    logger.info(f"Processed metadata for {len(metadata)} documents")
    if metadata:
        sample_key = next(iter(metadata))
        logger.info(f"Sample metadata key: {sample_key}")
        logger.info("Sample metadata content:")
        for k, v in metadata[sample_key].items():
            logger.info(f"  {k}: {v}")

    # Initialize retriever
    dataset_path = os.path.join(data_dir, f'deeplake_dataset_chunk_{chunk_size}')
    logger.info(f"\n=== STAGE 2: INITIALIZING RETRIEVER ===")
    logger.info(f"Dataset path: {dataset_path}")

    delete_existing = input("Do you want to delete the existing dataset? (y/n): ").lower() == 'y'
    text_retriever = RAGRetriever(dataset_path=dataset_path, model_name='instructor', logger=logger)

    if delete_existing:
        text_retriever.delete_dataset()
        logger.info("Existing dataset deleted.")

    # Load documents
    logger.info("\n=== STAGE 3: LOADING DOCUMENTS ===")
    documents = text_retriever.load_data(data_dir, metadata)
    logger.info(f"Loaded {len(documents)} documents")
    if documents:
        logger.info("\nFirst document metadata:")
        for k, v in documents[0].metadata.items():
            logger.info(f"  {k}: {v}")

    # Chunk documents
    logger.info("\n=== STAGE 4: CHUNKING DOCUMENTS ===")
    chunked_documents = chunk_documents(documents, chunk_size, logger)
    logger.info(f"Created {len(chunked_documents)} chunks")
    if chunked_documents:
        logger.info("\nFirst chunk metadata:")
        for k, v in chunked_documents[0].metadata.items():
            logger.info(f"  {k}: {v}")

    # Generate embeddings
    logger.info("\n=== STAGE 5: EMBEDDING GENERATION ===")
    if text_retriever.is_empty():
        logger.info("Generating new embeddings...")
        text_retriever.generate_embeddings(chunked_documents)
        logger.info("Embeddings generated and saved.")
    else:
        logger.info("Using existing embeddings.")

    # Initialize RAG components
    logger.info("\n=== STAGE 6: INITIALIZING RAG COMPONENTS ===")
    qa_generator = RAGGenerator(model_name='llama')
    rag_pipeline = RAGPipeline(text_retriever, qa_generator)

    logger.info("\nEntering interactive query mode...")
    while True:
        query = input("\nEnter your question (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break

        start_time = time.time()
        retrieved_docs, most_relevant_passage, response = rag_pipeline.run(query)
        end_time = time.time()

        logger.info("\n=== Results ===")
        logger.info(f"Processing time: {end_time - start_time:.2f} seconds")
        logger.info(f"Retrieved {len(retrieved_docs)} documents")
        logger.info("\nMetadata from retrieved documents:")
        for i, doc in enumerate(retrieved_docs):
            logger.info(f"\nDocument {i + 1} metadata:")
            for k, v in doc.metadata.items():
                logger.info(f"  {k}: {v}")
        logger.info("\nResponse:")
        logger.info(response)
        logger.info("=" * 50)

    # Cleanup
    logger.info("\nCleaning up...")
    del text_retriever, qa_generator, rag_pipeline
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()