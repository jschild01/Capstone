import os
import sys
import time
import torch
import gc
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import re

# Add the src directory to the Python path
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, src_dir)

from component.rag_retriever_deeplake import RAGRetriever
from component.rag_generator_deeplake import RAGGenerator
from component.rag_pipeline_deeplake import RAGPipeline
from component.metadata_processor import process_metadata, clean_metadata_dict
from component.logging_config import setup_logging


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def chunk_documents(documents: List[Document], chunk_size: int, logger) -> List[Document]:
    """Chunk documents with enhanced metadata handling and logging."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_size * 15 // 100,
        length_function=len,
    )

    timecode_pattern = r'\[\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}\] '
    chunked_documents = []
    total_processed = 0

    for doc in documents:
        try:
            # Clean metadata
            clean_metadata = clean_metadata_dict(doc.metadata)

            # Process content based on file type
            is_transcript = clean_metadata.get('file_type') == 'transcript'
            if is_transcript:
                lines = doc.page_content.split('\n')
                cleaned_lines = [re.sub(timecode_pattern, '', line) for line in lines]
                content = '\n'.join(line for line in cleaned_lines if line.strip())
                logger.debug(f"Cleaned transcript content for {clean_metadata.get('original_filename', 'unknown')}")
            else:
                content = doc.page_content

            if not content.strip():
                logger.warning(
                    f"Empty content after cleaning for document: {clean_metadata.get('original_filename', 'unknown')}")
                continue

            # Split content into chunks
            chunks = text_splitter.split_text(content)

            # Create chunk documents
            for i, chunk in enumerate(chunks):
                chunk_metadata = {
                    **clean_metadata,
                    'chunk_id': i,
                    'total_chunks': len(chunks)
                }

                chunked_doc = Document(
                    page_content=chunk,
                    metadata=chunk_metadata
                )
                chunked_documents.append(chunked_doc)

            total_processed += 1
            if total_processed % 100 == 0:
                logger.info(f"Processed {total_processed} documents into {len(chunked_documents)} chunks")

        except Exception as e:
            logger.error(f"Error chunking document {doc.metadata.get('original_filename', 'unknown')}: {e}")
            continue

    logger.info(f"Final chunk count: {len(chunked_documents)} from {total_processed} documents")
    return chunked_documents


def main():
    """Main execution function with enhanced error handling and logging."""
    logger = setup_logging()
    logger.info("Starting RAG pipeline...")

    set_seed(42)
    data_dir = os.path.join(project_root, 'data')
    chunk_size = 1000

    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Chunk size: {chunk_size}")

    # Ask if user wants to delete existing data
    delete_existing = input("Do you want to delete the existing dataset? (y/n): ").lower() == 'y'

    try:
        dataset_path = os.path.join(data_dir, f'deeplake_dataset_chunk_{chunk_size}')
        logger.info(f"Dataset path: {dataset_path}")

        # Initialize components
        metadata = process_metadata(data_dir)
        logger.info(f"Number of documents with metadata: {len(metadata)}")

        # Initialize retriever and vectorstore
        text_retriever = RAGRetriever(
            dataset_path=dataset_path,
            model_name='titan',
            logger=logger
        )

        try:
            text_retriever.initialize_vectorstore(delete_existing=delete_existing)
            if delete_existing:
                logger.info("Dataset deleted and reinitialized successfully")
            else:
                logger.info("Using existing dataset")
        except Exception as e:
            logger.error(f"Error initializing vectorstore: {e}")
            return

        # Load and process documents
        documents = text_retriever.load_data(data_dir, metadata)
        if not documents:
            logger.error("No documents loaded")
            return

        logger.info(f"Loaded {len(documents)} documents")

        # Sample document logging
        logger.info("\nSample of loaded documents:")
        for i, doc in enumerate(documents[:3]):
            logger.info(f"\nDocument {i + 1}:")
            logger.info(f"Content preview: {doc.page_content[:200]}...")
            logger.info(f"Metadata: {doc.metadata}")

        # Generate embeddings if needed
        if text_retriever.is_empty():
            logger.info("Generating embeddings for documents...")
            chunked_docs = chunk_documents(documents, chunk_size, logger)
            if chunked_docs:
                logger.info(f"Generated {len(chunked_docs)} chunks")
                # Add credential refresh prompt before embedding generation
                if hasattr(text_retriever, 'credential_manager'):
                    input(
                        "Please ensure your Bedrock credentials in config.ini are fresh before starting embedding generation. Press Enter to continue...")
                text_retriever.process_with_checkpoints(
                    chunked_docs)  # Using process_with_checkpoints for checkpointing
                logger.info("Embeddings generated successfully")
            else:
                logger.error("No chunks generated")
                return
        else:
            logger.info("Using existing embeddings")

        # Initialize RAG components
        model_name = 'claude'  # Options: 'llama', 't5', 'claude'
        logger.info(f"Initializing RAG pipeline with {model_name} model")

        qa_generator = RAGGenerator(model_name=model_name)
        rag_pipeline = RAGPipeline(text_retriever, qa_generator)

        # Interactive query loop
        logger.info("\nEntering interactive query mode")
        while True:
            query = input("\nEnter your question (or 'quit' to exit): ")
            if query.lower() == 'quit':
                break

            start_time = time.time()
            try:
                # Process query
                if model_name == 'claude':
                    results = rag_pipeline.run_claude(query=query, top_k=3)
                else:
                    results = rag_pipeline.run(query=query, top_k=3)

                retrieved_docs, most_relevant_passage, raw_response, validated_response, structured_response, final_response = results

                # Log results
                end_time = time.time()
                logger.info("\n=== Query Results ===")
                logger.info(f"Processing time: {end_time - start_time:.2f} seconds")
                logger.info(f"Retrieved {len(retrieved_docs)} relevant documents")

                logger.info("\nFinal Response:")
                logger.info(final_response)
                logger.info("=" * 50)

                # Debug logging
                logger.debug("\nRaw Response:")
                logger.debug(raw_response)
                logger.debug("\nValidated Response:")
                logger.debug(validated_response)
                logger.debug("\nStructured Response:")
                logger.debug(structured_response)

            except Exception as e:
                logger.error(f"Error processing query: {e}")
                logger.info("Please try another question")

    except Exception as e:
        logger.error(f"Fatal error in main execution: {e}")

    finally:
        # Cleanup
        logger.info("\nCleaning up resources...")
        if 'text_retriever' in locals():
            del text_retriever
        if 'qa_generator' in locals():
            del qa_generator
        if 'rag_pipeline' in locals():
            del rag_pipeline
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Cleanup complete")


if __name__ == "__main__":
    main()