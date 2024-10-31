import os
import sys
import shutil
import random
import logging
from datetime import datetime
import torch
import gc
from typing import Dict, List
import json

# Add the src directory to the Python path
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, src_dir)

from component.rag_retriever_deeplake import RAGRetriever
from component.metadata_processor import process_metadata, clean_metadata_dict
from component.logging_config import setup_logging


def get_dataset_statistics(data_dir: str) -> Dict[str, int]:
    """Get statistics about the full dataset."""
    stats = {
        'txt': 0,
        'transcripts': 0,
        'pdf_ocr': 0,
        'total': 0
    }

    # Count files in txt directory
    txt_dir = os.path.join(data_dir, 'txt')
    if os.path.exists(txt_dir):
        txt_files = [f for f in os.listdir(txt_dir) if f.endswith('.txt')]
        stats['txt'] = len(txt_files)

    # Count files in transcripts directory
    transcripts_dir = os.path.join(data_dir, 'transcripts')
    if os.path.exists(transcripts_dir):
        transcript_files = [f for f in os.listdir(transcripts_dir) if f.endswith('.txt')]
        stats['transcripts'] = len(transcript_files)

    # Count files in PDF OCR directory
    ocr_dir = os.path.join(data_dir, 'pdf', 'txtConversion')
    if os.path.exists(ocr_dir):
        ocr_files = [f for f in os.listdir(ocr_dir) if f.endswith('.txt')]
        stats['pdf_ocr'] = len(ocr_files)

    stats['total'] = stats['txt'] + stats['transcripts'] + stats['pdf_ocr']
    return stats


def create_test_directory(original_data_dir: str, test_data_dir: str, test_percentage: float = 1.0) -> Dict[str, int]:
    """Create a test directory with a percentage of data."""
    logger = logging.getLogger('TestVectorstore')
    logger.info(f"Creating test directory at {test_data_dir}")

    # Get full dataset statistics
    full_stats = get_dataset_statistics(original_data_dir)
    logger.info("\nFull dataset statistics:")
    logger.info(f"Text files: {full_stats['txt']}")
    logger.info(f"Transcript files: {full_stats['transcripts']}")
    logger.info(f"PDF OCR files: {full_stats['pdf_ocr']}")
    logger.info(f"Total files: {full_stats['total']}")

    # Calculate sample sizes
    test_stats = {
        'txt': max(1, int(full_stats['txt'] * test_percentage / 100)),
        'transcripts': max(1, int(full_stats['transcripts'] * test_percentage / 100)),
        'pdf_ocr': max(1, int(full_stats['pdf_ocr'] * test_percentage / 100))
    }
    test_stats['total'] = test_stats['txt'] + test_stats['transcripts'] + test_stats['pdf_ocr']

    logger.info(f"\nTest sample statistics ({test_percentage}% of data):")
    logger.info(f"Text files: {test_stats['txt']} ({test_stats['txt'] / full_stats['txt'] * 100:.1f}%)")
    logger.info(
        f"Transcript files: {test_stats['transcripts']} ({test_stats['transcripts'] / full_stats['transcripts'] * 100:.1f}%)")
    logger.info(f"PDF OCR files: {test_stats['pdf_ocr']} ({test_stats['pdf_ocr'] / full_stats['pdf_ocr'] * 100:.1f}%)")
    logger.info(f"Total test files: {test_stats['total']} ({test_stats['total'] / full_stats['total'] * 100:.1f}%)")

    try:
        # Create directory structure
        os.makedirs(test_data_dir, exist_ok=True)
        os.makedirs(os.path.join(test_data_dir, 'txt'), exist_ok=True)
        os.makedirs(os.path.join(test_data_dir, 'transcripts'), exist_ok=True)
        os.makedirs(os.path.join(test_data_dir, 'pdf', 'txtConversion'), exist_ok=True)
        os.makedirs(os.path.join(test_data_dir, 'loc_dot_gov_data'), exist_ok=True)

        # Copy sample files
        for subdir, count in [
            ('txt', test_stats['txt']),
            ('transcripts', test_stats['transcripts']),
            (os.path.join('pdf', 'txtConversion'), test_stats['pdf_ocr'])
        ]:
            src_dir = os.path.join(original_data_dir, subdir)
            dst_dir = os.path.join(test_data_dir, subdir)

            if os.path.exists(src_dir):
                files = [f for f in os.listdir(src_dir) if f.endswith('.txt')]
                if files:
                    sample_files = random.sample(files, min(count, len(files)))
                    for file in sample_files:
                        src_file = os.path.join(src_dir, file)
                        dst_file = os.path.join(dst_dir, file)
                        shutil.copy2(src_file, dst_file)
                        logger.debug(f"Copied {file} to test directory")

        # Copy metadata files
        loc_data_src = os.path.join(original_data_dir, 'loc_dot_gov_data')
        loc_data_dst = os.path.join(test_data_dir, 'loc_dot_gov_data')

        if os.path.exists(loc_data_src):
            for item in os.listdir(loc_data_src):
                src_path = os.path.join(loc_data_src, item)
                dst_path = os.path.join(loc_data_dst, item)

                if os.path.isdir(src_path):
                    os.makedirs(dst_path, exist_ok=True)
                    for csv_file in ['file_list.csv', 'search_results.csv']:
                        src_csv = os.path.join(src_path, csv_file)
                        if os.path.exists(src_csv):
                            shutil.copy2(src_csv, os.path.join(dst_path, csv_file))
                            logger.debug(f"Copied {csv_file} to test directory")

    except Exception as e:
        logger.error(f"Error creating test directory: {e}", exc_info=True)
        raise

    return test_stats


def test_vectorstore_creation(test_data_dir: str, embedding_model: str, chunk_size: int = 100) -> bool:
    """Test vectorstore creation process with detailed logging."""
    logger = logging.getLogger('TestVectorstore')
    logger.info(f"\nTesting vectorstore creation with: Embedding={embedding_model}")

    try:
        # Set up test dataset path
        test_dataset_path = os.path.join(test_data_dir, f'test_{embedding_model}_dataset_chunk_{chunk_size}')
        logger.info(f"Test dataset path: {test_dataset_path}")

        # Process metadata with debug output
        logger.info("Starting metadata processing...")
        metadata = process_metadata(test_data_dir)
        logger.info(f"Processed metadata for {len(metadata)} test documents")

        # Debug print first metadata entry
        if metadata:
            first_key = next(iter(metadata))
            logger.info("\nFirst metadata entry:")
            logger.info(f"File: {first_key}")
            logger.info("Metadata contents:")
            for k, v in metadata[first_key].items():
                logger.info(f"{k}: {type(v)} = {v}")

        # Initialize retriever
        logger.info("\nInitializing retriever...")
        text_retriever = RAGRetriever(
            dataset_path=test_dataset_path,
            model_name=embedding_model,
            logger=logger
        )

        # Initialize vectorstore
        logger.info("Initializing vectorstore...")
        text_retriever.initialize_vectorstore(delete_existing=True)
        logger.info("Vectorstore initialized")

        # Load and process documents
        logger.info("Loading documents...")
        documents = text_retriever.load_data(test_data_dir, metadata)
        logger.info(f"Loaded {len(documents)} test documents")

        if not documents:
            logger.error("No test documents loaded")
            return False

        # Log sample document details
        logger.info("\nFirst document details:")
        sample_doc = documents[0]
        logger.info("Content preview (first 200 chars):")
        logger.info(sample_doc.page_content[:200])
        logger.info("\nMetadata:")
        for k, v in sample_doc.metadata.items():
            logger.info(f"{k}: {type(v)} = {v}")

        # Generate embeddings
        logger.info("\nGenerating embeddings...")
        text_retriever.generate_embeddings(documents)
        logger.info("Embeddings generated successfully")

        logger.info(f"\nVectorstore creation test completed successfully for {embedding_model}")
        return True

    except Exception as e:
        logger.error(f"Error in vectorstore creation for {embedding_model}: {str(e)}")
        logger.error("Full traceback:", exc_info=True)
        return False

    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def main():
    """Main test execution function with enhanced logging."""
    # Set up logging
    log_dir = os.path.join(project_root, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(
                os.path.join(log_dir, f'vectorstore_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger('TestVectorstore')

    # Test configurations
    embedding_models = ['instructor']
    test_percentage = 1.0  # Test with 1% of the data

    # Set up test environment
    original_data_dir = os.path.join(project_root, 'data')
    test_data_dir = os.path.join(project_root, 'test_data')

    try:
        # Create test directory with sample data
        logger.info("\n=== Starting Test Setup ===")
        stats = create_test_directory(original_data_dir, test_data_dir, test_percentage)
        logger.info("Test directory created successfully")

        # Run tests for each embedding model
        logger.info("\n=== Starting Vectorstore Tests ===")
        results = {}
        for embedding_model in embedding_models:
            logger.info(f"\nTesting embedding model: {embedding_model}")
            success = test_vectorstore_creation(
                test_data_dir,
                embedding_model=embedding_model,
                chunk_size=100
            )
            results[embedding_model] = success

        # Log results summary
        logger.info("\n=== Test Results Summary ===")
        for model, success in results.items():
            logger.info(f"{model} embedding model: {'SUCCESS' if success else 'FAILED'}")
        logger.info(f"\nTested on {stats['total']} files ({test_percentage}% of total dataset)")

    except Exception as e:
        logger.error("Test execution failed:", exc_info=True)

    finally:
        # Cleanup test directory
        logger.info("\n=== Cleaning Up ===")
        if os.path.exists(test_data_dir):
            try:
                shutil.rmtree(test_data_dir)
                logger.info("Test directory cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up test directory: {e}")
        logger.info("Test script completed")


if __name__ == "__main__":
    main()