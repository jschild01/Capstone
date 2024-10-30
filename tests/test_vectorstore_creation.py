import os
import sys
import shutil
import random
import logging
from datetime import datetime
import torch
import gc
from typing import Any, Dict, List, Optional
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import numpy as np
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


def chunk_documents(documents: List[Document], chunk_size: int, logger) -> List[Document]:
    """Chunk documents with enhanced metadata handling."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_size * 15 // 100,
        length_function=len,
    )

    chunked_documents = []
    for doc in documents:
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            chunk_metadata = doc.metadata.copy()
            chunk_metadata['chunk_id'] = i
            chunk_metadata['total_chunks'] = len(chunks)
            chunked_doc = Document(
                page_content=chunk,
                metadata=chunk_metadata
            )
            chunked_documents.append(chunked_doc)

    logger.info(f"Split {len(documents)} documents into {len(chunked_documents)} chunks")
    return chunked_documents


def inspect_documents(documents: List[Document], logger, num_samples: int = 3) -> None:
    """Inspect document content and metadata in detail."""
    logger.info(f"\n=== Document Inspection ({len(documents)} total documents) ===")

    # Analyze all documents for metadata consistency
    metadata_fields = set()
    content_lengths = []
    metadata_stats = {}

    for doc in documents:
        content_lengths.append(len(doc.page_content))
        metadata_fields.update(doc.metadata.keys())

        # Count frequency of each metadata field
        for field, value in doc.metadata.items():
            if field not in metadata_stats:
                metadata_stats[field] = {'count': 0, 'sample_values': set()}
            metadata_stats[field]['count'] += 1
            if len(metadata_stats[field]['sample_values']) < 3:  # Keep up to 3 sample values
                metadata_stats[field]['sample_values'].add(str(value)[:100])  # Truncate long values

    # Print overall statistics
    logger.info("\nDocument Statistics:")
    logger.info(f"Total documents: {len(documents)}")
    logger.info(f"Average content length: {sum(content_lengths) / len(documents):.0f} characters")
    logger.info(f"Total unique metadata fields: {len(metadata_fields)}")

    # Print metadata field statistics
    logger.info("\nMetadata Field Coverage:")
    for field, stats in sorted(metadata_stats.items()):
        coverage = (stats['count'] / len(documents)) * 100
        logger.info(f"\n{field}:")
        logger.info(f"  Present in {stats['count']}/{len(documents)} documents ({coverage:.1f}%)")
        logger.info(f"  Sample values: {', '.join(stats['sample_values'])}")

    # Sample document inspection
    logger.info(f"\n=== Detailed Sample Document Inspection ({num_samples} samples) ===")
    for i, doc in enumerate(random.sample(documents, min(num_samples, len(documents)))):
        logger.info(f"\nDocument Sample {i + 1}:")
        logger.info("Content Preview:")
        content_lines = doc.page_content.split('\n')
        preview_lines = content_lines[:5] if len(content_lines) > 5 else content_lines
        for line in preview_lines:
            if line.strip():
                logger.info(f"  {line[:100]}...")

        logger.info("\nMetadata:")
        for key, value in sorted(doc.metadata.items()):
            logger.info(f"  {key}: {type(value).__name__} = {str(value)[:100]}")


def parse_metadata(metadata_content: Any) -> Dict:
    """Parse metadata content from various possible formats."""
    if isinstance(metadata_content, np.ndarray):
        # Handle numpy array
        if metadata_content.dtype.kind == 'S' or metadata_content.dtype.kind == 'U':
            # Handle string or unicode array
            metadata_str = metadata_content.item().decode() if metadata_content.dtype.kind == 'S' else metadata_content.item()
        else:
            # Handle other numpy array types
            metadata_str = str(metadata_content.tolist())
    elif isinstance(metadata_content, bytes):
        metadata_str = metadata_content.decode('utf-8')
    else:
        metadata_str = str(metadata_content)

    try:
        return json.loads(metadata_str)
    except json.JSONDecodeError:
        return {"raw_metadata": metadata_str}


def format_content(content: Any) -> str:
    """Format content from various possible formats."""
    if isinstance(content, np.ndarray):
        if content.dtype.kind in ['S', 'U']:
            return content.item().decode() if content.dtype.kind == 'S' else content.item()
        return str(content.tolist())
    elif isinstance(content, bytes):
        return content.decode('utf-8')
    return str(content)


def display_sample_documents(vectorstore, num_samples: int = 3) -> None:
    """Display detailed information about sample documents from the vectorstore."""
    print("\n=== Sample Documents from Vectorstore ===")

    try:
        ds = vectorstore.vectorstore.dataset
        total_docs = len(ds)
        print(f"\nTotal documents in vectorstore: {total_docs}")

        sample_indices = random.sample(range(total_docs), min(num_samples, total_docs))

        for idx in sample_indices:
            print(f"\n{'=' * 80}")
            print(f"Sample {idx + 1} of {num_samples}:")
            print(f"{'=' * 80}")

            # Display text content
            print("\nContent:")
            print("-" * 40)
            text_content = format_content(ds.text[idx].numpy())
            content_preview = text_content[:300]
            for line in content_preview.split('\n'):
                if line.strip():
                    print(f"{line}")
            if len(text_content) > 300:
                print("...")
            print("-" * 40)

            # Display metadata
            print("\nMetadata:")
            print("-" * 40)
            try:
                metadata = parse_metadata(ds.metadata[idx].numpy())
                for key in sorted(metadata.keys()):
                    value = metadata[key]
                    if isinstance(value, str) and len(value) > 100:
                        value = value[:100] + "..."
                    print(f"{key}: {value}")
            except Exception as e:
                print(f"Error parsing metadata: {str(e)}")
            print("-" * 40)

            # Display embedding info
            embedding = ds.embedding[idx].numpy()
            print(f"\nEmbedding shape: {embedding.shape}")
            print(f"Embedding statistics:")
            print(f"  Min: {embedding.min():.3f}")
            print(f"  Max: {embedding.max():.3f}")
            print(f"  Mean: {embedding.mean():.3f}")
            print(f"  Std: {embedding.std():.3f}")

            print(f"\n{'=' * 80}")

    except Exception as e:
        print(f"Error displaying sample documents: {e}")
        raise

def test_vectorstore_creation(test_data_dir: str, embedding_model: str, chunk_size: int = 100) -> bool:
    """Test vectorstore creation process with document chunking."""
    logger = logging.getLogger('TestVectorstore')
    logger.info(f"\nTesting vectorstore creation with: Embedding={embedding_model}")

    try:
        test_dataset_path = os.path.join(test_data_dir, f'test_{embedding_model}_dataset_chunk_{chunk_size}')
        logger.info(f"Test dataset path: {test_dataset_path}")

        # Process metadata
        logger.info("Starting metadata processing...")
        metadata = process_metadata(test_data_dir)
        logger.info(f"Processed metadata for {len(metadata)} test documents")

        # Initialize retriever and vectorstore
        logger.info("\nInitializing retriever and vectorstore...")
        text_retriever = RAGRetriever(
            dataset_path=test_dataset_path,
            model_name=embedding_model,
            logger=logger
        )
        text_retriever.initialize_vectorstore(delete_existing=True)

        # Load documents
        logger.info("Loading documents...")
        documents = text_retriever.load_data(test_data_dir, metadata)

        if not documents:
            logger.error("No test documents loaded")
            return False

        logger.info(f"Loaded {len(documents)} original documents")

        # Chunk documents
        logger.info(f"\nChunking documents with size {chunk_size}...")
        chunked_documents = chunk_documents(documents, chunk_size, logger)

        # Inspect chunks
        logger.info("\nInspecting chunked documents...")
        inspect_documents(chunked_documents, logger)

        # Generate embeddings
        logger.info("\nGenerating embeddings for chunks...")
        text_retriever.generate_embeddings(chunked_documents)

        # Verify vectorstore contents
        logger.info("\nVerifying vectorstore contents...")
        if text_retriever.vectorstore is not None:
            ds = text_retriever.vectorstore.vectorstore.dataset
            logger.info("\nVectorstore Statistics:")
            logger.info(f"Original documents: {len(documents)}")
            logger.info(f"Total chunks: {len(chunked_documents)}")
            logger.info(f"Tensors in vectorstore: {len(ds)}")
            logger.info("\nTensor sizes:")
            for tensor_name in ds.tensors:
                logger.info(f"  {tensor_name}: {len(ds.tensors[tensor_name])}")

            # Display sample documents
            display_sample_documents(text_retriever.vectorstore)

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
    """Main test execution function with verbose output and document sampling."""
    print("\n=== Starting Vectorstore Creation Test ===")
    print(f"Current working directory: {os.getcwd()}")

    # Set up logging
    log_dir = os.path.join(project_root, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    print(f"Log directory: {log_dir}")

    # Configure logging to both file and console
    log_file = os.path.join(log_dir, f'vectorstore_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)  # Explicitly write to stdout
        ]
    )
    logger = logging.getLogger('TestVectorstore')
    logger.info("Logger initialized")
    print(f"Logging to: {log_file}")

    # Test configurations
    embedding_models = ['instructor']
    test_percentage = 1.0  # Test with 1% of the data
    chunk_size = 250
    print(f"\nTest Configuration:")
    print(f"Embedding models: {embedding_models}")
    print(f"Test percentage: {test_percentage}%")
    print(f"Chunk size: {chunk_size}")

    # Set up test environment
    original_data_dir = os.path.join(project_root, 'data')
    test_data_dir = os.path.join(project_root, 'test_data')
    print(f"\nDirectories:")
    print(f"Original data: {original_data_dir}")
    print(f"Test data: {test_data_dir}")

    try:
        print("\nStarting test setup...")
        stats = create_test_directory(original_data_dir, test_data_dir, test_percentage)
        print("Test directory created")
        print("\nTest directory statistics:")
        for key, value in stats.items():
            print(f"{key}: {value}")

        # Run tests for each embedding model
        print("\nStarting vectorstore tests...")
        results = {}
        text_retrievers = {}  # Store retriever instances for later use

        for embedding_model in embedding_models:
            print(f"\nTesting {embedding_model} embedding model...")
            success = test_vectorstore_creation(
                test_data_dir,
                embedding_model=embedding_model,
                chunk_size=chunk_size
            )
            results[embedding_model] = success
            print(f"{embedding_model} test completed: {'SUCCESS' if success else 'FAILED'}")

        # Results summary
        print("\n=== Test Results Summary ===")
        for model, success in results.items():
            print(f"{model} embedding model: {'SUCCESS' if success else 'FAILED'}")
        print(f"\nTested on {stats['total']} files ({test_percentage}% of total dataset)")

        # Pause for manual inspection
        print("\nCheck the log file for detailed inspection results:")
        print(f"Log file: {log_file}")
        input("Press Enter to proceed with cleanup...")

    except Exception as e:
        print(f"\nError during test execution: {str(e)}")
        logger.error("Test execution failed:", exc_info=True)

    finally:
        # Cleanup test directory
        print("\n=== Cleaning Up ===")
        if os.path.exists(test_data_dir):
            try:
                shutil.rmtree(test_data_dir)
                print("Test directory cleaned up")
            except Exception as e:
                print(f"Error cleaning up test directory: {str(e)}")
        print("\nTest script completed")


if __name__ == "__main__":
    main()