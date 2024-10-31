import os
import sys
from typing import Dict, Any, List, Tuple
import json
from datetime import datetime
import deeplake
from langchain.schema import Document
from langchain_community.vectorstores import DeepLake
import shutil

# Add the src directory to the Python path
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, src_dir)

from component.rag_retriever_deeplake import RAGRetriever
from component.logging_config import setup_logging

# Define paths
DATA_DIR = "/home/ubuntu/Capstone/data"
LOG_DIR = os.path.join(project_root, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

logger = setup_logging()


def get_directory_size(path: str) -> float:
    """Calculate total size of a directory in MB."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size / 1024 / 1024  # Convert to MB


def log_test_results(test_name: str, results: Dict[str, Any]):
    """Log test results to a JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"vectorstore_test_{timestamp}.json")

    log_data = {
        "test_name": test_name,
        "timestamp": timestamp,
        "results": results
    }

    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)

    logger.info(f"Test results logged to: {log_file}")


def inspect_dataset(dataset_path: str) -> Dict[str, Any]:
    """Inspect the DeepLake dataset contents in detail."""
    try:
        logger.info(f"\nInspecting dataset at: {dataset_path}")
        total_size = get_directory_size(dataset_path)
        logger.info(f"Total dataset size on disk: {total_size:.2f} MB")

        ds = deeplake.load(dataset_path)

        info = {
            "tensors": {},
            "total_samples": 0,
            "storage_info": {},
            "directory_structure": {}
        }

        # Get storage details
        storage_path = os.path.join(dataset_path, 'storage')
        if os.path.exists(storage_path):
            storage_files = os.listdir(storage_path)
            logger.info("\nStorage contents:")
            for file in storage_files:
                file_path = os.path.join(storage_path, file)
                size = os.path.getsize(file_path)
                logger.info(f"- {file}: {size / 1024 / 1024:.2f} MB")
                info["storage_info"][file] = size

        # Inspect tensors
        logger.info("\nTensor Details:")
        for tensor_name in ds.tensors:
            tensor = ds.tensors[tensor_name]
            shape = tensor.shape
            dtype = tensor.dtype if hasattr(tensor, 'dtype') else 'unknown'
            storage_size = 0

            # Try to get tensor storage size
            tensor_path = os.path.join(storage_path, tensor_name) if os.path.exists(storage_path) else None
            if tensor_path and os.path.exists(tensor_path):
                storage_size = get_directory_size(tensor_path)

            info["tensors"][tensor_name] = {
                "shape": shape,
                "dtype": str(dtype),
                "storage_size_mb": storage_size,
                "samples": []
            }

            logger.info(f"\nTensor: {tensor_name}")
            logger.info(f"Shape: {shape}")
            logger.info(f"Type: {dtype}")
            logger.info(f"Storage size: {storage_size:.2f} MB")

            if info["total_samples"] == 0 and len(shape) > 0:
                info["total_samples"] = shape[0]

            # Sample first few elements if tensor is not empty
            if shape and shape[0] > 0:
                try:
                    sample_size = min(2, shape[0])
                    logger.info(f"First {sample_size} elements:")
                    for i in range(sample_size):
                        sample = tensor[i].numpy()
                        if isinstance(sample, bytes):
                            sample = sample.decode('utf-8')
                        if isinstance(sample, (list, tuple)) and len(sample) > 100:
                            sample = f"{str(sample[:100])}... (truncated)"
                        logger.info(f"Element {i}: {sample}")
                        info["tensors"][tensor_name]["samples"].append(str(sample))
                except Exception as e:
                    logger.error(f"Error sampling tensor {tensor_name}: {e}")

        # Check dataset directory structure
        logger.info("\nDataset Directory Structure:")
        for root, dirs, files in os.walk(dataset_path):
            level = root.replace(dataset_path, '').count(os.sep)
            indent = ' ' * 4 * level
            subpath = root.replace(dataset_path, '').lstrip(os.sep)
            info["directory_structure"][subpath] = {"files": {}}

            logger.info(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 4 * (level + 1)
            for f in files:
                size = os.path.getsize(os.path.join(root, f))
                logger.info(f"{subindent}{f}: {size / 1024 / 1024:.2f} MB")
                info["directory_structure"][subpath]["files"][f] = size

        return info

    except Exception as e:
        logger.error(f"Error inspecting dataset: {e}")
        return {"error": str(e)}


def test_sample_queries(retriever: RAGRetriever) -> List[Tuple[str, int, bool]]:
    """Run a series of test queries and return results."""
    test_queries = [
        (
        "Complete this sentence from the Captain Pearl R. Nye collection (AFC 1937/002): 'My mules are not hungry. They're lively and'",
        "gay"),
        ("Complete this sentence: 'Take a trip on the canal if you want to have'", "fun"),
        (
        "What is the name of the female character mentioned in the song that begins 'In Scarlett town where I was born'?",
        "Barbrae Allen"),
        ("According to the transcript, what is Captain Pearl R. Nye's favorite ballad?", "Barbara Allen"),
        ("Complete this phrase from the gospel train song: 'The gospel train is'", "night"),
        ("In the song 'Barbara Allen,' where was Barbara Allen from?", "Scarlett town"),
        (
        "In the song 'Lord Lovele,' how long was Lord Lovele gone before returning?", "A year or two or three at most"),
        ("What instrument does Captain Nye mention loving?", "old fiddled mouth organ banjo"),
        ("In the song about pumping out Lake Erie, what will be on the moon when they're done?", "whiskers"),
        ("Complete this line from a song: 'We land this war down by the'", "river"),
        ("What does the singer say they won't do in the song 'I Won't Marry At All'?", "Marry at all"),
        ("What does the song say will 'outshine the sun'?", "We'll"),
        ("In the 'Dying Cowboy' song, where was the cowboy born?", "Boston")
    ]

    results = []
    for query, expected_content in test_queries:
        try:
            # Log search parameters
            logger.info("\n--- Search Parameters ---")
            logger.info(f"Query: {query}")
            logger.info(f"Expected content: {expected_content}")
            logger.info(f"Filter: None")
            logger.info(f"Top K: 3")

            # Perform search with detailed logging
            try:
                embedding_vector = retriever.embeddings.embed_query(query)
                logger.info(f"Generated query embedding shape: {len(embedding_vector)}")
            except Exception as e:
                logger.error(f"Error generating query embedding: {e}")
                continue

            try:
                search_results = retriever.search_vector_store(query, top_k=50)
                logger.info(f"Search completed. Found {len(search_results)} results")

                success = any(expected_content.lower() in doc.page_content.lower() for doc in search_results)
                results.append((query, len(search_results), success))

                # Log detailed results
                for i, doc in enumerate(search_results, 1):
                    logger.info(f"\nResult {i}:")
                    logger.info(f"Content preview: {doc.page_content[:200]}...")
                    if hasattr(doc, 'metadata'):
                        logger.info(f"Metadata: {json.dumps(doc.metadata, indent=2)}")
                    contains_expected = expected_content.lower() in doc.page_content.lower()
                    logger.info(f"Contains expected content: {contains_expected}")

            except Exception as e:
                logger.error(f"Error during similarity search: {e}")
                results.append((query, 0, False))

        except Exception as e:
            logger.error(f"Error processing query '{query}': {e}")
            results.append((query, 0, False))

    return results


def verify_dataset_integrity(dataset_path: str) -> bool:
    """Verify the integrity of the dataset."""
    try:
        logger.info(f"\nVerifying dataset integrity at: {dataset_path}")

        # Check if dataset exists
        if not os.path.exists(dataset_path):
            logger.error("Dataset path does not exist")
            return False

        # Load dataset
        ds = deeplake.load(dataset_path)

        # Check required tensors
        required_tensors = ['embedding', 'text', 'metadata']
        for tensor in required_tensors:
            if tensor not in ds.tensors:
                logger.error(f"Missing required tensor: {tensor}")
                return False
            if len(ds.tensors[tensor]) == 0:
                logger.error(f"Tensor {tensor} is empty")
                return False

        # Check consistency
        sizes = [len(ds.tensors[tensor]) for tensor in required_tensors]
        if len(set(sizes)) != 1:
            logger.error(f"Inconsistent tensor sizes: {dict(zip(required_tensors, sizes))}")
            return False

        # Sample check
        try:
            sample_idx = 0
            embedding = ds.embedding[sample_idx].numpy()
            text = ds.text[sample_idx].numpy()
            metadata = ds.metadata[sample_idx].numpy()

            logger.info("\nSample check results:")
            logger.info(f"Embedding shape: {embedding.shape}")
            logger.info(f"Text sample: {text[:100]}...")
            logger.info(f"Metadata sample: {metadata}")

            return True

        except Exception as e:
            logger.error(f"Error during sample check: {e}")
            return False

    except Exception as e:
        logger.error(f"Error verifying dataset: {e}")
        return False


def test_vectorstore_contents():
    """Test the vectorstore contents and functionality."""
    logger.info("\nStarting vectorstore content test...")

    dataset_path = os.path.join(DATA_DIR, 'deeplake_dataset_chunk_500')
    test_results = {
        "dataset_info": None,
        "query_results": []
    }

    try:
        # Verify dataset integrity
        if not verify_dataset_integrity(dataset_path):
            error_msg = "Dataset integrity check failed"
            logger.error(error_msg)
            return False, {"error": error_msg}

        # Inspect dataset
        logger.info("Inspecting dataset...")
        dataset_info = inspect_dataset(dataset_path)
        test_results["dataset_info"] = dataset_info

        if "error" in dataset_info:
            logger.error(f"Dataset inspection error: {dataset_info['error']}")
            return False, test_results

        total_samples = dataset_info.get("total_samples", 0)
        logger.info(f"\nTotal samples in dataset: {total_samples}")

        if total_samples == 0:
            error_msg = "Dataset is empty"
            logger.error(error_msg)
            return False, {"error": error_msg}

        # Initialize retriever
        logger.info("\nInitializing retriever...")
        retriever = RAGRetriever(
            dataset_path=dataset_path,
            model_name='instructor',
            logger=logger
        )

        # Load existing vectorstore in read-only mode
        logger.info("Loading existing vectorstore in read-only mode...")
        retriever.vectorstore = DeepLake(
            dataset_path=dataset_path,
            embedding_function=retriever.embeddings,
            read_only=True
        )

        # Run test queries
        logger.info("\nExecuting test queries...")
        query_results = test_sample_queries(retriever)
        test_results["query_results"] = [
            {
                "query": query,
                "num_results": num_results,
                "found_expected": success
            }
            for query, num_results, success in query_results
        ]

        # Log results
        log_test_results("vectorstore_content_test", test_results)

        # Calculate success metrics
        successful_queries = sum(1 for _, _, success in query_results if success)
        total_queries = len(query_results)
        success_rate = (successful_queries / total_queries) * 100 if total_queries > 0 else 0

        logger.info("\nTest Results Summary:")
        logger.info(f"Total queries: {total_queries}")
        logger.info(f"Successful queries: {successful_queries}")
        logger.info(f"Success rate: {success_rate:.2f}%")

        return True, test_results

    except Exception as e:
        logger.error(f"Error in test execution: {e}", exc_info=True)
        test_results["error"] = str(e)
        return False, test_results


if __name__ == "__main__":
    logger.info("Starting vectorstore tests...")
    logger.info("=" * 80)

    success, results = test_vectorstore_contents()

    if success:
        logger.info("\nVectorstore content test completed successfully.")

        if "dataset_info" in results:
            logger.info(f"Dataset size: {results['dataset_info'].get('total_samples', 0)} samples")

        if "query_results" in results:
            logger.info("\nDetailed Query Results:")
            for result in results["query_results"]:
                logger.info(f"\nQuery: {result['query']}")
                logger.info(f"Results found: {result['num_results']}")
                logger.info(f"Found expected content: {'Yes' if result['found_expected'] else 'No'}")
    else:
        logger.error("\nVectorstore content test failed.")
        if "error" in results:
            logger.error(f"Error message: {results['error']}")

    logger.info("\nTest execution completed. Check the logs directory for detailed results.")