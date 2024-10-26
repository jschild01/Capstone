import os
import sys
from typing import Dict, Any
import json
from datetime import datetime
import deeplake

# Add the src directory to the Python path
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, src_dir)

from component.rag_retriever_deeplake import RAGRetriever

# Define paths
DATA_DIR = "/home/ubuntu/Capstone/data"
LOG_DIR = os.path.join(project_root, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)


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

    print(f"Test results logged to: {log_file}")


def inspect_dataset(dataset_path: str) -> Dict[str, Any]:
    """Inspect the DeepLake dataset contents."""
    try:
        ds = deeplake.load(dataset_path)

        info = {
            "tensors": {},
            "total_samples": 0
        }

        print("\nDataset Structure:")
        for tensor_name in ds.tensors:
            tensor = ds.tensors[tensor_name]
            shape = tensor.shape
            info["tensors"][tensor_name] = {
                "shape": shape,
                "samples": []
            }
            print(f"\nTensor: {tensor_name}")
            print(f"Shape: {shape}")

            if info["total_samples"] == 0 and len(shape) > 0:
                info["total_samples"] = shape[0]

            if tensor_name in ['text', 'metadata'] and shape[0] > 0:
                num_samples = min(5, shape[0])
                print(f"\nSampling {num_samples} entries from {tensor_name}:")
                for i in range(num_samples):
                    try:
                        sample = tensor[i].numpy()
                        if isinstance(sample, bytes):
                            sample = sample.decode('utf-8')
                        info["tensors"][tensor_name]["samples"].append(str(sample))
                        print(f"\nSample {i + 1}:")
                        print(str(sample)[:200] + "..." if len(str(sample)) > 200 else str(sample))
                    except Exception as e:
                        print(f"Error accessing sample {i} from {tensor_name}: {e}")

        return info

    except Exception as e:
        print(f"Error inspecting dataset: {e}")
        return {"error": str(e)}


def test_vectorstore_contents():
    print("\nStarting vectorstore content test...")

    dataset_path = os.path.join(DATA_DIR, 'deeplake_dataset_chunk_100')
    test_results = {
        "dataset_info": None,
        "queries": {}
    }

    try:
        print("Inspecting dataset...")
        dataset_info = inspect_dataset(dataset_path)
        test_results["dataset_info"] = dataset_info

        total_samples = dataset_info.get("total_samples", 0)
        print(f"\nTotal samples in dataset: {total_samples}")

        if total_samples == 0:
            print("Warning: No samples found in dataset")
            return False, test_results

        print("\nInitializing retriever for testing...")
        retriever = RAGRetriever(
            dataset_path=dataset_path,
            model_name='instructor'
        )
        test_queries = [
            ("Complete this sentence: 'My mules are not hungry. They're lively and'", "gay"),
            ("Complete this sentence: 'Take a trip on the canal if you want to have'", "fun"),
            ("What is the name of the female character mentioned in the song that begins 'In Scarlett town where I was born'?", "Barbrae Allen"),
            ("According to the transcript, what is Captain Pearl R. Nye's favorite ballad?", "Barbara Allen"),
            ("Complete this phrase from the gospel train song: 'The gospel train is'", "night"),
            ("In the song 'Barbara Allen,' where was Barbara Allen from?", "Scarlett town"),
            ("In the song 'Lord Lovele,' how long was Lord Lovele gone before returning?", "A year or two or three at most"),
            ("What instrument does Captain Nye mention loving?", "old fiddled mouth organ banjo"),
            ("In the song about pumping out Lake Erie, what will be on the moon when they're done?", "whiskers"),
            ("Complete this line from a song: 'We land this war down by the'", "river"),
            ("What does the singer say they won't do in the song 'I Won't Marry At All'?", "Marry at all"),
            ("What does the song say will 'outshine the sun'?", "We'll"),
            ("In the 'Dying Cowboy' song, where was the cowboy born?", "Boston")
        ]

        print("\nExecuting test queries...")
        for query, expected_content in test_queries:
            print(f"\nQuery: {query}")
            try:
                results = retriever.search_vector_store(query, top_k=3)

                query_results = {
                    "num_results": len(results),
                    "results": []
                }

                for i, doc in enumerate(results, 1):
                    result_info = {
                        "content_preview": doc.page_content[:200],
                        "metadata": doc.metadata if hasattr(doc, 'metadata') else None,
                        "contains_expected": expected_content.lower() in doc.page_content.lower()
                    }
                    query_results["results"].append(result_info)

                    print(f"\nResult {i}:")
                    print(f"Content preview: {doc.page_content[:200]}...")
                    if hasattr(doc, 'metadata') and doc.metadata:
                        print("Metadata:", doc.metadata)
                    print(f"Contains expected content: {expected_content.lower() in doc.page_content.lower()}")

                test_results["queries"][query] = query_results

            except Exception as e:
                print(f"Error during query '{query}': {e}")
                test_results["queries"][query] = {"error": str(e)}

        log_test_results("vectorstore_content_test", test_results)
        return True, test_results

    except Exception as e:
        print(f"Error in test execution: {e}")
        test_results["error"] = str(e)
        return False, test_results


if __name__ == "__main__":
    print("Starting vectorstore tests...")
    print("=" * 80)

    success, results = test_vectorstore_contents()

    if success:
        print("\nVectorstore content test completed.")
        if "dataset_info" in results:
            print(f"Total samples in dataset: {results['dataset_info'].get('total_samples', 0)}")

        print("\nQuery Results Summary:")
        for query, info in results.get("queries", {}).items():
            print(f"\nQuery: {query}")
            if "error" in info:
                print(f"Error: {info['error']}")
            else:
                print(f"Number of results: {info['num_results']}")
                matched = sum(1 for r in info['results'] if r.get('contains_expected', False))
                print(f"Results containing expected content: {matched}/{info['num_results']}")
    else:
        print("\nVectorstore content test failed.")

    print("\nTest execution completed. Check the logs directory for detailed results.")