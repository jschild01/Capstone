import os
import sys

# Set base root directory; this should be the dir that holds the main config, src, data directories
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
data_dir = os.path.join(project_root, 'data')
demo_dir = os.path.join(project_root, 'demo')
#vstore_dir = os.path.join(demo_dir, 'vector_stores')
config_dir = os.path.join(project_root, 'config')
src_dir = os.path.join(project_root, 'src')
components_dir = os.path.join(src_dir, 'components')

# Add paths to required component/app files
sys.path.append(demo_dir)
sys.path.append(data_dir)
sys.path.append(src_dir)
sys.path.append(components_dir)

# Get file classes/components
from retriever import retriever

# Initialize retriever
retriever_instance = retriever(project_root=project_root)

# Test retriever
def test_retriever():
    # Parameters
    query = 'Complete the following sentence with one word: "The mules are not hungry, they are lively and"'
    top_k = 3
    vstore_name = 'vectorstore_sample_250_instruct'
    correct_filename = 'sr22a_en.txt'
    
    # Run retriever
    query, bedrock_client, rr_results, rr_filenames, rr_best_filename = retriever_instance.runRetriever(
        query=query, top_k=top_k, vstore_name=vstore_name)

    # Check if the number of results is equal to top_k
    try:
        assert len(rr_results) == top_k
        print(f"SUCCESS: Number of results ({len(rr_results)}) matches top_k ({top_k}).")
    except AssertionError:
        print(f"FAILURE: Number of results ({len(rr_results)}) does not match top_k ({top_k}).")
        raise
    
    # Check if the number of filenames is equal to top_k
    try:
        assert len(rr_filenames) == top_k
        print(f"SUCCESS: Number of filenames ({len(rr_filenames)}) matches top_k ({top_k}).")
    except AssertionError:
        print(f"FAILURE: Number of filenames ({len(rr_filenames)}) does not match top_k ({top_k}).")
        raise

    # Check if the best filename is in the list of filenames
    try:
        assert correct_filename in rr_filenames
        print(f"SUCCESS: Correct filename ({correct_filename}) is in the list of filenames.")
    except AssertionError:
        print(f"FAILURE: Correct filename ({correct_filename}) is not in the list of filenames.")
        raise









