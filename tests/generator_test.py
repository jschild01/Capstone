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
from generator import generator

# Parameters
query = 'Complete the following sentence with one word: "The mules are not hungry, they are lively and"'
top_k = 3
vstore_name = 'vectorstore_sample_250_instruct'
correct_filename = 'sr22a_en.txt'

# Implement HyDER
retriever_instance = retriever(project_root=project_root)
uery, bedrock_client, rr_results, rr_filenames, rr_best_filename = retriever_instance.runRetriever(query=query, 
                                                                                                   top_k=top_k, 
                                                                                                   vstore_name=vstore_name)

# Initiate generator
generator_instance = generator(query=query, bedrock_client=bedrock_client, rr_results=rr_results, rr_filenames=rr_filenames, rr_best_filename=rr_best_filename,)

# Generate the response
response, text_response = generator_instance.generator(query, bedrock_client, rr_results)

# Check if the response is a string
try:
    assert isinstance(text_response, str), "Response is not a string."
    print("\nSUCCESS: Response is a valid string. Actual generated response:]\n")
    print(text_response)
    print(f"\n\nGenerator test complete.\n")
except AssertionError:
    print(f"\nFAILURE: Response is not a string. Got type {type(response).__name__}.")
    raise


