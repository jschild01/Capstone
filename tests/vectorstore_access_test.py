import os
import sys

# Set base root directory; this should be the dir that holds the main config, src, data directories
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
data_dir = os.path.join(project_root, 'data')
demo_dir = os.path.join(project_root, 'demo')
vstore_dir = os.path.join(data_dir, 'vector_stores')
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

# Initiate retriever instance for functions
retriever_instance = retriever(project_root=project_root)

# Parameters/requirements
config = retriever_instance.load_configuration(config_dir)
bedrock_client = retriever_instance.create_bedrock_client(config)

# Loop through the vector stores in the vstore_dir and print their size and length
def list_vector_store_info():
    # Ensure the directory exists
    if not os.path.exists(vstore_dir):
        print(f"Vector store directory does not exist: {vstore_dir}")
        return

    # List available vector store directories
    vector_stores = [name for name in os.listdir(vstore_dir) if os.path.isdir(os.path.join(vstore_dir, name))]
    
    # Display vstore information for each one available in the data/vector_store directory
    print(f"\nAvailable vector stores:")
    for store_name in vector_stores:
        #vectorstore, embeddor = retriever_instance.get_vector_store(store_name, bedrock_client)
        print(f"- {store_name} available for use.")
    print()
 
# Run the function
list_vector_store_info()
