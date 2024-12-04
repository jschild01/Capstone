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

# Get bedrock configuration
config = retriever_instance.load_configuration(config_dir)

# Assertions for configuration
assert 'BedRock_LLM_API' in config, "Config should contain a 'BedRock_LLM_API' section."
assert 'aws_access_key_id' in config['BedRock_LLM_API'], "Config should contain 'aws_access_key_id' under 'BedRock_LLM_API'."
assert 'aws_secret_access_key' in config['BedRock_LLM_API'], "Config should contain 'aws_secret_access_key' under 'BedRock_LLM_API'."
assert 'aws_session_token' in config['BedRock_LLM_API'], "Config should contain 'aws_session_token' under 'BedRock_LLM_API'."
assert config['BedRock_LLM_API']['aws_access_key_id'], "AWS access key ID should not be empty."
assert config['BedRock_LLM_API']['aws_secret_access_key'], "AWS secret access key should not be empty."
assert config['BedRock_LLM_API']['aws_session_token'], "AWS session token should not be empty."

# Establish bedrock connection
bedrock_client = retriever_instance.create_bedrock_client(config)

# Assertions for bedrock client
assert bedrock_client is not None, "Bedrock client should not be None."
assert hasattr(bedrock_client, 'invoke_model'), "Bedrock client should have an 'invoke_model' method."
assert callable(bedrock_client.invoke_model), "'invoke_model' should be callable."

# Test the bedrock connection
try:
    client_description = bedrock_client.meta.service_model.service_name
    assert client_description == "bedrock-runtime", "Bedrock client should describe itself correctly."
except Exception as e:
    raise AssertionError(f"Bedrock client creation test failed: {e}")

print("\nAll tests passed successfully. Bedrock connected.\n")
