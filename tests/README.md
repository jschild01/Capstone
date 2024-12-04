
# Testing Overview

The `/tests` directory contains four scripts designed to test the functionality of this project's main code. 

## Files and Descriptions

1. **`bedrock_connection_test.py`**:
   - Tests the connection to Amazon Bedrock and verifies the Bedrock LLM API configuration.
   - Performs connection tests to validate Bedrock runtime availability.

2. **`generator_test.py`**:
   - Validates the text generation component of the RAG system.
   - Tests the generation of responses using a pre-configured query.
   - Confirms that the generator produces a valid string response, and provides the response for manual inspection.

3. **`retriever_test.py`**:
   - Evaluates the functionality of the retrieval component.
   - Runs queries against a vector store and retrieves top-k results.
   - Checks for expected filenames among the retrieval results to ensure accuracy.

4. **`vectorstore_access_test.py`**:
   - Tests access to vector stores located in the `/data/vector_stores` directory.
   - Lists available vector stores.


## Requirements

The scripts assume the following directory structure:
```
project_root/
├── config/
├── data/
│   └── vector_stores/
├── demo/
└── src/
    └── components/
```

## Usage

1. **Testing Bedrock Connection**:
   Run `bedrock_connection_test.py` to verify AWS configuration and Bedrock connectivity.

2. **Generator Test**:
   Run `generator_test.py` to test text generation capabilities.

3. **Retriever Test**:
   Execute `retriever_test.py` to validate retrieval processes and ensure correct results.

4. **Vector Store Access Test**:
   Use `vectorstore_access_test.py` to list and verify vector stores.

## Notes

- Ensure the correct setup of AWS credentials in the `/config` directory before running the scripts.
- Verify that the required dependencies are installed and paths are correctly configured. Refer to the `requirements.txt` file in this project's root directory.

## License
This project is licensed under the MIT License. See the LICENSE file in the root directory for details.

