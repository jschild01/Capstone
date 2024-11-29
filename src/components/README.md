# Retriever

This project provides a Python-based retriever module for information retrieval from vector stores using embedding models. The retriever leverages AWS Bedrock, DeepLake, and LangChain for advanced query processing and document search functionalities.

## Features

- **Embedding Models**: Supports embedding models `instructor-xl` and Amazon Bedrock.
- **Vector Store**: Uses DeepLake for vector storage and retrieval.
- **HYDE Generator**: Generates hypothetical document embeddings for better query matching.
- **TF-IDF-based Reranking**: Improves retrieval accuracy by reranking results based on term relevance, freshness, and keyword coverage.
- **Customizable Vector Stores**: Configures different vector store directories with predefined settings.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/retriever.git
   cd retriever
   ```
2. Install required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Set up AWS Bedrock credentials in the `config/config.ini` file:
    ```bash
    [BedRock_LLM_API]
    AWS_ACCESS_KEY_ID=your_access_key
    AWS_SECRET_ACCESS_KEY=your_secret_key
    AWS_SESSION_TOKEN=your_session_token
    ```
4. Set up AWS Bedrock credentions in the `config/.env` file:
    ```bash
    CONFIG_FILE=config.ini
    ```

## Usage
Run the following command in the terminal to execute the Streamlit application.
```bash
streamlit run src/main.py
```    



# Retriever

### Vector Store
A vector store is selected by the user and loaded in by the application. There are five options available to choose from:
* Vector store constructed using `instructor-xl` embeddings and chunk sizes of 250 on **all** of the data. This is not widely available due to size.
* Vector store constructed using `instructor-xl` embeddings and chunk sizes of 250 on **all** of the data. This is not widely available due to size.
* Vector store constructed using `amazon/titan` embeddings and chunk sizes of 250 on **all** of the data.This is not widely available due to size.
* Vector store constructed using `instructor-xl` embeddings and chunk sizes of 250 on a **small sample** of the data. This is widely available on the git repo.
* Vector store constructed using `instructor-xl` embeddings and chunk sizes of 1,000 on a **small sample**  of the data. This is widely available on the git repo.

### HyDE Generator
The `hyde_generator` function uses AWS Bedrock's Claude model to generate and process variations of the users' query and their responses. The LLM generates an initial response for the original query, rewrites the query into slightly different variations, and generates answers for each variation. Each query-and-response is generated using progressively higher temperatures (i.e., 0.7, 0.8, 0.9) to balance coherence and creativity while ensuring variety. Finally, each query-and-response is combined into a single string, or "document", for a total of three hypothetical documents that will each be searched against the vector store. 

### Document Retrieval
The `test_document_retrieval` 

### TF-IDF Reranker
The `rerank_with_custom_tfidf` function enhances document retrieval in a Retrieval-Augmented Generation (RAG) system by scoring and ranking documents based on their relevance to a query. It first extracts the metadata and content from raw documents and calculates relevance using a combination of TF-IDF cosine similarity, document freshness (favoring newer documents), and keyword coverage from the query. These scores are weighted (50% TF-IDF, 30% freshness, 20% keyword coverage) to generate a total score for each document. The top-ranked documents are returned along with their filenames, prioritizing the most relevant and up-to-date information for query augmentation.









# License
This project is licensed under the MIT License. See the LICENSE file for more information.
