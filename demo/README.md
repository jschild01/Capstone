# FoldRAG: Library of Congress Research Assistant

**FolkRAG** is a Streamlit-based application that acts as a research assistant for exploring resources from the Library of Congress. It leverages state-of-the-art retriever and generator models to process user queries and return relevant information from a vector store.

## Features

- **Interactive UI**: Simple and intuitive interface using Streamlit.
- **Customizable Query Parameters**:
  - Enter a query for retrieval.
  - Select the top-k documents to process.
  - Choose a vector store for query matching.
- **Retriever Integration**: Fetches the most relevant documents using pre-configured vector stores.
- **Generative Response**: Generates a human-readable response based on the retrieved documents.

## Usage
Run the following command in the terminal to execute the Streamlit application.
```bash
streamlit run src/main.py
``` 
