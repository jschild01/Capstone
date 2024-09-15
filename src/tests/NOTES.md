# TO DO and Be aware of
1. Try different more powerful SOTA models on AWS GPUs
2. More Q&A testing on more documents
3. Directory needs to be set to where the txt data DataFrame is
4. This DOES NOT do the text extraction. The pandas dataframe containing all of the text data from txt files was generated before, so we still need this. 


# Text Processing, Retrieval, and Generation using Python

This document explains a Python script that involves text processing, retrieval, and generation using various natural language processing (NLP) libraries. The script is organized into four main classes:

1. **TextProcessor**
2. **TextRetriever**
3. **QAGenerator**
4. **RAGPipeline**

## Libraries and Dependencies

The script imports several libraries and dependencies, including:
- `os`, `re`, `string`, `glob`: Standard Python libraries for file handling, regular expressions, and string manipulation.
- `pandas`, `numpy`: Libraries for data manipulation.
- `sentence_transformers`: For generating text embeddings.
- `faiss`: A library for efficient similarity search.
- `spacy`, `nltk`: Libraries for natural language processing.
- `transformers`: For using pre-trained models for text generation.
- `fuzzywuzzy`: For fuzzy string matching.
- `warnings`: To manage warning messages.

## Class Breakdown

### 1. TextProcessor

This class is responsible for preprocessing text data. It removes unnecessary characters, URLs, tags, and more. It provides the user the option to lowercase the text, remove punctuation and stopwords, lemmatize, and filter out invalid sentences that do not have a subject or verb. Testing showed the optimal preprocessing was to set each of those to False. 

#### Methods:
- `__init__()`: Initializes the `WordNetLemmatizer`, stop words, and the `spaCy` NLP pipeline.
- `preprocess(text, lowercase, filter_invalid_sents, remove_punctuation, remove_stopwords, use_lemma)`: Cleans the text data with options for lowercasing, removing punctuation, stopwords, lemmatization, and filtering invalid sentences.
- `is_valid_sentence(sentence)`: Checks if a sentence is valid by verifying the presence of a subject and verb.
- `filter_valid_sentences(text)`: Filters out sentences that are not valid.

### 2. TextRetriever

This class is responsible for loading data, generating embeddings for the text, and searching a vector store for relevant text passages based on a query.

#### Methods:
- `__init__()`: Initializes the `SentenceTransformer` model for generating text embeddings.
- `load_data(filepath)`: Loads data from a CSV file.
- `generate_embeddings()`: Generates embeddings for the text data and stores them in a `faiss` index.
- `search_vector_store(query, top_k)`: Searches for the top `k` relevant text passages based on a query using cosine similarity.

### 3. QAGenerator

This class is responsible for generating text responses using pre-trained models from the `transformers` library.

#### Methods:
- `__init__(model_name)`: Initializes the tokenizer and model based on the provided model name.
- `generate_response(query, most_relevant_passage, max_new_tokens, temperature, top_p)`: Generates a response based on a query and the most relevant passage.

### 4. RAGPipeline

The `RAGPipeline` (Retrieval-Augmented Generation Pipeline) class combines text retrieval and text generation to create a cohesive question-answering pipeline.

#### Methods:
- `__init__(text_retriever, qa_generator)`: Initializes the text retriever and question-answering generator.
- `run(query, top_k)`: Runs the pipeline for a given query. It retrieves relevant passages, identifies the most relevant passage, and generates a response.

## Script Setup and Execution

1. **Setup**: Change the working directory to the location of the data files.
2. **Text Processing**: Initialize `TextProcessor`, `TextRetriever`, and `QAGenerator` classes.
3. **Data Loading**: Load the text data from a CSV file.
4. **Preprocessing**: Apply text preprocessing to clean the text data.
5. **Embedding Generation**: Generate embeddings for the cleaned text data.
6. **Pipeline Execution**: Create an instance of `RAGPipeline` and run it with a sample query.

### Example Usage

The script is tested with a query about the launch dates of Voyager I and II:

```python
query = 'When were the Voyager I and Voyager II launched?'
relevant_passages, most_relevant_passage, response, most_relevant_passage_filename = rag_pipeline.run(query, top_k=3)

print(f"Retrieved Passages (3x):\n", relevant_passages)
print()
print(f"Most Relevant Passage Used for Response from file {most_relevant_passage_filename}:\n", most_relevant_passage)

print()
print(f"RAG Response:\n", response)
