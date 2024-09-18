import os
import sys
import time
import pandas as pd
import torch
import numpy as np
import random

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import from the component directory
from component.text_processor import TextProcessor
from component.text_retriever import TextRetriever
from component.qa_generator import QAGenerator
from component.rag_pipeline import RAGPipeline

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def test_chunking_method(chunking_method, chunking_params, df, query):
    text_retriever = TextRetriever(chunking_method=chunking_method, chunking_params=chunking_params)
    text_retriever.load_data(df)
    text_retriever.generate_embeddings()

    qa_generator = QAGenerator(model_name='google/flan-t5-small')
    rag_pipeline = RAGPipeline(text_retriever, qa_generator)

    start_time = time.time()
    relevant_passages, most_relevant_passage, response, most_relevant_passage_filename = rag_pipeline.run(query, top_k=3)
    end_time = time.time()
    total_time = end_time - start_time

    print(f"\nChunking Method: {chunking_method}")
    print(f"Chunking Params: {chunking_params}")
    print(f"Retrieved Passages (3x):\n", relevant_passages)
    print(f"Most Relevant Passage Used for Response from file {most_relevant_passage_filename}:\n", most_relevant_passage)
    print(f"RAG Response:\n", response)
    print(f"Total processing time: {total_time:.2f} seconds")
    print("-" * 50)

def main():
    # Set seed for reproducibility
    set_seed(42)

    # Configuration
    input_csv = 'subset_for_examine100.csv'

    # Set up
    text_processor = TextProcessor()

    # Load the CSV file
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found.")
        return

    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)

    # Preprocess the text content for RAG
    print("Preprocessing text content...")
    df['clean_text'] = df['text'].apply(text_processor.preprocess)

    # Test different chunking methods
    query = input("Enter your question: ")

    chunking_methods = [
        ('sentence', {'max_chunk_size': 1000, 'overlap': 100}),
        ('words', {'max_words': 200, 'overlap': 20}),
        ('paragraphs', {'max_paragraphs': 3, 'overlap': 1}),
        ('fixed_size', {'chunk_size': 1000, 'overlap': 100})
    ]

    for method, params in chunking_methods:
        test_chunking_method(method, params, df, query)

if __name__ == "__main__":
    main()
