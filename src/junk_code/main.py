import os
import sys
import time
import pandas as pd
import torch
import numpy as np
import random
import gc

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import from the component directory
#from component.rag_text_processor import TextProcessor
from component.rag_retriever import RAGRetriever
from component.rag_generator import RAGGenerator
from component.rag_pipeline import RAGPipeline

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def test_chunking_method(chunk_size, chunk_overlap, df, query):
    # model can be 'mpnet' or 'instructor-xl'; use AWS GPU for instructor xl 
    text_retriever = RAGRetriever(model_name='instructor-xl', chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    text_retriever.load_data(df)
    text_retriever.generate_embeddings()

    qa_generator = RAGGenerator(model_name='google/flan-t5-small')
    rag_pipeline = RAGPipeline(text_retriever, qa_generator)

    start_time = time.time()
    relevant_passages, most_relevant_passage, response, most_relevant_passage_filename = rag_pipeline.run(query, top_k=3)
    end_time = time.time()
    total_time = end_time - start_time

    print(f"\nChunk Size: {chunk_size}, Chunk Overlap: {chunk_overlap}")
    print(f"\nRetrieved Passages (3x):\n", relevant_passages)
    print(f"\nMost Relevant Passage Used for Response from file {most_relevant_passage_filename}:\n", most_relevant_passage)
    print(f"\nRAG Response:\n", response)
    print(f"\nTotal processing time: {total_time:.2f} seconds")
    print("-" * 50)

    # Clear memory
    del text_retriever, qa_generator, rag_pipeline
    gc.collect()
    torch.cuda.empty_cache()

def main():
    # Set seed for reproducibility
    set_seed(42)

    # Configuration
    # direcotries
    base_path = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(base_path)
    grandparent = os.path.dirname(parent)
    input_csv = os.path.join(parent, 'data', 'subset_for_examine100.csv')

    # Load the CSV file
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found.")
        return

    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)

    # Test different chunking parameters
    query = input("\nEnter your question: ")

    chunking_params = [
        (1000, 200),
        (500, 100),
        (1500, 300),
    ]

    for chunk_size, chunk_overlap in chunking_params:
        test_chunking_method(chunk_size, chunk_overlap, df, query)
        # Clear some memory after each method
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
