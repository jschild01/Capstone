import os
import sys
import time
import pandas as pd

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import from the component directory
from component.text_processor import TextProcessor
from component.text_retriever import TextRetriever
from component.qa_generator import QAGenerator
from component.rag_pipeline import RAGPipeline

def main():
    # Configuration
    input_csv = 'subset_for_examine100.csv'

    # Set up
    text_processor = TextProcessor()
    text_retriever = TextRetriever()
    qa_generator = QAGenerator(model_name='google/flan-t5-small')

    # Load the CSV file
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found.")
        return

    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)

    # Preprocess the text content for RAG
    print("Preprocessing text content...")
    df['clean_text'] = df['text'].apply(text_processor.preprocess)

    # Load and process data for RAG pipeline
    text_retriever.load_data(df)
    text_retriever.generate_embeddings()

    # Set up RAG pipeline
    rag_pipeline = RAGPipeline(text_retriever, qa_generator)

    # Test the RAG pipeline
    while True:
        query = input("\nEnter your question (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break

        start_time = time.time()

        relevant_passages, most_relevant_passage, response, most_relevant_passage_filename = rag_pipeline.run(query, top_k=3)

        end_time = time.time()
        total_time = end_time - start_time

        print(f"\nRetrieved Passages (3x):\n", relevant_passages)
        print(f"\nMost Relevant Passage Used for Response from file {most_relevant_passage_filename}:\n", most_relevant_passage)
        print(f"\nRAG Response:\n", response)
        print(f"\nTotal processing time: {total_time:.2f} seconds")

    # Print summary
    print(f'\nTotal number of items processed: {len(df)}')

if __name__ == "__main__":
    main()
