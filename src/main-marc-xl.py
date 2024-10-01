import os
import sys
import time
import torch
import gc
from component.rag_retriever_marc_xl import RAGRetriever
from component.rag_generator_marc_xl import RAGGenerator
from component.rag_pipeline_marc_xl import RAGPipeline
from component.metadata_processor import process_metadata


def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def test_rag_system(data_dir: str, query: str):
    try:
        print("Starting metadata processing...")
        metadata = process_metadata(data_dir)
        print("Metadata processing completed.")

        print("Initializing RAG Retriever...")
        vectorstore_path = os.path.join(data_dir, 'vectorstore')
        text_retriever = RAGRetriever(
            model_name='hkunlp/instructor-xl',
            chunk_size=1000,
            chunk_overlap=200,
            vectorstore_path=vectorstore_path,
            allow_deserialization=True  # Enable deserialization
        )

        if text_retriever.vectorstore is None:
            print("Loading documents...")
            documents = text_retriever.load_data(data_dir, metadata)
            if not documents:
                print("No documents were loaded. Please check your data directory and file names.")
                return
            print(f"Loaded {len(documents)} documents.")

            print("Generating embeddings...")
            text_retriever.generate_embeddings(documents)
            print("Embeddings generated and saved.")
        else:
            print("Using existing vectorstore.")

        # Print sample documents to verify metadata
        text_retriever.print_sample_documents(num_samples=5)

        print("Initializing RAG Generator...")
        qa_generator = RAGGenerator(model_name='google/flan-t5-base')
        rag_pipeline = RAGPipeline(text_retriever, qa_generator)

        print(f"Processing query: {query}")
        start_time = time.time()
        retrieved_docs, most_relevant_passage, response = rag_pipeline.run(query, top_k=3)
        end_time = time.time()
        total_time = end_time - start_time

        print(f"\nRetrieved Passages (3x):")
        for doc, score in retrieved_docs:
            print(f"Passage: {doc.page_content[:100]}...")
            print(f"Metadata:")
            for key, value in doc.metadata.items():
                print(f"  {key}: {value}")
            print(f"Score: {score}\n")

        print(f"\nMost Relevant Passage Used for Response:")
        print(most_relevant_passage)

        most_relevant_metadata = retrieved_docs[0][0].metadata
        print(f"\nMetadata for Most Relevant Passage:")
        for key, value in most_relevant_metadata.items():
            print(f"  {key}: {value}")

        print(f"\nRAG Response:")
        print(response)

        print(f"\nTotal processing time: {total_time:.2f} seconds")
        print("-" * 50)

        del text_retriever, qa_generator, rag_pipeline
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        print("Please check your data files and directory structure.")


if __name__ == "__main__":
    set_seed(42)

    # Set data directory
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_path, 'data', 'marc-xl-data')

    query = input("\nEnter your question: ")
    test_rag_system(data_dir, query)
