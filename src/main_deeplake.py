import os
import sys
import time
import torch
import gc
from component.rag_retriever_deeplake import RAGRetriever
from component.rag_generator_deeplake import RAGGenerator
from component.rag_pipeline_deeplake import RAGPipeline
from component.metadata_processor import process_metadata


def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def test_rag_system(data_dir: str, query: str, delete_existing: bool = False):
    try:
        print("Starting metadata processing...")
        metadata = process_metadata(data_dir)
        print(f"Metadata processing completed. Found metadata for {len(metadata)} files.")

        print("\n--- Initializing RAG Retriever ---")
        dataset_path = os.path.join(data_dir, 'deeplake_dataset')
        text_retriever = RAGRetriever(dataset_path=dataset_path, model_name='all-MiniLM-L6-v2')

        if delete_existing:
            text_retriever.delete_dataset()

        is_empty = text_retriever.is_empty()
        print(f"Is dataset empty? {is_empty}")

        if is_empty:
            print("\n--- Loading and Embedding Documents ---")
            documents = text_retriever.load_data(data_dir, metadata)
            if not documents:
                print("No documents were loaded. Please check your data directory and file names.")
                return
            print(f"Loaded {len(documents)} documents.")

            print("Generating embeddings...")
            text_retriever.generate_embeddings(documents)
            print("Embeddings generated and saved.")
        else:
            print("Using existing DeepLake dataset.")
            text_retriever.load_data(data_dir, metadata)  # Load documents into memory

        print("\n--- Initializing RAG Generator ---")
        qa_generator = RAGGenerator(model_name='llama3')  # enter either 't5' or 'llama3'
        rag_pipeline = RAGPipeline(text_retriever, qa_generator)

        print("\n--- Processing Query ---")
        print(f"Query: {query}")

        # Handle metadata queries
        if any(keyword in query.lower() for keyword in ['print', 'show', 'list', 'what are']):
            try:
                if 'from' in query.lower():
                    year = query.lower().split('from')[-1].strip()
                    results = text_retriever.query_metadata(filter={"date": {"$regex": f".*{year}.*"}})
                else:
                    results = text_retriever.query_metadata()

                print(f"\nResults for query: '{query}'")
                for i, doc in enumerate(results, 1):
                    print(f"\nResult {i}:")
                    for key, value in doc.metadata.items():
                        if key.lower() == 'title' or ('song' in query.lower() and 'title' in key.lower()):
                            print(f"  {key}: {value}")
                    print(f"  Date: {doc.metadata.get('date', 'N/A')}")
                    print("-" * 50)
            except Exception as e:
                print(f"Error querying metadata: {e}")
                print("Attempting to print sample documents...")
                text_retriever.print_sample_documents(num_samples=5)
        else:
            try:
                print("\n--- Running RAG Pipeline ---")
                start_time = time.time()
                retrieved_docs, most_relevant_passage, response = rag_pipeline.run(query, top_k=3)
                end_time = time.time()
                total_time = end_time - start_time

                print(f"\nRetrieved Passages ({len(retrieved_docs)}):")
                for i, doc in enumerate(retrieved_docs, 1):
                    print(f"\nDocument {i}:")
                    print(f"Passage: {doc.page_content[:100]}...")
                    print(f"Metadata:")
                    for key, value in doc.metadata.items():
                        print(f"  {key}: {value}")

                print(f"\nStructured RAG Response:")
                print(response)

                print(f"\nTotal processing time: {total_time:.2f} seconds")

            except Exception as e:
                print(f"An error occurred: {str(e)}")
                import traceback
                traceback.print_exc()

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

    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_path, 'data', 'marc-xl-data')

    delete_existing = input("Do you want to delete the existing dataset? (y/n): ").lower() == 'y'
    query = input("\nEnter your question: ")

    test_rag_system(data_dir, query, delete_existing)