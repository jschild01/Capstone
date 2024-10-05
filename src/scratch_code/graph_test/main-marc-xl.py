import os
import sys
import time
import torch
import gc
from py2neo import Graph

# Add the parent and grandparent directories to the sys path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname((os.path.dirname(os.path.abspath(__file__)))))))
from component.rag_retriever_marc_xl import RAGRetriever
from component.rag_generator_marc_xl import RAGGenerator
from component.rag_pipeline_marc_xl import RAGPipeline
from component.metadata_processor import process_metadata

# Set up a connection to your Neo4j instance
# Replace 'localhost', '7687', 'neo4j', and 'password' with your own Neo4j details
graph = Graph("neo4j+s://767b9d2a.databases.neo4j.io", auth=("neo4j", "JdfbXB1ZSMap1D1L91VmahNhwUAntGAk6PQ8iUXSVU0"))


def query_knowledge_graph(entity: str) -> str:
    """
    Queries the Neo4j knowledge graph for entities related to the given entity.
    
    :param entity: The entity for which to find related entities.
    :return: A string listing related entities or a message indicating no relations found.
    """
    # Cypher query to find related entities
    query = """
    MATCH (e:Entity {name: $entity})-[:RELATED_TO]->(related:Entity)
    RETURN related.name as related_entity
    """
    
    # Run the query and fetch results from the Neo4j graph database
    results = graph.run(query, entity=entity)
    
    # Collect related entities from the query result
    related_entities = [record["related_entity"] for record in results]
    
    if related_entities:
        # If related entities are found, return them as a comma-separated list
        return f"Related Entities: {', '.join(related_entities)}"
    else:
        # If no related entities are found, return a message indicating so
        return "No related entities found in the knowledge graph."

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

        # Enrich the metadata with knowledge graph information
        print("\nEnriching metadata with knowledge graph data...")
        for doc_id, doc_metadata in metadata.items():
            entity = doc_metadata.get('title', 'Unknown Title')
            related_entities = query_knowledge_graph(entity)
            doc_metadata['related_entities'] = related_entities  # Add the knowledge graph info

        # Print sample documents to verify metadata
        text_retriever.print_sample_documents(num_samples=5)

        # Example: Sort documents by date
        print("\nSorting documents by date:")
        text_retriever.print_sorted_documents(sort_key='date', reverse=True, limit=5)

        # Example: Filter documents by a specific subject
        print("\nFiltering documents with subject 'Music':")
        music_documents = text_retriever.query_metadata(
            filter_func=lambda x: 'Music' in x.get('subjects', []),
            limit=5
        )
        for i, doc in enumerate(music_documents, 1):
            print(f"\nDocument {i}:")
            print(f"  Title: {doc.get('title', 'N/A')}")
            print(f"  Date: {doc.get('date', 'N/A')}")
            print(f"  Subjects: {doc.get('subjects', 'N/A')}")
            print(f"  Related Entities (from knowledge graph): {doc.get('related_entities', 'N/A')}")

        print("Initializing RAG Generator...")
        qa_generator = RAGGenerator(model_name='llama3') # choose either 't5' or 'llama3'
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
        print(f"  Related Entities (from knowledge graph): {most_relevant_metadata.get('related_entities', 'N/A')}")

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
    base_path = '/home/ubuntu/Capstone-5'
    #base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_path, 'data', 'marc-xl-data')

    query = input("\nEnter your question: ")
    test_rag_system(data_dir, query)
