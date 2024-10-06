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

# Simplify the sys path append logic
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

# Set up a connection to your Neo4j instance (use environment variables for security)
neo4j_url = os.getenv("NEO4J_URL", "neo4j+s://767b9d2a.databases.neo4j.io")
neo4j_user = os.getenv("NEO4J_USER", "neo4j")
neo4j_password = os.getenv("NEO4J_PASSWORD", "JdfbXB1ZSMap1D1L91VmahNhwUAntGAk6PQ8iUXSVU0")
graph = Graph(neo4j_url, auth=(neo4j_user, neo4j_password))

def query_knowledge_graph(entity: str) -> str:
    query = """
    MATCH (e:Entity)
    WHERE trim(toLower(e.name)) = trim(toLower($entity))
    OPTIONAL MATCH (e)-[:RELATED_TO]->(related:Entity)
    RETURN related.name as related_entity
    """
    
    try:
        results = graph.run(query, entity=entity)
        related_entities = [record["related_entity"] for record in results if record["related_entity"]]
        return f"Related Entities: {', '.join(related_entities)}" if related_entities else "No related entities found."
    except Exception as e:
        print(f"Error querying Neo4j: {str(e)}")
        return "Error querying knowledge graph."

def enrich_metadata_with_kg(metadata):
    print("\nEnriching metadata with knowledge graph data...")
    for doc_id, doc_metadata in metadata.items():
        entity = doc_metadata.get('title', 'Unknown Title')
        print(f"Querying knowledge graph for entity: {entity}")
        related_entities = query_knowledge_graph(entity)
        print(f"Related entities found: {related_entities}")
        doc_metadata['related_entities'] = related_entities
    return metadata

def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def test_rag_system(data_dir: str, query: str):
    """
    Test the RAG (Retrieval-Augmented Generation) system with the specified query.
    """
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
            allow_deserialization=True
        )

        if text_retriever.vectorstore is None:
            print("Loading documents...")
            documents = text_retriever.load_data(data_dir, metadata)
            if not documents:
                print("No documents were loaded. Please check your data directory.")
                return
            print(f"Loaded {len(documents)} documents.")
            print("Generating embeddings...")
            text_retriever.generate_embeddings(documents)
            print("Embeddings generated and saved.")
        else:
            print("Using existing vectorstore.")

        # Enrich metadata with knowledge graph information once
        metadata = enrich_metadata_with_kg(metadata)

        text_retriever.print_sample_documents(num_samples=5)
        text_retriever.print_sorted_documents(sort_key='date', reverse=True, limit=5)

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
            print(f"  Related Entities: {doc.get('related_entities', 'N/A')}")

        print("Initializing RAG Generator...")
        qa_generator = RAGGenerator(model_name='llama3')
        rag_pipeline = RAGPipeline(text_retriever, qa_generator)

        print(f"Processing query: {query}")
        start_time = time.time()
        retrieved_docs, most_relevant_passage, response = rag_pipeline.run(query, top_k=3)
        total_time = time.time() - start_time

        print(f"\nRetrieved Passages:")
        for doc, score in retrieved_docs:
            print(f"Passage: {doc.page_content[:100]}...")
            for key, value in doc.metadata.items():
                print(f"  {key}: {value}")
            print(f"Score: {score}\n")

        print(f"\nMost Relevant Passage: {most_relevant_passage}")
        print(f"\nRAG Response: {response}")
        print(f"\nTotal processing time: {total_time:.2f} seconds")
        print("-" * 50)

        # Clean up
        del text_retriever, qa_generator, rag_pipeline
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    set_seed(42)
    base_path = '/home/ubuntu/Capstone-5'
    data_dir = os.path.join(base_path, 'data', 'marc-xl-data')
    query = input("\nEnter your question: ")
    test_rag_system(data_dir, query)
