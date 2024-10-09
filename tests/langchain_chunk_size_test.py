import os
import sys
import time
import torch
import gc
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Add the src directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, src_dir)

from src.component.rag_retriever_deeplake import RAGRetriever
from src.component.rag_generator_deeplake import RAGGenerator
from src.component.rag_pipeline_deeplake import RAGPipeline
from src.component.metadata_processor import process_metadata
from src.component.rag_utils import generate_prompt, structure_response, integrate_metadata, validate_response


def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def evaluate_response(query: str, response: str, retrieved_docs: List[Document]) -> Dict:
    # Compute relevance score (cosine similarity between query and response)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([query, response])
    relevance_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    # Compute coherence score (average cosine similarity between consecutive sentences in the response)
    sentences = response.split('.')
    if len(sentences) > 1:
        sentence_vectors = vectorizer.transform(sentences)
        coherence_scores = [cosine_similarity(sentence_vectors[i:i + 1], sentence_vectors[i + 1:i + 2])[0][0]
                            for i in range(len(sentences) - 1)]
        coherence_score = np.mean(coherence_scores)
    else:
        coherence_score = 1.0  # Perfect coherence for single-sentence responses

    # Compute coverage score (fraction of retrieved documents referenced in the response)
    doc_contents = [doc.page_content for doc in retrieved_docs]
    doc_vectors = vectorizer.transform(doc_contents)
    response_vector = vectorizer.transform([response])
    doc_relevance_scores = cosine_similarity(response_vector, doc_vectors)[0]
    coverage_score = np.mean(doc_relevance_scores > 0.1)  # Threshold can be adjusted

    return {
        'relevance_score': relevance_score,
        'coherence_score': coherence_score,
        'coverage_score': coverage_score
    }


def test_chunk_sizes(data_dir: str, query: str, chunk_sizes: List[int], top_k: int = 3):
    results = {}

    for size in chunk_sizes:
        print(f"\nTesting chunk size: {size}")

        # Initialize components
        metadata = process_metadata(data_dir)
        dataset_path = os.path.join(data_dir, f'deeplake_dataset_chunk_{size}')
        text_retriever = RAGRetriever(dataset_path=dataset_path, model_name='all-MiniLM-L6-v2')

        # Load documents
        documents = text_retriever.load_data(data_dir, metadata)

        # Use LangChain's text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=size // 10,  # 10% overlap
            length_function=len,
        )

        chunked_documents = []
        for doc in documents:
            chunks = text_splitter.split_text(doc.page_content)
            for i, chunk in enumerate(chunks):
                chunked_doc = Document(
                    page_content=chunk,
                    metadata={**doc.metadata, 'chunk_id': i}
                )
                chunked_documents.append(chunked_doc)

        # Generate embeddings for chunked documents
        text_retriever.generate_embeddings(chunked_documents)

        # Initialize RAG components
        qa_generator = RAGGenerator(model_name='llama3')
        rag_pipeline = RAGPipeline(text_retriever, qa_generator)

        # Run query
        start_time = time.time()
        retrieved_docs, most_relevant_passage, response = rag_pipeline.run(query, top_k=top_k)
        end_time = time.time()

        # Evaluate response
        evaluation_metrics = evaluate_response(query, response, retrieved_docs)

        results[size] = {
            'response': response,
            'time': end_time - start_time,
            'num_chunks': len(chunked_documents),
            'retrieved_docs': retrieved_docs,
            'evaluation_metrics': evaluation_metrics
        }

        # Clean up
        del text_retriever, qa_generator, rag_pipeline
        gc.collect()
        torch.cuda.empty_cache()

    return results


def main():
    set_seed(42)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(project_root, 'data', 'marc-xl-data')

    query = input("Enter your question: ")
    chunk_sizes = [100, 500, 1000, 2000]  # Adjusted for character-based splitting

    results = test_chunk_sizes(data_dir, query, chunk_sizes)

    print("\n--- Results ---")
    for size, result in results.items():
        print(f"\nChunk size: {size}")
        print(f"Number of chunks: {result['num_chunks']}")
        print(f"Processing time: {result['time']:.2f} seconds")
        print(f"Number of retrieved documents: {len(result['retrieved_docs'])}")
        print("Evaluation metrics:")
        for metric, value in result['evaluation_metrics'].items():
            print(f"  {metric}: {value:.4f}")
        print("Response:")
        print(result['response'])
        print("-" * 50)


if __name__ == "__main__":
    main()