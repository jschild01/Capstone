import os
import sys
import time
import torch
import gc
from typing import List, Dict, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt

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
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([query, response])
    relevance_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    sentences = response.split('.')
    if len(sentences) > 1:
        sentence_vectors = vectorizer.transform(sentences)
        coherence_scores = [cosine_similarity(sentence_vectors[i:i+1], sentence_vectors[i+1:i+2])[0][0]
                            for i in range(len(sentences) - 1)]
        coherence_score = np.mean(coherence_scores)
    else:
        coherence_score = 1.0

    doc_contents = [doc.page_content for doc in retrieved_docs]
    doc_vectors = vectorizer.transform(doc_contents)
    response_vector = vectorizer.transform([response])
    doc_relevance_scores = cosine_similarity(response_vector, doc_vectors)[0]
    coverage_score = np.mean(doc_relevance_scores > 0.1)

    return {
        'relevance_score': relevance_score,
        'coherence_score': coherence_score,
        'coverage_score': coverage_score
    }


def test_rag_accuracy(rag_pipeline: RAGPipeline, questions: List[Tuple[str, str]], top_k: int) -> float:
    correct = 0
    total = len(questions)
    for question, expected_answer in questions:
        _, _, response = rag_pipeline.run(question, top_k=top_k)
        if expected_answer.lower() in response.lower():
            correct += 1
        else:
            print(f"Incorrect answer for question: {question}")
            print(f"Expected: {expected_answer}")
            print(f"Got: {response}\n")
    accuracy = correct / total
    print(f"Accuracy for top-k={top_k}: {accuracy:.2f} ({correct}/{total})")
    return accuracy


def test_top_k_values(data_dir: str, test_questions: List[Tuple[str, str]], top_k_values: List[int], chunk_size: int = 500):
    results = {}

    # Initialize components
    metadata = process_metadata(data_dir)
    dataset_path = os.path.join(data_dir, f'deeplake_dataset_chunk_{chunk_size}')
    text_retriever = RAGRetriever(dataset_path=dataset_path, model_name='all-MiniLM-L6-v2')

    # Load and chunk documents
    documents = text_retriever.load_data(data_dir, metadata)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_size // 10,
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

    for k in top_k_values:
        print(f"\nTesting top-k value: {k}")

        # Initialize RAG components
        qa_generator = RAGGenerator(model_name='llama3')
        rag_pipeline = RAGPipeline(text_retriever, qa_generator)

        # Test accuracy
        accuracy = test_rag_accuracy(rag_pipeline, test_questions, k)

        results[k] = {
            'accuracy': accuracy
        }

        # Clean up
        del qa_generator, rag_pipeline
        gc.collect()
        torch.cuda.empty_cache()

    # Clean up text_retriever
    del text_retriever
    gc.collect()
    torch.cuda.empty_cache()

    return results


def plot_results(results: Dict):
    k_values = list(results.keys())
    accuracies = [result['accuracy'] for result in results.values()]

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, accuracies, marker='o')
    plt.title('Accuracy vs. Top-K')
    plt.xlabel('Top-K Value')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('top_k_results.png')
    plt.close()


def main():
    set_seed(42)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(project_root, 'data', 'marc-xl-data')

    test_questions = [
        ("Complete this sentence: 'My mules are not hungry. They're lively and'", "gay"),
        ("Complete this sentence: 'Take a trip on the canal if you want to have'", "fun"),
        ("What is the name of the female character mentioned in the song that begins 'In Scarlett town where I was born'?", "Barbrae Allen"),
        ("According to the transcript, what is Captain Pearl R. Nye's favorite ballad?", "Barbara Allen"),
        ("Complete this phrase from the gospel train song: 'The gospel train is'", "night"),
        ("In the song 'Barbara Allen,' where was Barbara Allen from?", "Scarlett town"),
        ("In the song 'Lord Lovele,' how long was Lord Lovele gone before returning?", "A year or two or three at most"),
        ("What instrument does Captain Nye mention loving?", "old fiddled mouth organ banjo"),
        ("In the song about pumping out Lake Erie, what will be on the moon when they're done?", "whiskers"),
        ("Complete this line from a song: 'We land this war down by the'", "river"),
        ("What does the singer say they won't do in the song 'I Won't Marry At All'?", "Marry at all"),
        ("What does the song say will 'outshine the sun'?", "We'll"),
        ("In the 'Dying Cowboy' song, where was the cowboy born?", "Boston")
    ]

    top_k_values = [1, 3, 5, 10, 20, 50]

    results = test_top_k_values(data_dir, test_questions, top_k_values)

    print("\n--- Results ---")
    for k, result in results.items():
        print(f"\nTop-k value: {k}")
        print(f"Accuracy: {result['accuracy']:.4f}")
        print("-" * 50)

    plot_results(results)
    print("\nResults plot saved as 'top_k_results.png'")


if __name__ == "__main__":
    main()