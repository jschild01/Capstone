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

def test_rag_accuracy(rag_pipeline: RAGPipeline, questions: List[Tuple[str, str]], top_k: int) -> Tuple[float, List[Dict], List[Dict]]:
    correct = 0
    total = len(questions)
    all_metrics = []
    incorrect_answers = []
    for question, expected_answer in questions:
        retrieved_docs, _, response = rag_pipeline.run(question, top_k=top_k)
        if expected_answer.lower() in response.lower():
            correct += 1
        else:
            incorrect_answers.append({
                'question': question,
                'expected': expected_answer,
                'got': response
            })
        metrics = evaluate_response(question, response, retrieved_docs)
        all_metrics.append(metrics)
    accuracy = correct / total
    return accuracy, all_metrics, incorrect_answers

def test_chunk_sizes(data_dir: str, test_questions: List[Tuple[str, str]], chunk_sizes: List[int], top_k: int = 3):
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
            chunk_overlap=size // 10,
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

        # Test accuracy and get evaluation metrics
        start_time = time.time()
        accuracy, all_metrics, incorrect_answers = test_rag_accuracy(rag_pipeline, test_questions, top_k)
        end_time = time.time()

        avg_time = (end_time - start_time) / len(test_questions)
        avg_evaluation_metrics = {
            metric: np.mean([m[metric] for m in all_metrics])
            for metric in all_metrics[0].keys()
        }

        results[size] = {
            'accuracy': accuracy,
            'avg_time': avg_time,
            'num_chunks': len(chunked_documents),
            'evaluation_metrics': avg_evaluation_metrics,
            'incorrect_answers': incorrect_answers
        }

        # Clean up
        del text_retriever, qa_generator, rag_pipeline
        gc.collect()
        torch.cuda.empty_cache()

    return results

def plot_results(results: Dict):
    chunk_sizes = list(results.keys())
    accuracies = [result['accuracy'] for result in results.values()]
    avg_times = [result['avg_time'] for result in results.values()]
    relevance_scores = [result['evaluation_metrics']['relevance_score'] for result in results.values()]
    coherence_scores = [result['evaluation_metrics']['coherence_score'] for result in results.values()]
    coverage_scores = [result['evaluation_metrics']['coverage_score'] for result in results.values()]

    fig, axs = plt.subplots(3, 2, figsize=(15, 20))
    fig.suptitle('Chunk Size Test Results')

    axs[0, 0].plot(chunk_sizes, accuracies, marker='o')
    axs[0, 0].set_title('Accuracy vs. Chunk Size')
    axs[0, 0].set_xlabel('Chunk Size')
    axs[0, 0].set_ylabel('Accuracy')

    axs[0, 1].plot(chunk_sizes, avg_times, marker='o')
    axs[0, 1].set_title('Average Processing Time vs. Chunk Size')
    axs[0, 1].set_xlabel('Chunk Size')
    axs[0, 1].set_ylabel('Average Time (s)')

    axs[1, 0].plot(chunk_sizes, relevance_scores, marker='o')
    axs[1, 0].set_title('Relevance Score vs. Chunk Size')
    axs[1, 0].set_xlabel('Chunk Size')
    axs[1, 0].set_ylabel('Relevance Score')

    axs[1, 1].plot(chunk_sizes, coherence_scores, marker='o')
    axs[1, 1].set_title('Coherence Score vs. Chunk Size')
    axs[1, 1].set_xlabel('Chunk Size')
    axs[1, 1].set_ylabel('Coherence Score')

    axs[2, 0].plot(chunk_sizes, coverage_scores, marker='o')
    axs[2, 0].set_title('Coverage Score vs. Chunk Size')
    axs[2, 0].set_xlabel('Chunk Size')
    axs[2, 0].set_ylabel('Coverage Score')

    num_chunks = [result['num_chunks'] for result in results.values()]
    axs[2, 1].plot(chunk_sizes, num_chunks, marker='o')
    axs[2, 1].set_title('Number of Chunks vs. Chunk Size')
    axs[2, 1].set_xlabel('Chunk Size')
    axs[2, 1].set_ylabel('Number of Chunks')

    plt.tight_layout()
    plt.savefig('chunk_size_results.png')
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

    chunk_sizes = [100, 250, 500, 1000, 2000]

    results = test_chunk_sizes(data_dir, test_questions, chunk_sizes)

    print("\n--- Results ---")
    for size, result in results.items():
        print(f"\nChunk size: {size}")
        print(f"Number of chunks: {result['num_chunks']}")
        print(f"Accuracy: {result['accuracy']:.4f}")
        print(f"Average processing time: {result['avg_time']:.2f} seconds")
        print("Evaluation metrics:")
        for metric, value in result['evaluation_metrics'].items():
            print(f"  {metric}: {value:.4f}")
        print("\nIncorrect Answers:")
        for item in result['incorrect_answers']:
            print(f"Question: {item['question']}")
            print(f"Expected: {item['expected']}")
            print(f"Got: {item['got']}")
            print("-" * 50)
        print("=" * 50)

    plot_results(results)
    print("\nResults plot saved as 'chunk_size_results.png'")

if __name__ == "__main__":
    main()