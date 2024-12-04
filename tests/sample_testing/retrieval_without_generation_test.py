import os
import sys
import time
import torch
import gc
from typing import List, Dict, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import numpy as np
import matplotlib.pyplot as plt

# Add the src directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, src_dir)

from src.component.rag_retriever_deeplake import RAGRetriever
from src.component.metadata_processor import process_metadata

def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def test_retrieval_accuracy(text_retriever: RAGRetriever, questions: List[Tuple[str, str]], top_k: int) -> Tuple[float, List[Dict]]:
    correct = 0
    total = len(questions)
    incorrect_retrievals = []
    for question, expected_answer in questions:
        retrieved_docs = text_retriever.search_vector_store(question, top_k=top_k)

        # Check if the expected answer is in any of the retrieved documents
        answer_found = any(expected_answer.lower() in doc.page_content.lower() for doc in retrieved_docs)

        if answer_found:
            correct += 1
        else:
            incorrect_retrievals.append({
                'question': question,
                'expected': expected_answer,
                'retrieved_content': [doc.page_content for doc in retrieved_docs]
            })

    accuracy = correct / total
    return accuracy, incorrect_retrievals

def test_chunk_sizes(data_dir: str, test_questions: List[Tuple[str, str]], chunk_sizes: List[int], top_k: int = 3):
    results = {}

    for size in chunk_sizes:
        print(f"\nTesting chunk size: {size}")

        # Initialize components
        metadata = process_metadata(data_dir)
        dataset_path = os.path.join(data_dir, f'deeplake_dataset_chunk_{size}')
        text_retriever = RAGRetriever(dataset_path=dataset_path, model_name='all-MiniLM-L6-v2')

        # Load and chunk documents
        documents = text_retriever.load_data(data_dir, metadata)
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

        # Test retrieval accuracy
        start_time = time.time()
        accuracy, incorrect_retrievals = test_retrieval_accuracy(text_retriever, test_questions, top_k)
        end_time = time.time()

        avg_time = (end_time - start_time) / len(test_questions)

        results[size] = {
            'accuracy': accuracy,
            'avg_time': avg_time,
            'num_chunks': len(chunked_documents),
            'incorrect_retrievals': incorrect_retrievals
        }

        # Clean up
        del text_retriever
        gc.collect()
        torch.cuda.empty_cache()

    return results

def plot_results(results: Dict):
    chunk_sizes = list(results.keys())
    accuracies = [result['accuracy'] for result in results.values()]
    avg_times = [result['avg_time'] for result in results.values()]
    num_chunks = [result['num_chunks'] for result in results.values()]

    fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    fig.suptitle('Chunk Size Test Results (Retrieval Only)')

    axs[0, 0].plot(chunk_sizes, accuracies, marker='o')
    axs[0, 0].set_title('Retrieval Accuracy vs. Chunk Size')
    axs[0, 0].set_xlabel('Chunk Size')
    axs[0, 0].set_ylabel('Retrieval Accuracy')

    axs[0, 1].plot(chunk_sizes, avg_times, marker='o')
    axs[0, 1].set_title('Average Retrieval Time vs. Chunk Size')
    axs[0, 1].set_xlabel('Chunk Size')
    axs[0, 1].set_ylabel('Average Time (s)')

    axs[1, 0].plot(chunk_sizes, num_chunks, marker='o')
    axs[1, 0].set_title('Number of Chunks vs. Chunk Size')
    axs[1, 0].set_xlabel('Chunk Size')
    axs[1, 0].set_ylabel('Number of Chunks')

    axs[1, 1].axis('off')  # This subplot is left empty

    plt.tight_layout()
    plt.savefig('chunk_size_results_retrieval.png')
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
        print(f"Retrieval Accuracy: {result['accuracy']:.4f}")
        print(f"Average retrieval time: {result['avg_time']:.2f} seconds")
        print("\nIncorrect Retrievals:")
        for item in result['incorrect_retrievals']:
            print(f"Question: {item['question']}")
            print(f"Expected: {item['expected']}")
            print(f"Retrieved content snippets:")
            for i, content in enumerate(item['retrieved_content'], 1):
                print(f"  Snippet {i}: {content[:100]}...")  # Print first 100 chars of each retrieved document
            print("-" * 50)
        print("=" * 50)

    plot_results(results)
    print("\nResults plot saved as 'chunk_size_results_retrieval.png'")

if __name__ == "__main__":
    main()