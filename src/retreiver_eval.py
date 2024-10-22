import os
import sys
import time
import torch
import gc
import pandas as pd
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Add the src directory to the Python path
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, src_dir)

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


def chunk_documents(documents: List[Document], chunk_size: int) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_size * 15 // 100,
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
    return chunked_documents

def find_correct_chunk(documents: List[Document], answer: str, chunk_size: int) -> int:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_size * 15 // 100,  # Assuming 15% overlap as before
        length_function=len
    )
    for doc in documents:
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            if answer in chunk:
                return i
    return -1  # Return -1 if no chunk contains the answer

def get_chunk_text(document: Document, chunk_id: int, chunk_size: int) -> str:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_size * 15 // 100,  # Assuming 15% overlap as before
        length_function=len
    )
    chunks = text_splitter.split_text(document.page_content)
    if chunk_id < len(chunks):
        return chunks[chunk_id]
    return "Chunk ID out of range" 

def retriever_eval():
    set_seed(42)
    data_dir = os.path.join(project_root, 'data', 'marc-xl-data')

     # Ensure the retrieval_eval folder exists
    eval_dir = os.path.join(src_dir, 'retrieval_eval')
    os.makedirs(eval_dir, exist_ok=True)

    # Create a DataFrame to store aggregated results
    accuracy_df = pd.DataFrame(columns=["Top_k", "Chunk Size", "Embedding model", "Doc Match Count", "Chunk Match Count"])

    model_names = ['instructor'] # mini, instructor, titan
    chunk_sizes = [100, 250, 500, 5000]
    top_ks = [1, 2, 50]

    # iterations
    for top_k in top_ks:
        for chunk_size in chunk_sizes:
            for model_name in model_names:
                metadata = process_metadata(data_dir)
                dataset_path = os.path.join(data_dir, f'deeplake_dataset_chunk_{chunk_size}')
                text_retriever = RAGRetriever(dataset_path=dataset_path, model_name=model_name)
                    
                # remove old data
                text_retriever.delete_dataset()

                documents = text_retriever.load_data(data_dir, metadata)
                chunked_documents = chunk_documents(documents, chunk_size)

                if text_retriever.is_empty():
                    print("Generating embeddings for chunked documents...")
                    text_retriever.generate_embeddings(chunked_documents)

                queries_answers = [
                    ("Complete this sentence: 'The mules are not hungry. They're lively and'", "sr22a_en.txt", "gay"),
                    ("Complete this sentence: 'Take a trip on the canal if you want to have'", "sr28a_en.txt or sr13a_en.txt", "fun"),
                    ("What is the name of the female character mentioned in the song that begins 'In Scarlett town where I was born'?", "sr02b_en.txt", "Barbrae Allen"), 
                    ("According to the transcript, what is Captain Pearl R. Nye's favorite ballad?", "sr28a_en.txt", "Barbara Allen"),
                    ("Complete this phrase from the gospel train song: 'The gospel train is'", "sr26a_en.txt", "night"), 
                    ("In the song 'Barbara Allen,' where was Barbara Allen from?", "sr02b_en.txt", "Scarlett town"),
                    ("In the song 'Lord Lovele,' how long was Lord Lovele gone before returning?", "sr08a_en.txt", "A year or two or three at most"),
                    ("What instrument does Captain Nye mention loving?", "sr22a_en.txt", "old fiddled mouth organ banjo"), 
                    ("In the song about pumping out Lake Erie, what will be on the moon when they're done?", "sr27b_en.txt", "whiskers"),
                    ("Complete this line from a song: 'We land this war down by the'", "sr05a_en.txt", "river"),
                    ("What does the singer say they won't do in the song 'I Won't Marry At All'?", "sr01b_en.txt", "Marry/Mary at all"),
                    ("What does the song say will 'outshine the sun'?", "sr17b_en.txt", "We'll/not"),
                    ("In the 'Dying Cowboy' song, where was the cowboy born?", "sr20b_en.txt", "Boston")
                ]

                df_results = pd.DataFrame(columns=["Query", "Expected Answer", 
                                                "Expected Doc", "Retrieved Doc", "Doc Match", 
                                                "Expected Chunk ID", "Expected Chunk Text", "Retrieved Chunk", "Retrieved Content", "Chunk Match"])

                for query, doc_filenames, answer in queries_answers:
                    # in case there are multiple files that contain the answer
                    possible_filenames = [filename.strip() for filename in doc_filenames.split('or')]

                    # iterate through docs for comparing to retriever
                    for doc in documents:
                        if doc.metadata['original_filename'] in possible_filenames:
                            expected_chunk_id = find_correct_chunk([doc], answer, chunk_size)
                            expected_chunk_text = get_chunk_text(doc, expected_chunk_id, chunk_size)
                            query_result, original_filename, document_content, retrieved_chunk_id = text_retriever.test_document_retrieval(query, top_k=top_k)
                            doc_match = original_filename in possible_filenames
                            chunk_match = retrieved_chunk_id == expected_chunk_id
                            new_row = {
                                "Top_k": top_k,
                                "Query": query,
                                "Expected Answer": answer,
                                "Expected Doc": doc_filenames,
                                "Retrieved Doc": original_filename,
                                "Doc Match": doc_match,
                                "Expected Chunk ID": expected_chunk_id,
                                "Expected Chunk Text": expected_chunk_text,
                                "Retrieved Chunk": retrieved_chunk_id,
                                "Retrieved Content": document_content,
                                "Chunk Match": chunk_match
                            }
                            df_results = pd.concat([df_results, pd.DataFrame([new_row])], ignore_index=True)

                # Handling duplicative queries caused by instances where the answer is found in multiple docs
                scores = []
                for index, row in df_results.iterrows():
                    score = 0
                    if row['Doc Match'] and row['Chunk Match']:
                        score = 3  # Highest priority for TRUE, TRUE
                    elif row['Doc Match']:
                        score = 2  # Second priority for TRUE, FALSE
                    elif row['Chunk Match']:
                        score = 1  # Third priority for FALSE, TRUE
                    scores.append(score)
                df_results['Score'] = scores

                # drop the duplicative column but keeping the 'best scoring' one
                df_results = df_results.sort_values('Score', ascending=False).drop_duplicates(subset=['Query'], keep='first').drop('Score', axis=1)

                # Counting and printing the number of TRUE values for Doc Match and Chunk Match
                doc_match_count = df_results['Doc Match'].sum()
                chunk_match_count = df_results['Chunk Match'].sum()

                # add counts to overall dataframe
                accuracy_df = pd.concat([accuracy_df, pd.DataFrame([[top_k, chunk_size, model_name, doc_match_count, chunk_match_count]], columns=accuracy_df.columns)])

                # save dataframe csv
                csv_path = os.path.join(eval_dir, f'query_results_{model_name}_{top_k}_{chunk_size}.csv')
                df_results.to_csv(csv_path, index=False)
                print(f"The DataFrame has been saved to '{csv_path}'.")
                print()

        # Save the accumulated results to a CSV
        accuracy_csv_path = os.path.join(eval_dir, 'query_results_chunkComparison.csv')
        accuracy_df.to_csv(accuracy_csv_path, index=False)
        print(f"Overall accuracy summary has been saved to '{csv_path}'.")

if __name__ == "__main__":
    retriever_eval()