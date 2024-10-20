import os
import sys
import time
import torch
import gc
import pandas as pd
import ast
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

def main():
    set_seed(42)
    data_dir = os.path.join(project_root, 'data', 'marc-xl-data')
    chunk_size = 250  # Fixed chunk size of 100
    model_names = ['llama', 't5'] # claude in the future when functional

    # test q&q
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

    combined_results = []
    for model_name in model_names:
        # Initialize components
        metadata = process_metadata(data_dir)
        dataset_path = os.path.join(data_dir, f'deeplake_dataset_chunk_{chunk_size}')
        text_retriever = RAGRetriever(dataset_path=dataset_path, model_name='instructor')
        text_retriever.delete_dataset() # delete old/previous data
        
        # Load and prepare documents
        documents = text_retriever.load_data(data_dir, metadata)
        if len(documents) == 0:
            print("No documents loaded. Check the load_data method in RAGRetriever.")
            return

        # Chunk documents
        chunked_documents = chunk_documents(documents, chunk_size)
        num_chunks = len(chunked_documents)
        print(f"Prepared {num_chunks} chunks with size {chunk_size}")

        if num_chunks == 0:
            print("No chunks created. Check the chunking process.")
            return

        # Generate embeddings if the dataset is empty (should be empty)
        if text_retriever.is_empty():
            print("Generating embeddings for chunked documents...")
            text_retriever.generate_embeddings(chunked_documents)
            print("Embeddings generated and saved.")
        else:
            print("Using existing embeddings.")

        # Initialize RAG components
        qa_generator = RAGGenerator(model_name=model_name)
        rag_pipeline = RAGPipeline(text_retriever, qa_generator)

        # List to store dataframes for concatenation
        results_list = []

        # iterate through q&a
        for query, file, answer in queries_answers:
            # apply rag
            retrieved_docs, most_relevant_passage, raw_response, validated_response, structured_response, final_response = rag_pipeline.run(query)
            
            # extract just the generated response
            rag_response = rag_pipeline.extract_answer(raw_response)

            temp_df = pd.DataFrame([{
                "Model": model_name,
                "Query": query,
                "Expected Docs": file,
                "Expected Answer": answer,
                "RAG Sole Response": rag_response,
                "RAG Raw Response": raw_response,
                "RAG Validated Response": validated_response,
                "RAG Structured Response": structured_response,
                "RAG Final Response": final_response,
                "RAG Relevant Passage": most_relevant_passage,
                "RAG Doc Retrieved": retrieved_docs
            }])
            results_list.append(temp_df)

        # Concatenate all result dataframes
        df_results = pd.concat(results_list, ignore_index=True)

        # Append to main comparison list
        combined_results.append(df_results)

        # save dataframe csv; ensure the retrieval_eval folder exists
        eval_dir = os.path.join(src_dir, 'generator_eval')
        os.makedirs(eval_dir, exist_ok=True)
        csv_path = os.path.join(eval_dir, f'generator_results_{model_name}.csv')
        df_results.to_csv(csv_path, index=False)

        # Cleanup
        del text_retriever, qa_generator, rag_pipeline
        gc.collect()
        torch.cuda.empty_cache()

    # save overall comparing dataframe
    final_df = pd.concat(combined_results, ignore_index=True)
    final_df = final_df.sort_values(by="Query")  # Sort by Query column to put like-queries together
    final_csv_path = os.path.join(eval_dir, 'generator_results_all.csv')
    final_df.to_csv(final_csv_path, index=False)
    print(f"The final combined DataFrame has been saved to '{final_csv_path}'.")    

if __name__ == "__main__":
    main()
    
    