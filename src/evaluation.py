import os
import sys
import time
import pandas as pd
import torch
import numpy as np
import random
import gc
from sklearn.metrics import precision_score, recall_score, f1_score
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Now import from the component directory
from component.rag_text_processor import TextProcessor
from component.rag_retriever import RAGRetriever
from component.rag_generator import RAGGenerator
from component.rag_pipeline import RAGPipeline

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    # Set seed for reproducibility
    set_seed(42)

    # get csv from data folder of the parent directory
    input_csv = os.path.join(parent_dir, 'data', 'afc_txtFiles_QA_filtered.csv')

    # Load the CSV file
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found.")
        return

    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)

    # get 10% subset of df to test functionality
    df = df.sample(frac=0.01, random_state=42).reset_index(drop=True)

    # Initialize the retriever and generator
    chunk_size = 100
    chunk_overlap = 30

    # model can be 'mpnet' or 'instructor-xl'; use AWS GPU for instructor xl
    text_retriever = RAGRetriever(model_name='mpnet', chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Load the data into the retriever
    text_retriever.load_data(df)
    text_retriever.generate_embeddings()

    qa_generator = RAGGenerator(model_name='google/flan-t5-small')
    rag_pipeline = RAGPipeline(text_retriever, qa_generator)

    # Prepare lists for references and hypotheses
    references = []
    hypotheses = []

    # Now process each row
    rag_answers = []
    smoothie = SmoothingFunction().method4  # For BLEU score smoothing

    for idx, row in df.iterrows():
        start_time = time.time()

        query = row['generated_question']
        filename = row['filename']

        # Run the pipeline, retrieving only from the same filename
        relevant_passages, most_relevant_passage, response, most_relevant_passage_filename = rag_pipeline.run(
            query, top_k=3)

        # Save the response
        rag_answers.append(response)

        # Prepare references and hypotheses for corpus BLEU
        reference = row['answer'].split()
        hypothesis = response.split()
        references.append([reference])  # corpus_bleu expects list of lists
        hypotheses.append(hypothesis)

        end_time = time.time()
        total_time = end_time - start_time

        print(f"Processed row {idx+1}/{len(df)} in {total_time:.2f} seconds")

    # Add the RAG answers to the dataframe
    df['rag_answer'] = rag_answers

    # Compute corpus-level BLEU score safely without the '_normalize' issue
    try:
        corpus_bleu_score = corpus_bleu(references, hypotheses, smoothing_function=smoothie)
    except TypeError:
        # Fallback approach to calculate BLEU manually or adjust settings
        print("Encountered TypeError in corpus BLEU calculation. Consider checking NLTK version or modifying score calculation.")
        corpus_bleu_score = corpus_bleu(references, hypotheses)  # without smoothing or customize further
        
    # get overall bleu score
    print(f"\nOverall Corpus BLEU score: {corpus_bleu_score:.2f}")

    # Save the dataframe with RAG answers to data folder in parent direction
    df.to_csv(os.path.join(parent_dir, 'data', 'afc_txtFiles_QA_eval.csv'), index=False)

if __name__ == "__main__":
    main()
