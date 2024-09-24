import re
import os
import sys
import pandas as pd
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import random
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
import torch

class QuestionGenerator:
    def __init__(self, input_csv, output_csv, model_name='mohammedaly22/t5-small-squad-qg'):
        self.input_csv = input_csv
        self.output_csv = output_csv
        self.output_csv_base = os.path.splitext(output_csv)[0] 
        self.model_name = model_name
        self.df = None
        self.pipe = pipeline('text2text-generation', 
                             model=self.model_name,
                             device=0 if torch.cuda.is_available() else -1)
        embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device=device)
        self.keybert_model = KeyBERT(embedding_model)
        #self.keybert_model = KeyBERT('t5-large')

    def load_data(self):
        if not os.path.exists(self.input_csv):
            print(f"Error: {self.input_csv} not found.")
            sys.exit()

    def custom_preprocess(self, text):
        # Remove specific text
        text = re.sub(r'transcribed and reviewed by contributors participating in the by the people project at crowd.loc.gov.', '', text, flags=re.IGNORECASE)
        
        # Add any other custom preprocessing steps here
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    @staticmethod
    def process_context_and_prepare_instruction(context, answer):
        sentences = re.split(r'(?<=[.!?]) +', context)
        sentence_index = next((i for i, sentence in enumerate(sentences) if answer in sentence), None)
        if sentence_index is not None:
            start = max(0, sentence_index - 2)
            end = min(len(sentences), sentence_index + 3)
            trimmed_context = ' '.join(sentences[start:end])
        else:
            trimmed_context = context

        highlighted_context = trimmed_context.replace(answer, f" <h> {answer} <h> ")
        instruction_prompt = (
            f"### Instruction ###\n"
            f"Generate a question directly related to the highlighted part marked by <h> in the provided context below.\n"
            f"Ignore all other parts of the context; focus only on the highlighted section as the answer reference.\n\n"
            f"### Context ###\n"
            f"```{highlighted_context}```"
        )
        return instruction_prompt

    @staticmethod
    def keywords_tfidf(text, n_components=1):
        vectorizer = TfidfVectorizer(stop_words='english', max_features=10)
        X = vectorizer.fit_transform([text])

        if X.shape[1] <= n_components:
            return '', 0.0  # Return empty string and zero score

        svd = TruncatedSVD(n_components=n_components)
        svd.fit(X)
        terms = vectorizer.get_feature_names_out()
        components = svd.components_[0]
        key_term_index = components.argmax()
        key_phrase = terms[key_term_index]
        score = components[key_term_index]
        return key_phrase, score

    def keyphrases_keybert(self, text, top_n=1):
        keyphrases_with_probs = self.keybert_model.extract_keywords(
            text, keyphrase_ngram_range=(1, 1), use_maxsum=True, nr_candidates=20, top_n=top_n
        )
        keyphrases = [phrase for phrase, _ in keyphrases_with_probs]
        keyphrases_probs = [prob for _, prob in keyphrases_with_probs]
        return keyphrases, keyphrases_probs

    @staticmethod
    def get_token_count(text):
        return len(text.split())

    def process_chunk(self, chunk, chunk_index):
        chunk['clean_text'] = chunk['text'].apply(self.custom_preprocess)
        chunk['token_count'] = chunk['clean_text'].apply(self.get_token_count)
        chunk['keyword_tfidf'], chunk['keyword_tfidf_score'] = zip(
            *chunk['clean_text'].apply(lambda text: self.keywords_tfidf(text, n_components=1))
        )
        chunk['keyphrase_keybert'], chunk['keyphrase_keybert_score'] = zip(
            *chunk['clean_text'].apply(lambda text: self.keyphrases_keybert(text, top_n=1))
        )
        chunk['keyphrase_keybert'] = chunk['keyphrase_keybert'].apply(lambda phrases: phrases[0] if phrases else '')
        chunk['instruction_prompt_keyword'] = chunk.apply(
            lambda row: self.process_context_and_prepare_instruction(row['clean_text'], row['keyword_tfidf']), axis=1
        )
        chunk['instruction_prompt_keyphrase'] = chunk.apply(
            lambda row: self.process_context_and_prepare_instruction(row['clean_text'], row['keyphrase_keybert']), axis=1
        )
        chunk['keyword_generated_question'] = chunk['instruction_prompt_keyword'].apply(
            lambda prompt: self.pipe(prompt, num_return_sequences=1, num_beams=2, num_beam_groups=2, diversity_penalty=1.0)[0]['generated_text']
        )
        chunk['keyphrase_generated_question'] = chunk['instruction_prompt_keyphrase'].apply(
            lambda prompt: self.pipe(prompt, num_return_sequences=1, num_beams=2, num_beam_groups=2, diversity_penalty=1.0)[0]['generated_text']
        )

        # Save the processed chunk to a CSV file
        chunk_output_csv = f"{self.output_csv_base}_chunk_{chunk_index}.csv"
        chunk.to_csv(chunk_output_csv, index=False)
        print(f"Chunk {chunk_index} processed and saved to {chunk_output_csv}")

    def run(self):
        # Read and process the CSV file in chunks of 500 rows
        chunk_size = 1000
        chunk_index = 0
        for chunk in pd.read_csv(self.input_csv, chunksize=chunk_size):
            self.process_chunk(chunk, chunk_index)
            chunk_index += 1


# run
if __name__ == "__main__":
    
    # set device to gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # direcotries
    base_path = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(base_path)
    grandparent = os.path.dirname(parent)

    # Set seed for reproducibility
    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    set_seed(42)
    
    # define input and output csv files
    input_csv = os.path.join(parent, 'data', 'afc_txtFiles.csv')
    output_csv = os.path.join(parent, 'data', 'afc_txtFiles_QA.csv')

    # instantiate QuestionGenerator
    question_generator = QuestionGenerator(input_csv, output_csv)
    question_generator.run()

    print(f"\nQuestions generated and saved to: {output_csv}\n")
