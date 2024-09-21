import re
import os
import sys
import pandas as pd
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from keybert import KeyBERT

from rag_text_processor import custom_preprocess


class QuestionGenerator:
    def __init__(self, input_csv, output_csv, model_name='mohammedaly22/t5-small-squad-qg'):
        self.input_csv = input_csv
        self.output_csv = output_csv
        self.model_name = model_name
        self.df = None
        self.pipe = pipeline('text2text-generation', model=self.model_name)
        self.keybert_model = KeyBERT('t5-large')

    def load_data(self):
        if not os.path.exists(self.input_csv):
            print(f"Error: {self.input_csv} not found.")
            sys.exit()
        self.df = pd.read_csv(self.input_csv)

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

    def generate_questions(self):
        self.df['token_count'] = self.df['clean_text'].apply(self.get_token_count)

        self.df['keyword_tfidf'], self.df['keyword_tfidf_score'] = zip(
            *self.df['clean_text'].apply(lambda text: self.keywords_tfidf(text, n_components=1))
        )

        self.df['keyphrase_keybert'], self.df['keyphrase_keybert_score'] = zip(
            *self.df['clean_text'].apply(lambda text: self.keyphrases_keybert(text, top_n=1))
        )
        self.df['keyphrase_keybert'] = self.df['keyphrase_keybert'].apply(lambda phrases: phrases[0] if phrases else '')

        self.df['instruction_prompt_keyword'] = self.df.apply(
            lambda row: self.process_context_and_prepare_instruction(row['clean_text'], row['keyword_tfidf']), axis=1
        )
        self.df['instruction_prompt_keyphrase'] = self.df.apply(
            lambda row: self.process_context_and_prepare_instruction(row['clean_text'], row['keyphrase_keybert']), axis=1
        )

        self.df['keyword_generated_question'] = self.df['instruction_prompt_keyword'].apply(
            lambda prompt: self.pipe(prompt, num_return_sequences=1, num_beams=2, num_beam_groups=2, diversity_penalty=1.0)[0]['generated_text']
        )
        self.df['keyphrase_generated_question'] = self.df['instruction_prompt_keyphrase'].apply(
            lambda prompt: self.pipe(prompt, num_return_sequences=1, num_beams=2, num_beam_groups=2, diversity_penalty=1.0)[0]['generated_text']
        )

    def save_results(self):
        self.df.to_csv(self.output_csv, index=False)

    def run(self):
        self.load_data()
        self.generate_questions()   
        self.save_results()


# Example usage
if __name__ == "__main__":

    # direcotries
    base_path = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(base_path)
    grandparent = os.path.dirname(parent)

    # define input and output csv files
    input_csv = os.path.join(parent, 'data', 'afc_txtFiles.csv')
    output_csv = os.path.join(parent, 'data', 'afc_txtFiles_QA.csv')

    # instantiate QuestionGenerator
    question_generator = QuestionGenerator(input_csv, output_csv)
    question_generator.run()

    print(f"\nQuestions generated and saved to: {output_csv}\n")

