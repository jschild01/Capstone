import os
import re
import string
import glob
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import spacy
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import warnings
from fuzzywuzzy import process

warnings.filterwarnings('ignore')

class TextProcessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(nltk_stopwords.words('english'))
        self.nlp = spacy.load("en_core_web_sm")
    
    def preprocess(self, text, lowercase=False, filter_invalid_sents=False, remove_punctuation=False, remove_stopwords=False, use_lemma=False):
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r'([{}])\1+'.format(re.escape(string.punctuation)), r'\1', text)
        text = re.sub(r'-', ' ', text)
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^A-Za-z0-9.!? ]+', '', text)
        text = re.sub(r'transcribed and reviewed by contributors participating in the by the people project at crowd.loc.gov.', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text).strip()

        if filter_invalid_sents:
            text = self.filter_valid_sentences(text)

        if lowercase:
            text = text.lower()

        if remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))

        words = text.split()
        if remove_stopwords:
            words = [word for word in words if word not in self.stop_words]

        if use_lemma:
            words = [self.lemmatizer.lemmatize(word) for word in words]

        return ' '.join(words)

    def is_valid_sentence(self, sentence):
        doc = self.nlp(sentence)
        has_subject = any(token.dep_ in ("nsubj", "nsubjpass") for token in doc)
        has_verb = any(token.pos_ == "VERB" for token in doc)
        return has_subject and has_verb

    def filter_valid_sentences(self, text):
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        valid_sentences = [sent for sent in sentences if self.is_valid_sentence(sent)]
        return " ".join(valid_sentences)


class TextRetriever:
    def __init__(self):
        self.model = SentenceTransformer('all-mpnet-base-v2')  # Embedding model
        self.index = None
        self.df = None

    def load_data(self, filepath):
        self.df = pd.read_csv(filepath)
        self.df = self.df.head(100)  # Subset for testing

    def generate_embeddings(self):
        embeddings = self.model.encode(self.df['clean_text'].tolist(), convert_to_tensor=True, show_progress_bar=False)
        embeddings_np = embeddings.cpu().numpy()
        dimension = embeddings_np.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings_np)

    def search_vector_store(self, query, top_k=10):
        query_embedding = self.model.encode([query], convert_to_tensor=True).squeeze(0)
        query_embedding_np = query_embedding.cpu().numpy().reshape(1, -1)
        
        distances, indices = self.index.search(query_embedding_np.astype('float32'), top_k)
        retrieved_docs = self.df['clean_text'].iloc[indices[0]].tolist()

        relevant_passages = []
        for doc in retrieved_docs:
            doc_sentences = doc.split('. ')
            best_match = max(
                doc_sentences,
                key=lambda sentence: self.model.encode([sentence], convert_to_tensor=True).squeeze(0).dot(query_embedding).item()
            )
            relevant_passages.append(best_match)

        return relevant_passages


class QAGenerator:
    def __init__(self, model_name='google/flan-t5-small'):
        self.model_name = model_name
        self.model_dict = {
            'google/flan-t5-small': (AutoTokenizer.from_pretrained, AutoModelForSeq2SeqLM.from_pretrained),
            'google/flan-t5-large': (AutoTokenizer.from_pretrained, AutoModelForSeq2SeqLM.from_pretrained),
            'gpt2': (AutoTokenizer.from_pretrained, AutoModelForCausalLM.from_pretrained)
        }
        
        # Initialize tokenizer and model based on the chosen model
        if self.model_name in self.model_dict:
            tokenizer_class, model_class = self.model_dict[self.model_name]
            self.tokenizer = tokenizer_class(self.model_name)
            self.generatorModel = model_class(self.model_name)
        else:
            raise ValueError(f"Model '{self.model_name}' is not supported.")

    def generate_response(self, query, most_relevant_passage, max_new_tokens=50, temperature=0.4, top_p=0.8):
        input_text = query + " " + most_relevant_passage
        input_text = input_text[:2000]
        
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt')
        output = self.generatorModel.generate(input_ids, 
                                              max_length=max_new_tokens + len(input_ids[0]), 
                                              temperature=temperature,
                                              top_p=top_p,
                                              num_return_sequences=1,
                                              do_sample=True)
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        return response



class RAGPipeline:
    def __init__(self, text_retriever, qa_generator):
        self.text_retriever = text_retriever
        self.qa_generator = qa_generator

    def run(self, query, top_k=10):
        # Retrieve relevant passages
        relevant_passages = self.text_retriever.search_vector_store(query, top_k=top_k)
        
        # Find most relevant passage
        query_embedding = self.text_retriever.model.encode([query], convert_to_tensor=True).squeeze(0)
        most_relevant_passage = max(
            relevant_passages,
            key=lambda passage: self.text_retriever.model.encode([passage], convert_to_tensor=True).squeeze(0).dot(query_embedding).item()
        )

        # Get filename of most relevant passage
        matched_rows = self.text_retriever.df[self.text_retriever.df['clean_text'].str.contains(most_relevant_passage, na=False)]

        if not matched_rows.empty:
            most_relevant_passage_filename = matched_rows['filename'].values[0]
        else:
            most_relevant_passage_filename = None  # Handle case where no match is found


        # Generate response
        response = self.qa_generator.generate_response(query, most_relevant_passage)
        
        return relevant_passages, most_relevant_passage, response, most_relevant_passage_filename




# Set up
os.chdir(r'C:\Users\schil\OneDrive\Desktop\Grad SChool\Capstone\LOC')
text_processor = TextProcessor()
text_retriever = TextRetriever()
qa_generator = QAGenerator(model_name='google/flan-t5-small')

# Load and process data
text_retriever.load_data('afc_txtFiles.csv')
text_retriever.df['clean_text'] = text_retriever.df['text'].apply(text_processor.preprocess)
text_retriever.generate_embeddings()

# Test the RAG pipeline with vector store
rag_pipeline = RAGPipeline(text_retriever, qa_generator)
query = 'When were the Voyager I and Voyager II launched?'
relevant_passages, most_relevant_passage, response, most_relevant_passage_filename = rag_pipeline.run(query, top_k=3)

print(f"Retrieved Passages (3x):\n", relevant_passages)
print()
print(f"Most Relevant Passage Used for Response from file {most_relevant_passage_filename}:\n", most_relevant_passage)

print()
print(f"RAG Response:\n", response)
