#%%
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
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Function to get all text; might take a while because of the number of files
def getText():
    files = glob.glob('text/*.txt')
    file_data = []
    df = pd.DataFrame(columns=['filename', 'text'])

    for file in files:
        with open(file, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        file_data.append({'filename': file, 'text': text})

    df = pd.DataFrame(file_data)
    return df

# Function to clean the text
def preprocess(text, lowercase=False, filter_invalid_sents=False, remove_punctuation=False, remove_stopwords=False, use_lemma=False):
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'([{}])\1+'.format(re.escape(string.punctuation)), r'\1', text)
    text = re.sub(r'-', ' ', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^A-Za-z0-9.!? ]+', '', text)
    text = re.sub(r'transcribed and reviewed by contributors participating in the by the people project at crowd.loc.gov.', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()

    if filter_invalid_sents:
        text = filter_valid_sentences(text)

    if lowercase:
        text = text.lower()

    if remove_punctuation:
        text = text.translate(str.maketrans('', '', string.punctuation))

    words = text.split()
    if remove_stopwords:
        words = [word for word in words if word not in stop_words]

    if use_lemma:
        words = [lemmatizer.lemmatize(word) for word in words]

    return ' '.join(words)

def is_valid_sentence(sentence):
    doc = nlp(sentence)
    has_subject = any(token.dep_ in ("nsubj", "nsubjpass") for token in doc)
    has_verb = any(token.pos_ == "VERB" for token in doc)
    return has_subject and has_verb

def filter_valid_sentences(text):
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    valid_sentences = [sent for sent in sentences if is_valid_sentence(sent)]
    return " ".join(valid_sentences)

# Implement vector store search
def search_vector_store(query, top_k=10):
    query_embedding = model.encode([query], convert_to_tensor=False)
    query_embedding_np = np.array(query_embedding).astype('float32')
    distances, indices = index.search(query_embedding_np, top_k)
    retrieved_docs = df['clean_text'].iloc[indices[0]].tolist()

    # Rank passages by relevance (e.g., based on cosine similarity or additional scoring)
    # This is a placeholder; you can implement more sophisticated ranking logic
    return retrieved_docs

# Generate text response using a QA model like Flan-T5
def generate_response(query, retrieved_docs, max_new_tokens=50, temperature=0.3, top_p=0.85):
    # Use only the top 2-3 most relevant documents to generate response
    input_text = query + " ".join(retrieved_docs[:3])
    #input_text = " ".join(retrieved_docs[:3])
    
    # Truncate input text to ensure it fits within the model's context window
    input_text = input_text[:2000]  # Adjust the limit based on the model's max input size
    
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    output = generatorModel.generate(input_ids, 
                                     max_length=max_new_tokens + len(input_ids[0]), 
                                     temperature=temperature,
                                     top_p=top_p,
                                     num_return_sequences=1)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# RAG pipeline with vector store
def rag_pipeline_with_vectorstore(query, top_k=10):
    retrieved_docs = search_vector_store(query, top_k=top_k)
    response = generate_response(query, retrieved_docs)
    return retrieved_docs, response










# Set directory
os.chdir(r'C:\Users\schil\OneDrive\Desktop\Grad SChool\Capstone\LOC')

# Set up
lemmatizer = WordNetLemmatizer()
stop_words = set(nltk_stopwords.words('english'))
nlp = spacy.load("en_core_web_sm")

# Get the text; commented out because it takes a while and previously saved as a csv
# df = getText()
df = pd.read_csv('afc_txtFiles.csv')
df = df.head(100)  # Subset for testing

# Clean the text
df['clean_text'] = df['text'].apply(preprocess, lowercase=False, filter_invalid_sents=False, remove_punctuation=False, remove_stopwords=False, use_lemma=False)
df['clean_text_lower'] = df['clean_text'].apply(preprocess, lowercase=True, filter_invalid_sents=False, remove_punctuation=False, remove_stopwords=False, use_lemma=False)
df['clean_text_sents'] = df['text'].apply(preprocess, lowercase=False, filter_invalid_sents=True, remove_punctuation=False, remove_stopwords=False, use_lemma=False)
df['clean_text_lower_sents'] = df['text'].apply(preprocess, lowercase=True, filter_invalid_sents=True, remove_punctuation=False, remove_stopwords=False, use_lemma=False)

# Generate embeddings
model = SentenceTransformer('all-mpnet-base-v2')  # Embedding model
embeddings = model.encode(df['clean_text_lower'].tolist(), convert_to_tensor=True, show_progress_bar=False)

# Build FAISS vector store
embeddings_np = embeddings.cpu().numpy()
dimension = embeddings_np.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings_np)

# Load QA model (Flan-T5) for generation
tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-small')
generatorModel = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-small')

# Test the RAG pipeline with vector store
query = 'What date did Voyager I and Voyager II launch?'
retrieved_docs, response = rag_pipeline_with_vectorstore(query, top_k=3)
print('Number of Retrieved Docs:', len(retrieved_docs))
print()
print("Retrieved Docs Text:", retrieved_docs)
print()
print("RAG Response:", response)
