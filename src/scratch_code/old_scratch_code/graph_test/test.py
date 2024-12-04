import re
import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
import node2vec
from node2vec import Node2Vec
import networkx as nx

def custom_preprocess(text):
    # Remove specific text
    text = re.sub(r'transcribed and reviewed by contributors participating in the by the people project at crowd.loc.gov.', '', text, flags=re.IGNORECASE)
    
    # Additional preprocessing steps
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Lowercase
    #text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    return text

def sentence_aware_splitter(text, max_tokens=1000, overlap_sentences=5):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_tokens = 0

    for i, sentence in enumerate(sentences):
        sentence_tokens = word_tokenize(sentence)
        sentence_length = len(sentence_tokens)

        if current_tokens + sentence_length <= max_tokens:
            current_chunk.append(sentence)
            current_tokens += sentence_length
        else:
            chunks.append(' '.join(current_chunk))
            overlap = sentences[max(0, i - overlap_sentences):i]
            current_chunk = overlap + [sentence]
            current_tokens = sum(len(word_tokenize(s)) for s in current_chunk)

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def generate_graph_embeddings(graph):
    # Use Node2Vec or similar to generate graph embeddings
    node2vec = Node2Vec(graph, dimensions=128, walk_length=30, num_walks=200, workers=1)
    model = node2vec.fit(window=10, min_count=3, batch_words=4)
    
    # Create a dictionary of node embeddings
    node_embeddings = {str(node): model.wv[str(node)] for node in graph.nodes()}
    return node_embeddings

def combine_embeddings(text_embedding, graph_embedding, text_weight=0.7, graph_weight=0.3):
    combined_emb = (text_weight * text_embedding) + (graph_weight * graph_embedding)
    return combined_emb

def index_embeddings(df, embedding_model, node_embeddings):
    chunk_texts = [chunk for chunks in df['text_chunks'] for chunk in chunks]
    text_embeddings = embedding_model.encode(chunk_texts, batch_size=64, convert_to_tensor=False)

    # Combine text embeddings with graph embeddings (averaging or concatenation)
    combined_embeddings = [] # testing above
    for text_emb, chunk in zip(text_embeddings, chunk_texts):
        #node_key = str(chunk[:30])  # Simplified, ensure a proper mapping strategy
        node_key = str(hash(chunk))  # Alternative: Use a hash of the chunk text
        graph_emb = node_embeddings.get(node_key)
        #graph_emb = node_embeddings.get(node_key, np.zeros(len(text_emb)))  # Default to zero if not found
        
        if graph_emb is None:
            graph_emb = text_emb  # Default to text embedding if no graph embedding found

        combined_emb = combine_embeddings(text_emb, graph_emb)
        combined_embeddings.append(combined_emb)

    # Set up the FAISS index with appropriate dimension
    dimension = len(combined_embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    #index = faiss.IndexIVFFlat(dimension)

    # Convert combined embeddings to a NumPy array and add them to the index
    embeddings_array = np.array(combined_embeddings).astype('float32')
    index.add(embeddings_array)

    chunks_data = []
    chunk_counter = 0
    for idx, chunks in df['text_chunks'].items():
        for _ in chunks:
            chunks_data.append({
                'document_id': idx,
                'chunk': chunk_texts[chunk_counter],
                'embedding': combined_embeddings[chunk_counter]
            })
            chunk_counter += 1

    # Create a DataFrame from chunks data and merge it with the original DataFrame
    df_chunks = pd.DataFrame(chunks_data)
    df_merged = pd.merge(df, df_chunks, left_index=True, right_on='document_id')

    # Save the index and merged DataFrame
    faiss.write_index(index, 'test/embeddings/faiss_index_combined.bin')
    df_merged.to_parquet('test/embeddings/df_with_combined_embeddings.parquet', index=False)
    df_merged.to_csv('test/embeddings/df_with_combined_embeddings.csv', index=False)

    print(f"Indexed {len(embeddings_array)} chunks with combined embeddings for {len(df)} documents.")
    return df_merged, index, df_chunks

def enhanced_query_embedding(query, cross_encoder, embedding_model):
    # Use the cross-encoder to get a refined embedding of the query
    context_score_pairs = [(query, chunk) for chunk in df_chunks['chunk'].values]
    scores = cross_encoder.predict(context_score_pairs)
    
    # Select the highest scoring context for improved query embedding
    best_context = df_chunks.iloc[np.argmax(scores)]['chunk']
    refined_query_embedding = embedding_model.encode(best_context, convert_to_tensor=False)
    
    return refined_query_embedding

def retrieve_and_rerank(query, embedding_model, index, df_chunks, cross_encoder, top_k=5, text_embedding_size=768, graph_embedding_size=768):
    # Encode the query to get its text embedding
    text_embedding = embedding_model.encode(query, convert_to_tensor=False).astype('float32')
    
    # Ensure text embedding size matches expected size; adjust if necessary
    assert len(text_embedding) == text_embedding_size, f"Text embedding size is {len(text_embedding)}, expected {text_embedding_size}."
    
    # Create a placeholder graph embedding of zeros to match the graph embedding size used during indexing
    graph_embedding_placeholder = np.zeros(graph_embedding_size, dtype='float32')
    
    # Concatenate the text and graph placeholder embeddings to match the indexed dimensions
    query_embedding = enhanced_query_embedding(user_query, cross_encoder, embedding_model)
    #query_embedding = np.concatenate([text_embedding, graph_embedding_placeholder])
    
    # Ensure the combined query embedding matches the index dimensions
    assert query_embedding.shape[0] == index.d, f"Query embedding dimension {query_embedding.shape[0]} does not match index dimension {index.d}."
    
    # Perform the search in the FAISS index
    _, indices = index.search(np.array([query_embedding]), top_k)
    
    # Retrieve the most relevant chunks
    retrieved_chunks = df_chunks.iloc[indices[0]]
    
    # Re-rank retrieved chunks using cross-encoder
    chunk_texts = retrieved_chunks['chunk'].values.tolist()
    scores = cross_encoder.predict([(query, chunk) for chunk in chunk_texts])
    
    # Sort by scores in descending order to get the most relevant chunks first
    reranked_indices = np.argsort(scores)[::-1]
    reranked_chunks = retrieved_chunks.iloc[reranked_indices[:3]]
    
    return reranked_chunks

def generate_response(query, relevant_chunks, generative_model, generative_tokenizer, max_length=512):
    context = " ".join(relevant_chunks['chunk'].values)
    prompt = (
        f"Generate a response to the below Query based on the below Context:\n\n"
        f"Query: {query}\n\n"
        f"Context:\n{context}\n\n"
    )

    inputs = generative_tokenizer(prompt, return_tensors='pt', truncation=True, padding=True, max_length=max_length)
    output = generative_model.generate(**inputs, max_length=max_length, num_return_sequences=1, do_sample=True, temperature=0.9, top_p=0.9)
    response = generative_tokenizer.decode(output[0], skip_special_tokens=True)

    return response





# Load CSV file
base = '/home/ubuntu'
os.chdir(base)
df = pd.read_csv('test/afc_txtFiles.csv')
df = df.head(100)

# Initiate models
embedding_model = SentenceTransformer('hkunlp/instructor-xl') # requires embedding size of 768 in retrieve_and_rerank function
#embedding_model = SentenceTransformer('intfloat/e5-large') # requires embedding size of 1024 in retrieve_and_rerank function
generative_model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-large')
generative_tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-large')
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Preprocess text
df['text_clean'] = df['text'].apply(custom_preprocess)
df['text_chunks'] = df['text_clean'].apply(sentence_aware_splitter)

# Create a sample graph and generate graph embeddings (modify as needed)
G = nx.Graph()
# Add nodes and edges based on your knowledge graph
G.add_edges_from([(1, 2), (2, 3)])  # Simplified example; build this based on your knowledge graph data
node_embeddings = generate_graph_embeddings(G)

# Index combined embeddings and merge back into DataFrame
df_merged, index, df_chunks = index_embeddings(df, embedding_model, node_embeddings)

# Query processing
#user_query = "What are Voyager I and Voyager II?" # correct
#user_query = "When will Voyager I and Voyager II be launched?" # correct
#user_query = "How long after the Pioneer vehicles will Voyager vehicles be launched?" # wrong
user_query = "How many years after the Pioneer vehicles will Voyager vehicles be launched?" # correct

relevant_chunks = retrieve_and_rerank(user_query, embedding_model, index, df_chunks, cross_encoder, top_k=3)
response = generate_response(user_query, relevant_chunks, generative_model, generative_tokenizer)

print(f"\nGenerated Response:\n", response)