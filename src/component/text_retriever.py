import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

class TextRetriever:
    def __init__(self):
        self.model = SentenceTransformer('all-mpnet-base-v2')  # Embedding model
        self.index = None
        self.df = None

    def load_data(self, df):
        self.df = df

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

        return relevant_passages, indices[0]
