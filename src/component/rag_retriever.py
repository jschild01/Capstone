import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import torch
from .text_processor import TextProcessor

class RAGRetriever:
    def __init__(self, model_name='mpnet', chunk_size=1000, chunk_overlap=200):
        # Allow selection between 'mpnet' and 'instructxl'
        if model_name == 'mpnet':
            self.model = SentenceTransformer('all-mpnet-base-v2')
        elif model_name == 'instructor-xl':
            self.model = SentenceTransformer('hkunlp/instructor-xl')
        else:
            raise ValueError("Invalid model_name. Choose 'mpnet' or 'instructor-xl'.")

        self.index = None
        self.df = None
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_processor = TextProcessor()

        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)

    def load_data(self, df):
        self.df = df

    def generate_embeddings(self):
        all_chunks = []
        chunk_to_doc_mapping = []

        for idx, row in self.df.iterrows():
            chunks = self.text_processor.preprocess_and_split(row['text'])
            all_chunks.extend(chunks)
            chunk_to_doc_mapping.extend([idx] * len(chunks))

        # Process embeddings in batches
        batch_size = 32
        embeddings_list = []
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i+batch_size]
            with torch.no_grad():
                embeddings = self.model.encode(batch, convert_to_tensor=True, show_progress_bar=False)
            embeddings_list.append(embeddings.cpu().numpy())

        embeddings_np = np.vstack(embeddings_list)
        dimension = embeddings_np.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings_np)

        self.chunk_to_doc_mapping = chunk_to_doc_mapping
        self.all_chunks = all_chunks

        # Clear some memory
        del embeddings_list, embeddings_np
        torch.cuda.empty_cache()

    def search_vector_store(self, query, top_k=10):
        query_embedding = self.model.encode([query], convert_to_tensor=True).squeeze(0)
        query_embedding_np = query_embedding.cpu().numpy().reshape(1, -1)
        
        distances, indices = self.index.search(query_embedding_np.astype('float32'), top_k)
        retrieved_chunks = [self.all_chunks[i] for i in indices[0]]
        retrieved_doc_indices = [self.chunk_to_doc_mapping[i] for i in indices[0]]

        return retrieved_chunks, retrieved_doc_indices
