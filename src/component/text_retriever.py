import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import torch
from .text_chunker import TextChunker

class TextRetriever:
    def __init__(self, chunking_method: str = 'sentence', chunking_params: dict = None):
        self.model = SentenceTransformer('all-mpnet-base-v2')
        self.index = None
        self.df = None
        self.chunking_method = chunking_method
        self.chunking_params = chunking_params or {}
        self.chunker = TextChunker()

        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)

    def load_data(self, df):
        self.df = df

    def chunk_text(self, text):
        if self.chunking_method == 'sentence':
            return self.chunker.chunk_by_sentence(text, **self.chunking_params)
        elif self.chunking_method == 'words':
            return self.chunker.chunk_by_words(text, **self.chunking_params)
        elif self.chunking_method == 'paragraphs':
            return self.chunker.chunk_by_paragraphs(text, **self.chunking_params)
        elif self.chunking_method == 'fixed_size':
            return self.chunker.chunk_by_fixed_size(text, **self.chunking_params)
        else:
            raise ValueError(f"Unknown chunking method: {self.chunking_method}")

    def generate_embeddings(self):
        all_chunks = []
        chunk_to_doc_mapping = []

        for idx, row in self.df.iterrows():
            chunks = self.chunk_text(row['clean_text'])
            all_chunks.extend(chunks)
            chunk_to_doc_mapping.extend([idx] * len(chunks))

        embeddings = self.model.encode(all_chunks, convert_to_tensor=True, show_progress_bar=False)
        embeddings_np = embeddings.cpu().numpy()
        dimension = embeddings_np.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings_np)

        self.chunk_to_doc_mapping = chunk_to_doc_mapping
        self.all_chunks = all_chunks

    def search_vector_store(self, query, top_k=10):
        query_embedding = self.model.encode([query], convert_to_tensor=True).squeeze(0)
        query_embedding_np = query_embedding.cpu().numpy().reshape(1, -1)
        
        distances, indices = self.index.search(query_embedding_np.astype('float32'), top_k)
        retrieved_chunks = [self.all_chunks[i] for i in indices[0]]
        retrieved_doc_indices = [self.chunk_to_doc_mapping[i] for i in indices[0]]

        return retrieved_chunks, retrieved_doc_indices
