import os
import sys
import time
import pandas as pd
import torch
import numpy as np
import random
import gc
import re
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, T5ForConditionalGeneration
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Hardcoded CSV file paths
subset_csv = '/home/ubuntu/Capstone/src/data/subset_for_examine100.csv'
file_list_csv = '/home/ubuntu/Capstone/src/data/file_list.csv'
search_results_csv = '/home/ubuntu/Capstone/src/data/search_results.csv'


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class TextProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def custom_preprocess(self, text):
        text = re.sub(
            r'transcribed and reviewed by contributors participating in the by the people project at crowd.loc.gov.',
            '', text, flags=re.IGNORECASE)
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def preprocess_and_split(self, text):
        preprocessed_text = self.custom_preprocess(text)
        return self.text_splitter.split_text(preprocessed_text)


class EnhancedRAGRetriever:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.model = SentenceTransformer('hkunlp/instructor-xl')
        self.text_processor = TextProcessor(chunk_size, chunk_overlap)
        self.doc_index = None
        self.chunk_index = None
        self.df = None
        self.chunks = []
        self.chunk_to_doc_mapping = []

    def load_data(self, df):
        self.df = df

    def generate_embeddings(self):
        all_chunks = []
        chunk_to_doc_mapping = []
        metadata_enhanced_chunks = []

        for idx, row in self.df.iterrows():
            chunks = self.text_processor.preprocess_and_split(row['combined_text'])
            all_chunks.extend(chunks)
            chunk_to_doc_mapping.extend([idx] * len(chunks))

            metadata = self.get_metadata_for_document(row['filename'])
            metadata_text = self.metadata_to_text(metadata)
            metadata_enhanced_chunks.extend([f"{chunk} {metadata_text}" for chunk in chunks])

        # Use batching for embedding generation to manage memory
        batch_size = 32
        chunk_embeddings = []
        for i in range(0, len(metadata_enhanced_chunks), batch_size):
            batch = metadata_enhanced_chunks[i:i + batch_size]
            embeddings = self.model.encode(batch, show_progress_bar=True)
            chunk_embeddings.append(embeddings)

        chunk_embeddings = np.vstack(chunk_embeddings)
        self.chunk_index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
        self.chunk_index.add(chunk_embeddings.astype('float32'))

        # Generate document-level embeddings
        doc_embeddings = self.model.encode(self.df['combined_text'].tolist(), show_progress_bar=True)
        self.doc_index = faiss.IndexFlatL2(doc_embeddings.shape[1])
        self.doc_index.add(doc_embeddings.astype('float32'))

        self.chunks = all_chunks
        self.chunk_to_doc_mapping = chunk_to_doc_mapping

    def get_metadata_for_document(self, filename):
        resource, metadata = find_document_by_filename(filename)
        return metadata if metadata is not None else {}

    def metadata_to_text(self, metadata):
        return " ".join([f"{key}: {value}" for key, value in metadata.items() if pd.notna(value)])

    def hierarchical_search(self, query, top_k_docs=3, top_k_passages=5):
        query_vector = self.model.encode([query])[0]

        # Search for top documents
        _, doc_indices = self.doc_index.search(query_vector.reshape(1, -1).astype('float32'), top_k_docs)

        # Search for top passages within those documents
        relevant_passages = []
        passage_doc_mapping = []
        for doc_idx in doc_indices[0]:
            doc_chunks = [self.chunks[i] for i in range(len(self.chunks)) if self.chunk_to_doc_mapping[i] == doc_idx]
            chunk_vectors = self.model.encode(doc_chunks)
            temp_index = faiss.IndexFlatL2(chunk_vectors.shape[1])
            temp_index.add(chunk_vectors.astype('float32'))

            _, passage_indices = temp_index.search(query_vector.reshape(1, -1).astype('float32'), top_k_passages)
            relevant_passages.extend([doc_chunks[i] for i in passage_indices[0]])
            passage_doc_mapping.extend([self.df.iloc[doc_idx]['filename']] * len(passage_indices[0]))

        # Sort passages by relevance
        passage_vectors = self.model.encode(relevant_passages)
        _, sorted_indices = faiss.IndexFlatL2(passage_vectors.shape[1]).search(
            query_vector.reshape(1, -1).astype('float32'), len(relevant_passages))

        sorted_passages = [relevant_passages[i] for i in sorted_indices[0]]
        sorted_doc_mapping = [passage_doc_mapping[i] for i in sorted_indices[0]]

        return sorted_passages, sorted_doc_mapping


class EnhancedRAGGenerator:
    def __init__(self, model_name='google/flan-t5-base'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

    def generate(self, query, context, max_length=150):
        input_text = f"Question: {query}\nContext: {context}\nAnswer:"
        input_ids = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).input_ids

        outputs = self.model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
            top_k=10,
            top_p=0.95,
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


class EnhancedRAGPipeline:
    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.generator = generator

    def run(self, query, top_k_docs=3, top_k_passages=5, max_context_length=1000):
        relevant_passages, doc_filenames = self.retriever.hierarchical_search(query, top_k_docs, top_k_passages)

        # Build context from multiple chunks
        context = ""
        metadata_used = []
        for passage, filename in zip(relevant_passages, doc_filenames):
            if len(context) + len(passage) > max_context_length:
                break
            context += passage + " "

            resource, metadata = find_document_by_filename(filename)
            if metadata is not None and filename not in [m[0] for m in metadata_used]:
                metadata_info = " ".join(
                    [f"{col}: {metadata[col]}" for col in metadata.index if pd.notna(metadata[col])])
                metadata_used.append((filename, metadata_info))

        # Add metadata to context
        for filename, metadata_info in metadata_used:
            context += f"\nMetadata for {filename}: {metadata_info}"

        response = self.generator.generate(query, context)
        return relevant_passages, doc_filenames, response, metadata_used


def find_document_by_filename(filename):
    matching_row = file_list_df[file_list_df['source_url'].str.endswith(filename, na=False)]
    if matching_row.empty:
        return None, None
    doc_id = matching_row['id'].iloc[0]
    metadata = search_results_df[search_results_df['id'] == doc_id]
    if metadata.empty:
        return matching_row.iloc[0], None
    return matching_row.iloc[0], metadata.iloc[0]


def get_document_metadata(filename):
    resource, metadata = find_document_by_filename(filename)
    if resource is None:
        print(f"No resource found for filename: {filename}")
        return
    if metadata is None:
        print(f"Resource found, but no metadata for filename: {filename}")
        print("Resource info:")
        print(resource.to_dict())
        return
    print(f"Metadata for {filename}:")
    for column in metadata.index:
        value = metadata[column]
        if pd.notna(value):
            print(f"{column}: {value}")
    print("\nResource info:")
    for column in resource.index:
        value = resource[column]
        if pd.notna(value):
            print(f"{column}: {value}")


def test_chunking_method(chunk_size, chunk_overlap, df, query, top_k_docs=3, top_k_passages=5, max_context_length=1000):
    retriever = EnhancedRAGRetriever(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    retriever.load_data(df)
    retriever.generate_embeddings()

    generator = EnhancedRAGGenerator()
    rag_pipeline = EnhancedRAGPipeline(retriever, generator)

    start_time = time.time()
    relevant_passages, doc_filenames, response, metadata_used = rag_pipeline.run(query, top_k_docs, top_k_passages,
                                                                                 max_context_length)
    end_time = time.time()
    total_time = end_time - start_time

    print(f"\nChunk Size: {chunk_size}, Chunk Overlap: {chunk_overlap}")
    print(f"Top-K Documents: {top_k_docs}, Top-K Passages: {top_k_passages}, Max Context Length: {max_context_length}")

    print(f"\nRelevant Documents:")
    for filename in set(doc_filenames):
        print(f"- {filename}")

    print(f"\nRelevant Passages:")
    for i, (passage, filename) in enumerate(zip(relevant_passages, doc_filenames), 1):
        print(f"{i}. [{filename}] {passage[:100]}...")

    print(f"\nEnhanced RAG Response:\n{response}")

    print(f"\nMetadata Used:")
    for filename, metadata in metadata_used:
        print(f"- {filename}: {metadata}")

    print(f"\nTotal processing time: {total_time:.2f} seconds")
    print("-" * 50)

    # Clear memory
    del retriever, generator, rag_pipeline
    gc.collect()
    torch.cuda.empty_cache()


def main():
    # Set seed for reproducibility
    set_seed(42)

    # Load the CSV files
    try:
        subset_df = pd.read_csv(subset_csv)
        global file_list_df, search_results_df
        file_list_df = pd.read_csv(file_list_csv)
        search_results_df = pd.read_csv(search_results_csv)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Check the file paths and make sure the files are in the right location.")
        return

    # Combine text from relevant columns and apply preprocessing
    text_processor = TextProcessor()
    subset_df['combined_text'] = subset_df['text'] + " " + subset_df['clean_text']
    subset_df['combined_text'] = subset_df['combined_text'].apply(text_processor.custom_preprocess)

    print("Testing metadata retrieval:")
    get_document_metadata('001.txt')
    print("\n" + "=" * 50 + "\n")

    query = input("\nEnter your question: ")

    chunking_params = [
        (1000, 200),
        (500, 100),
        (1500, 300),
    ]

    for chunk_size, chunk_overlap in chunking_params:
        test_chunking_method(chunk_size, chunk_overlap, subset_df, query, top_k_docs=3, top_k_passages=5,
                             max_context_length=1000)


if __name__ == "__main__":
    main()