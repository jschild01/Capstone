import pandas as pd
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import faiss

# Hardcoded CSV file paths
subset_csv = '/home/ubuntu/Capstone/src/data/subset_for_examine100.csv'
file_list_csv = '/home/ubuntu/Capstone/src/data/file_list.csv'
search_results_csv = '/home/ubuntu/Capstone/src/data/search_results.csv'

# Load the CSV files
try:
    subset_df = pd.read_csv(subset_csv)
    file_list_df = pd.read_csv(file_list_csv)
    search_results_df = pd.read_csv(search_results_csv)
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Check the file paths and make sure the files are in the right location.")
    exit(1)


class RAGRetriever:
    def __init__(self, model_name='all-MiniLM-L6-v2', chunk_size=1000, chunk_overlap=200):
        self.model = SentenceTransformer(model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.index = None
        self.df = None

    def load_data(self, df):
        self.df = df

    def generate_embeddings(self):
        texts = self.df['combined_text'].tolist()
        embeddings = self.model.encode(texts, show_progress_bar=True)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings.astype('float32'))

    def search(self, query, top_k=5):
        query_vector = self.model.encode([query])[0]
        distances, indices = self.index.search(query_vector.reshape(1, -1).astype('float32'), top_k)
        return [self.df.iloc[i]['combined_text'] for i in indices[0]]


class RAGGenerator:
    def __init__(self, model_name='google/flan-t5-base'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def generate(self, query, context, max_length=150):
        # Use context, which now includes metadata if available
        input_text = f"Question: {query}\nContext: {context}\nAnswer:"
        input_ids = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).input_ids
        outputs = self.model.generate(input_ids, max_length=max_length, num_return_sequences=1, do_sample=True)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


class RAGPipeline:
    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.generator = generator

    def run(self, query, top_k=3):
        # Retrieve the relevant passages
        relevant_passages = self.retriever.search(query, top_k)
        context = " ".join(relevant_passages)

        # Get the filename for the most relevant passage
        most_relevant_filename = self.retriever.df.iloc[0]['filename']

        # Retrieve metadata for the most relevant file
        resource, metadata = find_document_by_filename(most_relevant_filename)

        # Combine metadata and context for answer generation
        if metadata is not None:
            metadata_info = " ".join([f"{col}: {metadata[col]}" for col in metadata.index if pd.notna(metadata[col])])
            context += f"\nMetadata: {metadata_info}"

        # Generate the final response
        response = self.generator.generate(query, context)
        return relevant_passages, relevant_passages[0], response, most_relevant_filename


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


def setup_rag_pipeline():
    # Combine text from relevant columns in subset_df
    subset_df['combined_text'] = subset_df['text'] + " " + subset_df['clean_text']

    # Initialize RAG components
    text_retriever = RAGRetriever(model_name='all-MiniLM-L6-v2', chunk_size=1000, chunk_overlap=200)
    text_retriever.load_data(subset_df[['filename', 'combined_text']])
    text_retriever.generate_embeddings()

    qa_generator = RAGGenerator(model_name='google/flan-t5-base')
    rag_pipeline = RAGPipeline(text_retriever, qa_generator)

    return rag_pipeline


def rag_query(pipeline, query, top_k=3):
    relevant_passages, most_relevant_passage, response, most_relevant_passage_id = pipeline.run(query, top_k=top_k)

    print(f"Query: {query}")
    print(f"\nRAG Response: {response}")
    print(f"\nMost Relevant Passage (from document {most_relevant_passage_id}):\n{most_relevant_passage}")
    print("\nRelevant Passages:")
    for i, passage in enumerate(relevant_passages, 1):
        print(f"{i}. {passage[:100]}...")  # Print first 100 characters of each passage


def main():
    rag_pipeline = setup_rag_pipeline()

    print("Testing get_document_metadata:")
    get_document_metadata('001.txt')

    print("\nTesting RAG query:")
    rag_query(rag_pipeline, "What can you tell me about John Lomax's work on folk songs?")

    print("\nTesting another RAG query:")
    rag_query(rag_pipeline, "Describe the contents of the Voyager record.")


if __name__ == "__main__":
    main()
