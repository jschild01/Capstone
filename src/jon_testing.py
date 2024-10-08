import os
import re
import time
import torch
import gc
import shutil
import csv
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, pipeline
from langchain.schema import Document
from langchain.vectorstores import DeepLake
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

# Set seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Metadata processor
def process_metadata(data_dir: str) -> Dict[str, Dict]:
    def parse_csv(file_path: str, fields: List[str]) -> Dict[str, Dict]:
        metadata_dict = {}
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                metadata = {field: row.get(field, '') for field in fields}
                metadata_dict[row['id']] = metadata
        return metadata_dict

    search_results_path = os.path.join(data_dir, 'search_results.csv')
    id_to_metadata = parse_csv(search_results_path, fields=['title', 'date', 'contributors', 'type', 'language'])
    return id_to_metadata

# RAG Retriever Class
class RAGRetriever:
    def __init__(self, dataset_path: str = './my_deeplake', model_name: str = 'all-MiniLM-L6-v2'):
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.dataset_path = dataset_path
        self.vectorstore = self.load_vectorstore()
        self.documents = []

    def load_vectorstore(self):
        if os.path.exists(self.dataset_path):
            return DeepLake(dataset_path=self.dataset_path, embedding_function=self.embeddings, read_only=False)
        return self.create_new_vectorstore()

    def create_new_vectorstore(self):
        return DeepLake(dataset_path=self.dataset_path, embedding_function=self.embeddings)

    def load_data(self, data_dir: str, metadata: Dict[str, Dict]) -> List[Document]:
        txt_dir = os.path.join(data_dir, 'txt')
        for filename in os.listdir(txt_dir):
            if filename.endswith('.txt'):
                file_path = os.path.join(txt_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                doc_metadata = metadata.get(filename, {})
                doc_metadata['original_filename'] = filename
                doc = Document(page_content=content, metadata=doc_metadata)
                self.documents.append(doc)
        return self.documents

    def generate_embeddings(self, documents: List[Document]):
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        self.vectorstore.add_texts(texts, metadatas=metadatas)

    def search_vector_store(self, query: str, top_k: int = 3):
        return self.vectorstore.similarity_search(query, k=top_k)
    
    def is_empty(self):
        try:
            return len(self.vectorstore) == 0
        except Exception as e:
            print(f"Error checking if vectorstore is empty: {e}")
            return True

# RAG Generator Class
class RAGGenerator:
    def __init__(self, model_name):
        if model_name == 'llama3':
            model_name = 'meta-llama/Llama-3.2-3B-Instruct'
            hf_token = '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'
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_token)
        elif model_name == 't5':
            model_name = 'google/flan-t5-xl'
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        else:
            raise ValueError("Invalid model name provided. Input either 'llama' or 't5' as model name.")

    def generate_response(self, prompt: str, max_length: int = 300) -> str:
        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
        attention_mask = inputs['attention_mask']
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        outputs = self.model.generate(inputs['input_ids'], attention_mask=attention_mask, max_new_tokens=max_length, num_return_sequences=1, do_sample=True, temperature=0.7, pad_token_id=self.tokenizer.pad_token_id)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

# RAG Pipeline Class
class RAGPipeline:
    def __init__(self, text_retriever: RAGRetriever, qa_generator: RAGGenerator):
        self.text_retriever = text_retriever
        self.qa_generator = qa_generator
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    def run(self, query: str, top_k: int = 3) -> Tuple[List[Document], str]:
        retrieved_docs = self.text_retriever.search_vector_store(query, top_k=top_k)
        combined_context = " ".join([doc.page_content[:500] for doc in retrieved_docs])  # Limit context to 500 chars per document
        summarized_context = self.summarizer(combined_context, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
        prompt = f"Answer the question concisely based on the given context.\nQuestion: {query}\nContext: {summarized_context}\nAnswer:"
        response = self.qa_generator.generate_response(prompt)
        return retrieved_docs, response

# Main function to run the RAG system
def test_rag_system(data_dir: str, query: str, delete_existing: bool = False):
    metadata = process_metadata(data_dir)
    text_retriever = RAGRetriever(dataset_path=os.path.join(data_dir, 'deeplake_dataset'))

    if delete_existing:
        shutil.rmtree(text_retriever.dataset_path)
        text_retriever.vectorstore = text_retriever.create_new_vectorstore()

    if text_retriever.is_empty():
        documents = text_retriever.load_data(data_dir, metadata)
        text_retriever.generate_embeddings(documents)

    qa_generator = RAGGenerator(model_name='llama3')
    rag_pipeline = RAGPipeline(text_retriever, qa_generator)

    retrieved_docs, response = rag_pipeline.run(query, top_k=3)
    print(f"Retrieved Documents: {[doc.metadata['original_filename'] for doc in retrieved_docs]}\n")
    print(f"Generated Response: {response}\n")

if __name__ == "__main__":
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_path, 'data', 'marc-xl-data')

    set_seed(42)
    #data_dir = input("Enter the data directory: ")
    query = input("Enter your question: ")
    delete_existing = input("Do you want to delete the existing dataset? (y/n): ").lower() == 'y'
    test_rag_system(data_dir, query, delete_existing)
