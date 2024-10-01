from typing import List, Tuple
from langchain.schema import Document

class RAGPipeline:
    def __init__(self, text_retriever, qa_generator):
        self.text_retriever = text_retriever
        self.qa_generator = qa_generator

    def run(self, query: str, top_k: int = 3) -> Tuple[List[Tuple[Document, float]], str, str]:
        # Retrieve relevant passages
        retrieved_docs = self.text_retriever.search_vector_store(query, top_k=top_k)
        
        # Find most relevant passage
        most_relevant_doc, _ = min(retrieved_docs, key=lambda x: x[1])
        
        # Generate response
        response = self.qa_generator.generate_response(query, most_relevant_doc.page_content)
        
        return retrieved_docs, most_relevant_doc.page_content, response
