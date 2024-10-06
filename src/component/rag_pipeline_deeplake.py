from typing import List, Tuple
from langchain.schema import Document

class RAGPipeline:
    def __init__(self, text_retriever, qa_generator):
        self.text_retriever = text_retriever
        self.qa_generator = qa_generator

    def run(self, query: str, top_k: int = 3) -> Tuple[List[Document], str, str]:
        # Retrieve relevant passages
        retrieved_docs = self.text_retriever.search_vector_store(query, top_k=top_k)

        # Find most relevant passage (assuming the first one is the most relevant)
        most_relevant_doc = retrieved_docs[0] if retrieved_docs else None

        if most_relevant_doc:
            # Generate response
            response = self.qa_generator.generate_response(query, most_relevant_doc.page_content)
            return retrieved_docs, most_relevant_doc.page_content, response
        else:
            return [], "", "No relevant documents found."