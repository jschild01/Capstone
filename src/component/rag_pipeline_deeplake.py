from typing import List, Tuple
from langchain.schema import Document
from .rag_generator_deeplake import RAGGenerator
from .rag_retriever_deeplake import RAGRetriever
from .rag_utils import generate_prompt, structure_response, integrate_metadata, validate_response

class RAGPipeline:
    def __init__(self, text_retriever: RAGRetriever, qa_generator: RAGGenerator):
        self.text_retriever = text_retriever
        self.qa_generator = qa_generator

    def run(self, query: str, top_k: int = 3) -> Tuple[List[Document], str, str]:
        retrieved_docs = self.text_retriever.search_vector_store(query, top_k=top_k)

        if not retrieved_docs:
            return [], "", "No relevant documents found."

        combined_context = " ".join([doc.page_content for doc in retrieved_docs])
        most_relevant_doc = retrieved_docs[0]

        prompt = generate_prompt(query, combined_context, most_relevant_doc.metadata)
        raw_response = self.qa_generator.generate_response(prompt)
        validated_response = validate_response(raw_response, most_relevant_doc.metadata)
        structured_response = structure_response(validated_response)
        final_response = integrate_metadata(structured_response, most_relevant_doc.metadata)

        return retrieved_docs, most_relevant_doc.page_content, raw_response, validated_response, structured_response, final_response