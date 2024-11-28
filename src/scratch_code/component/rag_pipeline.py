class RAGPipeline:
    def __init__(self, text_retriever, qa_generator):
        self.text_retriever = text_retriever
        self.qa_generator = qa_generator

    def run(self, query, top_k=10):
        # Retrieve relevant passages
        relevant_passages, indices = self.text_retriever.search_vector_store(query, top_k=top_k)
        
        # Find most relevant passage
        query_embedding = self.text_retriever.model.encode([query], convert_to_tensor=True).squeeze(0)
        most_relevant_passage = max(
            relevant_passages,
            key=lambda passage: self.text_retriever.model.encode([passage], convert_to_tensor=True).squeeze(0).dot(query_embedding).item()
        )

        # Get filename of most relevant passage
        most_relevant_passage_index = indices[relevant_passages.index(most_relevant_passage)]
        most_relevant_passage_filename = self.text_retriever.df.iloc[most_relevant_passage_index]['filename']

        # Generate response
        response = self.qa_generator.generate_response(query, most_relevant_passage)
        
        return relevant_passages, most_relevant_passage, response, most_relevant_passage_filename
