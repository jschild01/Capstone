import deeplake
from sentence_transformers import SentenceTransformer
import torch

class DeeplakeRAGRetriever:
    def __init__(self, dataset_path, embedding_model='hkunlp/instructor-xl'):
        self.dataset = deeplake.load(dataset_path)
        self.model = SentenceTransformer(embedding_model)

    def add_document(self, text, metadata):
        with self.dataset:
            self.dataset.append({
                'text': text,
                'metadata': metadata,
                'embedding': self.model.encode(text)
            })

    def search(self, query, k=5):
        query_embedding = self.model.encode(query)
        
        # Assuming the dataset has an 'embedding' tensor
        similarities = torch.nn.functional.cosine_similarity(
            torch.tensor(query_embedding).unsqueeze(0),
            torch.tensor(self.dataset.embedding.numpy()),
            dim=1
        )
        
        top_k_indices = similarities.argsort(descending=True)[:k]
        
        results = []
        for idx in top_k_indices:
            results.append({
                'text': self.dataset.text[idx].data()['value'],
                'metadata': self.dataset.metadata[idx].data()['value'],
                'similarity': similarities[idx].item()
            })
        
        return results

class EnhancedRAGPipeline:
    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.generator = generator

    def run(self, query, k=5):
        results = self.retriever.search(query, k)
        
        context = ""
        for result in results:
            context += f"{result['text']}\n"
            context += f"Metadata: {result['metadata']}\n\n"
        
        response = self.generator.generate(query, context)
        return results, response

# Usage example
dataset_path = '/path/to/your/deeplake/dataset'
retriever = DeeplakeRAGRetriever(dataset_path)
generator = EnhancedRAGGenerator()  # Assuming we keep the existing generator
pipeline = EnhancedRAGPipeline(retriever, generator)

# Add documents to the dataset
retriever.add_document("Document text 1", {"author": "John Doe", "date": "2023-01-01"})
retriever.add_document("Document text 2", {"author": "Jane Smith", "date": "2023-02-15"})

# Run a query
query = "Your question here"
results, response = pipeline.run(query)

print(f"Response: {response}")
print("\nRelevant Documents:")
for result in results:
    print(f"Text: {result['text'][:100]}...")
    print(f"Metadata: {result['metadata']}")
    print(f"Similarity: {result['similarity']}")
    print()
