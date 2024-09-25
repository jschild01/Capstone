import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle
import json
import xml.etree.ElementTree as ET
import fitz  # PyMuPDF for PDF processing
from PIL import Image
import torch
from transformers import AutoFeatureExtractor, AutoModel

class AdvancedFAISSRetriever:
    def __init__(self, text_model='hkunlp/instructor-xl', image_model='google/vit-base-patch16-224'):
        self.text_model = SentenceTransformer(text_model)
        self.image_feature_extractor = AutoFeatureExtractor.from_pretrained(image_model)
        self.image_model = AutoModel.from_pretrained(image_model)
        self.index = None
        self.documents = []
        self.metadata = []

    def process_text(self, text):
        return self.text_model.encode(text)

    def process_pdf(self, pdf_path):
        text = ""
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
        return self.process_text(text)

    def process_image(self, image_path):
        image = Image.open(image_path)
        inputs = self.image_feature_extractor(images=image, return_tensors="pt")
        outputs = self.image_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()

    def process_metadata(self, metadata):
        if isinstance(metadata, str):
            if metadata.endswith('.json'):
                with open(metadata, 'r') as f:
                    return json.load(f)
            elif metadata.endswith('.xml'):
                tree = ET.parse(metadata)
                return {elem.tag: elem.text for elem in tree.iter()}
        return metadata  # Assume it's already a dictionary

    def add_documents(self, documents, metadata_list):
        embeddings = []
        for doc in documents:
            if isinstance(doc, str):
                if doc.endswith('.txt'):
                    with open(doc, 'r') as f:
                        embeddings.append(self.process_text(f.read()))
                elif doc.endswith('.pdf'):
                    embeddings.append(self.process_pdf(doc))
                elif doc.lower().endswith(('.png', '.jpg', '.jpeg')):
                    embeddings.append(self.process_image(doc))
                else:
                    embeddings.append(self.process_text(doc))
            else:
                raise ValueError(f"Unsupported document type: {type(doc)}")

        embeddings = np.array(embeddings)

        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
        
        self.index.add(embeddings.astype('float32'))
        self.documents.extend(documents)
        self.metadata.extend([self.process_metadata(m) for m in metadata_list])

    def search(self, query, k=5):
        if isinstance(query, str):
            query_vector = self.process_text(query)
        elif query.lower().endswith(('.png', '.jpg', '.jpeg')):
            query_vector = self.process_image(query)
        else:
            raise ValueError(f"Unsupported query type: {type(query)}")

        distances, indices = self.index.search(query_vector.reshape(1, -1).astype('float32'), k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            results.append({
                'document': self.documents[idx],
                'metadata': self.metadata[idx],
                'distance': distances[0][i]
            })
        
        return results

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadata': self.metadata
            }, f)
        faiss.write_index(self.index, f"{file_path}.index")

    def load(self, file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.metadata = data['metadata']
        self.index = faiss.read_index(f"{file_path}.index")

class AdvancedRAGPipeline:
    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.generator = generator

    def run(self, query, k=5):
        results = self.retriever.search(query, k)
        
        context = ""
        for result in results:
            context += f"Document: {result['document']}\n"
            context += f"Metadata: {json.dumps(result['metadata'])}\n\n"
        
        response = self.generator.generate(query, context)
        return results, response

# Usage example
retriever = AdvancedFAISSRetriever()
generator = EnhancedRAGGenerator()  # Assuming we keep the existing generator
pipeline = AdvancedRAGPipeline(retriever, generator)

# Add documents to the index
documents = ["document1.txt", "document2.pdf", "image1.jpg"]
metadata_list = [
    "metadata1.json",
    "metadata2.xml",
    {"author": "John Doe", "date": "2023-01-01"}
]
retriever.add_documents(documents, metadata_list)

# Run a query
query = "Your question here"  # This could also be an image path for image queries
results, response = pipeline.run(query)

print(f"Response: {response}")
print("\nRelevant Documents:")
for result in results:
    print(f"Document: {result['document']}")
    print(f"Metadata: {result['metadata']}")
    print(f"Distance: {result['distance']}")
    print()
