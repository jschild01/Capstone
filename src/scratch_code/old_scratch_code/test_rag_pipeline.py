
import unittest
from unittest.mock import Mock
import sys
import os

# Add parent directory to sys path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Add the src folder in parent directory to sys path, which contains the component files
src_dir = os.path.join(parent_dir, 'src')
sys.path.append(src_dir)

# Import the RAGRetriever class for test file
from component.rag_pipeline import RAGPipeline



class TestRAGPipeline(unittest.TestCase):

    def setUp(self):
        # Mocking text retriever and QA generator
        self.mock_text_retriever = Mock()
        self.mock_qa_generator = Mock()

        # Sample data for mocking
        self.mock_text_retriever.search_vector_store.return_value = (["Passage 1", "Passage 2"], [0, 1])
        self.mock_text_retriever.model.encode.return_value = [0.5, 0.5]
        self.mock_text_retriever.df = Mock()
        self.mock_text_retriever.df.iloc.__getitem__.return_value = {'filename': 'file1.txt'}

        self.mock_qa_generator.generate_response.return_value = "Generated Answer"

        # Initialize the pipeline
        self.pipeline = RAGPipeline(self.mock_text_retriever, self.mock_qa_generator)

    def test_run(self):
        query = "Sample query"
        relevant_passages, most_relevant_passage, response, filename = self.pipeline.run(query)

        # Assertions
        self.assertEqual(len(relevant_passages), 2)
        self.assertEqual(most_relevant_passage, "Passage 1")
        self.assertEqual(response, "Generated Answer")
        self.assertEqual(filename, 'file1.txt')

        # Check if the correct methods were called
        self.mock_text_retriever.search_vector_store.assert_called_once_with(query, top_k=10)
        self.mock_qa_generator.generate_response.assert_called_once_with(query, "Passage 1")

if __name__ == '__main__':
    unittest.main()
