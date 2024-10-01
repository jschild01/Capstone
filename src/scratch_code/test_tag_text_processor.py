import unittest
import os
import sys

# Add parent directory to sys path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Add the src folder in parent directory to sys path, which contains the component files
src_dir = os.path.join(parent_dir, 'src')
sys.path.append(src_dir)

# Import the RAGRetriever class for test file
from component.rag_text_processor import TextProcessor

class TestTextProcessor(unittest.TestCase):
    def setUp(self):
        # Initialize the TextProcessor instance
        self.processor = TextProcessor()

    def test_custom_preprocess(self):
        # Test the custom_preprocess method
        text = "This text was transcribed and reviewed by contributors participating in the by the people project at crowd.loc.gov.\nNew line here."
        expected = "This text was New line here."
        result = self.processor.custom_preprocess(text)
        self.assertEqual(result, expected)

    def test_preprocess_and_split(self):
        # Test the preprocess_and_split method
        text = "# Header 1\nThis is some text.\n## Header 2\nMore text here."
        chunks = self.processor.preprocess_and_split(text)
        # Check if the splitting is correct
        self.assertTrue(len(chunks) > 0)
        self.assertTrue(all(isinstance(chunk, str) for chunk in chunks))

    def test_chunk_text(self):
        # Test the chunk_text method
        text = "This is a long text. " * 100
        chunks = self.processor.chunk_text(text, chunk_size=50, chunk_overlap=10)
        # Check if the chunking works correctly
        self.assertTrue(len(chunks) > 0)
        self.assertTrue(all(len(chunk) <= 50 for chunk in chunks))

if __name__ == "__main__":
    unittest.main()
