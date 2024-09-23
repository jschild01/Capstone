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
from component.rag_chunker import TextChunker

class TestTextChunker(unittest.TestCase):

    def setUp(self):
        self.text = (
            "This is the first sentence. Here is the second sentence. "
            "This is the third sentence. This should test chunking by sentences. "
            "Let's see how it works! "
        )

        self.paragraph_text = (
            "Paragraph one.\n\nParagraph two.\n\nParagraph three.\n\n"
            "Paragraph four.\n\n"
        )

    def test_chunk_by_sentence(self):
        chunks = TextChunker.chunk_by_sentence(self.text, max_chunk_size=50)
        self.assertTrue(len(chunks) > 1)
        self.assertIn("This is the first sentence.", chunks[0])

    def test_chunk_by_words(self):
        chunks = TextChunker.chunk_by_words(self.text, max_words=10)
        self.assertTrue(len(chunks) > 1)
        self.assertIn("This is the first sentence.", chunks[0])

    def test_chunk_by_paragraphs(self):
        chunks = TextChunker.chunk_by_paragraphs(self.paragraph_text, max_paragraphs=2)
        self.assertTrue(len(chunks) > 1)
        self.assertIn("Paragraph one.", chunks[0])

    def test_chunk_by_fixed_size(self):
        chunks = TextChunker.chunk_by_fixed_size(self.text, chunk_size=50)
        self.assertTrue(len(chunks) > 1)
        self.assertEqual(chunks[0], self.text[:50])

if __name__ == '__main__':
    unittest.main()

