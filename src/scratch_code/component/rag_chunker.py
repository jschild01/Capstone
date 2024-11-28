import re
from typing import List, Tuple

class TextChunker:
    @staticmethod
    def chunk_by_sentence(text: str, max_chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

    @staticmethod
    def chunk_by_words(text: str, max_words: int = 200, overlap: int = 20) -> List[str]:
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), max_words - overlap):
            chunk = " ".join(words[i:i + max_words])
            chunks.append(chunk)
        
        return chunks

    @staticmethod
    def chunk_by_paragraphs(text: str, max_paragraphs: int = 3, overlap: int = 1) -> List[str]:
        paragraphs = text.split('\n\n')
        chunks = []
        
        for i in range(0, len(paragraphs), max_paragraphs - overlap):
            chunk = "\n\n".join(paragraphs[i:i + max_paragraphs])
            chunks.append(chunk)
        
        return chunks

    @staticmethod
    def chunk_by_fixed_size(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + chunk_size
            if end > text_length:
                end = text_length
            chunks.append(text[start:end])
            start = end - overlap

        return chunks
