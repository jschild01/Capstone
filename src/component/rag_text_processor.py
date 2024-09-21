import re
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

class TextProcessor:
    def __init__(self):
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ]
        )
        self.char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
    
    def custom_preprocess(self, text):
        # Remove specific text
        text = re.sub(r'transcribed and reviewed by contributors participating in the by the people project at crowd.loc.gov.', '', text, flags=re.IGNORECASE)
        
        # Add any other custom preprocessing steps here
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def preprocess_and_split(self, text):
        # First, apply custom preprocessing
        text = self.custom_preprocess(text)
        
        # Then, split by markdown headers
        md_splits = self.markdown_splitter.split_text(text)
        
        # Finally, further split each section
        chunks = []
        for doc in md_splits:
            chunks.extend(self.char_splitter.split_text(doc.page_content))
        
        return chunks

    def chunk_text(self, text, chunk_size=1000, chunk_overlap=200):
        # First, apply custom preprocessing
        text = self.custom_preprocess(text)
        
        # Then, use LangChain's splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        return splitter.split_text(text)
