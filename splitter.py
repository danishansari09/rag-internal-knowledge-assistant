from langchain_text_splitters import RecursiveCharacterTextSplitter # For splitting documents into chunks
from pre_processing import remove_space_and_newlines_for_text # For cleaning up text by removing extra spaces and newlines
from langchain_core.documents import Document # For working with document objects

#==========================Text Splitter Definition===========================
class TextSplitter:
    # Initialize the Text Splitter with chunk size and overlap
    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 400):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
    
    # Split documents into chunks for better retrieval
    def split_text(self, documents) -> list:
        # Clean each document before splitting
        cleaned_documents = [remove_space_and_newlines_for_text(doc.page_content) for doc in documents]
        # Create new Document objects with cleaned content
        cleaned_docs = [Document(page_content=content) for content in cleaned_documents]
        return self.splitter.split_documents(cleaned_docs)
