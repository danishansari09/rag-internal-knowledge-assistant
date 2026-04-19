#==========================Text Splitter Module===========================
# This module defines the TextSplitter class, which is responsible for splitting documents into smaller chunks for better retrieval performance in the RAG agent. #
# It uses the RecursiveCharacterTextSplitter from LangChain to split the text based on specified chunk size and overlap. The module also includes a function to clean the
# text by removing extra spaces and newlines before splitting. This ensures that the chunks are well-formed and can be effectively used for embedding and retrieval in the RAG agent.
#==========================Importing Required Libraries===========================
from langchain_text_splitters import RecursiveCharacterTextSplitter # For splitting documents into chunks
from pre_processing import process_text_document # For cleaning up text by removing extra spaces and newlines
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
        cleaned_documents = [process_text_document(doc.page_content) for doc in documents]
        # print(cleaned_documents)  # Print the cleaned content of the first document for verification
        # Create new Document objects with cleaned content
        #cleaned_docs = [Document(page_content=content) for content in cleaned_documents]
        cleaned_docs = []
        for content in cleaned_documents:
            cleaned_docs.append(Document(page_content=content.get("text", "")))  # Use get to avoid KeyError if "text" is missing
        #print(cleaned_docs)  # Print the cleaned Document objects for verification
        # Split the cleaned documents into chunks using the splitter    
        return self.splitter.split_documents(cleaned_docs)
