#==========================Document Loader Module===========================
# This module defines the loader class, which is responsible for loading documents from various sources such as text files, PDF files, and web pages. 
# The class provides static methods to load documents from these sources using the appropriate loaders from the LangChain library. 
# The loaded documents are returned as lists, which    can then be processed and used in the RAG agent for retrieval and response generation. 
# This module is essential for ensuring that the RAG agent has access to the necessary information from different sources to provide accurate and relevant responses to user queries.

from langchain_community.document_loaders import TextLoader, PDFMinerLoader, WebBaseLoader # For loading text and PDF documents and web pages
from bs4 import SoupStrainer # For parsing web pages with BeautifulSoup

class loader:
    @staticmethod
    def load_txt(path: str) -> list:
        return TextLoader(path, encoding="utf-8").load()
    
    @staticmethod
    def load_pdf(path: str) -> list:
        return PDFMinerLoader(path).load()
    
    @staticmethod
    def load_webpages(url: str) -> list:
        strainer = SoupStrainer(class_=("post-content", "post-title", "post-header"))
        loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs={"parse_only": strainer},
        )
        return loader.load()