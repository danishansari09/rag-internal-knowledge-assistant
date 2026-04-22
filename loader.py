#==========================Document Loader Module===========================
# This module defines the loader class, which is responsible for loading documents from various sources such as text files, PDF files, and web pages. 
# The class provides static methods to load documents from these sources using the appropriate loaders from the LangChain library. 
# The loaded documents are returned as lists, which    can then be processed and used in the RAG agent for retrieval and response generation. 
# This module is essential for ensuring that the RAG agent has access to the necessary information from different sources to provide accurate and relevant responses to user queries.

from pathlib import Path
from langchain_community.document_loaders import TextLoader, PDFMinerLoader, WebBaseLoader
from bs4 import SoupStrainer


class loader:
    @staticmethod
    def load_txt(path: str) -> list:
        docs = TextLoader(path, encoding="utf-8").load()
        for doc in docs:
            doc.metadata["source_type"] = "text"
            doc.metadata["source_name"] = Path(path).name
            doc.metadata["source_path"] = str(path)
        return docs

    @staticmethod
    def load_pdf(path: str) -> list:
        docs = PDFMinerLoader(path).load()
        for doc in docs:
            doc.metadata["source_type"] = "pdf"
            doc.metadata["source_name"] = Path(path).name
            doc.metadata["source_path"] = str(path)
        return docs

    @staticmethod
    def load_webpages(url: str) -> list:
        strainer = SoupStrainer(class_=("post-content", "post-title", "post-header"))
        docs = WebBaseLoader(
            web_paths=(url,),
            bs_kwargs={"parse_only": strainer},
        ).load()

        for doc in docs:
            doc.metadata["source_type"] = "html"
            doc.metadata["source_name"] = url
            doc.metadata["source_path"] = url
            doc.metadata["url"] = url
        return docs