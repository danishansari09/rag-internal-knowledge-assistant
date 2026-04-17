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