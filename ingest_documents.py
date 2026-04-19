#==========================Document Ingestion Script===========================
# This script is responsible for ingesting documents into the RAG agent's knowledge base. 
# It loads documents from specified sources (text files, PDF files, and web pages), splits them into smaller chunks for better retrieval performance, generates embeddings for the chunks using HuggingFace models, and creates a vector store for efficient retrieval. 
# The script also includes a test retrieval section to verify that the retriever is set up correctly and can retrieve relevant documents based on a sample query. 
# This ingestion process is crucial for ensuring that the RAG agent has access to the necessary information to provide accurate and relevant responses to user queries.

from pathlib import Path
from loader import loader
from splitter import TextSplitter
from embedding import EmbeddingBuilder
import keycredentials
from vector_store import VectorStore
from retriever import Retriever
#===============Key Configurations===========================
HF_MODEL_EMBEDDING = keycredentials.hf_model_embedding
HF_KEY = keycredentials.hf_token    

DATA_PATH = Path("./docs")
DATA_CHROMA_DIR = Path("./chroma_db")
TEXT_EXTENSIONS = {".txt"}
HTML_URL = "https://lilianweng.github.io/posts/2023-06-23-agent/"

def main():
    vector_store = None
    retriever = None
    #============load documents from the data path===========================
    loaded_documents = []
    for file_path in sorted(DATA_PATH.glob("*")):
        suffix = file_path.suffix.lower()
        if suffix in TEXT_EXTENSIONS:
            # print(f"Loading text file: {file_path.name}")
            document = loader.load_txt(str(file_path))
            loaded_documents.extend(document)
            # print(f"Loaded {len(document)} documents from {file_path.name}")
        else:
            document = loader.load_pdf(str(file_path))
            loaded_documents.extend(document)
            print(f"Loaded {len(document)} documents from {file_path.name}")

    # Load web page content
    # if HTML_URL:
    #     print(f"Loading web page: {HTML_URL}")
    #     web_docs = loader.load_webpages(HTML_URL)
    #     loaded_documents.extend(web_docs)
    #     print(f"Loaded {len(web_docs)} documents from web page: {HTML_URL}")    

    #====================Split the loaded documents into chunks=========================
    all_chunks = []
    splitter = TextSplitter(chunk_size=2000, chunk_overlap=200)
    for doc in loaded_documents:
        chunks = splitter.split_text([doc])
        all_chunks.extend(chunks)
    
    #====================embed the chunks using HuggingFace Embeddings and create a vector store=========================
    embedder = EmbeddingBuilder.build_embeddings(HF_MODEL_EMBEDDING, HF_KEY)
    
    if  not DATA_CHROMA_DIR.exists():
        # print("Creating data directory for Chroma vector store...")
        DATA_CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        # print(f"Loading existing vector store from: {DATA_CHROMA_DIR.resolve()}")
        vector_store = VectorStore.create_vector_store(all_chunks, embedder, DATA_CHROMA_DIR)
    else:
        vector_store = VectorStore.load_vector_store(embedder, DATA_CHROMA_DIR)
    
    #====================To test retrieval =========================
    # retriever = Retriever()
    # print("Retriever is set up and ready to use.")
    # retriever = vector_store.as_retriever()
    # docs = retriever.invoke("What is generative AI?")
    # print(f"Retrieved {len(docs)} documents")
    # for d in docs:
    #     print(f"Document content: {d.page_content}...")  # Print a preview of the retrieved document
if __name__ == "__main__":
    main()