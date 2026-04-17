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

    #====================Split the loaded documents into chunks=========================
    all_chunks = []
    splitter = TextSplitter(chunk_size=300, chunk_overlap=30)
    for doc in loaded_documents:
        chunks = splitter.split_text([doc])
        all_chunks.extend(chunks)
    
    #====================embed the chunks using HuggingFace Embeddings and create a vector store=========================
    embedder = EmbeddingBuilder.build_embeddings(HF_MODEL_EMBEDDING, HF_KEY)
    
    if  DATA_CHROMA_DIR.exists():
        # print("Creating data directory for Chroma vector store...")
        DATA_CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        # print(f"Loading existing vector store from: {DATA_CHROMA_DIR.resolve()}")
        vector_store = VectorStore.create_vector_store(all_chunks, embedder, DATA_CHROMA_DIR)
    
    #====================Set up the retriever for the RAG Agent=========================
    # retriever = Retriever()
    # print("Retriever is set up and ready to use.")
    # docs = retriever.retrieve(vector_store, "Based on the text, what is the purpose of the MNR-4?")
    # print(f"Retrieved {len(docs)} documents")
    # for d in docs:
    #     print(f"Document content: {d.page_content}...")  # Print a preview of the retrieved document
if __name__ == "__main__":
    main()