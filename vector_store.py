#==========================Vector Store Module===========================
# This module defines the VectorStore class, which is responsible for creating and loading a vector store for the RAG agent. 
# The class provides static methods to create a Chroma vector store from the given documents and embedding model, as well as to load an existing vector store from disk. 
# The vector store is essential for efficient retrieval of relevant documents based on user queries in the RAG agent. 
# The module uses the Chroma library for handling vector stores and the pathlib library for managing file paths.
from langchain_chroma import Chroma # For creating a vector store for retrieval
from pathlib import Path # For handling file paths

class VectorStore:
    @staticmethod
    def create_vector_store(documents, embed_model, persist_directory) -> Chroma:
        """
        Create a Chroma vector store from the given documents and embedding model.
        """
        persist_path = persist_directory
        persist_path.mkdir(parents=True, exist_ok=True)

        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embed_model,
            persist_directory=str(persist_path),
            collection_name="internal_docs"
        )
        return vectorstore
    
    @staticmethod
    def load_vector_store(embed_model, persist_directory: Path) -> Chroma:
        """
        Load an existing store from disk.
        """
        return Chroma(
            persist_directory=str(persist_directory),
            embedding_function=embed_model,
            collection_name="internal_docs"
        )