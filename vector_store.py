#==========================Vector Store Module===========================
# This module defines the VectorStore class, which is responsible for creating and loading a vector store for the RAG agent. 
# The class provides static methods to create a Chroma vector store from the given documents and embedding model, as well as to load an existing vector store from disk. 
# The vector store is essential for efficient retrieval of relevant documents based on user queries in the RAG agent. 
# The module uses the Chroma library for handling vector stores and the pathlib library for managing file paths.
from langchain_chroma import Chroma
from pathlib import Path
import shutil


class VectorStore:
    COLLECTION_NAME = "internal_docs"

    @staticmethod
    def create_vector_store(documents, embed_model, persist_directory) -> Chroma:
        """
        Create a new Chroma vector store from documents.
        """
        persist_path = Path(persist_directory)
        persist_path.mkdir(parents=True, exist_ok=True)

        return Chroma.from_documents(
            documents=documents,
            embedding=embed_model,
            persist_directory=str(persist_path),
            collection_name=VectorStore.COLLECTION_NAME
        )

    @staticmethod
    def load_vector_store(embed_model, persist_directory: Path) -> Chroma:
        """
        Load an existing Chroma vector store.
        """
        return Chroma(
            persist_directory=str(persist_directory),
            embedding_function=embed_model,
            collection_name=VectorStore.COLLECTION_NAME
        )

    @staticmethod
    def load_or_create(documents, embed_model, persist_directory: Path) -> Chroma:
        """
        Load if exists, otherwise create.
        """
        persist_path = Path(persist_directory)

        if persist_path.exists() and any(persist_path.iterdir()):
            return VectorStore.load_vector_store(embed_model, persist_path)

        return VectorStore.create_vector_store(documents, embed_model, persist_path)

    @staticmethod
    def add_documents(vectorstore: Chroma, documents) -> None:
        """
        Add new documents/chunks to an existing collection.
        """
        if documents:
            vectorstore.add_documents(documents)

    @staticmethod
    def reset_store(persist_directory: Path) -> None:
        """
        Delete the persisted Chroma directory completely.
        Useful for full reindexing during experiments.
        """
        persist_path = Path(persist_directory)
        if persist_path.exists():
            shutil.rmtree(persist_path)