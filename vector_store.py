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