from langchain_community.vectorstores import Chroma # For creating a vector store for retrieval

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
        vectorstore.persist()
        return vectorstore
