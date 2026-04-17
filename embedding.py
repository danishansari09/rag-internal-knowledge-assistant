from langchain_huggingface import HuggingFaceEmbeddings # For generating embeddings using HuggingFace models

class EmbeddingBuilder:
    @staticmethod
    def build_embeddings(model_name: str, hf_key: str) -> HuggingFaceEmbeddings:
        """
        Initialize HuggingFace Embeddings with the specified model and key.
        """
        print(f"Initializing HuggingFace Embeddings with model: {model_name}")
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu", "use_auth_token": hf_key}
        )