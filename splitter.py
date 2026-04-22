#==========================Text Splitter Module===========================
# This module defines the TextSplitter class, which is responsible for splitting documents into smaller chunks for better retrieval performance in the RAG agent. #
# It uses the RecursiveCharacterTextSplitter from LangChain to split the text based on specified chunk size and overlap. The module also includes a function to clean the
# text by removing extra spaces and newlines before splitting. This ensures that the chunks are well-formed and can be effectively used for embedding and retrieval in the RAG agent.
#==========================Importing Required Libraries===========================
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pre_processing import process_text_document
from langchain_core.documents import Document


class TextSplitter:
    def __init__(self, chunk_size: int , chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

    def split_text(self, documents) -> list:
        cleaned_docs = []

        for doc_index, doc in enumerate(documents):
            # Clean and enrich text
            processed = process_text_document(
                text=doc.page_content,
                source_name=doc.metadata.get("source") if getattr(doc, "metadata", None) else None,
                source_path=doc.metadata.get("source_path") if getattr(doc, "metadata", None) else None,
            )

            # Merge original metadata with cleaned-text metadata
            original_meta = doc.metadata.copy() if getattr(doc, "metadata", None) else {}
            merged_meta = {
                **original_meta,
                **processed.get("metadata", {}),
                "doc_index": doc_index,
            }

            cleaned_docs.append(
                Document(
                    page_content=processed.get("text", ""),
                    metadata=merged_meta
                )
            )

        # Split documents into chunks
        chunks = self.splitter.split_documents(cleaned_docs)

        # Add chunk-level metadata
        for chunk_index, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = chunk_index
            chunk.metadata["chunk_id"] = f'{chunk.metadata.get("sha1", "doc")}_chunk_{chunk_index}'
        #print(chunks)
        return chunks