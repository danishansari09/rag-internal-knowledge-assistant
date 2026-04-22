#=============Retrieval module for RAG Agent===========================
# This module defines the Retriever class, which is responsible for retrieving relevant documents based on a query.
# The Retriever class has a method to invoke the retrieval process using the retriever from the vectorstore. 
# The retrieved documents can then be used to generate a response in the RAG agent.
class Retriever:
    def __init__(self):
        self.retriever = None

    def retrieve(self, query: str) -> list:        
        return self.retriever.invoke(query)
    
    def format_source_label(self, doc, idx: int) -> str:
        """
        Build a clean citation label from LangChain Document metadata.
        Example:
        [Source 1: generative_ai_sample_for_rag.pdf | page 3 | section: Applications]
        """
        md = doc.metadata or {}

        source = md.get("source") or md.get("file_name") or "unknown"
        page = md.get("page")
        section = md.get("section") or md.get("heading")
        chunk_id = md.get("chunk_id")
        file_type = md.get("file_type")

        parts = [f"Source {idx}: {source}"]

        if page is not None:
            parts.append(f"page {page}")

        if section:
            parts.append(f"section: {section}")

        if chunk_id is not None:
            parts.append(f"chunk {chunk_id}")

        if file_type:
            parts.append(file_type)

        return "[" + " | ".join(parts) + "]"
    
    def format_sources_block(self, docs) -> str:
        """
        Returns a Sources block for display under the answer.
        """
        if not docs:
            return "Sources:\n[No sources retrieved]"

        lines = ["Sources:"]
        for i, doc in enumerate(docs, start=1):
            lines.append(self.format_source_label(doc, i))

        return "\n".join(lines)
    # This method formats the sources block while deduplicating entries based on source, page, and section/heading.
    def format_sources_block_dedup(self, docs) -> str:
        if not docs:
            return "Sources:\n[No sources retrieved]"

        seen = set()
        lines = ["Sources:"]
        source_num = 1

        for doc in docs:
            md = doc.metadata or {}
            key = (
                md.get("source"),
                md.get("page"),
                md.get("section") or md.get("heading"),
            )

            if key in seen:
                continue

            seen.add(key)
            lines.append(self.format_source_label(doc, source_num))
            source_num += 1

        return "\n".join(lines)
    
    def build_final_response(self, answer_text: str, docs) -> str:
        sources_text = self.format_sources_block_dedup(docs)
        return f"{answer_text}\n\n{sources_text}"