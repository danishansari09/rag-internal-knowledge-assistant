#==========================RAG Agent Response Generation Script===========================
# This script defines the rag_qa function, which is responsible for generating a response to a user query using a Retrieval-Augmented Generation (RAG) approach. 
# The function takes a query and a vector store as input, retrieves relevant documents based on the query, constructs a prompt using the retrieved documents, and generates a response using a language model. 
# The script also includes an example usage of the rag_qa function, where a sample query is provided, and the generated answer is printed. 
# This script is essential for enabling the RAG agent to provide accurate and relevant responses to user queries by leveraging the information stored in the vector store. 
# It integrates the components defined in the model, retriever, and prompt modules to create a cohesive response generation process.

from langchain_chroma import Chroma
from prompt import PromptBuilder
import keycredentials
from model import llm
from retriever import Retriever
hf_key = keycredentials.hf_token
hf_model_llm = keycredentials.hf_model_llm
from vector_store import VectorStore
from pathlib import Path
def extract_sources(relevant_docs):
    seen = set()
    sources = []

    for doc in relevant_docs:
        meta = getattr(doc, "metadata", {}) or {}

        item = {
            "source_name": meta.get("source_name") or meta.get("source") or "Unknown",
            "source_type": meta.get("source_type"),
            "title": meta.get("title"),
            "chunk_id": meta.get("chunk_id"),
            "page": meta.get("page"),
            "section": meta.get("section") or meta.get("heading"),
        }

        key = (item["source_name"], item["chunk_id"])
        if key not in seen:
            seen.add(key)
            sources.append(item)

    return sources


def format_context(relevant_docs):
    parts = []
    for doc in relevant_docs:
        meta = getattr(doc, "metadata", {}) or {}
        title = meta.get("title") or meta.get("source_name") or meta.get("source") or "Untitled"
        parts.append(f"Source: {title}\nContent:\n{doc.page_content}")
    return "\n\n".join(parts)


def rag_qa(query: str, vectorstore: Chroma) -> dict:
    llm_model = llm(hf_model_llm, hf_key)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    relevant_docs = retriever.invoke(query)

    prompt = PromptBuilder.build_prompt()
    final_prompt = prompt.format(
        question=query,
        context=format_context(relevant_docs)
    )

    answer = llm_model.generate_response(final_prompt)
    answer_text = answer.content if hasattr(answer, "content") else str(answer)

    retriever_helper = Retriever()
    final_answer_with_sources = retriever_helper.build_final_response(answer_text, relevant_docs)
    return {
        "answer": answer_text,
        "answer_with_sources": final_answer_with_sources,
        "sources": extract_sources(relevant_docs),
    }
    # return {
    #     "answer": answer,
    #     "sources": extract_sources(relevant_docs),
    # }

def debug_retrieval(query: str, vectorstore, k: int = 4):
    results = vectorstore.similarity_search_with_score(query, k=k)

    print(f"\nQuery: {query}")
    print(f"Retrieved {len(results)} results\n")

    for i, (doc, score) in enumerate(results, 1):
        print(f"===== Result {i} | Score: {score} =====")
        print("Metadata:", getattr(doc, "metadata", {}))
        print("Content preview:")
        print(doc.page_content[:700])
        print()

# if __name__ == "__main__":
#     DATA_CHROMA_DIR = Path("./chroma_db")
#     print("Running debug retrieval test...")
#     vectorstore = VectorStore.load_vector_store(None, DATA_CHROMA_DIR)
#     debug_retrieval("What is generative AI?", vectorstore)