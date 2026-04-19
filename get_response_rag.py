#==========================RAG Agent Response Generation Script===========================
# This script defines the rag_qa function, which is responsible for generating a response to a user query using a Retrieval-Augmented Generation (RAG) approach. 
# The function takes a query and a vector store as input, retrieves relevant documents based on the query, constructs a prompt using the retrieved documents, and generates a response using a language model. 
# The script also includes an example usage of the rag_qa function, where a sample query is provided, and the generated answer is printed. 
# This script is essential for enabling the RAG agent to provide accurate and relevant responses to user queries by leveraging the information stored in the vector store. 
# It integrates the components defined in the model, retriever, and prompt modules to create a cohesive response generation process.

import os
from embedding import EmbeddingBuilder
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from prompt import PromptBuilder
from openai import OpenAI
import keycredentials
from model import llm


hf_mdodel_embedding = keycredentials.hf_model_embedding
hf_key = keycredentials.hf_token
hf_model_llm = keycredentials.hf_model_llm

def rag_qa(query: str, vectorstore: Chroma) -> str:
    llm_model = llm(hf_model_llm, hf_key)
    retriever = vectorstore.as_retriever()
    relevant_docs = retriever.invoke(query)
    prompt = PromptBuilder.build_prompt()
    final_prompt = prompt.format(question=query, context="\n\n".join(d.page_content for d in relevant_docs))
    response = llm_model.generate_response(final_prompt)
    return response

# Example usage
if __name__ == "__main__":
    query = "What is the purpose of the MNR-4?"
    answer = rag_qa(query)
    print(answer)