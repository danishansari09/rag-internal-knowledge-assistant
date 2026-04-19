#=============Retrieval module for RAG Agent===========================
# This module defines the Retriever class, which is responsible for retrieving relevant documents based on a query.
# The Retriever class has a method to invoke the retrieval process using the retriever from the vectorstore. 
# The retrieved documents can then be used to generate a response in the RAG agent.
class Retriever:
    def __init__(self):
        self.retriever = None

    def retrieve(self, query: str) -> list:        
        return self.retriever.invoke(query)