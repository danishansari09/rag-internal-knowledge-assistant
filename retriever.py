
#=============Retrieval module for RAG Agent===========================
class Retriever:
    def __init__(self):
        self.retriever = None

    def retrieve(self, query: str) -> list:
        
        return self.retriever.invoke(query)