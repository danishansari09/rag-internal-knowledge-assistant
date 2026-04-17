from langchain_core.prompts import PromptTemplate # For creating prompt templates

#==========================Prompt Template Definition===========================
class PromptBuilder:
    @staticmethod
    def build_prompt() -> PromptTemplate:
        """
        Create a prompt template for the RAG Agent.
        """
        prompt_template = """"You are a helpful assistant for danish-ansari.com. 
        You are an assistant for answering questions based on the retrieved documents. 
        Use only the following retrieved documents to answer the question. If you don't know the answer, say you don't know.

        {context}

        Question: {question}
        Answer:"""
        
        return PromptTemplate.from_template(prompt_template)