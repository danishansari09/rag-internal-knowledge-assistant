#==========================Prompt Template Module===========================
# This module defines the PromptBuilder class, which is responsible for creating a prompt template for the RAG Agent. 
# The prompt template is designed to guide the language model in generating responses based on the retrieved documents. 
# The template includes instructions for the assistant to use only the provided context to answer the question and to admit when it does not know the answer. 
# The PromptBuilder class has a static method that returns a PromptTemplate object created from the defined template string. 
# This module is essential for ensuring that the language model generates accurate and relevant responses based on the retrieved information.

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