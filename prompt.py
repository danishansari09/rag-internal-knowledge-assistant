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
        prompt_template = """You are a grounded question-answering assistant.

                                Answer the user's question using only the provided context.
                                Write the answer in a natural and direct way.
                                Do not say phrases like:
                                - according to document
                                - based on document
                                - quoted from document
                                - the provided context states

                                Do not mention document numbers in the answer.

                                If the answer is supported by the context, answer clearly and concisely.
                                If the context is insufficient, say so briefly.

                                Use the context for grounding, but do not copy its formatting unless necessary.

        {context}

        Question: {question}
        Answer:"""
        
        return PromptTemplate.from_template(prompt_template)