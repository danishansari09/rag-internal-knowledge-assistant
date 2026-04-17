#==========================RAG Agent Implementation=========================

#=============== Import necessary libraries for RAG Agent================
from unittest import loader

from langchain_community.document_loaders import TextLoader, PDFMinerLoader, WebBaseLoader # For loading text and PDF documents and web pages
from langchain_community.vectorstores import FAISS, Chroma # For creating a vector store for retrieval
from langchain_community.embeddings import HuggingFaceEmbeddings # For generating embeddings using HuggingFace models
from langchain_text_splitters import RecursiveCharacterTextSplitter # For splitting documents into chunks
from langchain_core.prompts import PromptTemplate # For creating prompt templates
from openai import OpenAI # For interacting with the OpenAI API
from bs4 import SoupStrainer # For parsing web pages with BeautifulSoup


#==========================RAG Agent Definition===========================
class RAGAgent:
    # Initialize the RAG Agent with HuggingFace model and key
    def __init__(self, hf_model: str, hf_key: str):
        self.hf_model = hf_model
        self.hf_key = hf_key

        self.embed_model = self.init_embeddings()
        self.prompt = self.build_prompt()
        self.retriever = None

        self.client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=self.hf_key
        )

    # Initialize HuggingFace Embeddings
    def init_embeddings(self) -> HuggingFaceEmbeddings:
        return HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu", "use_auth_token": self.hf_key}
        )
    
    # Load text documents from a given path
    def load_txt(self, path: str) -> list:
        return TextLoader(path).load()
    
    # Load PDF documents from a given path
    def load_pdf(self, path: str) -> list:
        return PDFMinerLoader(path).load()
    
    # Load web pages from a given URL
    def load_webpages(self, url: str) -> list:
        strainer = SoupStrainer(class_=("post-content", "post-title", "post-header"))
        loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs={"parse_only": strainer},
        )
        return loader.load()
    
    # Split documents into chunks for better retrieval
    def split_docs(self, documents) -> list:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=400
        )
        
        return splitter.split_documents(documents)
    
    # Create a vector store from the given documents
    def create_vector_store(self, documents) -> FAISS:
        return FAISS.from_documents(documents, self.embed_model)
    
    # Set the retriever for the RAG Agent
    def set_retriever(self, retriever):
        self.retriever = retriever

    # Build a prompt template for the RAG Agent
    def build_prompt(self) -> PromptTemplate:
        template = """
                You are a highly precise Knowledge Assistant only for danish-ansari.com.
                Use the following context to answer the question.
                Don't make up answers. If the answer is not in the context, say you don't know.

                Context:
                {context}

                Question:
                {question}

                Answer:
            """
        return PromptTemplate.from_template(template)
    
    # Perform RAG-based question answering
    def rag_qa(self, query: str) -> str:
        if not self.retriever:
            raise RuntimeError("Retriever not initialized")
        
        # Retrieve relevant documents based on the query
        docs = self.retriever.invoke(query)
        for i, t in enumerate(docs):
            print(f"\n--- Retrieved Document {i+1} ---")
            print(f"Content Length: {len(t.page_content)}")
            print(f"Preview: {t.page_content}...")
        
        # Format the final prompt with retrieved context and the query
        final_prompt = self.prompt.format(
            context="\n\n".join(d.page_content for d in docs),
            question=query,
        )

        # Generate a response using the HuggingFace model via OpenAI API
        completion = self.client.chat.completions.create(
            model=self.hf_model,
            messages=[{"role": "user", "content": final_prompt}],
            max_tokens=220,
            temperature=0.6,
        )
        # Return the generated answer from the model
        return completion.choices[0].message.content or ""