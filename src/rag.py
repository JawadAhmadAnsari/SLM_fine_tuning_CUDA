# src/rag.py

import os
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

class RAGEngine:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initializes the RAGEngine with a specified embedding model.
        """
        self.embedding_model = HuggingFaceEmbeddings(model_name=model_name)
        self.vector_store = None

    def load_documents(self, path: str):
        """
        Loads documents from a file path. Supports PDF and Excel files.
        Returns an empty list if the file is not found.
        """
        _, file_extension = os.path.splitext(path)
        
        try:
            if file_extension.lower() == '.pdf':
                loader = PyPDFLoader(path)
                return loader.load()
            
            elif file_extension.lower() == '.xlsx':
                df = pd.read_excel(path)
                documents = []
                for _, row in df.iterrows():
                    text = ", ".join([f"{col}: {val}" for col, val in row.items()])
                    documents.append(text)
                from langchain_core.documents import Document
                return [Document(page_content=doc) for doc in documents]
            
            else:
                print(f"Warning: Unsupported file type: {file_extension}")
                return []
        
        except FileNotFoundError:
            print(f"Warning: File not found at {path}. Returning empty list.")
            return []

    def build_index(self, documents, chunk_size=1000, chunk_overlap=200):
        """
        Builds a Chroma vector store index from the loaded documents.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        splits = text_splitter.split_documents(documents)
        
        print("Building Chroma vector store...")
        self.vector_store = Chroma.from_documents(documents=splits, embedding=self.embedding_model)
        print("Vector store built successfully.")
        return self.vector_store

    def retrieve(self, query: str, k: int = 3) -> str:
        """
        Retrieves the top-k most relevant documents from the vector store.
        """
        if self.vector_store is None:
            raise RuntimeError("Vector store is not built. Please call build_index() first.")
        
        print(f"Retrieving top-{k} documents for query: '{query}'")
        docs = self.vector_store.similarity_search(query, k=k)
        
        # Format the retrieved documents into a single context string
        context = "\n---\n".join([doc.page_content for doc in docs])
        return context
