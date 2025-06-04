import os
from typing import List, Optional
from langchain_core.chat_models import BaseChatModel
from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.documents import Document
from langchain.chains import RetrievalQA

VECTOR_DIR = os.path.expanduser("~/.shandu/vectorstore")

class RAGRetriever:
    """Simple retrieval wrapper around a persistent Chroma vector store."""
    def __init__(self, llm: Optional[BaseChatModel] = None, vector_dir: str = VECTOR_DIR):
        self.embeddings = FastEmbedEmbeddings()
        os.makedirs(vector_dir, exist_ok=True)
        self.vectorstore = Chroma(persist_directory=vector_dir, embedding_function=self.embeddings)
        self.llm = llm

    def add_documents(self, docs: List[Document]) -> None:
        if not docs:
            return
        self.vectorstore.add_documents(docs)

    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        return retriever.get_relevant_documents(query)

    def run_qa(self, query: str, k: int = 4) -> str:
        if self.llm is None:
            raise ValueError("LLM must be provided to run QA")
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        chain = RetrievalQA.from_chain_type(llm=self.llm, retriever=retriever)
        return chain.run(query)


_retriever: Optional[RAGRetriever] = None


def get_retriever(llm: Optional[BaseChatModel] = None) -> RAGRetriever:
    global _retriever
    if _retriever is None:
        _retriever = RAGRetriever(llm=llm)
    return _retriever
