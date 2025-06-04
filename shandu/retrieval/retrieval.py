import os
from typing import List, Optional
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.chains import RetrievalQA

VECTOR_DIR = os.path.expanduser("~/.shandu/vectorstore")

class RAGRetriever:
    """Simple retrieval wrapper around a persistent Chroma vector store."""
    def __init__(self, llm: Optional[ChatOpenAI] = None, vector_dir: str = VECTOR_DIR):
        self.embeddings = OpenAIEmbeddings()
        os.makedirs(vector_dir, exist_ok=True)
        self.vectorstore = Chroma(persist_directory=vector_dir, embedding_function=self.embeddings)
        self.vectorstore.persist()
        self.llm = llm or ChatOpenAI()

    def add_documents(self, docs: List[Document]) -> None:
        if not docs:
            return
        self.vectorstore.add_documents(docs)
        self.vectorstore.persist()

    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        return retriever.get_relevant_documents(query)

    def run_qa(self, query: str, k: int = 4) -> str:
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        chain = RetrievalQA.from_chain_type(llm=self.llm, retriever=retriever)
        return chain.run(query)

_retriever: Optional[RAGRetriever] = None


def get_retriever(llm: Optional[ChatOpenAI] = None) -> RAGRetriever:
    global _retriever
    if _retriever is None:
        _retriever = RAGRetriever(llm=llm)
    return _retriever
