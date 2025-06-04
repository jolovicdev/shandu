"""
Shandu Deep Research System
A powerful research tool combining multiple search engines with LangChain integration.

Copyright (c) 2025 Dušan Jolović
Licensed under the MIT License. See LICENSE file for details.
"""

from .search.search import UnifiedSearcher, SearchResult
from .research.researcher import DeepResearcher, ResearchResult
from .agents.agent import ResearchAgent
from .retrieval import RAGRetriever, get_retriever

__version__ = "2.0.0"
__all__ = [
    "UnifiedSearcher",
    "SearchResult",
    "DeepResearcher",
    "ResearchResult",
    "ResearchAgent"
    ,"RAGRetriever"
    ,"get_retriever"
]
