"""
Shandu Deep Research System
A powerful research tool combining multiple search engines with LangChain integration.

Copyright (c) 2025 Dušan Jolović
Licensed under the MIT License. See LICENSE file for details.
"""

from .search.search import UnifiedSearcher, SearchResult
from .research.researcher import DeepResearcher, ResearchResult
from .agents.agent import ResearchAgent

__version__ = "0.1.1"
__all__ = [
    "UnifiedSearcher",
    "SearchResult",
    "DeepResearcher",
    "ResearchResult",
    "ResearchAgent"
]
