"""
Research agent implementation using LangGraph.
"""
import time
import asyncio
from datetime import datetime
from typing import List, Dict, Optional, Any, Callable
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from rich.console import Console
from rich.panel import Panel
from ..search.search import UnifiedSearcher, SearchResult
from ..scraper import WebScraper, ScrapedContent
from ..retrieval import get_retriever
from ..research.researcher import ResearchResult
from ..config import config, get_current_date
from .processors import AgentState
from .utils.agent_utils import (
    get_user_input,
    clarify_query,
    display_research_progress,
    is_shutdown_requested
)
from .nodes import (
    initialize_node,
    reflect_node,
    generate_queries_node,
    search_node,
    smart_source_selection,
    format_citations_node,
    generate_initial_report_node,
    enhance_report_node,
    expand_key_sections_node,
    report_node
)
from .graph import build_graph, create_node_wrapper

console = Console()

class ResearchGraph:
    """Research workflow graph implementation."""
    def __init__(
        self, 
        llm: Optional[ChatOpenAI] = None, 
        searcher: Optional[UnifiedSearcher] = None, 
        scraper: Optional[WebScraper] = None, 
        temperature: float = 0.5,
        date: Optional[str] = None
    ):
        api_base = config.get("api", "base_url")
        api_key = config.get("api", "api_key")
        model = config.get("api", "model")
        
        self.llm = llm or ChatOpenAI(
            base_url=api_base,
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=16384  # Significantly increased max tokens to support much more comprehensive responses
        )
        self.searcher = searcher or UnifiedSearcher()
        self.scraper = scraper or WebScraper(retriever=get_retriever(self.llm))
        self.date = date or get_current_date()
        self.progress_callback = None
        self.include_objective = False
        self.detail_level = "high"
        self.graph = self._build_graph()

    def _build_graph(self):
        """Build the research graph."""

        init_node = create_node_wrapper(lambda state: initialize_node(self.llm, self.date, self.progress_callback, state))
        reflect = create_node_wrapper(lambda state: reflect_node(self.llm, self.progress_callback, state))
        gen_queries = create_node_wrapper(lambda state: generate_queries_node(self.llm, self.progress_callback, state))
        search = create_node_wrapper(lambda state: search_node(self.llm, self.searcher, self.scraper, self.progress_callback, state))
        source_selection = create_node_wrapper(lambda state: smart_source_selection(self.llm, self.progress_callback, state))
        citations = create_node_wrapper(lambda state: format_citations_node(self.llm, self.progress_callback, state))
        initial_report = create_node_wrapper(lambda state: generate_initial_report_node(self.llm, self.include_objective, self.progress_callback, state))
        enhance = create_node_wrapper(lambda state: enhance_report_node(self.llm, self.progress_callback, state))
        expand_sections = create_node_wrapper(lambda state: expand_key_sections_node(self.llm, self.progress_callback, state))
        final_report = create_node_wrapper(lambda state: report_node(self.llm, self.progress_callback, state))
        
        # Build graph with these node functions
        return build_graph(
            init_node,
            reflect,
            gen_queries,
            search,
            source_selection,
            citations,
            initial_report,
            enhance,
            expand_sections,
            final_report
        )

    async def research(
        self, 
        query: str, 
        depth: int = 2, 
        breadth: int = 4, 
        progress_callback: Optional[Callable[[AgentState], None]] = None,
        include_objective: bool = False,
        detail_level: str = "high" 
    ) -> ResearchResult:
        """Execute research process on a query."""
        self.progress_callback = progress_callback
        self.include_objective = include_objective
        self.detail_level = detail_level

        depth = max(1, min(5, depth))  # Ensure depth is between 1 and 5
        breadth = max(1, min(10, breadth))  # Ensure breadth is between 1 and 10

        state = AgentState(
            messages=[HumanMessage(content=f"Starting research on: {query}")],
            query=query,
            depth=depth,
            breadth=breadth,
            current_depth=0,
            findings="",
            sources=[],
            selected_sources=[],
            formatted_citations="",
            subqueries=[],
            content_analysis=[],
            start_time=time.time(),
            chain_of_thought=[],
            status="Starting",
            current_date=get_current_date(),
            detail_level=detail_level,
            identified_themes="",
            initial_report="",
            enhanced_report="",
            final_report=""
        )
        
        try:
            # Invoke the graph with increased recursion limit
            final_state = await self.graph.ainvoke(state, {"recursion_limit": 50})
            
            elapsed_time = time.time() - final_state["start_time"]
            minutes, seconds = divmod(int(elapsed_time), 60)
            
            return ResearchResult(
                query=query,
                summary=final_state["findings"],
                sources=final_state["sources"],
                subqueries=final_state["subqueries"],
                depth=depth,
                content_analysis=final_state["content_analysis"],
                chain_of_thought=final_state["chain_of_thought"],
                research_stats={
                    "elapsed_time": elapsed_time,
                    "elapsed_time_formatted": f"{minutes}m {seconds}s",
                    "sources_count": len(final_state["sources"]),
                    "subqueries_count": len(final_state["subqueries"]),
                    "depth": depth,
                    "breadth": breadth,
                    "detail_level": detail_level
                }
            )
        except KeyboardInterrupt:
            console.print("\n[yellow]Research interrupted by user. Generating report with current findings...[/]")

            elapsed_time = time.time() - state["start_time"]
            minutes, seconds = divmod(int(elapsed_time), 60)
            
            return ResearchResult(
                query=query,
                summary=state["findings"] + "\n\n*Note: Research was interrupted before completion.*",
                sources=state["sources"],
                subqueries=state["subqueries"],
                depth=state["current_depth"],
                content_analysis=state["content_analysis"],
                chain_of_thought=state["chain_of_thought"],
                research_stats={
                    "elapsed_time": elapsed_time,
                    "elapsed_time_formatted": f"{minutes}m {seconds}s",
                    "sources_count": len(state["sources"]),
                    "subqueries_count": len(state["subqueries"]),
                    "depth": state["current_depth"],
                    "breadth": breadth,
                    "detail_level": detail_level,
                    "interrupted": True
                }
            )
    
    def research_sync(
        self, 
        query: str, 
        depth: int = 2, 
        breadth: int = 4, 
        progress_callback: Optional[Callable[[AgentState], None]] = None,
        include_objective: bool = False,
        detail_level: str = "high"
    ) -> ResearchResult:
        """Synchronous wrapper for research."""
        try:
            return asyncio.run(self.research(query, depth, breadth, progress_callback, include_objective, detail_level))
        except KeyboardInterrupt:
            console.print("\n[yellow]Research interrupted by user.[/]")
            raise
