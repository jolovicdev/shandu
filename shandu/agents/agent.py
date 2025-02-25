"""
Agent module for Shandu deep research system.
Implements LangChain agents for intelligent research orchestration.
"""
from typing import List, Dict, Optional, Union, Any
from dataclasses import dataclass
from datetime import datetime
import asyncio
import json
import time

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain.agents import AgentType, initialize_agent
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.tools import Tool, DuckDuckGoSearchResults, DuckDuckGoSearchRun

from ..search.search import UnifiedSearcher, SearchResult
from ..research.researcher import ResearchResult
from ..scraper import WebScraper, ScrapedContent

SYSTEM_PROMPT = """You are an expert research agent tasked with deeply investigating topics.
Your goal is to:
1. Break down complex queries into subqueries
2. Search multiple sources for information
3. Analyze and verify findings by examining source content
4. Generate insights through self-reflection
5. Produce comprehensive research reports

Always explain your reasoning process and reflect on the quality of information found.
If you find contradictory information, highlight it and explain the discrepancies.
When examining sources, look for:
- Primary sources and official documents
- Recent and up-to-date information
- Expert analysis and commentary
- Cross-verification of key claims

Current query: {query}
Research depth: {depth}
Research breadth: {breadth}"""

REFLECTION_PROMPT = """Based on the current findings:

1. What are the key insights discovered?
2. What important questions remain unanswered?
3. Are there any contradictions in the findings?
4. What additional angles should we explore?
5. How reliable are the sources found?

Current findings:
{findings}

Think step by step and be specific in your analysis.
"""

QUERY_GENERATION_PROMPT = """Generate {breadth} specific search queries to investigate:

Main query: {query}
Current findings: {findings}
Remaining questions: {questions}

Generate queries that will:
1. Fill gaps in current knowledge
2. Verify important claims
3. Explore new angles
4. Find primary sources

Format each query to be specific and targeted."""

class ResearchAgent:
    """
    LangChain-based agent for intelligent research orchestration with deep content analysis.
    """
    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        searcher: Optional[UnifiedSearcher] = None,
        scraper: Optional[WebScraper] = None,
        temperature: float = 0,
        max_depth: int = 2,
        breadth: int = 4,
        max_urls_per_query: int = 3,
        proxy: Optional[str] = None
    ):
        # Initialize components
        self.llm = llm or ChatOpenAI(
            temperature=temperature,
            model="gpt-4"  # Using GPT-4 for better reasoning
        )
        self.searcher = searcher or UnifiedSearcher()
        self.scraper = scraper or WebScraper(proxy=proxy)
        
        # Research parameters
        self.max_depth = max_depth
        self.breadth = breadth
        self.max_urls_per_query = max_urls_per_query
        
        # Initialize prompts
        self.system_prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)
        self.reflection_prompt = ChatPromptTemplate.from_template(REFLECTION_PROMPT)
        self.query_gen_prompt = ChatPromptTemplate.from_template(QUERY_GENERATION_PROMPT)
        
        # Initialize tools
        self.tools = self._setup_tools()
        
        # Initialize agent
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )

    def _setup_tools(self) -> List[Tool]:
        """Setup research tools for the agent."""
        return [
            Tool(
                name="search",
                func=self.searcher.search_sync,
                description="Search multiple sources for information about a topic"
            ),
            DuckDuckGoSearchResults(
                name="ddg_results",
                description="Get detailed search results from DuckDuckGo"
            ),
            DuckDuckGoSearchRun(
                name="ddg_search",
                description="Search DuckDuckGo for a quick answer"
            ),
            Tool(
                name="reflect",
                func=self._reflect_on_findings,
                description="Analyze and reflect on current research findings"
            ),
            Tool(
                name="generate_queries",
                func=self._generate_subqueries,
                description="Generate targeted subqueries for deeper research"
            )
        ]

    async def _reflect_on_findings(self, findings: str) -> str:
        """
        Analyze and reflect on current research findings.
        """
        reflection_chain = self.reflection_prompt | self.llm | StrOutputParser()
        return await reflection_chain.ainvoke({"findings": findings})

    async def _generate_subqueries(
        self,
        query: str,
        findings: str,
        questions: str
    ) -> List[str]:
        """
        Generate targeted subqueries based on current findings and questions.
        """
        query_chain = self.query_gen_prompt | self.llm | StrOutputParser()
        result = await query_chain.ainvoke({
            "query": query,
            "findings": findings,
            "questions": questions,
            "breadth": self.breadth
        })
        
        # Parse the result into a list of queries
        queries = [q.strip() for q in result.split("\n") if q.strip()]
        return queries[:self.breadth]

    async def _extract_urls_from_results(
        self,
        search_results: List[SearchResult],
        max_urls: int = 3
    ) -> List[str]:
        """Extract top URLs from search results for scraping."""
        urls = []
        seen = set()
        
        for result in search_results:
            if len(urls) >= max_urls:
                break
                
            url = result.url
            if url and url not in seen and url.startswith('http'):
                urls.append(url)
                seen.add(url)
        
        return urls

    async def _analyze_content(
        self,
        query: str,
        content: List[ScrapedContent]
    ) -> Dict[str, Any]:
        """Analyze scraped content using LLM."""
        # Prepare content for analysis
        content_text = ""
        for item in content:
            content_text += f"\nSource: {item.url}\nTitle: {item.title}\n"
            content_text += f"Content Summary:\n{item.text[:1000]}...\n"
        
        # Analyze with LLM
        analysis_prompt = f"""Analyze the following content related to: "{query}"

Content:
{content_text}

Provide:
1. Key findings and facts
2. Any contradictions or discrepancies
3. Reliability assessment of sources
4. Suggested areas for deeper investigation

Analysis:"""
        
        analysis_chain = PromptTemplate.from_template(analysis_prompt) | self.llm | StrOutputParser()
        analysis = await analysis_chain.ainvoke({})
        
        return {
            "analysis": analysis,
            "sources": [c.url for c in content]
        }

    async def research(
        self,
        query: str,
        depth: Optional[int] = None,
        engines: List[str] = ["google", "duckduckgo"]
    ) -> ResearchResult:
        """
        Perform intelligent research using LangChain agents and tools.
        
        Args:
            query: The research query
            depth: How many levels deep to explore (defaults to self.max_depth)
            engines: Which search engines to use
            
        Returns:
            ResearchResult object containing findings and sources
        """
        depth = depth if depth is not None else self.max_depth
        
        # Initialize research context
        context = {
            "query": query,
            "depth": depth,
            "breadth": self.breadth,
            "findings": "",
            "sources": [],
            "subqueries": [],
            "content_analysis": []
        }
        
        # Initial system prompt to set up the research
        system_chain = self.system_prompt | self.llm | StrOutputParser()
        context["findings"] = await system_chain.ainvoke(context)
        
        # Iterative deepening research process
        for current_depth in range(depth):
            # Reflect on current findings
            reflection = await self._reflect_on_findings(context["findings"])
            
            # Generate new subqueries based on reflection
            new_queries = await self._generate_subqueries(
                query=query,
                findings=context["findings"],
                questions=reflection
            )
            context["subqueries"].extend(new_queries)
            
            # Search and analyze for each new query
            for subquery in new_queries:
                # Use the agent to decide how to approach this subquery
                agent_result = await self.agent.arun(
                    f"Research this specific aspect: {subquery}\n\n"
                    f"Current findings: {context['findings']}\n\n"
                    "Think step by step about what tools to use and how to verify the information."
                )
                
                # Perform the search
                search_results = await self.searcher.search(
                    subquery,
                    engines=engines
                )
                
                # Extract URLs for scraping
                urls_to_scrape = await self._extract_urls_from_results(
                    search_results,
                    self.max_urls_per_query
                )
                
                # Scrape and analyze content
                if urls_to_scrape:
                    scraped_content = await self.scraper.scrape_urls(
                        urls_to_scrape,
                        dynamic=True
                    )
                    
                    if scraped_content:
                        # Analyze the content
                        analysis = await self._analyze_content(subquery, scraped_content)
                        context["content_analysis"].append({
                            "subquery": subquery,
                            "analysis": analysis["analysis"],
                            "sources": analysis["sources"]
                        })
                
                # Add results to context
                for r in search_results:
                    if isinstance(r, SearchResult):
                        context["sources"].append(r.to_dict())
                    elif isinstance(r, dict):
                        context["sources"].append(r)
                    else:
                        print(f"Warning: Skipping non-serializable search result: {type(r)}")
                context["findings"] += f"\n\nFindings for '{subquery}':\n{agent_result}"
                
                # Add content analysis if available
                if context["content_analysis"]:
                    latest_analysis = context["content_analysis"][-1]
                    context["findings"] += f"\n\nDetailed Analysis:\n{latest_analysis['analysis']}"
        
        # Final reflection and summary
        final_reflection = await self._reflect_on_findings(context["findings"])
        
        # Prepare detailed sources with content analysis
        detailed_sources = []
        for source in context["sources"]:
            # Source is already a dictionary at this point
            source_dict = source.copy()  # Make a copy to avoid modifying the original
            
            # Add any content analysis related to this source
            for analysis in context["content_analysis"]:
                if source.get("url", "") in analysis["sources"]:
                    source_dict["detailed_analysis"] = analysis["analysis"]
            
            detailed_sources.append(source_dict)
        
        return ResearchResult(
            query=query,
            summary=final_reflection,
            sources=detailed_sources,
            subqueries=context["subqueries"],
            depth=depth,
            content_analysis=context["content_analysis"]
        )

    def research_sync(
        self,
        query: str,
        depth: Optional[int] = None,
        engines: List[str] = ["google", "duckduckgo"]
    ) -> ResearchResult:
        """
        Synchronous version of research method.
        """
        return asyncio.run(self.research(query, depth, engines))
