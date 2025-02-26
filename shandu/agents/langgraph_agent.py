from typing import List, Dict, Optional, Any, Annotated, TypedDict, Sequence, Union, Callable
from dataclasses import dataclass
import json
import time
import asyncio
from datetime import datetime
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.tree import Tree
from rich.progress import Progress, SpinnerColumn, TextColumn
from langgraph.graph import Graph, StateGraph, START, END
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from ..search.search import UnifiedSearcher, SearchResult
from ..scraper import WebScraper, ScrapedContent
from ..research.researcher import ResearchResult
from ..config import config, get_current_date, get_current_datetime

console = Console()

def get_user_input(prompt: str) -> str:
    console.print(Panel(prompt, style="yellow"))
    return input("> ").strip()

class AgentState(TypedDict):
    messages: Sequence[Union[HumanMessage, AIMessage]]
    query: str
    depth: int
    breadth: int
    current_depth: int
    findings: str
    sources: List[Dict[str, Any]]
    subqueries: List[str]
    content_analysis: List[Dict[str, Any]]
    start_time: float
    chain_of_thought: List[str]
    status: str
    current_date: str
    detail_level: str

def should_continue(state: AgentState) -> str:
    if state["current_depth"] < state["depth"]:
        return "continue"
    return "end"

class ResearchGraph:
    def __init__(
        self, 
        llm: Optional[ChatOpenAI] = None, 
        searcher: Optional[UnifiedSearcher] = None, 
        scraper: Optional[WebScraper] = None, 
        temperature: float = 0.7,
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
            max_tokens=8192  # Increased max tokens to support more comprehensive responses
        )
        self.searcher = searcher or UnifiedSearcher()
        self.scraper = scraper or WebScraper()
        self.date = date or get_current_date()
        self.progress_callback = None
        self.include_objective = False
        self.detail_level = "high"
        self.graph = self._build_graph()

    async def initialize_node(self, state: AgentState) -> AgentState:
        console.print(Panel(f"[bold blue]Starting Research:[/] {state['query']}", title="Research Process", border_style="blue"))
        state["start_time"] = time.time()
        state["status"] = "Initializing research"
        state["current_date"] = self.date or get_current_date()
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are an expert research agent tasked with deeply investigating topics.
Current date: {state['current_date']}
Your goal is to create a detailed research plan for the query.
Break down the query into key aspects that need investigation.
Identify potential sources of information and approaches.
Consider different perspectives and potential biases.
Think about how to verify information from multiple sources.

IMPORTANT: Do not use markdown formatting like "**Objective:**" or similar formatting in your response.
Format your response as plain text with clear section headings without special formatting."""),
            ("user", """Create a detailed research plan for investigating:
{query}
Your plan should include:
1. Key aspects to investigate
2. Potential sources of information
3. Specific questions to answer
4. Potential challenges and how to address them

IMPORTANT: Do not use markdown formatting like "**Objective:**" or similar formatting in your response.
Format your response as plain text with clear section headings without special formatting.""")
        ])
        
        chain = prompt | self.llm
        plan = chain.invoke({"query": state["query"]})
        
        # Clean up any markdown formatting that might have been included
        cleaned_plan = plan.content.replace("**", "").replace("# ", "").replace("## ", "")
        
        state["messages"].append(HumanMessage(content=f"Planning research on: {state['query']}"))
        state["messages"].append(AIMessage(content=cleaned_plan))
        state["findings"] = f"# Research Plan\n\n{cleaned_plan}\n\n# Initial Findings\n\n"
        
        self.log_chain_of_thought(state, f"Created research plan for query: {state['query']}")
        if self.progress_callback:
            await self._call_progress_callback(state)
        return state

    async def reflect_node(self, state: AgentState) -> AgentState:
        state["status"] = "Reflecting on findings"
        console.print("[bold yellow]Reflecting on current findings...[/]")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are analyzing research findings to generate insights, identify gaps, and flag irrelevant content.
Current date: {state['current_date']}
Your analysis should be thorough, critical, and balanced.
Look for patterns, contradictions, unanswered questions, and content that is not directly relevant to the main query.
Assess the reliability and potential biases of sources.
Identify areas where more information is needed and suggest how to refine the research focus."""),
            ("user", """Based on the current findings, provide a detailed analysis:
{findings}
Your analysis should include:
1. Key insights discovered so far, including the strength of evidence for each.
2. Important questions that remain unanswered, specifying why they are critical.
3. Assessment of source reliability and potential biases, considering the source's reputation, author's expertise, and presence of citations.
4. Specific areas that need deeper investigation, with suggestions on how to approach them.
5. Identification of any irrelevant or tangential content that should be excluded from further consideration, explaining why.
Think step by step and be specific in your analysis. Consider multiple perspectives and evaluate the overall quality of the information gathered so far.""")
        ])
        
        chain = prompt | self.llm
        reflection = chain.invoke({"findings": state["findings"]})
        
        state["messages"].append(HumanMessage(content="Analyzing current findings..."))
        state["messages"].append(AIMessage(content=reflection.content))
        state["findings"] += f"\n\n## Reflection on Current Findings\n\n{reflection.content}\n\n"
        
        self.log_chain_of_thought(state, "Completed reflection on current findings")
        if self.progress_callback:
            await self._call_progress_callback(state)
        return state

    async def generate_queries_node(self, state: AgentState) -> AgentState:
        state["status"] = "Generating research queries"
        console.print("[bold yellow]Generating targeted search queries...[/]")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are generating targeted search queries to explore specific aspects of a research topic.
Current date: {state['current_date']}
Create queries that are specific, focused, and likely to yield relevant results.
Craft queries that a human would use when searching for information.
Avoid overly technical or complex queries unless necessary.
Each query should address a specific aspect or question, particularly targeting the gaps and unanswered questions identified in the current findings.
DO NOT use formatting like "**Category:**" in your queries.
DO NOT number your queries or add prefixes.
Just return plain, direct search queries that could be typed into a search engine."""),
            ("user", """Generate {breadth} specific search queries to investigate:
Main query: {query}
Current findings and reflection: {findings}
Based on the reflection, particularly the identified gaps, unanswered questions, and areas needing deeper investigation, create queries that will help address these issues.
Your queries should be directly related to the main query and should help:
1. Find specific facts and figures about the topic
2. Verify important claims from reliable sources
3. Explore different perspectives on the issue
4. Find primary sources and official documentation
5. Specifically target the gaps and unanswered questions mentioned in the reflection
IMPORTANT: Format each query as a plain text search query without any prefixes, numbering, or formatting.
BAD: "**Explore New Angles:** renewable energy funding from government"
GOOD: "total US government funding for renewable energy since 2022"
Each query should be on a separate line and should be specific enough to return relevant results.""")
        ])
        
        chain = prompt | self.llm
        result = chain.invoke({"query": state["query"], "findings": state["findings"], "breadth": state["breadth"]})
        
        new_queries = [line.strip() for line in result.content.split("\n") if line.strip() and not line.startswith(("#", "-", "*", "Query", "1.", "2."))]
        new_queries = [line[line.find(". ") + 2:] if line[0].isdigit() and ". " in line[:4] else line for line in new_queries]
        new_queries = new_queries[:state["breadth"]]
        
        state["messages"].append(HumanMessage(content="Generating new research directions..."))
        state["messages"].append(AIMessage(content="Generated queries:\n" + "\n".join(new_queries)))
        state["subqueries"].extend(new_queries)
        
        console.print("[bold green]Generated search queries:[/]")
        for i, query in enumerate(new_queries, 1):
            console.print(f"  {i}. {query}")
        
        self.log_chain_of_thought(state, f"Generated {len(new_queries)} search queries for investigation")
        if self.progress_callback:
            await self._call_progress_callback(state)
        return state

    async def search_node(self, state: AgentState) -> AgentState:
        state["status"] = "Searching and analyzing content"
        recent_queries = state["subqueries"][-state["breadth"]:]
        processed_queries = set()
        
        # Create a simple LRU cache for URL relevance checks to reduce redundant LLM calls
        url_relevance_cache = {}
        max_cache_size = 100
        
        async def is_relevant_url(url, title, snippet, query):
            # First use simple heuristics to avoid LLM calls for obviously irrelevant domains
            irrelevant_domains = ["pinterest", "instagram", "facebook", "twitter", "youtube", "tiktok",
                                "reddit", "quora", "linkedin", "amazon.com", "ebay.com", "etsy.com",
                                "walmart.com", "target.com"]
            if any(domain in url.lower() for domain in irrelevant_domains):
                return False
            
            # Check cache before making LLM call
            cache_key = f"{url}:{query}"
            if cache_key in url_relevance_cache:
                return url_relevance_cache[cache_key]
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are evaluating if a search result is relevant to a query.
Respond with a single word: either "RELEVANT" or "IRRELEVANT"."""),
                ("user", """Query: {query}
Search Result:
Title: {title}
URL: {url}
Snippet: {snippet}
Is this result relevant to the query? Consider:
1. Does the title or snippet directly address the query topic?
2. Does the source seem authoritative for this type of information?
3. Is the content likely to provide factual information rather than opinion or marketing?
IMPORTANT: Respond with only one word: RELEVANT or IRRELEVANT""")
            ])
            chain = prompt | self.llm | StrOutputParser()
            result = await chain.ainvoke({"query": query, "title": title, "url": url, "snippet": snippet})
            is_relevant = "RELEVANT" in result.upper()
            
            # Update cache with result
            if len(url_relevance_cache) >= max_cache_size:
                # Remove a random item if cache is full
                url_relevance_cache.pop(next(iter(url_relevance_cache)))
            url_relevance_cache[cache_key] = is_relevant
            
            return is_relevant
        
        # Process multiple queries in parallel
        async def process_query(subquery, query_task, progress):
            if subquery in processed_queries:
                return
            
            processed_queries.add(subquery)
            
            try:
                self.log_chain_of_thought(state, f"Searching for: {subquery}")
                console.print(f"[dim]Executing search for: {subquery}[/dim]")
                search_results = await self.searcher.search(subquery)
                progress.update(query_task, advance=0.3, description=f"[yellow]Found {len(search_results)} results for: {subquery}")
                
                urls = []
                seen = set()
                
                # Batch URL relevance checks with asyncio.gather
                relevance_tasks = []
                for i, result in enumerate(search_results):
                    if len(urls) >= 5:  # Maximum number of URLs to analyze
                        break
                    if (result.url and isinstance(result.url, str) and result.url not in seen and 
                        result.url.startswith('http')):
                        relevance_tasks.append((i, result, is_relevant_url(result.url, result.title, result.snippet, subquery)))
                
                # Wait for all relevance checks to complete
                for i, result, relevance_task in relevance_tasks:
                    is_relevant = await relevance_task
                    if is_relevant and len(urls) < 5:
                        urls.append(result.url)
                        seen.add(result.url)
                        self.log_chain_of_thought(state, f"Selected relevant URL: {result.url}")
                        console.print(f"[green]Selected for analysis:[/green] {result.title}")
                        console.print(f"[blue]URL:[/blue] {result.url}")
                        if result.snippet:
                            console.print(f"[dim]{result.snippet[:150]}{'...' if len(result.snippet) > 150 else ''}[/dim]")
                        console.print("")
                
                if urls:
                    progress.update(query_task, advance=0.2, description=f"[yellow]Scraping {len(urls)} pages for: {subquery}")
                    scraped = await self.scraper.scrape_urls(urls, dynamic=True)
                    successful_scraped = [s for s in scraped if s.is_successful()]
                    
                    if successful_scraped:
                        progress.update(query_task, advance=0.2, description=f"[yellow]Analyzing content for: {subquery}")
                        content_text = ""
                        
                        # Process content extraction and reliability evaluation in parallel
                        async def process_scraped_item(item):
                            main_content = await self.scraper.extract_main_content(item)
                            
                            # Combined reliability evaluation and content extraction
                            prompt = ChatPromptTemplate.from_messages([
                                ("system", """Analyze this source in two parts:
PART 1: Evaluate the reliability of this source based on domain reputation, author expertise, citations, objectivity, and recency.
PART 2: Extract comprehensive detailed information relevant to the query."""),
                                ("user", """Source URL: {url}
Title: {title}
Query: {query}
Content: {content}

Provide your response in two clearly separated sections:
RELIABILITY: [HIGH/MEDIUM/LOW] followed by a brief justification (1-2 sentences)
EXTRACTED_CONTENT: Detailed facts, statistics, data points, examples, and key information relevant to the query.""")
                            ])
                            chain = prompt | self.llm
                            result = await chain.ainvoke({
                                "url": item.url, 
                                "title": item.title, 
                                "query": subquery,
                                "content": main_content[:8000]  # Reduced from 10000 to 8000 for faster processing
                            })
                            
                            # Parse the combined response
                            response_text = result.content
                            reliability_section = ""
                            content_section = ""
                            
                            if "RELIABILITY:" in response_text and "EXTRACTED_CONTENT:" in response_text:
                                parts = response_text.split("EXTRACTED_CONTENT:")
                                reliability_section = parts[0].replace("RELIABILITY:", "").strip()
                                content_section = parts[1].strip()
                            else:
                                # Fallback if format wasn't followed
                                reliability_section = "MEDIUM (Unable to parse reliability assessment)"
                                content_section = response_text
                            
                            # Extract rating
                            rating = "MEDIUM"
                            if "HIGH" in reliability_section.upper():
                                rating = "HIGH"
                            elif "LOW" in reliability_section.upper():
                                rating = "LOW"
                            
                            justification = reliability_section.replace("HIGH", "").replace("MEDIUM", "").replace("LOW", "").strip()
                            if justification.startswith("(") and justification.endswith(")"):
                                justification = justification[1:-1].strip()
                            
                            return {
                                "item": item,
                                "rating": rating,
                                "justification": justification,
                                "content": content_section
                            }
                        
                        # Process all items in parallel
                        processing_tasks = [process_scraped_item(item) for item in successful_scraped]
                        processed_items = await asyncio.gather(*processing_tasks)
                        
                        # Build content text from processed items
                        relevant_items = []
                        for processed in processed_items:
                            if processed["rating"] == "LOW":
                                console.print(f"[yellow]Skipping low-reliability source: {processed['item'].url}[/yellow]")
                                continue
                            
                            relevant_items.append(processed)
                            content_text += f"\nSource: {processed['item'].url}\nTitle: {processed['item'].title}\nReliability: {processed['rating']}\nRelevant Content:\n{processed['content']}\n\n"
                            console.print(f"\n[bold cyan]Analyzing page:[/bold cyan] {processed['item'].title}")
                            console.print(f"[blue]URL:[/blue] {processed['item'].url}")
                            console.print(f"[dim]Extracted Content: {processed['content'][:150]}{'...' if len(processed['content']) > 150 else ''}[/dim]")
                        
                        if relevant_items:
                            # Improved content analysis prompt for more cohesive organization
                            analysis_prompt = ChatPromptTemplate.from_messages([
                                ("system", """You are analyzing web content to extract comprehensive information and organize it thematically.
Your analysis should be thorough and well-structured, focusing on evidence assessment and in-depth exploration.
Group information by themes and integrate data from different sources into unified sections.
Avoid contradictions or redundancy in your analysis.

For evidence assessment:
- Be concise when evaluating source reliability - focus on the highest and lowest credibility sources only
- Briefly note bias or conflicts of interest in sources
- Prioritize original research, peer-reviewed content, and official publications 

For in-depth analysis:
- Provide extensive exploration of key concepts and technologies
- Highlight current trends, challenges and future directions
- Present technical details when relevant to understanding the topic
- Include comparative analysis of different methodologies or approaches"""),
                                ("user", """Analyze the following content related to: "{query}"
Content:
{content}

Provide a comprehensive analysis including:
1. Key themes and concepts identified across sources
2. Detailed evidence and statistics organized by theme
3. Patterns and trends evident across the sources
4. Detailed exploration of important concepts

IMPORTANT:
Each piece of information must only appear ONCE in your analysis.
If two sources mention the same fact, present it once with multiple citations.
Avoid stating the same information multiple times in different sections.
Ensure each paragraph has unique content not repeated elsewhere.

Guidelines:
- Group information by themes rather than by source
- Integrate related information into cohesive sections
- Include specific facts and technical details
- Ensure proper citation of sources using [n] notation
- Use markdown formatting to organize content effectively
- Cite multiple sources for the same information where appropriate""")
                            ])
                            
                            # Use more tokens but with a timeout to avoid hanging
                            analysis_llm = self.llm.with_config({"max_tokens": 4096, "timeout": 120})
                            analysis_chain = analysis_prompt | analysis_llm
                            analysis = analysis_chain.invoke({"query": subquery, "content": content_text})
                            
                            state["content_analysis"].append({
                                "subquery": subquery,
                                "analysis": analysis.content,
                                "sources": [item["item"].url for item in relevant_items]
                            })
                            
                            # Store findings with thematic headers
                            state["findings"] += f"\n\n## Research on '{subquery}':\n{analysis.content}\n"
                            self.log_chain_of_thought(state, f"Analyzed content for query: {subquery}")
                
                for r in search_results:
                    if isinstance(r, SearchResult):
                        state["sources"].append(r.to_dict())
                    elif isinstance(r, dict):
                        state["sources"].append(r)
                
                progress.update(query_task, completed=True, description=f"[green]Completed: {subquery}")
                return True
                
            except Exception as e:
                progress.update(query_task, completed=True, description=f"[red]Error: {subquery} - {str(e)}")
                state["messages"].append(HumanMessage(content=f"Failed to process subquery: {subquery}"))
                self.log_chain_of_thought(state, f"Error processing query '{subquery}': {str(e)}")
                console.print(f"[dim red]Error processing {subquery}: {str(e)}[/dim red]")
                return False
        
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            # Create all tasks first
            tasks = {}
            for i, subquery in enumerate(recent_queries):
                if subquery not in processed_queries:
                    task_id = progress.add_task(f"[yellow]Searching: {subquery}", total=1)
                    tasks[subquery] = task_id
            
            # Process queries in parallel batches to avoid overwhelming the system
            batch_size = min(3, len(tasks))  # Process up to 3 queries at once
            
            for i in range(0, len(tasks), batch_size):
                batch_queries = list(tasks.keys())[i:i+batch_size]
                batch_tasks = [process_query(query, tasks[query], progress) for query in batch_queries]
                await asyncio.gather(*batch_tasks)
        
        state["current_depth"] += 1
        elapsed_time = time.time() - state["start_time"]
        minutes, seconds = divmod(int(elapsed_time), 60)
        state["status"] = f"Completed depth {state['current_depth']}/{state['depth']} ({minutes}m {seconds}s elapsed)"
        
        if self.progress_callback:
            await self._call_progress_callback(state)
        return state

    async def report_node(self, state: AgentState) -> AgentState:
        state["status"] = "Generating final report"
        console.print("[bold blue]Research complete. Generating final report...[/]")

        current_date = state["current_date"]

        # Improved system prompt for detailed, cohesive reports
        system_prompt = f"""You are synthesizing research findings into a comprehensive, detailed, and insightful report.
Today's date is {current_date}.

Your report should be:
- Well-structured with clear sections and logical flow
- Extremely comprehensive and detailed (minimum 7000+ words), providing in-depth analysis
- Written with long, cohesive paragraphs that thoroughly explore each topic (avoid one-paragraph-per-finding structure)
- Rich in detail, with extensive elaboration on each point rather than brief summaries
- Balanced and objective, presenting multiple perspectives when relevant
- Evidence-based, with proper citations to sources using [n] format, consistently linked to the reference section
- Free of repetition while being thorough and extensive in covering the topic
- Tailored to the user's desired detail level: {state['detail_level']}

IMPORTANT REGARDING REPETITION AND CONTRADICTIONS:
- Each key fact, statistic, or piece of information must appear only ONCE in your report
- If you need to refer to the same information in different sections, reference it without repeating all the details
- Check each paragraph to ensure it contains unique information not stated elsewhere
- Ensure consistent treatment of facts across the entire report
- If sources present contradictory information, acknowledge this explicitly rather than presenting both as fact
- Do not repeat the same example, statistic, or explanation in different sections

Use advanced markdown formatting to organize content effectively:
- Create tables for comparing data or options
- Use bullet points for listings where appropriate
- Include section headers (# and ##) to organize content
- Use blockquotes for important quotations
- Add horizontal rules (---) to separate major sections
- Consider including a table of contents for longer reports

IMPORTANT: For citations, use numbered references in square brackets [1], [2], etc. when citing information.
Always ensure each citation corresponds to the correct entry in the References section.
Verify that every source cited in the text appears in the References section with matching numbers.

Ensure the report includes:
- All relevant statistics, trends, and data points with detailed analysis
- Multiple paragraphs of exploration for each major topic (rather than one paragraph per source)
- Long-form content that deeply explores each topic with proper transitions between related ideas
- Detailed assessment of source reliability and evidence strength
- Comprehensive discussion of any contradictions or discrepancies in the findings"""

        if not self.include_objective:
            system_prompt += """

IMPORTANT: DO NOT include an "Objective" section at the beginning of the report. Start directly with an Executive Summary section."""

        # Improved prompt for cohesive, long-form report generation
        report_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", """Create a comprehensive research report for the query: {query}

Analyzed Findings: {analyzed_findings}
Number of sources: {num_sources}

Your report should include the following sections:
1. Executive Summary - A concise overview of key findings (300-500 words)
2. In-Depth Analysis - Detailed, long-form examination of all relevant aspects
   - Write multiple paragraphs for each subtopic (not just one paragraph per finding)
   - Integrate information from multiple sources within cohesive sections
   - Ensure thorough exploration of each aspect (minimum 3000+ words for this section)
3. Evidence Assessment - CONCISE evaluation of source reliability
   - Focus only on the most and least reliable sources
   - Briefly note any potential biases or conflicts of interest
   - Keep this section focused and brief (maximum 500 words)
4. Uncertainties and Open Questions - Gaps in knowledge that remain
5. Recommendations for Further Research - Based on identified gaps
6. Additional Insights - Any other relevant perspectives

Important guidelines:
- GENERATE A LONG REPORT (minimum 7000+ words) with extensive exploration of each topic
- Write in a cohesive, flowing narrative style rather than disconnected paragraphs
- Group related information into substantive sections even if from different sources
- Integrate information from different sources into unified, thematic sections
- AVOID one-paragraph-per-finding structure - instead, develop rich, multi-paragraph explorations
- Use proper transitions between related topics to ensure smooth reading
- EXTRACT AND INCLUDE extensive data from websites - don't summarize when detail is available
- Allocate most content to the In-Depth Analysis section
- Use numbered citations [1], [2], etc. consistently when referencing sources
- Each source citation should correspond exactly to the numbered entry in the references
- Utilize advanced markdown features (tables, blockquotes, etc.) to enhance readability
- Create tables for comparing data points or options when appropriate
""")
        ])
        
        # Use a higher token limit for the report generation
        report_llm = self.llm.with_config({"max_tokens": 8192})
        report_chain = report_prompt | report_llm
        final_report = report_chain.invoke({
            "query": state["query"],
            "analyzed_findings": state["findings"],
            "num_sources": len(state["sources"])
        }).content

        elapsed_time = time.time() - state["start_time"]
        minutes, seconds = divmod(int(elapsed_time), 60)

        state["messages"].append(AIMessage(content="Research complete. Generating final report..."))
        state["findings"] = final_report
        state["status"] = "Complete"

        self.log_chain_of_thought(state, f"Generated final report after {minutes}m {seconds}s")
        if self.progress_callback:
            await self._call_progress_callback(state)
        return state

    def log_chain_of_thought(self, state: AgentState, thought: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        state["chain_of_thought"].append(f"[{timestamp}] {thought}")

    async def _call_progress_callback(self, state: AgentState) -> None:
        if self.progress_callback:
            if asyncio.iscoroutinefunction(self.progress_callback):
                await self.progress_callback(state)
            else:
                self.progress_callback(state)

    def _build_graph(self) -> Graph:
        workflow = StateGraph(AgentState)
        workflow.add_node("initialize", self.initialize_node)
        workflow.add_node("reflect", self.reflect_node)
        workflow.add_node("generate_queries", self.generate_queries_node)
        workflow.add_node("search", self.search_node)
        workflow.add_node("report", self.report_node)
        workflow.add_edge("initialize", "generate_queries")
        workflow.add_edge("reflect", "generate_queries")
        workflow.add_edge("generate_queries", "search")
        workflow.add_conditional_edges("search", should_continue, {"continue": "reflect", "end": "report"})
        workflow.set_entry_point("initialize")
        workflow.set_finish_point("report")
        return workflow.compile()

    async def research(
        self, 
        query: str, 
        depth: int = 2, 
        breadth: int = 4, 
        progress_callback: Optional[Callable[[AgentState], None]] = None,
        include_objective: bool = False,
        detail_level: str = "high" 
    ) -> ResearchResult:
        self.progress_callback = progress_callback
        self.include_objective = include_objective
        self.detail_level = detail_level

        state = AgentState(
            messages=[HumanMessage(content=f"Starting research on: {query}")],
            query=query,
            depth=depth,
            breadth=breadth,
            current_depth=0,
            findings="",
            sources=[],
            subqueries=[],
            content_analysis=[],
            start_time=time.time(),
            chain_of_thought=[],
            status="Starting",
            current_date=get_current_date(),
            detail_level=detail_level
        )
        
        final_state = await self.graph.ainvoke(state)
        
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

    def research_sync(
        self, 
        query: str, 
        depth: int = 2, 
        breadth: int = 4, 
        progress_callback: Optional[Callable[[AgentState], None]] = None,
        include_objective: bool = False,
        detail_level: str = "high"
    ) -> ResearchResult:
        import asyncio
        return asyncio.run(self.research(query, depth, breadth, progress_callback, include_objective, detail_level))

def display_research_progress(state: AgentState) -> Tree:
    elapsed_time = time.time() - state["start_time"]
    minutes, seconds = divmod(int(elapsed_time), 60)
    
    tree = Tree(f"[bold blue]Research Progress: {state['status']}")
    stats_node = tree.add(f"[cyan]Stats")
    stats_node.add(f"[blue]Time Elapsed:[/] {minutes}m {seconds}s")
    stats_node.add(f"[blue]Current Depth:[/] {state['current_depth']}/{state['depth']}")
    stats_node.add(f"[blue]Sources Found:[/] {len(state['sources'])}")
    stats_node.add(f"[blue]Subqueries Explored:[/] {len(state['subqueries'])}")
    
    if state["subqueries"]:
        queries_node = tree.add("[green]Current Research Paths")
        for query in state["subqueries"][-state["breadth"]:]:
            queries_node.add(query)
    
    if state["chain_of_thought"]:
        thoughts_node = tree.add("[yellow]Recent Thoughts")
        for thought in state["chain_of_thought"][-3:]:
            thoughts_node.add(thought)
    
    if state["findings"]:
        findings_node = tree.add("[magenta]Latest Findings")
        sections = state["findings"].split("\n\n")
        for section in sections[-2:]:
            if section.strip():
                findings_node.add(section.strip()[:100] + "..." if len(section.strip()) > 100 else section.strip())
    
    return tree

async def clarify_query(query: str, llm: Optional[ChatOpenAI] = None, date: Optional[str] = None) -> str:
    if llm is None:
        api_base = config.get("api", "base_url")
        api_key = config.get("api", "api_key")
        model = config.get("api", "model")
        temperature = config.get("api", "temperature", 0.5)
        llm = ChatOpenAI(base_url=api_base, api_key=api_key, temperature=0.5)
    
    current_date = date or get_current_date()
    console.print(Panel(f"[bold blue]Initial Query:[/] {query}", title="Research Setup"))
    console.print("\n[bold]I'll ask a few questions to better understand your research needs.[/]")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are a research assistant helping to clarify research queries.
Today's date is {current_date}.
Your goal is to ask questions that will help refine the scope, focus, and direction of the research.
Ask questions that will help understand:
1. The specific aspects the user wants to explore
2. The level of detail needed
3. Any specific sources or perspectives to include or exclude
4. The time frame or context relevant to the query
5. The user's background knowledge on the topic"""),
        ("user", """Generate 3 follow-up questions to better understand the research needs for the query: "{query}"
Questions should be concise, specific, and help refine the scope, focus, and direction of the research.
Each question should address a different aspect of the research needs.""")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"query": query})
    questions = [q.strip() for q in response.content.split("\n") if q.strip() and "?" in q]
    
    answers = []
    for q in questions[:3]:
        answer = get_user_input(q)
        answers.append(answer)
    
    refinement_prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are refining a research query based on user responses.
Today's date is {current_date}.
Your goal is to create a comprehensive research context that includes:
1. A clear objective statement that begins with "Objective:" 
2. Key aspects to focus on based on user responses
3. Any constraints or preferences mentioned
4. Specific areas to explore in depth

IMPORTANT: Do not use markdown formatting like "**Objective:**" or similar formatting in your response.
Format your response as plain text with clear section headings without special formatting.
Do not use bold text, italics, or other markdown formatting."""),
        ("user", """Original query: {query}
Follow-up questions and answers:
{qa}
Based on these, provide a refined research query or context that captures all the important details.
The refined query should be comprehensive but focused, incorporating all relevant information from the user's responses.

IMPORTANT: Do not use markdown formatting like "**Objective:**" or similar formatting in your response.
Format your response as plain text with clear section headings without special formatting.""")
    ])
    
    qa_text = "\n".join([f"Q: {q}\nA: {a}" for q, a in zip(questions, answers)])
    refined_query = refinement_prompt | llm
    refined_context_raw = refined_query.invoke({"query": query, "qa": qa_text}).content
    
    # Clean up any markdown formatting that might have been included
    refined_context = refined_context_raw.replace("**", "").replace("# ", "").replace("## ", "")
    
    console.print(Panel(Markdown(f"{refined_context}"), title="Research Plan", border_style="green"))
    return refined_context
