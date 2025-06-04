"""Report generation nodes with modular, robust processing."""
import re
import time
import asyncio
import traceback
from typing import List, Dict, Any, Optional, Tuple
from rich.console import Console
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field
from ..processors.content_processor import AgentState
from ..processors.report_generator import (
    generate_title,
    extract_themes,
    generate_initial_report,
    enhance_report,
    expand_key_sections,
    format_citations
)
from ...retrieval import get_retriever
from ..utils.agent_utils import log_chain_of_thought, _call_progress_callback, is_shutdown_requested
from ..utils.citation_registry import CitationRegistry
from ..utils.citation_manager import CitationManager, SourceInfo, Learning

console = Console()

# Structured output models for report generation
class ReportSection(BaseModel):
    """Structured output for a report section."""
    title: str = Field(description="Title of the section")
    content: str = Field(description="Content of the section")
    order: int = Field(description="Order of the section in the report", default=0)
    status: str = Field(description="Processing status of the section", default="pending")

class FinalReport(BaseModel):
    """Structured output for the final report."""
    title: str = Field(description="Title of the report")
    sections: list[ReportSection] = Field(
        description="List of report sections",
        min_items=1
    )
    references: list[str] = Field(
        description="List of references in the report",
        min_items=0
    )
    
# Maximum retry attempts for report generation processes
MAX_RETRIES = 3

async def prepare_report_data(state: AgentState) -> Tuple[CitationManager, CitationRegistry, Dict[str, Any]]:
    """
    Prepare all necessary data for report generation, ensuring sources are correctly registered.
    
    Args:
        state: The current agent state
        
    Returns:
        Tuple containing the citation manager, citation registry, and citation statistics
    """
    # Initialize or retrieve citation manager
    if "citation_manager" not in state:
        citation_manager = CitationManager()
        state["citation_manager"] = citation_manager
        # For backward compatibility
        state["citation_registry"] = citation_manager.citation_registry
    else:
        citation_manager = state["citation_manager"]

    citation_registry = citation_manager.citation_registry

    # Pre-register all selected sources and extract learnings
    if "selected_sources" in state and state["selected_sources"]:
        for url in state["selected_sources"]:
            source_meta = next((s for s in state["sources"] if s.get("url") == url), {})

            source_info = SourceInfo(
                url=url,
                title=source_meta.get("title", ""),
                snippet=source_meta.get("snippet", ""),
                source_type="web",
                content_type=source_meta.get("content_type", "article"),
                access_time=time.time(),
                domain=url.split("//")[1].split("/")[0] if "//" in url else "unknown",
                reliability_score=0.8,  # Default score, could be more dynamic
                metadata=source_meta
            )

            citation_manager.add_source(source_info)

            for analysis in state["content_analysis"]:
                if url in analysis.get("sources", []):
                    citation_manager.extract_learning_from_text(
                        analysis.get("analysis", ""),
                        url,
                        context=f"Analysis for query: {analysis.get('query', '')}"
                    )

            # For backward compatibility with citation registry
            cid = citation_registry.register_citation(url)
            citation_registry.update_citation_metadata(cid, {
                "title": source_meta.get("title", "Untitled"),
                "date": source_meta.get("date", "n.d."),
                "url": url
            })

    citation_stats = citation_manager.get_learning_statistics()
    console.print(f"[bold green]Processed {citation_stats.get('total_learnings', 0)} learnings from {citation_stats.get('total_sources', 0)} sources[/]")
    
    return citation_manager, citation_registry, citation_stats


async def generate_initial_report_node(llm, include_objective, progress_callback, state: AgentState) -> AgentState:
    """Generate the initial report with enhanced citation tracking using a modular approach."""
    state["status"] = "Generating initial report with enhanced source attribution"
    console.print("[bold blue]Generating comprehensive report with dynamic structure and source tracking...[/]")

    current_date = state["current_date"]
    
    # Prepare all citation data
    citation_manager, citation_registry, citation_stats = await prepare_report_data(state)

    # Retrieve additional context from indexed documents
    retriever = get_retriever(llm)
    retrieved_docs = retriever.retrieve(state["query"], k=5)
    retrieved_passages = "\n\n".join(doc.page_content for doc in retrieved_docs)
    state["retrieved_passages"] = retrieved_passages

    # Step 1: Generate report title (with retries)
    report_title = None
    for attempt in range(MAX_RETRIES):
        try:
            report_title = await generate_title(llm, state['query'])
            console.print(f"[bold green]Generated title: {report_title}[/]")
            break
        except Exception as e:
            console.print(f"[yellow]Title generation attempt {attempt+1} failed: {str(e)}[/]")
            if attempt == MAX_RETRIES - 1:
                report_title = f"Research on {state['query']}"
                console.print(f"[yellow]Using fallback title: {report_title}[/]")

    # Step 2: Extract themes to structure the report (with retries)
    extracted_themes = None
    for attempt in range(MAX_RETRIES):
        try:
            extracted_themes = await extract_themes(llm, state['findings'])
            break
        except Exception as e:
            console.print(f"[yellow]Theme extraction attempt {attempt+1} failed: {str(e)}[/]")
            if attempt == MAX_RETRIES - 1:
                # Create fallback themes if all attempts fail
                extracted_themes = "## Main Concepts\nCore concepts related to the topic.\n\n## Applications\nPractical applications and implementations.\n\n## Challenges\nChallenges and limitations in the field.\n\n## Future Directions\nEmerging trends and future possibilities."
                console.print("[yellow]Using fallback themes structure[/]")

    # Step 3: Format citations (with retries)
    formatted_citations = None
    for attempt in range(MAX_RETRIES):
        try:
            formatted_citations = await format_citations(
                llm,
                state.get('selected_sources', []),
                state["sources"],
                citation_registry
            )
            break
        except Exception as e:
            console.print(f"[yellow]Citation formatting attempt {attempt+1} failed: {str(e)}[/]")
            if attempt == MAX_RETRIES - 1:
                # Create basic citations if all attempts fail
                formatted_citations = "\n".join([f"[{i+1}] {url}" for i, url in enumerate(state.get('selected_sources', []))])
                console.print("[yellow]Using fallback citation format[/]")

    # Step 4: Generate the initial report with progress tracking
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Generating report..."),
        console=console
    ) as progress:
        task = progress.add_task("Generating", total=1)
        
        initial_report = None
        for attempt in range(MAX_RETRIES):
            try:
                initial_report = await generate_initial_report(
                    llm,
                    state['query'],
                    state['findings'],
                    state.get('retrieved_passages', ''),
                    extracted_themes,
                    report_title,
                    state['selected_sources'],
                    formatted_citations,
                    current_date,
                    state['detail_level'],
                    include_objective,
                    citation_registry
                )
                progress.update(task, completed=1)
                break
            except Exception as e:
                console.print(f"[yellow]Report generation attempt {attempt+1} failed: {str(e)}[/]")
                if attempt == MAX_RETRIES - 1:
                    # Create a minimal report if all attempts fail
                    console.print("[yellow]Creating fallback report structure[/]")
                    initial_report = f"# {report_title}\n\n## Executive Summary\n\nThis report explores {state['query']}.\n\n"
                    
                    # Extract sections from themes
                    section_matches = re.findall(r"##\s+([^\n]+)(?:\n([^#]+))?", extracted_themes)
                    for title, content in section_matches:
                        initial_report += f"## {title}\n\n{content.strip() if content else 'Information on this topic.'}\n\n"
                    
                    initial_report += "## References\n\n" + formatted_citations
                    progress.update(task, completed=1)

    # Store data for later stages
    state["identified_themes"] = extracted_themes
    state["initial_report"] = initial_report
    state["formatted_citations"] = formatted_citations
    state["report_title"] = report_title

    log_chain_of_thought(
        state,
        f"Generated initial report with {len(citation_registry.citations)} properly tracked citations and {citation_stats.get('total_learnings', 0)} learnings"
    )
    
    if progress_callback:
        await _call_progress_callback(progress_callback, state)
    return state

async def enhance_report_node(llm, progress_callback, state: AgentState) -> AgentState:
    """
    Enhance the report by processing each section individually to improve reliability.
    """
    if is_shutdown_requested():
        state["status"] = "Shutdown requested, skipping report enhancement"
        log_chain_of_thought(state, "Shutdown requested, skipping report enhancement")
        return state

    state["status"] = "Enhancing report sections"
    console.print("[bold blue]Enhancing report with more detailed information...[/]")
    
    initial_report = state.get("initial_report", "")
    if not initial_report or len(initial_report.strip()) < 500:
        log_chain_of_thought(state, "Initial report too short or missing, skipping enhancement")
        state["enhanced_report"] = initial_report
        return state
    
    # Extract report title and sections
    title_match = re.match(r'# ([^\n]+)', initial_report)
    original_title = title_match.group(1) if title_match else state.get("report_title", "Research Report")
    
    # Extract sections using regex pattern
    section_pattern = re.compile(r'(#+\s+[^\n]+)(\n\n[^#]+?)(?=\n#+\s+|\Z)', re.DOTALL)
    sections = section_pattern.findall(initial_report)
    
    if not sections:
        log_chain_of_thought(state, "No sections found in report, using initial report as is")
        state["enhanced_report"] = initial_report
        return state

    # Process each section in parallel for better reliability
    enhanced_sections = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Enhancing sections..."),
        console=console
    ) as progress:
        task = progress.add_task("Enhancing", total=len(sections))
        
        # Prepare citation information for enhancement
        citation_registry = state.get("citation_registry")
        formatted_citations = state.get("formatted_citations", "")
        current_date = state.get("current_date", "")
        
        # Process each section (excluding references)
        for i, (section_header, section_content) in enumerate(sections):
            # Skip enhancing references section
            if "References" in section_header or "references" in section_header.lower():
                enhanced_sections.append((i, f"{section_header}{section_content}"))
                progress.update(task, advance=1)
                continue
                
            for attempt in range(MAX_RETRIES):
                try:
                    # Generate available sources text for this section
                    available_sources_text = ""
                    if citation_registry:
                        available_sources = []
                        for cid in sorted(citation_registry.citations.keys()):
                            citation_info = citation_registry.citations[cid]
                            url = citation_info.get("url", "")
                            title = citation_info.get("title", "")
                            available_sources.append(f"[{cid}] - {title} ({url})")
                        
                        if available_sources:
                            available_sources_text = "\n\nAVAILABLE SOURCES FOR CITATION:\n" + "\n".join(available_sources)
                    
                    # Configure LLM for this section enhancement
                    enhance_llm = llm.with_config({"max_tokens": 4096, "temperature": 0.2})
                    
                    # Create section-specific enhancement prompt
                    section_prompt = f"""Enhance this section of a research report with additional depth and detail:

{section_header}{section_content}{available_sources_text}

Your task is to:
1. Add more detailed explanations to key concepts
2. Expand on examples and case studies
3. Enhance the analysis and interpretation of findings
4. Improve the flow within this section
5. Add relevant statistics, data points, or evidence
6. Ensure proper citation [n] format throughout
7. Maintain scientific accuracy and up-to-date information (current as of {current_date})

CITATION REQUIREMENTS:
- ONLY use the citation IDs provided in the AVAILABLE SOURCES list above
- Format citations as [n] where n is the exact ID of the source
- Place citations at the end of the relevant sentences or paragraphs
- Do not make up your own citation numbers
- Do not cite sources that aren't in the available sources list

IMPORTANT:
- DO NOT change the section heading
- DO NOT add information not supported by the research
- DO NOT use academic-style citations like "Journal of Medicine (2020)"
- DO NOT include PDF/Text/ImageB/ImageC/ImageI tags or any other markup
- Return ONLY the enhanced section with the original heading

Return the enhanced section with the exact same heading but with expanded content.
"""
                    # Enhance the section
                    response = await enhance_llm.ainvoke(section_prompt)
                    section_text = response.content
                    
                    # Clean up any markup errors
                    section_text = re.sub(r'\[\/?(?:PDF|Text|ImageB|ImageC|ImageI)(?:\/?|\])(?:[^\]]*\])?', '', section_text)
                    
                    # Ensure the section starts with the correct header
                    if not section_text.strip().startswith(section_header.strip()):
                        section_text = f"{section_header}\n\n{section_text}"
                    
                    # Store the enhanced section with its original position
                    enhanced_sections.append((i, section_text))
                    break
                    
                except Exception as e:
                    console.print(f"[yellow]Error enhancing section '{section_header.strip()}' (Attempt {attempt+1}): {str(e)}[/]")
                    if attempt == MAX_RETRIES - 1:
                        # If all enhancement attempts fail, use the original section
                        enhanced_sections.append((i, f"{section_header}{section_content}"))
            
            progress.update(task, advance=1)
    
    # Sort sections by their original order and combine into the enhanced report
    enhanced_sections.sort(key=lambda x: x[0])
    enhanced_report = f"# {original_title}\n\n" + "\n\n".join([section for _, section in enhanced_sections])
    
    # Update state with the enhanced report
    state["enhanced_report"] = enhanced_report
    log_chain_of_thought(state, f"Enhanced report with {len(sections)} sections processed")
    
    if progress_callback:
        await _call_progress_callback(progress_callback, state)
    return state

async def expand_key_sections_node(llm, progress_callback, state: AgentState) -> AgentState:
    """
    Expand key sections of the report to provide more comprehensive information.
    """
    if is_shutdown_requested():
        state["status"] = "Shutdown requested, skipping section expansion"
        log_chain_of_thought(state, "Shutdown requested, skipping section expansion")
        return state
    
    state["status"] = "Expanding key report sections"
    console.print("[bold blue]Expanding key sections with more comprehensive information...[/]")
    
    enhanced_report = state.get("enhanced_report", "")
    if not enhanced_report or len(enhanced_report.strip()) < 500:
        log_chain_of_thought(state, "Enhanced report too short or missing, using as is")
        state["final_report"] = enhanced_report
        return state
    
    # Get report title and sections
    title_match = re.match(r'# ([^\n]+)', enhanced_report)
    original_title = title_match.group(1) if title_match else state.get("report_title", "Research Report")
    
    # Extract sections using regex pattern (only level 2 headings - main content sections)
    section_pattern = re.compile(r'(##\s+[^\n]+)(\n\n[^#]+?)(?=\n##\s+|\Z)', re.DOTALL)
    sections = section_pattern.findall(enhanced_report)
    
    if not sections:
        log_chain_of_thought(state, "No expandable sections found, using enhanced report as is")
        state["final_report"] = enhanced_report
        return state
    
    # Identify important sections to expand (excluding Executive Summary, Introduction, Conclusion, References)
    important_sections = []
    for i, (section_header, section_content) in enumerate(sections):
        title = section_header.replace('#', '').strip().lower()
        if title not in ["executive summary", "introduction", "conclusion", "references"]:
            important_sections.append((i, section_header, section_content))
    
    # Limit to 3 most important sections
    important_sections = important_sections[:3]
    if not important_sections:
        log_chain_of_thought(state, "No key sections to expand, using enhanced report as is")
        state["final_report"] = enhanced_report
        return state
    
    # Create a copy of the report that we'll modify
    expanded_report = enhanced_report
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Expanding key sections..."),
        console=console
    ) as progress:
        task = progress.add_task("Expanding", total=len(important_sections))
        
        # Prepare citation information for expansion
        citation_registry = state.get("citation_registry")
        current_date = state.get("current_date", "")
        
        # Process each important section
        for i, section_header, section_content in important_sections:
            section_title = section_header.replace('#', '').strip()
            
            for attempt in range(MAX_RETRIES):
                try:
                    # Generate available sources text for this section
                    available_sources_text = ""
                    if citation_registry:
                        available_sources = []
                        for cid in sorted(citation_registry.citations.keys()):
                            citation_info = citation_registry.citations[cid]
                            url = citation_info.get("url", "")
                            title = citation_info.get("title", "")
                            available_sources.append(f"[{cid}] - {title} ({url})")
                        
                        if available_sources:
                            available_sources_text = "\n\nAVAILABLE SOURCES FOR CITATION:\n" + "\n".join(available_sources)
                    
                    # Configure LLM for this section expansion
                    expand_llm = llm.with_config({"max_tokens": 6144, "temperature": 0.2})
                    
                    # Create section-specific expansion prompt
                    section_prompt = f"""Expand this section of a research report with much greater depth and detail:

{section_header}{section_content}{available_sources_text}

EXPANSION REQUIREMENTS:
1. Triple the length and detail of the section while maintaining accuracy
2. Add specific examples, case studies, or data points to support claims
3. Include additional context and background information
4. Add nuance, caveats, and alternative perspectives
5. Use proper citation format [n] throughout
6. Maintain the existing section structure but add subsections if appropriate
7. Ensure all information is accurate as of {current_date}

CITATION REQUIREMENTS:
- ONLY use the citation IDs provided in the AVAILABLE SOURCES list above
- Format citations as [n] where n is the exact ID of the source
- Place citations at the end of the relevant sentences or paragraphs
- Do not make up your own citation numbers
- Do not cite sources that aren't in the available sources list
- Ensure each major claim or statistic has an appropriate citation

IMPORTANT:
- DO NOT change the section heading
- DO NOT add information not supported by the research
- DO NOT use academic-style citations like "Journal of Medicine (2020)"
- DO NOT include PDF/Text/ImageB/ImageC/ImageI tags or any other markup
- Return ONLY the expanded section with the original heading

Return the expanded section with the exact same heading but with expanded content.
"""
                    # Expand the section
                    response = await expand_llm.ainvoke(section_prompt)
                    expanded_content = response.content
                    
                    # Clean up any markup errors
                    expanded_content = re.sub(r'\[\/?(?:PDF|Text|ImageB|ImageC|ImageI)(?:\/?|\])(?:[^\]]*\])?', '', expanded_content)
                    
                    # Ensure the section starts with the correct header
                    if not expanded_content.strip().startswith(section_header.strip()):
                        expanded_content = f"{section_header}\n\n{expanded_content}"
                    
                    # Replace the original section with the expanded one in the report
                    section_pattern_to_replace = re.escape(section_header) + r'\s*\n\n' + re.escape(section_content.strip())
                    expanded_report = re.sub(section_pattern_to_replace, expanded_content, expanded_report, flags=re.DOTALL)
                    
                    break
                except Exception as e:
                    console.print(f"[yellow]Error expanding section '{section_title}' (Attempt {attempt+1}): {str(e)}[/]")
                    if attempt == MAX_RETRIES - 1:
                        # If all expansion attempts fail, keep the original section
                        console.print(f"[yellow]Failed to expand section '{section_title}', keeping original[/]")
            
            progress.update(task, advance=1)
    
    # Update state with the expanded report
    state["final_report"] = expanded_report
    log_chain_of_thought(state, f"Expanded {len(important_sections)} key sections in the report")
    
    if progress_callback:
        await _call_progress_callback(progress_callback, state)
    return state

async def report_node(llm, progress_callback, state: AgentState) -> AgentState:
    """Finalize the report."""
    state["status"] = "Finalizing report"
    console.print("[bold blue]Research complete. Finalizing report...[/]")

    has_report = False
    if "final_report" in state and state["final_report"]:
        final_report = state["final_report"]
        has_report = True
    elif "enhanced_report" in state and state["enhanced_report"]:
        final_report = state["enhanced_report"]
        has_report = True
    elif "initial_report" in state and state["initial_report"]:
        final_report = state["initial_report"]
        has_report = True
    
    # If we have a report but it's broken or too short, regenerate it
    if has_report and (len(final_report.strip()) < 1000):
        console.print("[bold yellow]Existing report appears broken or incomplete. Regenerating...[/]")
        has_report = False
        
    # If we don't have a report, regenerate initial, enhanced, and expanded reports
    if not has_report:
        console.print("[bold yellow]No valid report found. Regenerating report from scratch...[/]")

        report_title = await generate_title(llm, state['query'])
        console.print(f"[bold green]Generated title: {report_title}[/]")

        extracted_themes = await extract_themes(llm, state['findings'])

        initial_report = await generate_initial_report(
            llm,
            state['query'],
            state['findings'],
            state.get('retrieved_passages', ''),
            extracted_themes,
            report_title,
            state['selected_sources'],
            state.get('formatted_citations', ''),
            state['current_date'],
            state['detail_level'],
            False, # Don't include objective in fallback
            state.get('citation_registry') # Use existing citation registry if available
        )
        
        # Store the initial report
        state["initial_report"] = initial_report
        
        # Skip enhancement and expansion steps to maintain consistent report structure
        enhanced_report = initial_report
        state["enhanced_report"] = enhanced_report
        
        # Use the initial report directly as the final report
        final_report = initial_report

        used_source_urls = []
        for analysis in state["content_analysis"]:
            if "sources" in analysis and isinstance(analysis["sources"], list):
                for url in analysis["sources"]:
                    if url not in used_source_urls:
                        used_source_urls.append(url)
        
        # If we don't have enough used sources, also grab from selected_sources
        if len(used_source_urls) < 5 and "selected_sources" in state:
            for url in state["selected_sources"]:
                if url not in used_source_urls:
                    used_source_urls.append(url)
                    if len(used_source_urls) >= 15:
                        break

        sources_info = []
        for url in used_source_urls[:20]:  # Limit to 20 sources
            source_meta = next((s for s in state["sources"] if s.get("url") == url), {})
            sources_info.append({
                "url": url,
                "title": source_meta.get("title", ""),
                "snippet": source_meta.get("snippet", "")
            })

    # Apply comprehensive cleanup of artifacts and unwanted sections
    final_report = re.sub(r'Completed:.*?\n', '', final_report)
    final_report = re.sub(r'Here are.*?(search queries|queries to investigate).*?\n', '', final_report)
    final_report = re.sub(r'Generated search queries:.*?\n', '', final_report)
    final_report = re.sub(r'\*Generated on:.*?\*', '', final_report)
    
    # Remove "Refined Research Query" section which sometimes appears at the beginning
    final_report = re.sub(r'#\s*Refined Research Query:.*?(?=\n#|\Z)', '', final_report, flags=re.DOTALL)
    final_report = re.sub(r'Refined Research Query:.*?(?=\n\n)', '', final_report, flags=re.DOTALL)
    
    # Remove entire Research Framework sections (from start to first actual content section)
    if "Research Framework:" in final_report or "# Research Framework:" in final_report:

        framework_matches = re.search(r'(?:^|\n)(?:#\s*)?Research Framework:.*?(?=\n#|\n\*\*|\Z)', final_report, re.DOTALL)
        if framework_matches:
            framework_section = framework_matches.group(0)
            final_report = final_report.replace(framework_section, '')
    
    # Remove "Based on our discussion" title if it exists
    final_report = re.sub(r'^(?:#\s*)?Based on our discussion,.*?\n', '', final_report, flags=re.MULTILINE)
    
    # Also try to catch Objective sections and other framework components
    final_report = re.sub(r'^Objective:.*?\n\n', '', final_report, flags=re.MULTILINE | re.DOTALL)
    final_report = re.sub(r'^Key Aspects to Focus On:.*?\n\n', '', final_report, flags=re.MULTILINE | re.DOTALL)
    final_report = re.sub(r'^Constraints and Preferences:.*?\n\n', '', final_report, flags=re.MULTILINE | re.DOTALL)
    final_report = re.sub(r'^Areas to Explore in Depth:.*?\n\n', '', final_report, flags=re.MULTILINE | re.DOTALL)
    final_report = re.sub(r'^Preferred Sources, Perspectives, or Approaches:.*?\n\n', '', final_report, flags=re.MULTILINE | re.DOTALL)
    final_report = re.sub(r'^Scope, Boundaries, and Context:.*?\n\n', '', final_report, flags=re.MULTILINE | re.DOTALL)
    
    # Also remove any remaining individual problem framework lines
    final_report = re.sub(r'^Research Framework:.*?\n', '', final_report, flags=re.MULTILINE)
    final_report = re.sub(r'^Key Findings:.*?\n', '', final_report, flags=re.MULTILINE)
    final_report = re.sub(r'^Key aspects to focus on:.*?\n', '', final_report, flags=re.MULTILINE)

    report_title = await generate_title(llm, state['query'])
    
    # Remove the query or any long text description from the beginning of the report if present
    # This pattern removes lines that look like full query pasted as title or at the beginning
    if final_report.strip().startswith('# '):
        lines = final_report.split('\n')
        
        # Remove any extremely long title lines (likely a full query pasted as title)
        if len(lines) > 0 and len(lines[0]) > 80 and lines[0].startswith('# '):
            lines = lines[1:]  # Remove the first line
            final_report = '\n'.join(lines)
        
        # Also look for any text block before the actual title that might be the original query
        # or refined query description
        start_idx = 0
        title_idx = -1
        
        for i, line in enumerate(lines):
            if line.startswith('# ') and i > 0 and len(line) < 100:
                # Found what appears to be the actual title
                title_idx = i
                break
        
        # If we found a title after some text, remove everything before it
        if title_idx > 0:
            lines = lines[title_idx:]
            final_report = '\n'.join(lines)

    title_match = re.match(r'^#\s+.*?\n', final_report)
    if title_match:
        # Replace existing title with our generated one
        final_report = re.sub(r'^#\s+.*?\n', f'# {report_title}\n', final_report, count=1)
    else:

        final_report = f'# {report_title}\n\n{final_report}'
        
    # Also check for second line being the full query, which happens sometimes
    lines = final_report.split('\n')
    if len(lines) > 2 and len(lines[1]) > 80 and "query" not in lines[1].lower():
        lines.pop(1)  # Remove the second line if it looks like a query
        final_report = '\n'.join(lines)

    if "References" in final_report:

        references_match = re.search(r'#+\s*References.*?(?=#+\s+|\Z)', final_report, re.DOTALL)
        if references_match:
            references_section = references_match.group(0)
            
            # Always replace the references section with our properly formatted web citations
            console.print("[yellow]Ensuring references are properly formatted as web citations...[/]")

            citation_registry = state.get("citation_registry")
            citation_manager = state.get("citation_manager")
            formatted_citations = ""
            
            if citation_manager and citation_registry:

                citation_stats = citation_manager.get_learning_statistics()
                console.print(f"[bold green]Report references {len(citation_registry.citations)} sources with {citation_stats.get('total_learnings', 0)} tracked learnings[/]")

                validation_result = citation_registry.validate_citations(final_report)
                
                if not validation_result["valid"]:

                    out_of_range_count = len(validation_result.get("out_of_range_citations", set()))
                    other_invalid_count = len(validation_result["invalid_citations"]) - out_of_range_count
                    max_valid_id = validation_result.get("max_valid_id", 0)
                    
                    console.print(f"[bold yellow]Found {len(validation_result['invalid_citations'])} invalid citations in the report[/]")
                    
                    if out_of_range_count > 0:
                        console.print(f"[bold red]Found {out_of_range_count} out-of-range citations (exceeding max valid ID: {max_valid_id})[/]")
                    
                    # Remove invalid citations from the report
                    for invalid_cid in validation_result["invalid_citations"]:
                        # For out-of-range citations, replace with valid range indicator
                        if invalid_cid in validation_result.get("out_of_range_citations", set()):
                            replacement = f'[1-{max_valid_id}]'  # Suggest valid range
                            final_report = re.sub(f'\\[{invalid_cid}\\]', replacement, final_report)
                        else:
                            # Replace other invalid patterns like [invalid_cid] with [?]
                            final_report = re.sub(f'\\[{invalid_cid}\\]', '[?]', final_report)

                used_citations = validation_result["used_citations"]
                
                # If we have a citation manager, use its enhanced formatting
                if citation_manager and used_citations:

                    processed_text, bibliography_entries = citation_manager.get_citations_for_report(final_report)
                    
                    # Use the citation manager's bibliography formatter with APA style
                    if bibliography_entries:
                        formatted_citations = citation_manager.format_bibliography(bibliography_entries, "apa")
                        console.print(f"[bold green]Generated enhanced bibliography with {len(bibliography_entries)} entries[/]")
                # Fall back to regular citation formatting
                elif used_citations:

                    formatted_citations = await format_citations(
                        llm, 
                        state.get('selected_sources', []), 
                        state["sources"],
                        citation_registry
                    )
            
            # Replace references section with properly formatted ones
            if formatted_citations:
                new_references = f"# References\n\n{formatted_citations}\n"
                final_report = final_report.replace(references_section, new_references)
            elif state.get("formatted_citations"):
                new_references = f"# References\n\n{state['formatted_citations']}\n"
                final_report = final_report.replace(references_section, new_references)
            else:

                basic_references = []
                for i, url in enumerate(state.get("selected_sources", []), 1):
                    source_meta = next((s for s in state["sources"] if s.get("url") == url), {})
                    title = source_meta.get("title", "Untitled")
                    domain = url.split("//")[1].split("/")[0] if "//" in url else "Unknown Source"
                    date = source_meta.get("date", "n.d.")
                    
                    # Simpler citation format without the date
                    citation = f"[{i}] *{domain}*, \"{title}\", {url}"
                    basic_references.append(citation)
                
                new_references = f"# References\n\n" + "\n".join(basic_references) + "\n"
                final_report = final_report.replace(references_section, new_references)

    elapsed_time = time.time() - state["start_time"]
    minutes, seconds = divmod(int(elapsed_time), 60)

    state["messages"].append(AIMessage(content="Research complete. Generating final report..."))
    state["findings"] = final_report
    state["status"] = "Complete"

    if "citation_manager" in state:
        citation_stats = state["citation_manager"].get_learning_statistics()
        log_chain_of_thought(
            state, 
            f"Generated final report after {minutes}m {seconds}s with {citation_stats.get('total_sources', 0)} sources and {citation_stats.get('total_learnings', 0)} tracked learnings"
        )
    else:
        log_chain_of_thought(state, f"Generated final report after {minutes}m {seconds}s")
    
    if progress_callback:
        await _call_progress_callback(progress_callback, state)
    return state
