"""Report generation utilities with structured output."""
import os
from typing import List, Dict, Optional, Any, Union
import re
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from ..utils.citation_registry import CitationRegistry

# Structured output models
class ReportTitle(BaseModel):
    """Structured output for report title generation."""
    title: str = Field(description="A concise, professional title for the research report")
    
class ThemeExtraction(BaseModel):
    """Structured output for theme extraction."""
    themes: List[str] = Field(description="List of key themes identified in the research")
    descriptions: List[str] = Field(description="Brief descriptions of each theme")
    
class InitialReport(BaseModel):
    """Structured output for initial report generation."""
    executive_summary: str = Field(description="Executive summary of the research findings")
    introduction: str = Field(description="Introduction to the research topic")
    sections: List[Dict[str, str]] = Field(description="List of report sections with titles and content")
    conclusion: str = Field(description="Conclusion summarizing the key findings")
    
class EnhancedReport(BaseModel):
    """Structured output for enhanced report generation."""
    enhanced_sections: List[Dict[str, str]] = Field(description="Enhanced sections with additional details")
    
class ExpandedSection(BaseModel):
    """Structured output for section expansion."""
    section_title: str = Field(description="Title of the section being expanded")
    expanded_content: str = Field(description="Expanded content for the section")

async def generate_title(llm: ChatOpenAI, query: str) -> str:
    """Generate a professional title for the report using structured output."""
    # Use structured output for title generation
    structured_llm = llm.with_structured_output(ReportTitle, method="function_calling")
    
    # Use a direct system prompt without template variables
    system_prompt = """You are creating a professional, concise title for a research report.
    
    CRITICAL REQUIREMENTS - YOUR TITLE MUST:
    1. Be EXTREMELY CONCISE (8 words maximum)
    2. Be DESCRIPTIVE of the main topic
    3. Be PROFESSIONAL in tone
    4. NEVER start with words like "Evaluating", "Analyzing", "Assessment", "Study", "Investigation", etc.
    5. NEVER contain generic phrases like "A Comprehensive Overview", "An In-depth Look", etc.
    6. NEVER use the word "report", "research", "analysis", or similar meta-terms
    7. NEVER include prefixes like "Topic:", "Subject:", etc.
    8. NEVER be in question format
    9. NEVER be in sentence format - should be a noun phrase
    
    EXAMPLES OF GOOD TITLES:
    "NVIDIA RTX 5000 Series Gaming Performance"
    "Quantum Computing Applications in Cryptography"
    "Clean Energy Transition: Global Market Trends"
    """

    user_message = f"Create a professional, concise title (8 words max) for research about: {query}"
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_message)
    ])
    
    try:
        # Direct non-structured approach to avoid errors
        direct_prompt = f"""Create a professional, concise title (8 words max) for this research topic: {query}
        
        The title should be:
        - Extremely concise (8 words max)
        - Descriptive of the main topic
        - Professional in tone
        - NOT use words like "Evaluating", "Analyzing", "Assessment", "Report", etc.
        - NOT use generic phrases like "A Comprehensive Overview"
        - Be a noun phrase, not a question or sentence
        
        Return ONLY the title, nothing else. No quotation marks, no explanations.
        """
        
        # Use a direct, simplified approach
        simple_llm = llm.with_config({"temperature": 0.2})
        response = await simple_llm.ainvoke(direct_prompt)

        clean_title = response.content.replace("Title:", "").replace("\"", "").strip()
        return clean_title
    except Exception as e:
        print(f"Error in structured title generation: {str(e)}. Using simpler approach.")

        from ...utils.logger import log_error
        log_error("Error in structured title generation", e, 
                 context=f"Query: {query}, Function: generate_title")
        # Fallback to non-structured approach
        simple_prompt = ChatPromptTemplate.from_messages([
            ("system", "Create a professional, concise title (8 words max) for a research report."),
            ("user", f"Topic: {query}")
        ])
        
        simple_llm = llm.with_config({"temperature": 0.2})
        simple_chain = simple_prompt | simple_llm
        title = await simple_chain.ainvoke({})

        clean_title = title.content.replace("Title: ", "").replace("\"", "").strip()
        
        return clean_title

async def extract_themes(llm: ChatOpenAI, findings: str) -> str:
    """Extract key themes from research findings using structured output."""
    # Use structured output for theme extraction
    structured_llm = llm.with_structured_output(ThemeExtraction, method="function_calling")
    
    # Use a direct system prompt without template variables
    system_prompt = """You are analyzing research findings to identify key themes for a report structure.
    Extract 4-7 major themes (dont mention word Theme)from the content that would make logical report sections.
    These themes should emerge naturally from the content rather than following a predetermined structure.
    For each theme, provide a brief description of what content would be included."""

    user_message = f"Analyze these research findings and extract 4-7 key themes that should be used as main sections in a report:\n\n{findings}"
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_message)
    ])
    
    try:
        # Direct non-structured approach to avoid errors
        direct_prompt = f"""Extract 4-7 key themes from these research findings that would make logical report sections.
        
        Research findings:
        {findings[:3000]}
        
        Format your response as:
        
        ## Theme 1 (dont mention word Theme)
        Description of theme 1
        
        ## Theme 2 (dont mention word Theme)
        Description of theme 2
        
        And so on. Each theme should have a clear, concise title and a brief description.
        
        Do not include any other text or explanations outside of the themes and descriptions.
        """
        
        # Use direct approach
        simple_llm = llm.with_config({"temperature": 0.3})
        response = await simple_llm.ainvoke(direct_prompt)

        return response.content
    except Exception as e:
        print(f"Error in structured theme extraction: {str(e)}. Using simpler approach.")

        from ...utils.logger import log_error
        log_error("Error in structured theme extraction", e, 
                 context=f"Findings length: {len(findings)}, Function: extract_themes")
        
        # Fallback to non-structured approach
        simple_prompt = ChatPromptTemplate.from_messages([
            ("system", """Extract 4-7 key themes from research findings that would make logical report sections.
            Format your response as:
            
            ## Theme 1 (dont mention word Theme)
            Description of theme 1
            
            ## Topic 2 (dont mention word Topic)
            Description of topic 2
            
            And so on."""),
            ("user", f"Extract topics from these findings:\n\n{findings[:10000]}")
        ])
        
        simple_llm = llm.with_config({"temperature": 0.3})
        simple_chain = simple_prompt | simple_llm
        themes = await simple_chain.ainvoke({})
        
        return themes.content

async def format_citations(
    llm: ChatOpenAI, 
    selected_sources: List[str], 
    sources: List[Dict[str, Any]], 
    citation_registry: Optional[CitationRegistry] = None
) -> str:
    """
    Format citations for selected sources.
    
    Args:
        llm: The language model to use
        selected_sources: List of source URLs to format as citations
        sources: List of source metadata dictionaries
        citation_registry: Optional CitationRegistry instance to use for citation tracking
        
    Returns:
        String of formatted citations
    """
    if not selected_sources:
        return ""
    
    # If we have a citation registry, use it to get all properly cited sources
    if citation_registry:

        all_citations = citation_registry.get_all_citations()
        
        # Only format citations that were actually used
        # Sort by citation ID to ensure consistent ordering
        citations = []
        for cid in sorted(all_citations.keys()):
            citation_info = all_citations[cid]
            url = citation_info.get("url")

            source_meta = next((s for s in sources if s.get("url") == url), {})

            domain = url.split("//")[1].split("/")[0] if "//" in url else "Unknown Source"
            title = source_meta.get("title", citation_info.get("title", "Untitled"))
            date = source_meta.get("date", citation_info.get("date", "n.d."))
            
            citation = f"[{cid}] *{domain}*, \"{title}\", {url}"
            citations.append(citation)
        
        if citations:
            return "\n".join(citations)

    sources_text = ""
    for i, url in enumerate(selected_sources, 1):

        source_meta = {}
        for source in sources:
            if source.get("url") == url:
                source_meta = source
                break

        sources_text += f"Source {i}:\nURL: {url}\n"
        if source_meta.get("title"):
            sources_text += f"Title: {source_meta.get('title')}\n"
        if source_meta.get("source"):
            sources_text += f"Publication: {source_meta.get('source')}\n"
        if source_meta.get("date"):
            sources_text += f"Date: {source_meta.get('date')}\n"
        sources_text += "\n"
    
    # Use an enhanced citation formatter prompt with direct system message
    system_prompt = """Format the following source information into properly numbered citations for a research report.
    
    FORMATTING REQUIREMENTS:
    1. Each citation MUST start with [n] where n is the citation number
    2. Each citation MUST include the following elements (when available):
       - Website domain name in italics
       - Title of the article/page in quotes
       - Publication date
       - Complete URL
    
    EXAMPLE PROPER CITATIONS:
    [1] *techreview.com*, "Advances in GPU Architecture", 2024-01-15, https://techreview.com/articles/gpu-advances
    [2] *arxiv.org*, "Neural Network Performance Optimization", 2023-11-30, https://arxiv.org/papers/nn-optimization
    
    MISSING INFORMATION:
    - If website domain is missing, extract it from the URL
    - If title is missing, use "Untitled"
    - If date is missing, use "n.d." (no date)
    
    IMPORTANT: DO NOT generate citations from your training data. ONLY use the provided source information.
    DO NOT create academic-style citations like "Journal of Medicine (2020)". 
    ONLY create web citations with the exact format shown in the examples.
    
    FORMAT ALL CITATIONS IN A CONSISTENT STYLE.
    Number citations sequentially starting from [1].
    Place each citation on a new line.
    """

    user_message = f"Format these sources into proper citations:\n\n{sources_text}"
    
    citation_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_message)
    ])
    
    formatter_chain = citation_prompt | llm
    formatted_citations = (await formatter_chain.ainvoke({})).content
    
    # Verify citations have proper format [n] at the beginning
    if not re.search(r'\[\d+\]', formatted_citations):

        citations = []
        for i, url in enumerate(selected_sources, 1):
            source_meta = next((s for s in sources if s.get("url") == url), {})
            title = source_meta.get("title", "Untitled")
            domain = url.split("//")[1].split("/")[0] if "//" in url else "Unknown Source"
            date = source_meta.get("date", "n.d.")
            
            citation = f"[{i}] *{domain}*, \"{title}\", {date}, {url}"
            citations.append(citation)
        
        formatted_citations = "\n".join(citations)
    
    return formatted_citations

async def generate_initial_report(
    llm: ChatOpenAI,
    query: str,
    findings: str,
    retrieved_passages: str,
    extracted_themes: str,
    report_title: str,
    selected_sources: List[str],
    formatted_citations: str,
    current_date: str,
    detail_level: str,
    include_objective: bool,
    citation_registry: Optional[CitationRegistry] = None
) -> str:
    """Generate the initial report draft using structured output.

    retrieved_passages provides additional context retrieved from the vector store.
    """
    # Use structured output for initial report generation
    structured_llm = llm.with_structured_output(InitialReport, method="function_calling")

    objective_instruction = ""
    if not include_objective:
        objective_instruction = "\n\nIMPORTANT: DO NOT include an \"Objective\" section at the beginning of the report. Let your content and analysis naturally determine the structure."

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
    
    # Use a direct system prompt without template variables
    system_prompt = f"""You are generating a comprehensive research report based on extensive research.
    
    REPORT REQUIREMENTS:
    - The report should be thorough, detailed, and professionally formatted in Markdown.
    - Include headers, subheaders, and formatting for readability.
    - The level of detail should be {detail_level.upper()}.
    - Base the report ENTIRELY on the provided research findings.
    - As of {current_date}, incorporate the most up-to-date information available.
    - Create a dynamic structure based on the content themes rather than a rigid template.{objective_instruction}
    
    CITATION REQUIREMENTS:
    - ONLY use the citation IDs provided in the AVAILABLE SOURCES list
    - Format citations as [n] where n is the exact ID of the source
    - Place citations at the end of the relevant sentences or paragraphs
    - Do not make up your own citation numbers
    - Do not cite sources that aren't in the available sources list
    - Ensure each major claim or statistic has an appropriate citation
    """

    user_message = f"""Create an extensive, in-depth research report on this topic.

Title: {report_title}
Analyzed Findings: {augmented_findings[:5000]}
Number of sources: {len(selected_sources)}
Key themes identified in the research: 
{extracted_themes}{available_sources_text}

Organize your report around these key themes that naturally emerged from the research.
Create a dynamic, organic structure that best presents the findings, rather than forcing content into predetermined sections.
Ensure comprehensive coverage while maintaining a logical flow between topics.

Your report must be extensive, detailed, and grounded in the research. Include all relevant data, examples, and insights found in the research.
Use proper citations to the sources throughout, referring only to the available sources listed above.

IMPORTANT: Begin your report with the exact title provided: "{report_title}" - do not modify or rephrase it."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_message)
    ])

    sources_text = "\n\nSOURCES ANALYZED IN DETAIL:\n"
    if formatted_citations:
        sources_text += formatted_citations
    else:
        for i, url in enumerate(selected_sources, 1):
            sources_text += f"{i}. {url}\n"
    
    # Augment findings with selected source information
    retrieval_section = ""
    if retrieved_passages:
        retrieval_section = "\n\nRETRIEVED CONTEXT:\n" + retrieved_passages
    augmented_findings = findings + retrieval_section + sources_text

    try:
        # Direct non-structured approach to avoid errors
        # Direct non-structured approach to avoid errors
        direct_prompt = f"""Create an extremely comprehensive, detailed research report that is AT LEAST 5,000 words long.

Title: {report_title}
Analyzed Findings: {augmented_findings[:5000]}
Number of sources: {len(selected_sources)}
Key themes identified in the research: 
{extracted_themes}{available_sources_text}

REPORT REQUIREMENTS:
- The report should be thorough, detailed, and professionally formatted in Markdown.
- Include headers, subheaders, and formatting for readability.
- The level of detail should be {detail_level.upper()}.
- Base the report ENTIRELY on the provided research findings.
- As of {current_date}, incorporate the most up-to-date information available.
- Create a dynamic structure based on the content themes rather than a rigid template.
- CRITICALLY IMPORTANT: DO NOT include the original query text at the beginning of the report. Start directly with the title.

CITATION REQUIREMENTS:
- ONLY use the citation IDs provided in the AVAILABLE SOURCES list
- Format citations as [n] where n is the exact ID of the source
- Place citations at the end of the relevant sentences or paragraphs
- Do not make up your own citation numbers
- Do not cite sources that aren't in the available sources list
- Ensure each major claim or statistic has an appropriate citation

FORMAT (IMPORTANT):
- Always use full power of markdown (eg. tables for comparasions, links, citations, etc.)
- Start with the title as a level 1 heading: "# {report_title}"
- Include executive summary
- Include an introduction
- Include sections based on the key themes
- Include a conclusion
- Include references section
"""
        # Use direct approach with maximum token limit for very long reports
        report_llm = llm.with_config({"max_tokens": 32768, "temperature": 0.6})
        response = await report_llm.ainvoke(direct_prompt)
        
        return response.content
        
    except Exception as e:
        print(f"Error in structured report generation: {str(e)}. Using simpler approach.")

        from ...utils.logger import log_error
        log_error("Error in structured report generation", e, 
                 context=f"Query: {query}, Report title: {report_title}, Function: generate_initial_report")
        # Fallback to simpler report generation without structured output
        simple_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""Generate a comprehensive research report based on the provided findings.
            The report should be well-structured with clear sections and proper citations.
            Current date: {current_date}"""),
            ("user", f"Title: {report_title}\n\nFindings: {augmented_findings[:10000]}")
        ])
        
        simple_llm = llm.with_config({"max_tokens": 16000, "temperature": 0.6})
        simple_chain = simple_prompt | simple_llm
        report = await simple_chain.ainvoke({})
        
        return report.content

async def enhance_report(
    llm: ChatOpenAI, 
    initial_report: str, 
    current_date: str, 
    formatted_citations: str, 
    selected_sources: List, 
    sources: List[Dict],
    citation_registry: Optional[CitationRegistry] = None
) -> str:
    """Enhance the report with additional detail while preserving structure."""

    if not initial_report or len(initial_report.strip()) < 500:
        return initial_report

    title_match = re.match(r'# ([^\n]+)', initial_report)
    original_title = title_match.group(1) if title_match else "xxx"

    section_pattern = re.compile(r'(#+\s+[^\n]+)(\n\n[^#]+?)(?=\n#+\s+|\Z)', re.DOTALL)
    sections = section_pattern.findall(initial_report)
    
    if not sections:
        # If no sections found, return the initial report
        return initial_report

    enhanced_report = f"# {original_title}\n\n"
    
    for section_header, section_content in sections:
        # Skip enhancing references section
        if "References" in section_header:
            enhanced_report += f"{section_header}{section_content}\n\n"
            continue

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
        
        try:
            # Use a lower token limit for each section to avoid issues
            enhance_llm = llm.with_config({"max_tokens": 4096, "temperature": 0.2})
            response = await enhance_llm.ainvoke(section_prompt)

            section_text = response.content
            section_text = re.sub(r'\[/[^\]]*\]', '', section_text)  # Remove any malformed closing tags

            if not section_text.strip().startswith(section_header.strip()):
                section_text = f"{section_header}\n\n{section_text}"
                
            enhanced_report += f"{section_text}\n\n"
            
        except Exception as e:
            # If enhancement fails for a section, use the original
            print(f"Error enhancing section '{section_header.strip()}': {str(e)}")
            enhanced_report += f"{section_header}{section_content}\n\n"
    
    return enhanced_report

async def expand_key_sections(
    llm: ChatOpenAI, 
    report: str, 
    identified_themes: str, 
    current_date: str,
    citation_registry: Optional[CitationRegistry] = None
) -> str:
    """Expand key sections of the report while preserving structure and avoiding markup errors."""
    # Make sure we have a properly formatted report to start with
    if not report or len(report.strip()) < 1000:
        return report

    report = re.sub(r'\[\/?(?:PDF|Text|ImageB|ImageC|ImageI)(?:\/?|\])(?:[^\]]*\])?', '', report)

    section_pattern = re.compile(r'(## [^\n]+)(\n\n[^#]+?)(?=\n##|\Z)', re.DOTALL)
    sections = section_pattern.findall(report)
    
    if not sections:
        return report
    
    # Select 2-3 most important sections to expand
    important_sections = []
    for section_header, section_content in sections:
        title = section_header.replace('#', '').strip()
        # Skip executive summary, introduction, conclusion, and references
        if title.lower() in ["executive summary", "introduction", "conclusion", "references"]:
            continue
        important_sections.append((section_header, section_content))
    
    # Limit to 3 sections maximum
    important_sections = important_sections[:3]
    
    if not important_sections:
        return report
    
    # Expand each section
    expanded_report = report
    for section_header, section_content in important_sections:
        title = section_header.replace('#', '').strip()

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
        
        try:
            # Use a reasonable token limit for each section
            expand_llm = llm.with_config({"max_tokens": 6144, "temperature": 0.2})
            response = await expand_llm.ainvoke(section_prompt)

            expanded_content = response.content
            # More thorough cleanup of any PDF/markup tags
            expanded_content = re.sub(r'\[\/?(?:PDF|Text|ImageB|ImageC|ImageI)(?:\/?|\])(?:[^\]]*\])?', '', expanded_content)

            if not expanded_content.strip().startswith(section_header.strip()):
                expanded_content = f"{section_header}\n\n{expanded_content}"
            
            # Replace the original section with the expanded one
            # Use a more precise pattern to avoid partial replacements
            pattern = re.compile(f"{re.escape(section_header)}{re.escape(section_content)}", re.DOTALL)
            expanded_report = pattern.sub(expanded_content, expanded_report)
            
        except Exception as e:
            print(f"Error expanding section '{title}': {str(e)}")
            # Continue with other sections if one fails
    
    return expanded_report
