"""
Research module for Shandu deep research system.
Handles research orchestration and result management with improved reporting.
"""
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import os

@dataclass
class ResearchResult:
    """Container for research results with enhanced reporting capabilities."""
    query: str
    summary: str
    sources: List[Dict[str, Any]]
    subqueries: List[str]
    depth: int
    content_analysis: Optional[List[Dict[str, Any]]] = None
    chain_of_thought: Optional[List[str]] = None
    research_stats: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_markdown(self, include_chain_of_thought: bool = False, include_objective: bool = False) -> str:
        """
        Convert research results to markdown format with improved formatting.
        
        Args:
            include_chain_of_thought: Whether to include the chain of thought in the report
            include_objective: Whether to include the objective section in the report
            
        Returns:
            Markdown formatted research report
        """
        # Format timestamp
        timestamp_str = self.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        
        stats = self.research_stats or {}
        elapsed_time = stats.get("elapsed_time_formatted", "Unknown")
        sources_count = stats.get("sources_count", len(self.sources))
        subqueries_count = stats.get("subqueries_count", len(self.subqueries))
    
        source_refs = {}
        source_list = []
        
        # Deduplicate sources and assign numbers
        for i, source in enumerate(self.sources, 1):
            url = source.get('url', '')
            if url and url not in source_refs:
                source_refs[url] = i
                source['ref_id'] = i
                source_list.append(source)
        
        # Process the summary to remove formatting issues
        summary = self.summary
        
        # Fix the "Research Report: **Objective:**" formatting issue
        if summary.startswith("# Research Report: **Objective:**"):
            summary = summary.replace("# Research Report: **Objective:**", "# Research Report")
        
        # Remove objective section if not requested
        if not include_objective and "**Objective:**" in summary:
            # Split by sections
            parts = summary.split("## ")
            filtered_parts = []
            
            # Process each section
            for part in parts:
                # Keep executive summary or empty parts
                if part.startswith("Executive Summary") or not part.strip():
                    filtered_parts.append(part)
                    continue
                
                # Skip objective section
                if "**Objective:**" in part and "**Key Aspects to Focus On:**" in part:
                    continue
                
                # Keep other sections
                filtered_parts.append(part)
            
            # Reconstruct the summary
            if filtered_parts:
                if not filtered_parts[0].startswith("Executive Summary"):
                    summary = "## ".join(filtered_parts)
                else:
                    summary = filtered_parts[0] + "## " + "## ".join(filtered_parts[1:])
        
        md = [
            f"# Research Report: {self.query}\n",
            f"*Generated on: {timestamp_str}*\n",
            f"{summary}\n"
        ]
        
        md.append("## References\n")
        for source in source_list:
            ref_id = source.get('ref_id', '')
            title = source.get('title', 'Untitled Source')
            url = source.get('url', '')
            source_type = source.get('source', 'Unknown')
            date = source.get('date', '')
            
            # Improved reference formatting with markdown
            ref_text = f"**[{ref_id}]** "
            
            # Format title with emphasis if available
            if title:
                ref_text += f"*{title}*"
            
            # Format the reference details in a clean, consistent way
            details = []
            if source_type and source_type != "Unknown":
                details.append(f"**Source**: {source_type}")
            if date:
                details.append(f"**Date**: {date}")
            
            # Add details if we have any
            if details:
                ref_text += " - " + ", ".join(details)
            
            # Add URL as a proper markdown link if available
            if url:
                ref_text += f"\n  <{url}>"
            
            md.append(ref_text)
            md.append("") # Add blank line between references for better readability
        md.append("")
        
        # Add concise research process stats
        md.append("## Research Process\n")
        md.append(f"- **Depth**: {self.depth}")
        md.append(f"- **Breadth**: {stats.get('breadth', 'Not specified')}")
        md.append(f"- **Time Taken**: {elapsed_time}")
        md.append(f"- **Subqueries Explored**: {subqueries_count}")
        md.append(f"- **Sources Analyzed**: {sources_count}\n")
        
        if include_chain_of_thought and self.chain_of_thought:
            md.append("## Research Process: Chain of Thought\n")
            significant_thoughts = []
            
            for thought in self.chain_of_thought:
                # Skip generic or repetitive thoughts
                if any(x in thought.lower() for x in ["searching for", "selected relevant url", "completed"]):
                    continue
                significant_thoughts.append(thought)
            
            if len(significant_thoughts) > 20:
                selected_thoughts = (
                    significant_thoughts[:5] + 
                    significant_thoughts[len(significant_thoughts)//2-2:len(significant_thoughts)//2+3] + 
                    significant_thoughts[-5:]
                )
            else:
                selected_thoughts = significant_thoughts
                
            for thought in selected_thoughts:
                md.append(f"- {thought}")
            md.append("")
        
        return "\n".join(md)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format with all fields."""
        return {
            "query": self.query,
            "summary": self.summary,
            "sources": self.sources,
            "subqueries": self.subqueries,
            "depth": self.depth,
            "content_analysis": self.content_analysis,
            "chain_of_thought": self.chain_of_thought,
            "research_stats": self.research_stats,
            "timestamp": self.timestamp.isoformat()
        }
    
    def save_to_file(self, filepath: str, include_chain_of_thought: bool = False, include_objective: bool = False) -> None:
        """
        Save research results to a file.
        
        Args:
            filepath: Path to save the file
            include_chain_of_thought: Whether to include chain of thought in markdown
            include_objective: Whether to include the objective section in the report
        """
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        _, ext = os.path.splitext(filepath)
        ext = ext.lower()
        
        if ext == '.md':
            # Save as markdown
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(self.to_markdown(include_chain_of_thought, include_objective))
        elif ext == '.json':
            # Save as JSON
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, indent=2, default=str)
        else:
            # Default to markdown
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(self.to_markdown(include_chain_of_thought, include_objective))
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResearchResult':
        """
        Create a ResearchResult from a dictionary.
        
        Args:
            data: Dictionary with research result data
            
        Returns:
            ResearchResult object
        """
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
            
        return cls(**data)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'ResearchResult':
        """
        Load research results from a file.
        
        Args:
            filepath: Path to the file (must be JSON)
            
        Returns:
            ResearchResult object
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return cls.from_dict(data)

class DeepResearcher:
    """
    High-level research orchestrator with multiple research strategies.
    """
    def __init__(
        self,
        output_dir: Optional[str] = None,
        save_results: bool = True,
        auto_save_interval: Optional[int] = None
    ):
        """
        Initialize the researcher.
        
        Args:
            output_dir: Directory to save research results
            save_results: Whether to automatically save results
            auto_save_interval: Interval in seconds for auto-saving (None to disable)
        """
        self.output_dir = output_dir or os.path.expanduser("~/shandu_research")
        self.save_results = save_results
        self.auto_save_interval = auto_save_interval
        
        if self.save_results:
            os.makedirs(self.output_dir, exist_ok=True)
    
    def get_output_path(self, query: str, format: str = 'md') -> str:
        """
        Get output path for research results.
        
        Args:
            query: Research query
            format: File format ('md' or 'json')
            
        Returns:
            File path
        """
        sanitized = "".join(c if c.isalnum() or c in " -_" else "_" for c in query)
        sanitized = sanitized[:50]
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{sanitized}_{timestamp}.{format}"
        
        return os.path.join(self.output_dir, filename)
    
    async def research(
        self, 
        query: str,
        strategy: str = 'langgraph',
        **kwargs
    ) -> ResearchResult:
        """
        Perform research using the specified strategy.
        
        Args:
            query: Research query
            strategy: Research strategy ('langgraph' or 'agent')
            **kwargs: Additional arguments for the research strategy
            
        Returns:
            ResearchResult object
        """
        from ..agents.langgraph_agent import ResearchGraph
        from ..agents.agent import ResearchAgent
        
        result = None
        
        if strategy == 'langgraph':
            graph = ResearchGraph()
            result = await graph.research(query, **kwargs)
        elif strategy == 'agent':
            agent = ResearchAgent()
            result = await agent.research(query, **kwargs)
        else:
            raise ValueError(f"Unknown research strategy: {strategy}")
        
        if self.save_results and result:
            md_path = self.get_output_path(query, 'md')
            result.save_to_file(md_path)
            
            json_path = self.get_output_path(query, 'json')
            result.save_to_file(json_path)
        
        return result
    
    def research_sync(
        self, 
        query: str,
        strategy: str = 'langgraph',
        **kwargs
    ) -> ResearchResult:
        """Synchronous version of research method."""
        import asyncio
        return asyncio.run(self.research(query, strategy, **kwargs))
