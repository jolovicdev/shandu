"""
CLI interface for Shandu deep research system.
Provides commands for configuration and research with rich display.
"""
import os
import sys
import json
import asyncio
import time
from typing import Optional, Dict, Any, List
from datetime import datetime
import click
from rich.console import Console
from rich.panel import Panel
from rich.tree import Tree
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.markdown import Markdown
from rich.live import Live
from rich.table import Table
from rich.layout import Layout
from rich.syntax import Syntax
from langchain_openai import ChatOpenAI
from .config import config
from .agents.langgraph_agent import clarify_query, display_research_progress
from .agents.langgraph_agent import ResearchGraph, AgentState
from .search.search import UnifiedSearcher
from .search.ai_search import AISearcher
from .scraper import WebScraper
from .research.researcher import DeepResearcher

# Initialize console
console = Console()

def display_banner():
    """Display the Shandu banner."""
    banner = """
    ███████╗██╗  ██╗ █████╗ ███╗   ██╗██████╗ ██╗   ██╗
    ██╔════╝██║  ██║██╔══██╗████╗  ██║██╔══██╗██║   ██║
    ███████╗███████║███████║██╔██╗ ██║██║  ██║██║   ██║
    ╚════██║██╔══██║██╔══██║██║╚██╗██║██║  ██║██║   ██║
    ███████║██║  ██║██║  ██║██║ ╚████║██████╔╝╚██████╔╝
    ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═════╝  ╚═════╝ 
                Deep Research System
    """
    console.print(Panel(banner, style="bold blue"))

def create_research_dashboard(state: AgentState) -> Layout:
    """Create a rich dashboard layout for research progress."""
    layout = Layout()
    
    layout.split(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=3)
    )
    layout["body"].split_row(
        Layout(name="left", ratio=2),
        Layout(name="right", ratio=1)
    )
    layout["left"].split(
        Layout(name="status", size=3),
        Layout(name="progress", size=3),
        Layout(name="findings")
    )
    layout["right"].split(
        Layout(name="queries"),
        Layout(name="sources"),
        Layout(name="chain_of_thought")
    )
    
    elapsed_time = time.time() - state["start_time"]
    minutes, seconds = divmod(int(elapsed_time), 60)
    header_content = f"[bold blue]Research Query:[/] {state['query']}\n"
    header_content += f"[bold blue]Status:[/] {state['status']} | "
    header_content += f"[bold blue]Time:[/] {minutes}m {seconds}s | "
    header_content += f"[bold blue]Depth:[/] {state['current_depth']}/{state['depth']}"
    layout["header"].update(Panel(header_content, title="Shandu Deep Research"))
    
    status_table = Table(show_header=False, box=None)
    status_table.add_column("Metric", style="blue")
    status_table.add_column("Value")
    status_table.add_row("Current Depth", f"{state['current_depth']}/{state['depth']}")
    status_table.add_row("Sources Found", str(len(state['sources'])))
    status_table.add_row("Subqueries Explored", str(len(state['subqueries'])))
    layout["status"].update(Panel(status_table, title="Research Progress"))
    
    progress_percent = min(100, int((state['current_depth'] / max(1, state['depth'])) * 100))
    progress_bar = f"[{'#' * (progress_percent // 5)}{' ' * (20 - progress_percent // 5)}] {progress_percent}%"
    layout["progress"].update(Panel(progress_bar, title="Completion"))
    
    findings_text = state["findings"][-2000:] if state["findings"] else "No findings yet..."
    layout["findings"].update(Panel(Markdown(findings_text), title="Latest Findings"))
    queries_table = Table(show_header=True)
    queries_table.add_column("#", style="dim")
    queries_table.add_column("Query")
    for i, query in enumerate(state["subqueries"][-10:], 1):  # Show last 10 queries
        queries_table.add_row(str(i), query)
    layout["queries"].update(Panel(queries_table, title="Research Paths"))
    
    sources_table = Table(show_header=True)
    sources_table.add_column("Source", style="dim")
    sources_table.add_column("Count")
    source_counts = {}
    for source in state["sources"]:
        source_type = source.get("source", "Unknown")
        source_counts[source_type] = source_counts.get(source_type, 0) + 1
    for source_type, count in source_counts.items():
        sources_table.add_row(source_type, str(count))
    layout["sources"].update(Panel(sources_table, title="Sources"))
    
    cot_text = "\n".join(state["chain_of_thought"][-5:]) if state["chain_of_thought"] else "No thoughts recorded yet..."
    layout["chain_of_thought"].update(Panel(cot_text, title="Chain of Thought"))
    
    footer_text = "Press Ctrl+C to stop research"
    layout["footer"].update(Panel(footer_text, style="dim"))
    
    return layout

@click.group()
def cli():
    """Shandu deep research system."""
    display_banner()
    pass

@cli.command()
def configure():
    """Configure API settings."""
    console.print(Panel("Configure Shandu API Settings", style="bold blue"))
    
    # Get API settings
    api_base = click.prompt(
        "OpenAI API Base URL",
        default=config.get("api", "base_url")
    )
    api_key = click.prompt(
        "OpenAI API Key",
        default=config.get("api", "api_key"),
        hide_input=True
    )
    model = click.prompt(
        "Model Name",
        default=config.get("api", "model")
    )
    proxy = click.prompt(
        "Proxy URL (optional, press Enter to skip)",
        default="",
        show_default=False
    )
    
    user_agent = click.prompt(
        "User Agent",
        default=config.get("search", "user_agent")
    )
    
    # Save config
    config.set("api", "base_url", api_base)
    config.set("api", "api_key", api_key)
    config.set("api", "model", model)
    config.set("scraper", "proxy", proxy)
    config.set("search", "user_agent", user_agent)
    config.save()
    
    console.print(Panel("[green]Configuration saved successfully!", 
                       title="Success", 
                       border_style="green"))

@cli.command()
def info():
    """Display information about the current configuration."""
    console.print(Panel("Shandu Configuration Information", style="bold blue"))
    
    table = Table(title="Current Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("API Base URL", config.get("api", "base_url"))
    table.add_row("API Key", "****" + config.get("api", "api_key")[-4:] if config.get("api", "api_key") else "Not set")
    table.add_row("Model", config.get("api", "model"))
    
    table.add_row("Default Search Engines", ", ".join(config.get("search", "engines")))
    table.add_row("User Agent", config.get("search", "user_agent"))
    
    table.add_row("Default Depth", str(config.get("research", "default_depth")))
    table.add_row("Default Breadth", str(config.get("research", "default_breadth")))
    
    console.print(table)

@cli.command()
@click.option("--force", "-f", is_flag=True, help="Delete without confirmation")
@click.option("--cache-only", "-c", is_flag=True, help="Delete only cache files, keep configuration")
def clean(force: bool, cache_only: bool):
    """Delete configuration and cache files."""
    config_path = os.path.expanduser("~/.shandu")
    
    if not os.path.exists(config_path):
        console.print("[yellow]No configuration or cache files found.[/]")
        return
    
    if cache_only:
        cache_path = os.path.join(config_path, "cache")
        if os.path.exists(cache_path):
            if not force:
                confirm = click.confirm(f"Are you sure you want to delete all cache files in {cache_path}?")
                if not confirm:
                    console.print("[yellow]Operation cancelled.[/]")
                    return
            
            try:
                import shutil
                shutil.rmtree(cache_path)
                console.print(Panel("[green]Cache files deleted successfully!", 
                                   title="Success", 
                                   border_style="green"))
            except Exception as e:
                console.print(f"[red]Error deleting cache files: {e}[/]")
        else:
            console.print("[yellow]No cache files found.[/]")
    else:
        if not force:
            confirm = click.confirm(f"Are you sure you want to delete all configuration and cache files in {config_path}?")
            if not confirm:
                console.print("[yellow]Operation cancelled.[/]")
                return
        
        try:
            import shutil
            shutil.rmtree(config_path)
            console.print(Panel("[green]Configuration and cache files deleted successfully!", 
                               title="Success", 
                               border_style="green"))
        except Exception as e:
            console.print(f"[red]Error deleting configuration: {e}[/]")

@cli.command()
@click.argument("query")
@click.option("--depth", "-d", default=None, type=int, help="Research depth (1-5)")
@click.option("--breadth", "-b", default=None, type=int, help="Research breadth (3-10)")
@click.option("--output", "-o", help="Save report to file")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed progress")
@click.option("--strategy", "-s", default="langgraph", type=click.Choice(["langgraph", "agent"]), 
              help="Research strategy to use")
@click.option("--include-chain-of-thought", "-c", is_flag=True, help="Include chain of thought in report")
@click.option("--include-objective", "-i", is_flag=True, help="Include objective section in report")
def research(
    query: str, 
    depth: Optional[int], 
    breadth: Optional[int], 
    output: Optional[str], 
    verbose: bool,
    strategy: str,
    include_chain_of_thought: bool,
    include_objective: bool
):
    """Perform deep research on a topic."""
    if depth is None:
        depth = config.get("research", "default_depth", 2)
    if breadth is None:
        breadth = config.get("research", "default_breadth", 4)
    
    if depth < 1 or depth > 5:
        console.print("[red]Error: Depth must be between 1 and 5[/]")
        sys.exit(1)
    if breadth < 2 or breadth > 10:
        console.print("[red]Error: Breadth must be between 2 and 10[/]")
        sys.exit(1)
    
    api_base = config.get("api", "base_url")
    api_key = config.get("api", "api_key")
    model = config.get("api", "model")
    temperature = config.get("api", "temperature", 0)
    
    llm = ChatOpenAI(
        base_url=api_base,
        api_key=api_key,
        model=model,
        temperature=temperature
    )
    
    searcher = UnifiedSearcher()
    scraper = WebScraper(proxy=config.get("scraper", "proxy"))
    
    console.print(Panel(
        f"[bold blue]Query:[/] {query}\n"
        f"[bold blue]Depth:[/] {depth}\n"
        f"[bold blue]Breadth:[/] {breadth}\n"
        f"[bold blue]Strategy:[/] {strategy}\n"
        f"[bold blue]Model:[/] {model}",
        title="Research Parameters",
        border_style="blue"
    ))
    
    try:
        refined_query = asyncio.run(clarify_query(query, llm))
    except KeyboardInterrupt:
        console.print("\n[yellow]Query clarification cancelled. Using original query.[/]")
        refined_query = query
    
    # Create research graph
    if strategy == "langgraph":
        graph = ResearchGraph(llm=llm, searcher=searcher, scraper=scraper)
        
        # Create progress display
        with Live(console=console, auto_refresh=True, screen=False, transient=False) as live:
            try:
                def update_display(state):
                    if verbose:
                        dashboard = create_research_dashboard(state)
                        live.update(dashboard)
                    else:
                        tree = display_research_progress(state)
                        live.update(tree)
                
                console.print("[bold green]Starting research...[/]")
                console.print("[bold blue]This will show detailed information about the search process and pages being analyzed.[/]")
                console.print("[dim]The research process may take some time depending on depth and breadth settings.[/]")
                console.print("[dim]You'll see search queries, selected URLs, and content analysis in real-time.[/]")
                
                result = graph.research_sync(
                    refined_query,
                    depth=depth,
                    breadth=breadth,
                    progress_callback=update_display,
                    include_objective=include_objective
                )
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Research interrupted by user.[/]")
                sys.exit(0)
            except Exception as e:
                console.print(f"\n[red]Error during research: {e}[/]")
                sys.exit(1)
    else:
        # Use agent-based research
        from .agents.agent import ResearchAgent
        agent = ResearchAgent(llm=llm, searcher=searcher, scraper=scraper)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[green]Researching...", total=depth)
            
            try:
                result = agent.research_sync(
                    refined_query,
                    depth=depth,
                    engines=config.get("search", "engines")
                )
                
                progress.update(task, completed=depth)
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Research interrupted by user.[/]")
                sys.exit(0)
            except Exception as e:
                console.print(f"\n[red]Error during research: {e}[/]")
                sys.exit(1)
    
    # Display result
    console.print("\n[bold green]Research complete![/]")
    
    if output:
        result.save_to_file(output, include_chain_of_thought, include_objective)
        console.print(f"[green]Report saved to {output}[/]")
    else:
        console.print(Markdown(result.to_markdown(include_chain_of_thought, include_objective)))

@cli.command()
@click.argument("query")
@click.option("--engines", "-e", default=None, help="Comma-separated list of search engines to use")
@click.option("--max-results", "-m", default=10, type=int, help="Maximum number of results to return")
@click.option("--output", "-o", help="Save results to file")
@click.option("--detailed", "-d", is_flag=True, help="Generate a detailed analysis")
def aisearch(query: str, engines: Optional[str], max_results: int, output: Optional[str], detailed: bool):
    """Perform AI-powered search with analysis of results."""
    if engines:
        engine_list = [e.strip() for e in engines.split(",")]
    else:
        engine_list = config.get("search", "engines")
    
    api_base = config.get("api", "base_url")
    api_key = config.get("api", "api_key")
    model = config.get("api", "model")
    temperature = config.get("api", "temperature", 0)
    
    llm = ChatOpenAI(
        base_url=api_base,
        api_key=api_key,
        model=model,
        temperature=temperature
    )
    
    searcher = UnifiedSearcher(max_results=max_results)
    ai_searcher = AISearcher(llm=llm, searcher=searcher, max_results=max_results)
    
    console.print(Panel(
        f"[bold blue]Query:[/] {query}\n"
        f"[bold blue]Engines:[/] {', '.join(engine_list)}\n"
        f"[bold blue]Max Results:[/] {max_results}\n"
        f"[bold blue]Analysis:[/] {'Detailed' if detailed else 'Concise'}",
        title="AI Search Parameters",
        border_style="blue"
    ))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[green]Searching and analyzing...", total=1)
        
        try:
            result = ai_searcher.search_sync(query, engine_list, detailed)
            progress.update(task, completed=1)
            
        except Exception as e:
            console.print(f"[red]Error during AI search: {e}[/]")
            sys.exit(1)
    
    # Display result
    console.print("\n[bold green]Search and analysis complete![/]")
    
    if output:
        with open(output, 'w', encoding='utf-8') as f:
            f.write(result.to_markdown())
        console.print(f"[green]Results saved to {output}[/]")
    else:
        console.print(Markdown(result.to_markdown()))

@cli.command()
@click.argument("query")
@click.option("--engines", "-e", default=None, help="Comma-separated list of search engines to use")
@click.option("--max-results", "-m", default=10, type=int, help="Maximum number of results to return")
def search(query: str, engines: Optional[str], max_results: int):
    """Perform a quick search without deep research."""
    if engines:
        engine_list = [e.strip() for e in engines.split(",")]
    else:
        engine_list = config.get("search", "engines")
    searcher = UnifiedSearcher(max_results=max_results)
    console.print(Panel(
        f"[bold blue]Query:[/] {query}\n"
        f"[bold blue]Engines:[/] {', '.join(engine_list)}\n"
        f"[bold blue]Max Results:[/] {max_results}",
        title="Search Parameters",
        border_style="blue"
    ))
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[green]Searching...", total=1)
        
        try:
            results = searcher.search_sync(query, engine_list)
            progress.update(task, completed=1)
            
        except Exception as e:
            console.print(f"[red]Error during search: {e}[/]")
            sys.exit(1)
    
    console.print(f"\n[bold green]Found {len(results)} results:[/]")
    
    # Create table
    table = Table(title=f"Search Results for '{query}'")
    table.add_column("Source", style="cyan")
    table.add_column("Title", style="green")
    table.add_column("URL", style="blue")
    table.add_column("Snippet", style="dim")
    
    for result in results:
        table.add_row(
            result.source,
            result.title[:50] + "..." if len(result.title) > 50 else result.title,
            result.url[:50] + "..." if len(result.url) > 50 else result.url,
            result.snippet[:100] + "..." if len(result.snippet) > 100 else result.snippet
        )
    
    console.print(table)

@cli.command()
@click.argument("url")
@click.option("--dynamic", "-d", is_flag=True, help="Use dynamic rendering (for JavaScript-heavy sites)")
def scrape(url: str, dynamic: bool):
    """Scrape and analyze a webpage."""
    scraper = WebScraper(proxy=config.get("scraper", "proxy"))
    
    console.print(Panel(
        f"[bold blue]URL:[/] {url}\n"
        f"[bold blue]Dynamic Rendering:[/] {'Enabled' if dynamic else 'Disabled'}",
        title="Scrape Parameters",
        border_style="blue"
    ))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[green]Scraping...", total=1)
        
        try:
            result = asyncio.run(scraper.scrape_url(url, dynamic=dynamic))
            progress.update(task, completed=1)
            
        except Exception as e:
            console.print(f"[red]Error during scraping: {e}[/]")
            sys.exit(1)
    
    if result.is_successful():
        console.print(f"\n[bold green]Successfully scraped {url}[/]")
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body")
        )
        layout["body"].split_row(
            Layout(name="content", ratio=2),
            Layout(name="metadata", ratio=1)
        )
        
        header_content = f"[bold blue]Title:[/] {result.title}\n"
        header_content += f"[bold blue]URL:[/] {result.url}\n"
        header_content += f"[bold blue]Content Type:[/] {result.content_type}"
        layout["header"].update(Panel(header_content, title="Page Information"))
        content_preview = result.text[:2000] + "..." if len(result.text) > 2000 else result.text
        layout["content"].update(Panel(content_preview, title="Content Preview"))
        metadata_table = Table(show_header=True)
        metadata_table.add_column("Key", style="cyan")
        metadata_table.add_column("Value", style="green")
        
        for key, value in result.metadata.items():
            if value and isinstance(value, str):
                metadata_table.add_row(key, value[:50] + "..." if len(value) > 50 else value)
        
        layout["metadata"].update(Panel(metadata_table, title="Metadata"))
        
        console.print(layout)
    else:
        console.print(f"[red]Failed to scrape {url}: {result.error}[/]")

if __name__ == "__main__":
    cli()
