"""
Search module for Shandu deep research system.
Implements unified search using multiple search engines with caching and parallel processing.
"""
from typing import List, Dict, Optional, Union, Any, Tuple
from langchain_community.tools import DuckDuckGoSearchResults, DuckDuckGoSearchRun
from googlesearch import search as google_search
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import time
import hashlib
import json
import os
from pathlib import Path
import wikipedia
import arxiv
from ..config import config, get_user_agent

class SearchResult:
    """Container for search results from various engines."""
    def __init__(
        self, 
        title: str, 
        url: Union[str, None],  # Allow None but convert it
        snippet: str,
        source: str,
        date: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.title = title if title else "Untitled"
        self.url = url if isinstance(url, str) else ""  # Ensure url is a string
        self.snippet = snippet if snippet else ""
        self.source = source if source else "Unknown"
        self.date = date
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return f"SearchResult(title='{self.title}', url='{self.url}', source='{self.source}')"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source,
            "date": self.date,
            "metadata": self.metadata
        }

class SearchCache:
    """Cache for search results to improve performance."""
    def __init__(self, cache_dir: Optional[str] = None, ttl: int = 3600):
        self.cache_dir = cache_dir or os.path.expanduser("~/.shandu/cache/search")
        self.ttl = ttl  # Time to live in seconds
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _get_cache_key(self, query: str, engine: str) -> str:
        """Generate a cache key from query and engine."""
        hash_key = hashlib.md5(f"{query}:{engine}".encode()).hexdigest()
        return hash_key
    
    def _get_cache_path(self, key: str) -> str:
        """Get file path for cache key."""
        return os.path.join(self.cache_dir, f"{key}.json")
    
    def get(self, query: str, engine: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached results if available and not expired."""
        key = self._get_cache_key(query, engine)
        path = self._get_cache_path(key)
        
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                
                # Check if cache is expired
                if time.time() - data['timestamp'] <= self.ttl:
                    return data['results']
            except Exception as e:
                print(f"Error reading cache: {e}")
        
        return None
    
    def set(self, query: str, engine: str, results: List[Any]):
        """Cache search results."""
        try:
            key = self._get_cache_key(query, engine)
            path = self._get_cache_path(key)
            
            # Create serializable results
            serializable_results = []
            for r in results:
                if hasattr(r, 'to_dict'):
                    serializable_results.append(r.to_dict())
                elif isinstance(r, dict):
                    serializable_results.append(r)
                else:
                    # Skip non-serializable results
                    continue
            
            with open(path, 'w') as f:
                json.dump({
                    'timestamp': time.time(),
                    'results': serializable_results
                }, f)
                
        except Exception as e:
            print(f"Error caching search results: {e}")
            # Failures should be silent - don't impact the main functionality

class UnifiedSearcher:
    """
    Unified search interface combining results from multiple search engines.
    Supports DuckDuckGo, Google, Wikipedia, and arXiv with caching and parallel processing.
    Uses a shared ThreadPoolExecutor for better resource management.
    """
    # Class-level executor for thread pool reuse
    _executor = None
    _executor_lock = None
    
    def __init__(
        self,
        proxy: Optional[str] = None,
        max_results: int = 10,
        region: str = "wt-wt",
        safesearch: str = "moderate",
        timelimit: Optional[str] = None,
        backend: str = "news",
        cache_ttl: int = 3600,
        user_agent: Optional[str] = None
    ):
        self.max_results = max_results
        self.proxy = proxy or config.get("scraper", "proxy")
        self.user_agent = user_agent or get_user_agent()
        
        # Initialize DuckDuckGo searchers
        self.ddg_results = DuckDuckGoSearchResults(
            backend=backend,
            region=region,
            safesearch=safesearch,
            timelimit=timelimit,
            max_results=max_results
        )
        self.ddg_run = DuckDuckGoSearchRun()
        
        # Initialize cache
        self.cache = SearchCache(ttl=cache_ttl)
        
        # Use a shared executor for better resource management
        if not hasattr(UnifiedSearcher, '_executor'):
            UnifiedSearcher._executor_lock = threading.Lock()
            with UnifiedSearcher._executor_lock:
                if not hasattr(UnifiedSearcher, '_executor'):
                    UnifiedSearcher._executor = ThreadPoolExecutor(max_workers=6)
        
        self.executor = UnifiedSearcher._executor
        
        # Add request timeout to prevent hanging
        self.request_timeout = 10  # 10 second timeout for all requests

    async def _async_google_search(self, query: str, num_results: int) -> List[SearchResult]:
        """Perform Google search asynchronously with improved error handling and relevance filtering."""
        results = []
        try:
            search_query = self._humanize_query(query)
            
            cached_results = self.cache.get(search_query, "google")
            if cached_results:
                return [SearchResult(**r) for r in cached_results]
            
            headers = {'User-Agent': self.user_agent}
            
            for attempt in range(3):
                try:
                    print(f"Executing Google search for: {search_query}")
                    
                    search_results = list(google_search(
                        search_query, 
                        num_results=num_results * 2,
                        proxy=self.proxy,
                        timeout=15,
                        unique=True, 
                        advanced=True
                    ))
                    
                    if search_results and len(search_results) > 0:
                        first_result = search_results[0]
                        print(f"Google search result type: {type(first_result)}")
                        if hasattr(first_result, '__dict__'):
                            print(f"Google search result attributes: {first_result.__dict__}")
                        else:
                            print(f"Google search result dir: {dir(first_result)}")
                    
                    filtered_results = []
                    irrelevant_domains = [
                        "pinterest", "instagram", "facebook", "twitter", 
                        "youtube", "tiktok", "reddit", "quora", "linkedin",
                        "msn.com/en-us/money", "msn.com/en-us/lifestyle",
                        "msn.com/en-us/entertainment", "msn.com/en-us/travel",
                        "amazon.com", "ebay.com", "etsy.com", "walmart.com", 
                        "target.com", "netflix.com", "hulu.com", "spotify.com"
                    ]
                    
                    for result in search_results:
                        try:
                            if hasattr(result, 'url') and hasattr(result, 'title'):
                                url = result.url
                                if url and isinstance(url, str) and any(domain in url.lower() for domain in irrelevant_domains):
                                    continue
                                
                                snippet = ""
                                if hasattr(result, 'description'):
                                    snippet = result.description
                                
                                filtered_results.append(
                                    SearchResult(
                                        title=result.title,
                                        url=url,
                                        snippet=snippet,
                                        source="Google"
                                    )
                                )
                            elif isinstance(result, dict):
                                url = result.get('url', '')
                                if url and isinstance(url, str) and any(domain in url.lower() for domain in irrelevant_domains):
                                    continue
                                    
                                filtered_results.append(
                                    SearchResult(
                                        title=result.get('title', result.get('url', 'Untitled')),
                                        url=url,
                                        snippet=result.get('description', ''),
                                        source="Google",
                                        date=result.get('date')
                                    )
                                )
                            elif isinstance(result, str):
                                if any(domain in result.lower() for domain in irrelevant_domains):
                                    continue
                                    
                                filtered_results.append(
                                    SearchResult(
                                        title=result,
                                        url=result,
                                        snippet="",
                                        source="Google"
                                    )
                                )
                            else:
                                print(f"Skipping unknown result type: {type(result)}")
                        except Exception as e:
                            print(f"Error processing search result: {e}")
                            continue
                    
                    if filtered_results:
                        query_keywords = query.lower().split()
                        important_keywords = [word for word in query_keywords 
                                             if len(word) > 3 and word not in ["from", "with", "that", "this", "what", "when", "where", "which", "while"]]
                        
                        scored_results = []
                        for result in filtered_results:
                            score = 0
                            title_lower = result.title.lower() if result.title else ""
                            for keyword in important_keywords:
                                if keyword in title_lower:
                                    score += 3
                            
                            snippet_lower = result.snippet.lower() if result.snippet else ""
                            for keyword in important_keywords:
                                if keyword in snippet_lower:
                                    score += 1
                            
                            scored_results.append((score, result))
                        
                        scored_results.sort(reverse=True, key=lambda x: x[0])
                        results = [result for score, result in scored_results[:num_results]]
                    else:
                        results = []
                        for result in search_results[:num_results]:
                            try:
                                if hasattr(result, 'url') and hasattr(result, 'title'):
                                    snippet = ""
                                    if hasattr(result, 'description'):
                                        snippet = result.description
                                    
                                    results.append(SearchResult(
                                        title=result.title,
                                        url=result.url,
                                        snippet=snippet,
                                        source="Google"
                                    ))
                                elif isinstance(result, dict):
                                    results.append(SearchResult(
                                        title=result.get('title', result.get('url', 'Untitled')),
                                        url=result.get('url', ''),
                                        snippet=result.get('description', ''),
                                        source="Google"
                                    ))
                                elif isinstance(result, str):
                                    results.append(SearchResult(
                                        title=result,
                                        url=result,
                                        snippet="",
                                        source="Google"
                                    ))
                            except Exception as e:
                                print(f"Error converting search result: {e}")
                    
                    break
                except Exception as e:
                    print(f"Google search attempt {attempt+1} failed: {e}")
                    if attempt < 2:
                        await asyncio.sleep(2 ** attempt)
            
            if results:
                try:
                    result_dicts = []
                    for r in results:
                        if isinstance(r, SearchResult):
                            result_dicts.append(r.to_dict())
                        elif isinstance(r, dict):
                            result_dicts.append(r)
                        else:
                            print(f"Warning: Skipping non-serializable result: {type(r)}")
                    
                    self.cache.set(search_query, "google", result_dicts)
                except Exception as e:
                    print(f"Error caching Google search results: {e}")
                
        except Exception as e:
            print(f"Google search error: {e}")
            
        return results
    
    def _humanize_query(self, query: str) -> str:
        """Make search queries more human-like for better results."""
        # Remove excessive punctuation and formatting
        query = query.replace('?', ' ').replace('!', ' ').replace('"', ' ').replace("'", ' ')
        query = ' '.join(query.split())
        
        # If query is too long, truncate it but keep important parts
        if len(query) > 150:
            words = query.split()
            
            # Keep the first 5 words and the last 10 words to maintain context
            if len(words) > 15:
                query = ' '.join(words[:5] + words[-10:])
            else:
                query = ' '.join(words[:15])
            
        return query

    def _parse_ddg_results(self, results_str: str) -> List[SearchResult]:
        """Parse DuckDuckGo results with improved error handling."""
        results = []
        if not results_str or "snippet:" not in results_str:
            return results  # Return empty list if input is invalid
        
        entries = results_str.split("snippet:")
        for entry in entries[1:]:  # Skip first empty split
            try:
                snippet_end = entry.find(", title:")
                title_end = entry.find(", link:")
                link_end = entry.find(", date:") if ", date:" in entry else entry.find(", source:") 
                date_end = entry.find(", source:") if ", date:" in entry else -1
                
                snippet = entry[:snippet_end].strip() if snippet_end > 0 else ""
                title = (entry[entry.find(", title:") + len(", title:"):title_end].strip() 
                        if title_end > snippet_end else "Untitled")
                link = (entry[entry.find(", link:") + len(", link:"):link_end].strip() 
                        if link_end > title_end else "")
                
                date = None
                if ", date:" in entry and date_end > link_end:
                    date = entry[entry.find(", date:") + len(", date:"):date_end].strip()
                
                source = "DuckDuckGo"
                
                results.append(SearchResult(
                    title=title,
                    url=link,
                    snippet=snippet,
                    source=source,
                    date=date
                ))
            except Exception as e:
                print(f"Error parsing DuckDuckGo result: {e}")
                continue
        return results
    
    async def _search_wikipedia(self, query: str, max_results: int = 3) -> List[SearchResult]:
        """Search Wikipedia for information."""
        results = []
        
        # Check cache first
        cached_results = self.cache.get(query, "wikipedia")
        if cached_results:
            return [SearchResult(**r) for r in cached_results]
        
        try:
            # Make search more specific for Wikipedia
            search_query = f"{query} information facts"
            
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            search_results = await loop.run_in_executor(
                self.executor, 
                lambda: wikipedia.search(search_query, results=max_results)
            )
            
            for title in search_results:
                try:
                    # Get page content with more sentences for better context
                    page_summary = await loop.run_in_executor(
                        self.executor,
                        lambda t=title: wikipedia.summary(t, sentences=5, auto_suggest=False)
                    )
                    
                    # Get page URL
                    page_url = await loop.run_in_executor(
                        self.executor,
                        lambda t=title: wikipedia.page(t, auto_suggest=False).url
                    )
                    
                    # Get full page content for better research
                    try:
                        page_content = await loop.run_in_executor(
                            self.executor,
                            lambda t=title: wikipedia.page(t, auto_suggest=False).content
                        )
                        # Extract first 2000 chars of content for more comprehensive information
                        full_content = page_content[:2000] + "..." if len(page_content) > 2000 else page_content
                    except Exception:
                        full_content = page_summary
                    
                    results.append(SearchResult(
                        title=title,
                        url=page_url,
                        snippet=page_summary,
                        source="Wikipedia",
                        metadata={
                            "type": "encyclopedia",
                            "full_content": full_content
                        }
                    ))
                except (wikipedia.exceptions.DisambiguationError, 
                        wikipedia.exceptions.PageError) as e:
                    print(f"Wikipedia error for '{title}': {e}")
                    continue
                except Exception as e:
                    print(f"Error processing Wikipedia page '{title}': {e}")
                    continue
            
            # Cache results
            if results:
                try:
                    # Ensure we're storing dictionaries, not SearchResult objects
                    result_dicts = []
                    for r in results:
                        if isinstance(r, SearchResult):
                            result_dicts.append(r.to_dict())
                        elif isinstance(r, dict):
                            result_dicts.append(r)
                        else:
                            print(f"Warning: Skipping non-serializable result: {type(r)}")
                    
                    self.cache.set(query, "wikipedia", result_dicts)
                except Exception as e:
                    print(f"Error caching Wikipedia search results: {e}")
                
        except Exception as e:
            print(f"Wikipedia search error: {e}")
            
        return results
    
    async def _search_arxiv(self, query: str, max_results: int = 3) -> List[SearchResult]:
        """Search arXiv for academic papers."""
        results = []
        
        # Check cache first
        cached_results = self.cache.get(query, "arxiv")
        if cached_results:
            return [SearchResult(**r) for r in cached_results]
        
        try:
            # Make search more academic for arXiv
            search_query = f"{query} research paper"
            
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            search_results = await loop.run_in_executor(
                self.executor,
                lambda: list(arxiv.Search(
                    query=search_query,
                    max_results=max_results,
                    sort_by=arxiv.SortCriterion.Relevance
                ).results())
            )
            
            for paper in search_results:
                # Extract full summary for better research
                full_summary = paper.summary
                
                # Create a more comprehensive snippet
                snippet = full_summary[:500] + "..." if len(full_summary) > 500 else full_summary
                
                # Get both PDF and abstract URLs for more flexibility
                pdf_url = paper.pdf_url
                abstract_url = paper.entry_id.replace("http://", "https://") if paper.entry_id else pdf_url
                
                # Use abstract URL as primary since it's more readable in browser
                primary_url = abstract_url
                
                results.append(SearchResult(
                    title=paper.title,
                    url=primary_url,
                    snippet=snippet,
                    source="arXiv",
                    date=paper.published.strftime("%Y-%m-%d") if paper.published else None,
                    metadata={
                        "authors": [author.name for author in paper.authors],
                        "categories": paper.categories,
                        "type": "academic_paper",
                        "pdf_url": pdf_url,
                        "abstract_url": abstract_url,
                        "full_summary": full_summary,
                        "doi": paper.doi
                    }
                ))
            
            # Cache results
            if results:
                try:
                    # Ensure we're storing dictionaries, not SearchResult objects
                    result_dicts = []
                    for r in results:
                        if isinstance(r, SearchResult):
                            result_dicts.append(r.to_dict())
                        elif isinstance(r, dict):
                            result_dicts.append(r)
                        else:
                            print(f"Warning: Skipping non-serializable result: {type(r)}")
                    
                    self.cache.set(query, "arxiv", result_dicts)
                except Exception as e:
                    print(f"Error caching arXiv search results: {e}")
                
        except Exception as e:
            print(f"arXiv search error: {e}")
            
        return results

    async def search(
        self, 
        query: str,
        engines: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """
        Perform concurrent search across multiple search engines.
        
        Args:
            query: Search query string
            engines: List of search engines to use (supported: "google", "duckduckgo", "wikipedia", "arxiv")
            
        Returns:
            List of SearchResult objects
        """
        if engines is None:
            engines = config.get("search", "engines")
        
        results_per_engine = max(2, self.max_results // len(engines))
        all_results = []
        tasks = []
        
        # Start all searches concurrently
        if "duckduckgo" in engines:
            tasks.append(self._search_duckduckgo(query))
        
        if "google" in engines:
            tasks.append(self._async_google_search(query, results_per_engine))
        
        if "wikipedia" in engines:
            tasks.append(self._search_wikipedia(query, max(1, results_per_engine // 2)))
        
        if "arxiv" in engines:
            tasks.append(self._search_arxiv(query, max(1, results_per_engine // 2)))
        
        # Wait for all searches to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for result in results:
            if isinstance(result, Exception):
                print(f"Search error: {result}")
            elif isinstance(result, list):
                # Ensure all results are SearchResult objects
                for item in result:
                    try:
                        if isinstance(item, SearchResult):
                            all_results.append(item)
                        elif isinstance(item, dict) and 'title' in item and 'url' in item and 'snippet' in item and 'source' in item:
                            all_results.append(SearchResult(**item))
                        # Handle googlesearch.SearchResult objects
                        elif hasattr(item, 'title') and hasattr(item, 'url'):
                            # Convert googlesearch.SearchResult to our SearchResult
                            snippet = ""
                            if hasattr(item, 'description'):
                                snippet = item.description
                            
                            all_results.append(SearchResult(
                                title=item.title,
                                url=item.url,
                                snippet=snippet,
                                source="Google"
                            ))
                        else:
                            print(f"Warning: Skipping invalid search result: {type(item)}")
                    except Exception as e:
                        print(f"Error processing search result: {e}")
        
        # Merge and limit results
        return self.merge_results(all_results, "alternate")[:self.max_results]
    
    async def _search_duckduckgo(self, query: str) -> List[SearchResult]:
        """Perform DuckDuckGo search with caching and error handling."""
        # Check cache first
        cached_results = self.cache.get(query, "duckduckgo")
        if cached_results:
            return [SearchResult(**r) for r in cached_results]
        
        results = []
        
        # Use LangChain's DuckDuckGo tools
        try:
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            ddg_results_str = await loop.run_in_executor(
                self.executor,
                lambda: self.ddg_results.run(query)
            )
            
            ddg_results = self._parse_ddg_results(ddg_results_str)
            results.extend(ddg_results)
            
            # Cache results
            if results:
                try:
                    # Ensure we're storing dictionaries, not SearchResult objects
                    result_dicts = []
                    for r in results:
                        if isinstance(r, SearchResult):
                            result_dicts.append(r.to_dict())
                        elif isinstance(r, dict):
                            result_dicts.append(r)
                        else:
                            print(f"Warning: Skipping non-serializable result: {type(r)}")
                    
                    self.cache.set(query, "duckduckgo", result_dicts)
                except Exception as e:
                    print(f"Error caching DuckDuckGo search results: {e}")
                
        except Exception as e:
            print(f"DuckDuckGo search error: {e}")
            # Fallback to simpler DuckDuckGo run
            try:
                simple_result = await loop.run_in_executor(
                    self.executor,
                    lambda: self.ddg_run.run(query)
                )
                
                results.append(
                    SearchResult(
                        title="DuckDuckGo Result",
                        url="",
                        snippet=simple_result,
                        source="DuckDuckGo"
                    )
                )
            except Exception as e:
                print(f"DuckDuckGo fallback error: {e}")
        
        return results

    def search_sync(
        self,
        query: str,
        engines: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """
        Synchronous version of search method.
        """
        return asyncio.run(self.search(query, engines))

    @staticmethod
    def merge_results(
        results: List[SearchResult],
        strategy: str = "relevance"
    ) -> List[SearchResult]:
        """
        Merge results from different sources using various strategies with improved relevance filtering.
        
        Args:
            results: List of SearchResult objects
            strategy: Merging strategy ("alternate", "source_priority", "date", "relevance")
            
        Returns:
            Merged and reordered list of SearchResult objects
        """
        if not results:
            return []
            
        if strategy == "alternate":
            # Group by source
            sources = {}
            for r in results:
                if r.source not in sources:
                    sources[r.source] = []
                sources[r.source].append(r)
            
            # Alternate between sources
            merged = []
            max_len = max(len(v) for v in sources.values()) if sources else 0
            for i in range(max_len):
                for source in sources:
                    if i < len(sources[source]):
                        merged.append(sources[source][i])
            return merged
            
        elif strategy == "date":
            # Sort by date if available
            dated_results = [r for r in results if r.date]
            undated_results = [r for r in results if not r.date]
            dated_results.sort(key=lambda x: x.date, reverse=True)
            return dated_results + undated_results
            
        elif strategy == "relevance":
            # Enhanced relevance-based sorting
            # First, categorize by source type
            academic = []
            encyclopedia = []
            news = []
            official = []
            other = []
            
            for r in results:
                source_type = r.metadata.get("type", "").lower() if hasattr(r, "metadata") and r.metadata else ""
                source = r.source.lower()
                url = r.url.lower() if r.url else ""
                
                # Categorize by source type
                if source_type == "academic_paper" or source == "arxiv" or ".edu/" in url:
                    academic.append(r)
                elif source_type == "encyclopedia" or source == "wikipedia" or "encyclopedia" in url:
                    encyclopedia.append(r)
                elif "news" in source or any(domain in url for domain in ["bbc", "cnn", "nyt", "reuters", "ap", "npr"]):
                    news.append(r)
                elif any(domain in url for domain in [".gov/", ".org/", "un.org", "who.int", "worldbank.org"]):
                    official.append(r)
                else:
                    other.append(r)
            
            # Score results within each category
            def score_result(result):
                score = 0
                
                # Title length (prefer more descriptive titles)
                if result.title:
                    title_len = len(result.title)
                    if 20 <= title_len <= 100:
                        score += 2
                    
                # Snippet length (prefer more detailed snippets)
                if result.snippet:
                    snippet_len = len(result.snippet)
                    if snippet_len > 100:
                        score += 2
                    elif snippet_len > 50:
                        score += 1
                
                # Date (prefer more recent content)
                if result.date:
                    try:
                        # Simple check if date looks recent (contains current year or last year)
                        import datetime
                        current_year = str(datetime.datetime.now().year)
                        last_year = str(datetime.datetime.now().year - 1)
                        if current_year in result.date:
                            score += 3
                        elif last_year in result.date:
                            score += 2
                    except:
                        pass
                
                return score
            
            # Sort each category by score
            for category in [academic, encyclopedia, official, news, other]:
                category.sort(key=score_result, reverse=True)
            
            # Return in order of credibility with internal scoring
            return academic + encyclopedia + official + news + other
            
        else:  # source_priority
            # Keep original order which reflects source priority
            return results
    
    def get_available_engines(self) -> List[str]:
        """Get list of available search engines."""
        return ["google", "duckduckgo", "wikipedia", "arxiv"]
