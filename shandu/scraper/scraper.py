"""
Web scraping module for Shandu research system.
Provides functionality for scraping web pages with WebBaseLoader integration.
"""
import os
import re
import asyncio
import time
import random
from typing import List, Dict, Any, Optional, Union, Set
from dataclasses import dataclass
import logging
from urllib.parse import urlparse
import aiohttp
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from ..retrieval import get_retriever, RAGRetriever
from pydantic import BaseModel, Field
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to get USER_AGENT from environment, otherwise use a generic one
USER_AGENT = os.environ.get('USER_AGENT', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')

# Cache settings
CACHE_ENABLED = True
CACHE_DIR = os.path.expanduser("~/.shandu/cache/scraper")
CACHE_TTL = 86400  # 24 hours in seconds

if CACHE_ENABLED and not os.path.exists(CACHE_DIR):
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
    except Exception as e:
        logger.warning(f"Could not create cache directory: {e}")
        CACHE_ENABLED = False

# Domain reliability tracking
class DomainReliability:
    """Track reliability metrics for domains to optimize scraping."""
    def __init__(self):
        self.domain_metrics: Dict[str, Dict[str, Any]] = {}
        self.DEFAULT_TIMEOUT = 10.0
        
    def get_timeout(self, url: str) -> float:
        """Get appropriate timeout for a domain based on past performance."""
        domain = urlparse(url).netloc
        if domain in self.domain_metrics:
            # Use domain-specific timeout if available
            return self.domain_metrics[domain].get("timeout", self.DEFAULT_TIMEOUT)
        return self.DEFAULT_TIMEOUT
        
    def update_metrics(self, url: str, success: bool, response_time: float, status_code: Optional[int] = None) -> None:
        """Update metrics for a domain based on scraping results."""
        domain = urlparse(url).netloc
        if domain not in self.domain_metrics:
            self.domain_metrics[domain] = {
                "success_count": 0,
                "fail_count": 0,
                "avg_response_time": 0,
                "timeout": self.DEFAULT_TIMEOUT,
                "status_codes": {}
            }
            
        metrics = self.domain_metrics[domain]

        if success:
            metrics["success_count"] += 1
        else:
            metrics["fail_count"] += 1

        if response_time > 0:
            total_requests = metrics["success_count"] + metrics["fail_count"]
            metrics["avg_response_time"] = (
                (metrics["avg_response_time"] * (total_requests - 1) + response_time) / total_requests
            )

        if status_code:
            metrics["status_codes"][str(status_code)] = metrics["status_codes"].get(str(status_code), 0) + 1
            
        # Adjust timeout based on response times
        if metrics["success_count"] >= 3:

            metrics["timeout"] = min(30.0, max(5.0, metrics["avg_response_time"] * 1.5))

# Global domain reliability tracker
domain_reliability = DomainReliability()

@dataclass
class ScrapedContent:
    """Class to store scraped content from a web page."""
    url: str
    title: str
    text: str
    html: str
    content_type: str
    metadata: Dict[str, Any]
    error: Optional[str] = None
    scrape_time: float = 0.0
    
    def is_successful(self) -> bool:
        """Check if scraping was successful."""
        return self.error is None and len(self.text) > 0
    
    def get_cache_key(self) -> str:
        """Generate a cache key for this content."""
        domain = urlparse(self.url).netloc
        path = urlparse(self.url).path
        return f"{domain}{path}".replace("/", "_").replace(".", "_")

class WebScraper:
    """Web scraper for extracting content from web pages using WebBaseLoader."""

    def __init__(
        self,
        proxy: Optional[str] = None,
        timeout: int = 10,
        max_concurrent: int = 5,
        cache_enabled: bool = CACHE_ENABLED,
        cache_ttl: int = CACHE_TTL,
        retriever: Optional[RAGRetriever] = None,
    ):
        """
        Initialize the web scraper.
        
        Args:
            proxy: Optional proxy URL to use for requests
            timeout: Default timeout for requests in seconds
            max_concurrent: Maximum number of concurrent scraping operations
            cache_enabled: Whether to use caching for scraped content
            cache_ttl: Time-to-live for cached content in seconds
            retriever: Optional RAG retriever for indexing scraped content
        """
        self.proxy = proxy
        self.timeout = timeout
        self.max_concurrent = max(1, min(max_concurrent, 10))  # Clamp between 1 and 10
        self.user_agent = USER_AGENT
        self.cache_enabled = cache_enabled
        self.cache_ttl = cache_ttl
        self.in_progress_urls: Set[str] = set()  # Track URLs being scraped to prevent duplicates
        self._semaphores = {}  # Dictionary to store semaphores for each event loop
        self._semaphore_lock = asyncio.Lock()  # Lock for thread-safe access to semaphores
        self.retriever = retriever or get_retriever()
        
        # Try to use fake_useragent if available
        try:
            ua = UserAgent()
            self.user_agent = ua.random
        except Exception as e:
            logger.warning(f"Could not generate random user agent: {e}. Using default.")
    
    async def _check_cache(self, url: str) -> Optional[ScrapedContent]:
        """Check if content is available in cache and not expired."""
        if not self.cache_enabled:
            return None
            
        domain = urlparse(url).netloc
        path = urlparse(url).path
        cache_key = f"{domain}{path}".replace("/", "_").replace(".", "_")
        cache_path = os.path.join(CACHE_DIR, f"{cache_key}.json")
        
        if not os.path.exists(cache_path):
            return None
            
        try:

            if time.time() - os.path.getmtime(cache_path) > self.cache_ttl:
                return None
                
            # Load cached content
            import json
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            return ScrapedContent(
                url=data["url"],
                title=data["title"],
                text=data["text"],
                html=data["html"],
                content_type=data["content_type"],
                metadata=data["metadata"],
                error=data.get("error"),
                scrape_time=data.get("scrape_time", 0.0)
            )
        except Exception as e:
            logger.warning(f"Error loading cache for {url}: {e}")
            return None
    
    async def _save_to_cache(self, content: ScrapedContent) -> bool:
        """Save scraped content to cache."""
        if not self.cache_enabled or not content.is_successful():
            return False
            
        cache_key = content.get_cache_key()
        cache_path = os.path.join(CACHE_DIR, f"{cache_key}.json")
        
        try:

            import json
            data = {
                "url": content.url,
                "title": content.title,
                "text": content.text,
                "html": content.html[:50000],  # Limit HTML size to avoid huge cache files
                "content_type": content.content_type,
                "metadata": content.metadata,
                "error": content.error,
                "scrape_time": content.scrape_time or time.time()
            }
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=None)
            return True
        except Exception as e:
            logger.warning(f"Error saving cache for {content.url}: {e}")
            return False
    
    async def scrape_url(self, url: str, dynamic: bool = False, force_refresh: bool = False) -> ScrapedContent:
        """
        Scrape content from a URL using WebBaseLoader with smart timeouts and caching.
        
        Args:
            url: URL to scrape
            dynamic: Whether to use dynamic rendering (for JavaScript-heavy sites)
            force_refresh: Whether to ignore cache and force a fresh scrape
            
        Returns:
            ScrapedContent object with the scraped content
        """
        start_time = time.time()
        logger.info(f"Scraping URL: {url}")

        if not url.startswith(('http://', 'https://')):
            return ScrapedContent(
                url=url,
                title="",
                text="",
                html="",
                content_type="",
                metadata={},
                error="Invalid URL format",
                scrape_time=start_time
            )

        if url in self.in_progress_urls:
            return ScrapedContent(
                url=url,
                title="",
                text="",
                html="",
                content_type="",
                metadata={},
                error="URL is already being processed",
                scrape_time=start_time
            )
            
        # Mark as in progress
        self.in_progress_urls.add(url)
        
        try:

            if not force_refresh:
                cached_content = await self._check_cache(url)
                if cached_content:
                    logger.info(f"Using cached content for {url}")
                    if self.retriever and cached_content.text:
                        doc = Document(
                            page_content=cached_content.text,
                            metadata={"url": cached_content.url, "title": cached_content.title},
                        )
                        self.retriever.add_documents([doc])
                    self.in_progress_urls.remove(url)
                    return cached_content

            adaptive_timeout = domain_reliability.get_timeout(url)
            
            # Configure WebBaseLoader with appropriate settings
            requests_kwargs = {
                "headers": {"User-Agent": self.user_agent},
                "timeout": adaptive_timeout,
                "verify": True  # SSL verification
            }
            
            if self.proxy:
                requests_kwargs["proxies"] = {
                    "http": self.proxy,
                    "https": self.proxy
                }

            semaphore = await self._get_semaphore()
            
            # Acquire semaphore to limit concurrency
            async with semaphore:
                # If dynamic rendering is requested, use Playwright
                if dynamic:
                    try:
                        from playwright.async_api import async_playwright
                        
                        async with async_playwright() as p:
                            browser = await p.chromium.launch(headless=True)
                            page = await browser.new_page(user_agent=self.user_agent)

                            page.set_default_timeout(adaptive_timeout * 1000)
                            
                            # Navigate to the URL with timeout handling
                            try:
                                await asyncio.wait_for(
                                    page.goto(url, wait_until="networkidle"),
                                    timeout=adaptive_timeout
                                )

                                html_content = await page.content()

                                title = await page.title()
                                
                                # Close the browser
                                await browser.close()

                                soup = BeautifulSoup(html_content, "lxml")

                                metadata = self._extract_metadata(soup, url)

                                main_content = self._extract_main_content(soup)
                                
                                end_time = time.time()
                                scrape_time = end_time - start_time

                                domain_reliability.update_metrics(
                                    url=url, 
                                    success=True, 
                                    response_time=scrape_time, 
                                    status_code=200
                                )
                                
                                result = ScrapedContent(
                                    url=url,
                                    title=title,
                                    text=main_content,
                                    html=html_content,
                                    content_type="text/html",
                                    metadata=metadata,
                                    scrape_time=scrape_time
                                )
                                
                                # Cache the successful result
                                await self._save_to_cache(result)

                                # Index into vector store for retrieval
                                if self.retriever and result.text:
                                    doc = Document(
                                        page_content=result.text,
                                        metadata={"url": result.url, "title": result.title},
                                    )
                                    self.retriever.add_documents([doc])

                                # Remove from in-progress set
                                self.in_progress_urls.remove(url)

                                return result
                                
                            except asyncio.TimeoutError:
                                await browser.close()
                                logger.warning(f"Playwright timeout for {url}")

                                domain_reliability.update_metrics(
                                    url=url, 
                                    success=False, 
                                    response_time=adaptive_timeout
                                )
                                # Fall back to WebBaseLoader
                                
                    except ImportError:
                        logger.warning("Playwright not installed. Falling back to WebBaseLoader.")
                    except Exception as e:
                        logger.error(f"Error during dynamic rendering: {e}. Falling back to WebBaseLoader.")

                        domain_reliability.update_metrics(
                            url=url, 
                            success=False, 
                            response_time=time.time() - start_time
                        )
                
                # Use WebBaseLoader for scraping
                loader = WebBaseLoader(
                    web_path=url,
                    requests_kwargs=requests_kwargs,
                    bs_kwargs={},  # BeautifulSoup already gets features parameter internally
                    raise_for_status=True,
                    continue_on_failure=False,
                    autoset_encoding=True,
                    trust_env=True
                )
                
                try:
                    # Load the document using WebBaseLoader
                    documents = await asyncio.to_thread(loader.load)
                    
                    if not documents:
                        end_time = time.time()
                        scrape_time = end_time - start_time

                        domain_reliability.update_metrics(
                            url=url, 
                            success=False, 
                            response_time=scrape_time
                        )
                        
                        self.in_progress_urls.remove(url)
                        return ScrapedContent(
                            url=url,
                            title="",
                            text="",
                            html="",
                            content_type="",
                            metadata={},
                            error="No content found",
                            scrape_time=scrape_time
                        )

                    document = documents[0]

                    metadata = document.metadata

                    text_content = document.page_content

                    # Only split if very long to reduce processing overhead
                    if len(text_content) > 20000:
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=10000,
                            chunk_overlap=200,
                            length_function=len,
                        )
                        chunks = text_splitter.split_text(text_content)
                        text_content = "\n\n".join(chunks[:5])  # Use first 5 chunks for more comprehensive coverage

                    text_content = re.sub(r'\n{3,}', '\n\n', text_content)
                    text_content = re.sub(r'\s{2,}', ' ', text_content)
                    
                    # Remove problematic patterns that could conflict with Rich markup
                    # This prevents issues when this text is displayed in the console
                    text_content = re.sub(r'\[\]', ' ', text_content)  # Empty brackets
                    text_content = re.sub(r'\[\/?[^\]]*\]?', ' ', text_content)  # Incomplete/malformed tags
                    text_content = re.sub(r'\[[^\]]*\]', ' ', text_content)  # Any bracketed content

                    html_content = ""
                    if hasattr(loader, "_html_content") and loader._html_content:
                        html_content = loader._html_content

                    title = metadata.get("title", "")
                    if not title and html_content:
                        soup = BeautifulSoup(html_content, "lxml")
                        title_tag = soup.find("title")
                        if title_tag:
                            title = title_tag.text.strip()

                    content_type = metadata.get("content-type", "text/html")
                    
                    end_time = time.time()
                    scrape_time = end_time - start_time

                    domain_reliability.update_metrics(
                        url=url, 
                        success=True, 
                        response_time=scrape_time,
                        status_code=metadata.get("status_code")
                    )
                    
                    result = ScrapedContent(
                        url=url,
                        title=title,
                        text=text_content,
                        html=html_content,
                        content_type=content_type,
                        metadata=metadata,
                        scrape_time=scrape_time
                    )
                    
                    # Cache the successful result
                    await self._save_to_cache(result)

                    # Index into vector store for retrieval
                    if self.retriever and result.text:
                        doc = Document(
                            page_content=result.text,
                            metadata={"url": result.url, "title": result.title},
                        )
                        self.retriever.add_documents([doc])

                    # Remove from in-progress set
                    self.in_progress_urls.remove(url)

                    return result
                    
                except Exception as e:
                    logger.error(f"Error using WebBaseLoader: {e}")
                    raise  # Re-raise to be caught by outer try/except
            
        except Exception as e:
            end_time = time.time()
            scrape_time = end_time - start_time
            
            logger.error(f"Error scraping URL {url}: {str(e)}")

            domain_reliability.update_metrics(
                url=url, 
                success=False, 
                response_time=scrape_time
            )
            
            # Remove from in-progress set
            self.in_progress_urls.remove(url)
            
            return ScrapedContent(
                url=url,
                title="",
                text="",
                html="",
                content_type="",
                metadata={},
                error=str(e),
                scrape_time=scrape_time
            )
    
    def _extract_metadata(self, soup: BeautifulSoup, url: str) -> Dict[str, str]:
        """Extract metadata from a BeautifulSoup object."""
        metadata = {
            "url": url,
            "domain": urlparse(url).netloc
        }

        title_tag = soup.find("title")
        if title_tag:
            metadata["title"] = title_tag.text.strip()

        for meta in soup.find_all("meta"):
            name = meta.get("name", meta.get("property", ""))
            content = meta.get("content", "")
            if name and content:
                metadata[name.lower()] = content
        
        return metadata
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """
        Extract the main content from a BeautifulSoup object.
        
        This function tries to identify the main content area of a webpage
        and returns its text content with consistent formatting.
        """
        # First, try to remove common noise elements
        for noise_tag in soup.select('nav, header, footer, aside, .menu, .sidebar, .navigation, .ad, .advertisement, script, style, [role="banner"], [role="navigation"]'):
            if noise_tag:
                noise_tag.decompose()
                
        # Try to find main content containers
        main_tags = soup.find_all(
            ["main", "article", "div", "section"], 
            class_=lambda c: c and any(x in str(c).lower() for x in [
                "content", "main", "article", "body", "entry", "post", "text"
            ])
        )
        
        content = ""
        if main_tags:
            # Use the largest content container
            main_tag = max(main_tags, key=lambda tag: len(tag.get_text(strip=True)))
            content = main_tag.get_text(separator="\n", strip=True)
        else:
            # If no main content container found, use the body
            body = soup.find("body")
            if body:
                content = body.get_text(separator="\n", strip=True)
            else:
                # If no body found, use the entire HTML
                content = soup.get_text(separator="\n", strip=True)
        
        # Thorough cleanup of the content
        # Remove repetitive headers/footers
        content = re.sub(r'([^\n]+)(\n\1)+', r'\1', content)
        
        # Normalize whitespace 
        content = re.sub(r'\n{3,}', '\n\n', content)  # Replace 3+ newlines with 2
        content = re.sub(r'\s{2,}', ' ', content)     # Replace multiple spaces with 1
        
        # Remove very short lines that are likely menu items or noise
        content_lines = [line for line in content.split('\n') if len(line.strip()) > 3]
        content = '\n'.join(content_lines)
        
        return content.strip()
    
    async def _get_semaphore(self) -> asyncio.Semaphore:
        """
        Get or create a semaphore for the current event loop.
        
        This ensures that each thread has its own semaphore bound to the correct event loop.
        """
        try:

            loop = asyncio.get_event_loop()
            loop_id = id(loop)
            
            # If we already have a semaphore for this loop, return it
            if loop_id in self._semaphores:
                return self._semaphores[loop_id]
            
            # Otherwise, create a new one
            async with self._semaphore_lock:
                # Double-check if another task has created it while we were waiting
                if loop_id in self._semaphores:
                    return self._semaphores[loop_id]

                semaphore = asyncio.Semaphore(self.max_concurrent)
                self._semaphores[loop_id] = semaphore
                return semaphore
                
        except RuntimeError:
            # If we can't get the event loop, create a new one and a semaphore for it
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            semaphore = asyncio.Semaphore(self.max_concurrent)
            
            loop_id = id(loop)
            self._semaphores[loop_id] = semaphore
            return semaphore
            
    async def scrape_urls(self, urls: List[str], dynamic: bool = False, force_refresh: bool = False) -> List[ScrapedContent]:
        """
        Scrape multiple URLs concurrently with improved parallelism and error handling.
        
        Args:
            urls: List of URLs to scrape
            dynamic: Whether to use dynamic rendering
            force_refresh: Whether to ignore cache and force fresh scrapes
            
        Returns:
            List of ScrapedContent objects
        """
        if not urls:
            return []
            
        # Filter out duplicates while preserving order
        unique_urls = []
        seen = set()
        for url in urls:
            if url not in seen:
                unique_urls.append(url)
                seen.add(url)
                
        # Use asyncio.gather with concurrency control via the semaphore
        # The semaphore is handled inside scrape_url, so we don't need to apply it here
        tasks = [self.scrape_url(url, dynamic, force_refresh) for url in unique_urls]
        
        try:
            # Use as_completed pattern for better handling of individual timeouts
            results = []
            for task in asyncio.as_completed(tasks, timeout=60):  # Overall timeout
                try:
                    result = await task
                    results.append(result)
                except asyncio.TimeoutError:
                    logger.warning(f"Task timeout during batch scraping")
                except Exception as e:
                    logger.error(f"Error in scraping task: {e}")
                    
            # Sort results back to match input order
            result_map = {r.url: r for r in results if hasattr(r, 'url')}
            ordered_results = [result_map.get(url, None) for url in unique_urls]
            
            # Replace None values with error objects
            for i, result in enumerate(ordered_results):
                if result is None:
                    ordered_results[i] = ScrapedContent(
                        url=unique_urls[i],
                        title="",
                        text="",
                        html="",
                        content_type="",
                        metadata={},
                        error="Failed to complete scraping",
                        scrape_time=time.time()
                    )
            
            return ordered_results
            
        except asyncio.TimeoutError:
            logger.error("Overall timeout exceeded during batch scraping")

            completed_results = [task.result() if task.done() and not task.exception() else None for task in tasks]
            
            # Replace None values with error objects
            for i, result in enumerate(completed_results):
                if result is None:
                    completed_results[i] = ScrapedContent(
                        url=unique_urls[i] if i < len(unique_urls) else "unknown",
                        title="",
                        text="",
                        html="",
                        content_type="",
                        metadata={},
                        error="Timeout during batch scraping",
                        scrape_time=time.time()
                    )
            
            return completed_results
        except Exception as e:
            logger.error(f"Unexpected error during batch scraping: {e}")
            return [ScrapedContent(
                url=url,
                title="",
                text="",
                html="",
                content_type="",
                metadata={},
                error=f"Batch scraping error: {str(e)}",
                scrape_time=time.time()
            ) for url in unique_urls]

# Structured output models for scraping
class ScrapingResult(BaseModel):
    """Structured output for scraping results."""
    url: str = Field(description="URL of the scraped page")
    title: str = Field(description="Title of the page")
    content: str = Field(description="Extracted content from the page")
    success: bool = Field(description="Whether scraping was successful")
    error: Optional[str] = Field(description="Error message if scraping failed", default=None)
