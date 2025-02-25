"""
Scraper module for Shandu deep research system.
"""
from typing import List, Dict, Optional, Union, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import asyncio
import aiohttp
import time
import os
import json
import hashlib
import random
from fake_useragent import UserAgent
from pathlib import Path
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import trafilatura
from ..config import config

@dataclass
class ScrapedContent:
    """Container for scraped webpage content."""
    url: str
    title: str
    text: str
    html: str
    metadata: Dict[str, Any]
    timestamp: datetime = datetime.now()
    content_type: str = "text/html"
    status_code: Optional[int] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "url": self.url,
            "title": self.title,
            "text": self.text,
            "html": self.html,
            "metadata": self.metadata,
            "content_type": self.content_type,
            "status_code": self.status_code,
            "timestamp": self.timestamp.isoformat(),
            "error": self.error
        }
    
    def is_successful(self) -> bool:
        """Check if scraping was successful."""
        return self.error is None and bool(self.text.strip())
    
    @classmethod
    def from_error(cls, url: str, error: str) -> 'ScrapedContent':
        """Create an error result."""
        return cls(
            url=url,
            title="Error",
            text="",
            html="",
            metadata={},
            error=error
        )

class ScraperCache:
    """Cache for scraped content to improve performance."""
    def __init__(self, cache_dir: Optional[str] = None, ttl: int = 86400):
        self.cache_dir = cache_dir or os.path.expanduser("~/.shandu/cache/scraper")
        self.ttl = ttl
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _get_cache_key(self, url: str) -> str:
        """Generate a cache key from URL."""
        return hashlib.md5(url.encode()).hexdigest()
    
    def _get_cache_path(self, key: str) -> str:
        """Get file path for cache key."""
        return os.path.join(self.cache_dir, f"{key}.json")
    
    def get(self, url: str) -> Optional[ScrapedContent]:
        """Get cached content if available and not expired."""
        key = self._get_cache_key(url)
        path = self._get_cache_path(key)
        
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                
                if time.time() - data['timestamp'] <= self.ttl:
                    content_dict = data['content']
                    required_fields = ['url', 'title', 'text', 'html', 'metadata']
                    if all(field in content_dict for field in required_fields):
                        if 'timestamp' in content_dict:
                            content_dict['timestamp'] = datetime.fromisoformat(content_dict['timestamp'])
                        return ScrapedContent(**content_dict)
                    else:
                        print(f"Cache entry for {url} is missing required fields. Invalidating cache.")
                        os.remove(path)
            except Exception as e:
                print(f"Error reading cache for {url}: {e}")
        
        return None
    
    def set(self, content: ScrapedContent):
        """Cache scraped content."""
        if not isinstance(content, ScrapedContent):
            raise ValueError("Only ScrapedContent objects can be cached.")
        key = self._get_cache_key(content.url)
        path = self._get_cache_path(key)
        
        try:
            with open(path, 'w') as f:
                json.dump({
                    'timestamp': time.time(),
                    'content': content.to_dict()
                }, f)
        except Exception as e:
            print(f"Error writing cache for {content.url}: {e}")

class WebScraper:
    """
    Advanced web scraper with support for both static and dynamic pages.
    Features caching, parallel processing, and improved error handling.
    """
    def __init__(
        self,
        proxy: Optional[str] = None,
        timeout: int = 20,  # Reduced from 30 to 20 seconds
        max_retries: int = 2,  # Reduced from 3 to 2 retries
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        max_concurrent: int = 8,  # Increased from 5 to 8 for more parallel processing
        cache_ttl: int = 86400,  # 24 hours
        user_agent: Optional[str] = None
    ):
        self.proxy = proxy or config.get("scraper", "proxy")
        self.timeout = timeout or config.get("scraper", "timeout", 10)
        self.max_retries = max_retries or config.get("scraper", "max_retries", 2)
        self.chunk_size = chunk_size or config.get("scraper", "chunk_size", 1000)
        self.chunk_overlap = chunk_overlap or config.get("scraper", "chunk_overlap", 200)
        self.max_concurrent = max_concurrent
        if user_agent is None:
            self.user_agent = UserAgent().random
        else:
            self.user_agent = user_agent

        
        self.splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        self.cache = ScraperCache(ttl=cache_ttl)
        
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
    
    async def _get_page_simple(self, url: str) -> Tuple[Optional[str], Optional[str], Optional[int]]:
        """
        Get page content using aiohttp.
        
        Returns:
            Tuple of (html_content, content_type, status_code)
        """
        headers = {
            'User-Agent': self.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0',
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                kwargs = {
                    'timeout': aiohttp.ClientTimeout(total=self.timeout),
                    'headers': headers,
                    'allow_redirects': True
                }
                
                if self.proxy and self.proxy.strip():
                    kwargs['proxy'] = self.proxy
                
                async with session.get(url, **kwargs) as response:
                    content_type = response.headers.get('Content-Type', 'text/html')
                    status_code = response.status
                    
                    if 200 <= status_code < 300:
                        return await response.text(), content_type, status_code
                    else:
                        print(f"HTTP error {status_code} for {url}")
                        return None, content_type, status_code
            except asyncio.TimeoutError:
                print(f"Timeout fetching {url}")
                return None, None, None
            except Exception as e:
                print(f"Error fetching {url}: {e}")
                return None, None, None

    PROBLEMATIC_DOMAINS = []
    
    PROBLEMATIC_DOMAINS = ["msn.com", "evwind.es", "military.com", "statista.com", "yahoo.com"]
    
    async def _get_page_dynamic(
        self, 
        url: str, 
        wait_for_selector: Optional[str] = None,
        extra_wait: int = 0
    ) -> Tuple[Optional[str], Optional[str], Optional[int]]:
        """
        Get page content using Playwright for JavaScript rendering.
        
        Args:
            url: URL to fetch
            wait_for_selector: CSS selector to wait for before considering page loaded
            extra_wait: Additional time in seconds to wait after page load
            
        Returns:
            Tuple of (html_content, content_type, status_code)
        """
        is_problematic = any(domain in url for domain in self.PROBLEMATIC_DOMAINS)
        
        if is_problematic:
            print(f"URL {url} is from a problematic domain. Using simple fetching instead.")
            return await self._get_page_simple(url)
        
        browser = None
        context = None
        
        try:
            user_agent = self.user_agent
            
            async with async_playwright() as p:
                proxy_options = {"server": self.proxy} if self.proxy and self.proxy.strip() else None
                
                browser_args = ["--disable-dev-shm-usage", "--no-sandbox", "--disable-setuid-sandbox"]
                
                browser = await p.chromium.launch(
                    proxy=proxy_options,
                    headless=True,
                    args=browser_args
                )
                
                context = await browser.new_context(
                    user_agent=user_agent,
                    viewport={"width": 1280, "height": 800},
                    accept_downloads=True
                )
                
                context.set_default_timeout(self.timeout * 1000)
                
                page = await context.new_page()
                
                page.set_default_timeout(self.timeout * 1000)
                
                try:
                    wait_until = "commit" if is_problematic else "domcontentloaded"
                    
                    response = await page.goto(url, wait_until=wait_until, timeout=self.timeout * 1000)
                    
                    try:
                        await page.wait_for_load_state("networkidle", timeout=5000)
                    except PlaywrightTimeoutError:
                        print(f"Networkidle timeout for {url}, continuing anyway")
                    except Exception as e:
                        print(f"Error waiting for networkidle for {url}: {e}")
                    
                    if wait_for_selector:
                        try:
                            await page.wait_for_selector(wait_for_selector, timeout=5000)
                        except PlaywrightTimeoutError:
                            print(f"Selector '{wait_for_selector}' not found, continuing anyway")
                        except Exception as e:
                                print(f"Error waiting for selector for {url}: {e}")
                    
                    if extra_wait > 0:
                        await asyncio.sleep(min(extra_wait, 2))
                    
                    status_code = None
                    content_type = 'text/html'
                    
                    if response:
                        try:
                            status_code = response.status
                            content_type = response.headers.get('content-type', 'text/html')
                        except Exception as e:
                            print(f"Error getting response details for {url}: {e}")
                    
                    html = None
                    try:
                        html = await page.content()
                    except Exception as e:
                        print(f"Error getting page content for {url}: {e}")
                        return None, content_type, status_code
                    
                    return html, content_type, status_code
                    
                except PlaywrightTimeoutError as e:
                    print(f"Timeout error for {url}: {e}")
                    return None, None, None
                except Exception as e:
                    print(f"Error during page navigation for {url}: {e}")
                    return None, None, None
                finally:
                    await context.close()
                    await browser.close()
                
        except Exception as e:
            print(f"Error fetching {url} with Playwright: {e}")
            return None, None, None

    def _extract_content(self, html: str, url: str, content_type: str = "text/html") -> Dict[str, Any]:
        """
        Extract content from HTML/XML/JSON using trafilatura or appropriate parser.
        
        Args:
            html: The raw content (HTML, XML, JSON, etc.)
            url: The URL of the content
            content_type: The content type (MIME type)
            
        Returns:
            Dictionary with title, text, and metadata
        """
        if not html:
            return {
                "title": "No content",
                "text": "",
                "metadata": {"url": url}
            }
        
        if content_type and "json" in content_type.lower():
            try:
                json_data = json.loads(html)
                text = json.dumps(json_data, indent=2)
                return {
                    "title": url.split("/")[-1] or "JSON Content",
                    "text": text,
                    "metadata": {"url": url, "content_type": "json"}
                }
            except json.JSONDecodeError:
                pass
                
        elif content_type and "xml" in content_type.lower():
            try:
                soup = BeautifulSoup(html, 'xml')
                text = soup.get_text(separator="\n\n", strip=True)
                title = soup.find('title')
                title_text = title.get_text(strip=True) if title else url.split("/")[-1] or "XML Content"
                return {
                    "title": title_text,
                    "text": text,
                    "metadata": {"url": url, "content_type": "xml"}
                }
            except Exception:
                pass
        
        try:
            extracted_text = trafilatura.extract(
                html,
                url=url,
                include_comments=False,
                include_tables=True,
                include_images=False,
                include_links=False,
                output_format="txt"
            )
            
            if extracted_text and len(extracted_text.strip()) > 100:
                soup = BeautifulSoup(html, 'html.parser')
                title = soup.title.string if soup.title else url.split("/")[-1]
                
                return {
                    "title": title,
                    "text": extracted_text,
                    "metadata": {"url": url}
                }
            
            try:
                extracted = trafilatura.bare_extraction(
                    html,
                    url=url,
                    include_comments=False,
                    include_tables=True,
                    include_images=False,
                    include_links=False,
                    output_format="python"
                )
                
                if extracted and isinstance(extracted, dict):
                    title = extracted.get('title', '')
                    text = extracted.get('text', '')
                    
                    metadata = {
                        "url": url,
                        "description": extracted.get('description', ''),
                        "author": extracted.get('author', ''),
                        "date": extracted.get('date', ''),
                        "categories": extracted.get('categories', ''),
                        "tags": extracted.get('tags', ''),
                        "sitename": extracted.get('sitename', '')
                    }
                    
                    return {
                        "title": title,
                        "text": text,
                        "metadata": metadata
                    }
            except Exception as inner_e:
                print(f"Trafilatura bare_extraction failed for {url}: {inner_e}")
        except Exception as e:
            print(f"Trafilatura extraction failed for {url}: {e}")
            
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            title = ""
            if soup.title:
                title = soup.title.string
            
            for script in soup(["script", "style", "iframe", "noscript"]):
                script.decompose()
            
            lines = []
            
            for i, tag in enumerate(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                for heading in soup.find_all(tag):
                    text = heading.get_text(strip=True)
                    if text:
                        prefix = '#' * (i + 1)
                        lines.append(f"{prefix} {text}")
            
            for element in soup.find_all(['p', 'li']):
                text = element.get_text(strip=True)
                if text:
                    lines.append(text)
            
            for table in soup.find_all('table'):
                lines.append("TABLE:")
                for row in table.find_all('tr'):
                    row_data = []
                    for cell in row.find_all(['td', 'th']):
                        row_data.append(cell.get_text(strip=True))
                    if row_data:
                        lines.append(" | ".join(row_data))
                lines.append("END TABLE")
            
            text = "\n\n".join(lines)
            
            metadata = {
                "description": "",
                "keywords": "",
                "author": "",
                "date": "",
                "publisher": "",
                "language": "",
                "url": url
            }
            
            meta_tags = soup.find_all('meta')
            for tag in meta_tags:
                if tag.get('name') and tag.get('content'):
                    name = tag['name'].lower()
                    if name in metadata or name in ['description', 'keywords', 'author', 'date', 'publisher', 'language']:
                        metadata[name] = tag['content']
                
                if tag.get('property') and tag.get('content'):
                    prop = tag['property'].lower()
                    if 'og:title' in prop:
                        metadata['og_title'] = tag['content']
                    elif 'og:description' in prop:
                        metadata['og_description'] = tag['content']
                    elif 'og:site_name' in prop:
                        metadata['site_name'] = tag['content']
                    elif 'article:published_time' in prop:
                        metadata['date'] = tag['content']
            
            date_elements = soup.select('time, .date, .published, [itemprop="datePublished"]')
            if date_elements and not metadata.get('date'):
                metadata['date'] = date_elements[0].get_text(strip=True)
            
            return {
                "title": title,
                "text": text,
                "metadata": metadata
            }
        except Exception as e:
            print(f"BeautifulSoup extraction failed for {url}: {e}")
            
        return {
            "title": url.split("/")[-1] or "Unknown Title",
            "text": html[:1000] + "...",
            "metadata": {"url": url, "extraction_failed": True}
        }

    async def scrape_url(
        self,
        url: str,
        dynamic: bool = False,
        extract_images: bool = False,
        force_refresh: bool = False,
        wait_for_selector: Optional[str] = None,
        extra_wait: int = 0
    ) -> ScrapedContent:
        """
        Scrape content from a URL with caching and improved error handling.
        
        Args:
            url: The URL to scrape
            dynamic: Whether to use Playwright for JavaScript rendering
            extract_images: Whether to extract image data
            force_refresh: Whether to ignore cache and fetch fresh content
            wait_for_selector: CSS selector to wait for before considering page loaded
            extra_wait: Additional time in seconds to wait after page load
            
        Returns:
            ScrapedContent object (check is_successful() to verify success)
        """
        if not force_refresh:
            cached_content = self.cache.get(url)
            if cached_content:
                return cached_content
        
        html = None
        content_type = None
        status_code = None
        
        for attempt in range(self.max_retries):
            try:
                if dynamic:
                    html, content_type, status_code = await self._get_page_dynamic(
                        url, 
                        wait_for_selector=wait_for_selector,
                        extra_wait=extra_wait
                    )
                else:
                    html, content_type, status_code = await self._get_page_simple(url)
                
                if html:
                    break
                    
            except Exception as e:
                delay = (attempt + 1) * 2
                print(f"Attempt {attempt + 1} failed for {url}: {e}")
                print(f"Waiting {delay} seconds...")
                await asyncio.sleep(delay)
        
        if not html:
            error_msg = f"Failed to fetch content after {self.max_retries} attempts"
            return ScrapedContent.from_error(url, error_msg)
            
        content = self._extract_content(html, url, content_type)
        
        result = ScrapedContent(
            url=url,
            title=content["title"],
            text=content["text"],
            html=html,
            metadata=content["metadata"],
            content_type=content_type or "text/html",
            status_code=status_code
        )
        
        if result.is_successful():
            try:
                self.cache.set(result)
            except Exception as e:
                print(f"Failed to cache content for {url}: {e}")
            
        return result

    async def scrape_urls(
        self,
        urls: List[str],
        dynamic: bool = False,
        extract_images: bool = False,
        force_refresh: bool = False,
        wait_for_selector: Optional[str] = None,
        extra_wait: int = 0
    ) -> List[ScrapedContent]:
        """
        Scrape multiple URLs concurrently with improved error handling.
        
        Args:
            urls: List of URLs to scrape
            dynamic: Whether to use Playwright for JavaScript rendering
            extract_images: Whether to extract image data
            force_refresh: Whether to ignore cache and fetch fresh content
            wait_for_selector: CSS selector to wait for before considering page loaded
            extra_wait: Additional time in seconds to wait after page load
            
        Returns:
            List of ScrapedContent objects
        """
        unique_urls = []
        seen_urls = set()
        
        for url in urls:
            normalized_url = url.rstrip('/')
            if normalized_url not in seen_urls:
                seen_urls.add(normalized_url)
                unique_urls.append(url)
        
        if len(unique_urls) < len(urls):
            print(f"Removed {len(urls) - len(unique_urls)} duplicate URLs")
        
        async def scrape_with_semaphore(url: str) -> ScrapedContent:
            async with self.semaphore:
                return await self.scrape_url(
                    url, 
                    dynamic=dynamic, 
                    extract_images=extract_images, 
                    force_refresh=force_refresh,
                    wait_for_selector=wait_for_selector,
                    extra_wait=extra_wait
                )
        
        tasks = [scrape_with_semaphore(url) for url in unique_urls]
        results = await asyncio.gather(*tasks)
        
        return results

    def chunk_content(
        self,
        content: ScrapedContent,
        include_metadata: bool = True,
        max_chunks: Optional[int] = None
    ) -> List[str]:
        """
        Split content into chunks for processing with improved metadata handling.
        
        Args:
            content: ScrapedContent to chunk
            include_metadata: Whether to include metadata in chunks
            max_chunks: Maximum number of chunks to return
            
        Returns:
            List of text chunks
        """
        if not content.is_successful():
            return []
            
        text = content.text
        
        if include_metadata:
            header = f"Title: {content.title}\nURL: {content.url}\n\n"
            
            if content.metadata.get("description"):
                header += f"Description: {content.metadata['description']}\n\n"
            if content.metadata.get("date"):
                header += f"Date: {content.metadata['date']}\n\n"
            if content.metadata.get("author"):
                header += f"Author: {content.metadata['author']}\n\n"
                
            text = header + text
        
        chunks = self.splitter.split_text(text)
        
        if max_chunks and len(chunks) > max_chunks:
            return chunks[:max_chunks]
            
        return chunks

    @staticmethod
    def extract_links(html: str, base_url: Optional[str] = None) -> List[str]:
        """
        Extract all links from HTML content with improved handling of relative URLs.
        
        Args:
            html: HTML content
            base_url: Base URL for resolving relative links
            
        Returns:
            List of absolute URLs
        """
        if not html:
            return []
            
        soup = BeautifulSoup(html, 'html.parser')
        links = []
        
        base_tag = soup.find('base', href=True)
        if base_tag and not base_url:
            base_url = base_tag['href']
        
        for a in soup.find_all('a', href=True):
            href = a['href'].strip()
            
            if not href or href.startswith('javascript:') or href.startswith('#'):
                continue
                
            if not href.startswith('http'):
                if base_url:
                    if base_url.endswith('/'):
                        base_url = base_url[:-1]
                        
                    if href.startswith('/'):
                        href = base_url + href
                    else:
                        href = f"{base_url}/{href}"
                else:
                    continue
            
            links.append(href)
                
        return links

    @staticmethod
    def extract_text_by_selectors(
        html: str,
        selectors: List[str]
    ) -> Dict[str, List[str]]:
        """
        Extract text content by CSS selectors with improved error handling.
        
        Args:
            html: HTML content
            selectors: List of CSS selectors
            
        Returns:
            Dictionary mapping selectors to extracted text
        """
        if not html:
            return {selector: [] for selector in selectors}
            
        soup = BeautifulSoup(html, 'html.parser')
        results = {}
        
        for selector in selectors:
            try:
                elements = soup.select(selector)
                results[selector] = [
                    el.get_text(strip=True)
                    for el in elements
                    if el.get_text(strip=True)
                ]
            except Exception as e:
                print(f"Error extracting with selector '{selector}': {e}")
                results[selector] = []
            
        return results
    
    async def extract_main_content(self, content: ScrapedContent) -> str:
        """
        Extract main content from a webpage, filtering out navigation, ads, etc.
        Uses trafilatura for optimal content extraction with fallback to BeautifulSoup.
        
        Args:
            content: ScrapedContent object
            
        Returns:
            Extracted main content text
        """
        if not content.is_successful() or not content.html:
            return content.text
        
        try:
            extracted_text = trafilatura.extract(
                content.html,
                url=content.url,
                include_comments=False,
                include_tables=True,
                include_images=False,
                include_links=False,
                output_format="txt"
            )
            
            if extracted_text and len(extracted_text.strip()) > 100:
                return extracted_text
        except Exception as e:
            print(f"Trafilatura extraction failed for main content: {e}")
        
        try:
            soup = BeautifulSoup(content.html, 'html.parser')
            
            for selector in [
                'nav', 'header', 'footer', 'aside', 
                '.sidebar', '.navigation', '.menu', '.ad', '.advertisement',
                '.cookie-notice', '.popup', '#cookie-banner', '.banner',
                'script', 'style', 'iframe', 'noscript'
            ]:
                for element in soup.select(selector):
                    element.decompose()
            
            main_content = None
            for selector in [
                'article', 'main', '.content', '.main-content', '#content', '#main',
                '[role="main"]', '.post', '.entry', '.article-content'
            ]:
                main_content = soup.select_one(selector)
                if main_content:
                    break
            
            if not main_content:
                main_content = soup.body
            
            if not main_content:
                return content.text
                
            lines = []
            for tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li']:
                for element in main_content.find_all(tag):
                    text = element.get_text(strip=True)
                    if text:
                        if tag.startswith('h'):
                            # Add heading level indicator
                            level = int(tag[1])
                            prefix = '#' * level
                            lines.append(f"{prefix} {text}")
                        else:
                            lines.append(text)
            
            return "\n\n".join(lines)
            
        except Exception as e:
            print(f"BeautifulSoup extraction failed for main content: {e}")
            return content.text  # Return original text as fallback
