"""
Tests for the scraper module.
"""
import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio
import json
from datetime import datetime
from shandu.scraper.scraper import WebScraper, ScrapedContent, ScraperCache

class TestScrapedContent(unittest.TestCase):
    """Test cases for the ScrapedContent class."""
    
    def test_is_successful(self):
        """Test is_successful method."""
        # Successful content
        content = ScrapedContent(
            url="https://example.com",
            title="Example",
            text="Some content",
            html="<html><body>Some content</body></html>",
            metadata={}
        )
        self.assertTrue(content.is_successful())
        
        # Unsuccessful content (with error)
        content_with_error = ScrapedContent(
            url="https://example.com",
            title="Example",
            text="Some content",
            html="<html><body>Some content</body></html>",
            metadata={},
            error="Failed to fetch"
        )
        self.assertFalse(content_with_error.is_successful())
        
        # Unsuccessful content (empty text)
        content_empty = ScrapedContent(
            url="https://example.com",
            title="Example",
            text="",
            html="<html><body></body></html>",
            metadata={}
        )
        self.assertFalse(content_empty.is_successful())
    
    def test_to_dict(self):
        """Test to_dict method."""
        timestamp = datetime.now()
        content = ScrapedContent(
            url="https://example.com",
            title="Example",
            text="Some content",
            html="<html><body>Some content</body></html>",
            metadata={"key": "value"},
            timestamp=timestamp,
            content_type="text/html",
            status_code=200,
            error=None
        )
        
        content_dict = content.to_dict()
        
        self.assertEqual(content_dict["url"], "https://example.com")
        self.assertEqual(content_dict["title"], "Example")
        self.assertEqual(content_dict["text"], "Some content")
        self.assertEqual(content_dict["html"], "<html><body>Some content</body></html>")
        self.assertEqual(content_dict["metadata"], {"key": "value"})
        self.assertEqual(content_dict["content_type"], "text/html")
        self.assertEqual(content_dict["status_code"], 200)
        self.assertEqual(content_dict["error"], None)
        self.assertEqual(content_dict["timestamp"], timestamp.isoformat())
    
    def test_from_error(self):
        """Test from_error class method."""
        error_content = ScrapedContent.from_error(
            url="https://example.com",
            error="Failed to fetch"
        )
        
        self.assertEqual(error_content.url, "https://example.com")
        self.assertEqual(error_content.title, "Error")
        self.assertEqual(error_content.text, "")
        self.assertEqual(error_content.html, "")
        self.assertEqual(error_content.metadata, {})
        self.assertEqual(error_content.error, "Failed to fetch")
        self.assertFalse(error_content.is_successful())

class TestWebScraper(unittest.TestCase):
    """Test cases for the WebScraper class."""
    
    def setUp(self):
        """Set up test environment."""
        self.scraper = WebScraper()
    
    def test_extract_links(self):
        """Test extract_links method."""
        # Test with absolute URLs - no need to mock, let's use the actual implementation
        html = "<html><body><a href='https://example.com/page1'>Link 1</a><a href='https://example.com/page2'>Link 2</a><a href='javascript:void(0)'>Link 3</a></body></html>"
        links = WebScraper.extract_links(html)
        self.assertEqual(links, ["https://example.com/page1", "https://example.com/page2"])
        
        # Test with base URL for relative paths
        links = WebScraper.extract_links(
            "<html><body><a href='/relative/path'>Link 3</a></body></html>",
            base_url="https://example.com"
        )
        self.assertEqual(links, ["https://example.com/relative/path"])
        
        # Test with empty HTML
        links = WebScraper.extract_links("")
        self.assertEqual(links, [])
    
    @patch("shandu.scraper.scraper.BeautifulSoup")
    def test_extract_text_by_selectors(self, mock_bs):
        """Test extract_text_by_selectors method."""
        # Mock BeautifulSoup
        mock_soup = MagicMock()
        mock_bs.return_value = mock_soup
        
        # Mock select for selectors
        def mock_select(selector):
            if selector == "h1":
                mock_h1 = MagicMock()
                mock_h1.get_text.return_value = "Heading 1"
                return [mock_h1]
            elif selector == "p":
                mock_p1 = MagicMock()
                mock_p1.get_text.return_value = "Paragraph 1"
                mock_p2 = MagicMock()
                mock_p2.get_text.return_value = "Paragraph 2"
                return [mock_p1, mock_p2]
            else:
                return []
        
        mock_soup.select.side_effect = mock_select
        
        # Test with valid selectors
        results = WebScraper.extract_text_by_selectors(
            "<html><body><h1>Heading 1</h1><p>Paragraph 1</p><p>Paragraph 2</p></body></html>",
            ["h1", "p", "div"]
        )
        
        self.assertEqual(results["h1"], ["Heading 1"])
        self.assertEqual(results["p"], ["Paragraph 1", "Paragraph 2"])
        self.assertEqual(results["div"], [])
        
        # Test with empty HTML
        results = WebScraper.extract_text_by_selectors("", ["h1", "p"])
        self.assertEqual(results["h1"], [])
        self.assertEqual(results["p"], [])
    
    @patch("shandu.scraper.scraper.aiohttp.ClientSession")
    async def test_get_page_simple(self, mock_session):
        """Test _get_page_simple method."""
        # Mock ClientSession
        mock_session_instance = AsyncMock()
        mock_session.return_value.__aenter__.return_value = mock_session_instance
        
        # Mock response
        mock_response = AsyncMock()
        mock_response.headers = {"Content-Type": "text/html"}
        mock_response.status = 200
        mock_response.text.return_value = "<html><body>Test content</body></html>"
        
        mock_session_instance.get.return_value.__aenter__.return_value = mock_response
        
        # Test successful request
        html, content_type, status_code = await self.scraper._get_page_simple("https://example.com")
        
        self.assertEqual(html, "<html><body>Test content</body></html>")
        self.assertEqual(content_type, "text/html")
        self.assertEqual(status_code, 200)
        
        # Test error response
        mock_response.status = 404
        mock_session_instance.get.return_value.__aenter__.return_value = mock_response
        
        html, content_type, status_code = await self.scraper._get_page_simple("https://example.com/not-found")
        
        self.assertIsNone(html)
        self.assertEqual(content_type, "text/html")
        self.assertEqual(status_code, 404)

class TestScraperCache(unittest.TestCase):
    """Test cases for the ScraperCache class."""
    
    @patch("shandu.scraper.scraper.os.path.exists")
    @patch("shandu.scraper.scraper.open")
    @patch("shandu.scraper.scraper.json.load")
    @patch("shandu.scraper.scraper.time.time")
    def test_get_cache_hit(self, mock_time, mock_json_load, mock_open, mock_exists):
        """Test get method with cache hit."""
        # Mock file exists
        mock_exists.return_value = True
        
        # Mock time
        mock_time.return_value = 1000
        
        # Mock JSON data
        mock_json_load.return_value = {
            "timestamp": 900,  # Within TTL
            "content": {
                "url": "https://example.com",
                "title": "Example",
                "text": "Cached content",
                "html": "<html><body>Cached content</body></html>",
                "metadata": {},
                "content_type": "text/html",
                "status_code": 200,
                "timestamp": "2023-01-01T00:00:00"
            }
        }
        
        # Create cache with TTL of 200 seconds
        cache = ScraperCache(ttl=200)
        
        # Get cached content
        content = cache.get("https://example.com")
        
        # Check if content was retrieved from cache
        self.assertIsNotNone(content)
        self.assertEqual(content.url, "https://example.com")
        self.assertEqual(content.title, "Example")
        self.assertEqual(content.text, "Cached content")
    
    @patch("shandu.scraper.scraper.os.path.exists")
    def test_get_cache_miss(self, mock_exists):
        """Test get method with cache miss."""
        # Mock file does not exist
        mock_exists.return_value = False
        
        # Create cache
        cache = ScraperCache()
        
        # Get non-existent cached content
        content = cache.get("https://example.com")
        
        # Check if content is None
        self.assertIsNone(content)
    
    @patch("shandu.scraper.scraper.open")
    @patch("shandu.scraper.scraper.json.dump")
    @patch("shandu.scraper.scraper.time.time")
    def test_set(self, mock_time, mock_json_dump, mock_open):
        """Test set method."""
        # Mock time
        mock_time.return_value = 1000
        
        # Create cache
        cache = ScraperCache()
        
        # Create content to cache
        content = ScrapedContent(
            url="https://example.com",
            title="Example",
            text="Some content",
            html="<html><body>Some content</body></html>",
            metadata={}
        )
        
        # Cache content
        cache.set(content)
        
        # Check if content was cached correctly
        mock_json_dump.assert_called_once()
        args, _ = mock_json_dump.call_args
        data = args[0]
        
        self.assertEqual(data["timestamp"], 1000)
        self.assertEqual(data["content"]["url"], "https://example.com")
        self.assertEqual(data["content"]["title"], "Example")
        self.assertEqual(data["content"]["text"], "Some content")

if __name__ == "__main__":
    unittest.main()
