"""
Tests for the search module.
"""
import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio
import json
from datetime import datetime
from shandu.search.search import UnifiedSearcher, SearchResult, SearchCache

class TestSearchResult(unittest.TestCase):
    """Test cases for the SearchResult class."""
    
    def test_init(self):
        """Test initialization."""
        # Test with all parameters
        result = SearchResult(
            title="Test Title",
            url="https://example.com",
            snippet="Test snippet",
            source="Test Source",
            date="2023-01-01",
            metadata={"key": "value"}
        )
        
        self.assertEqual(result.title, "Test Title")
        self.assertEqual(result.url, "https://example.com")
        self.assertEqual(result.snippet, "Test snippet")
        self.assertEqual(result.source, "Test Source")
        self.assertEqual(result.date, "2023-01-01")
        self.assertEqual(result.metadata, {"key": "value"})
        
        # Test with minimal parameters
        result = SearchResult(
            title="Test Title",
            url="https://example.com",
            snippet="Test snippet",
            source="Test Source"
        )
        
        self.assertEqual(result.title, "Test Title")
        self.assertEqual(result.url, "https://example.com")
        self.assertEqual(result.snippet, "Test snippet")
        self.assertEqual(result.source, "Test Source")
        self.assertIsNone(result.date)
        self.assertEqual(result.metadata, {})
        
        # Test with None url
        result = SearchResult(
            title="Test Title",
            url=None,
            snippet="Test snippet",
            source="Test Source"
        )
        
        self.assertEqual(result.title, "Test Title")
        self.assertEqual(result.url, "")  # None should be converted to empty string
        self.assertEqual(result.snippet, "Test snippet")
        self.assertEqual(result.source, "Test Source")
    
    def test_to_dict(self):
        """Test to_dict method."""
        result = SearchResult(
            title="Test Title",
            url="https://example.com",
            snippet="Test snippet",
            source="Test Source",
            date="2023-01-01",
            metadata={"key": "value"}
        )
        
        result_dict = result.to_dict()
        
        self.assertEqual(result_dict["title"], "Test Title")
        self.assertEqual(result_dict["url"], "https://example.com")
        self.assertEqual(result_dict["snippet"], "Test snippet")
        self.assertEqual(result_dict["source"], "Test Source")
        self.assertEqual(result_dict["date"], "2023-01-01")
        self.assertEqual(result_dict["metadata"], {"key": "value"})
    
    def test_repr(self):
        """Test __repr__ method."""
        result = SearchResult(
            title="Test Title",
            url="https://example.com",
            snippet="Test snippet",
            source="Test Source"
        )
        
        repr_str = repr(result)
        
        self.assertIn("Test Title", repr_str)
        self.assertIn("https://example.com", repr_str)
        self.assertIn("Test Source", repr_str)

class TestUnifiedSearcher(unittest.TestCase):
    """Test cases for the UnifiedSearcher class."""
    
    def setUp(self):
        """Set up test environment."""
        self.searcher = UnifiedSearcher()
    
    def test_humanize_query(self):
        """Test _humanize_query method."""
        # Test with normal query
        query = "What is the capital of France?"
        humanized = self.searcher._humanize_query(query)
        self.assertEqual(humanized, "What is the capital of France")
        
        # Test with excessive punctuation
        query = "What is the capital of France?!?!"
        humanized = self.searcher._humanize_query(query)
        self.assertEqual(humanized, "What is the capital of France")
        
        # Test with quotes
        query = "What is the 'capital' of France?"
        humanized = self.searcher._humanize_query(query)
        self.assertEqual(humanized, "What is the capital of France")
        
        # Test with very long query
        query = "This is a very long query that should be truncated because it exceeds the maximum length allowed for a search query. It contains a lot of unnecessary information that is not relevant to the search."
        humanized = self.searcher._humanize_query(query)
        self.assertLess(len(humanized), len(query))
    
    def test_parse_ddg_results(self):
        """Test _parse_ddg_results method."""
        # Test with valid results
        results_str = """
        snippet: This is the first snippet, title: First Result, link: https://example.com/1, date: 2023-01-01, source: Example
        snippet: This is the second snippet, title: Second Result, link: https://example.com/2, source: Example
        """
        
        parsed = self.searcher._parse_ddg_results(results_str)
        
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0].title, "First Result")
        self.assertEqual(parsed[0].url, "https://example.com/1")
        self.assertEqual(parsed[0].snippet, "This is the first snippet")
        self.assertEqual(parsed[0].source, "DuckDuckGo")
        self.assertEqual(parsed[0].date, "2023-01-01")
        
        self.assertEqual(parsed[1].title, "Second Result")
        self.assertEqual(parsed[1].url, "https://example.com/2")
        self.assertEqual(parsed[1].snippet, "This is the second snippet")
        self.assertEqual(parsed[1].source, "DuckDuckGo")
        self.assertIsNone(parsed[1].date)
        
        # Test with invalid results
        results_str = "This is not a valid result format"
        parsed = self.searcher._parse_ddg_results(results_str)
        self.assertEqual(len(parsed), 0)
        
        # Test with empty results
        parsed = self.searcher._parse_ddg_results("")
        self.assertEqual(len(parsed), 0)
    
    @patch("shandu.search.search.google_search")
    async def test_async_google_search(self, mock_google_search):
        """Test _async_google_search method."""
        # Mock google_search
        mock_result1 = MagicMock()
        mock_result1.title = "First Result"
        mock_result1.url = "https://example.com/1"
        mock_result1.description = "This is the first result"
        
        mock_result2 = MagicMock()
        mock_result2.title = "Second Result"
        mock_result2.url = "https://example.com/2"
        mock_result2.description = "This is the second result"
        
        mock_google_search.return_value = [mock_result1, mock_result2]
        
        # Test search
        results = await self.searcher._async_google_search("test query", 2)
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].title, "First Result")
        self.assertEqual(results[0].url, "https://example.com/1")
        self.assertEqual(results[0].snippet, "This is the first result")
        self.assertEqual(results[0].source, "Google")
        
        self.assertEqual(results[1].title, "Second Result")
        self.assertEqual(results[1].url, "https://example.com/2")
        self.assertEqual(results[1].snippet, "This is the second result")
        self.assertEqual(results[1].source, "Google")
        
        # Test with exception
        mock_google_search.side_effect = Exception("Search failed")
        results = await self.searcher._async_google_search("test query", 2)
        self.assertEqual(len(results), 0)
    
    @patch("shandu.search.search.UnifiedSearcher._async_google_search")
    @patch("shandu.search.search.UnifiedSearcher._search_duckduckgo")
    @patch("shandu.search.search.UnifiedSearcher._search_wikipedia")
    @patch("shandu.search.search.UnifiedSearcher._search_arxiv")
    async def test_search(self, mock_arxiv, mock_wiki, mock_ddg, mock_google):
        """Test search method."""
        # Mock search results
        google_result = SearchResult(
            title="Google Result",
            url="https://example.com/google",
            snippet="Google snippet",
            source="Google"
        )
        
        ddg_result = SearchResult(
            title="DuckDuckGo Result",
            url="https://example.com/ddg",
            snippet="DuckDuckGo snippet",
            source="DuckDuckGo"
        )
        
        wiki_result = SearchResult(
            title="Wikipedia Result",
            url="https://example.com/wiki",
            snippet="Wikipedia snippet",
            source="Wikipedia"
        )
        
        arxiv_result = SearchResult(
            title="arXiv Result",
            url="https://example.com/arxiv",
            snippet="arXiv snippet",
            source="arXiv"
        )
        
        mock_google.return_value = [google_result]
        mock_ddg.return_value = [ddg_result]
        mock_wiki.return_value = [wiki_result]
        mock_arxiv.return_value = [arxiv_result]
        
        # Test with all engines
        results = await self.searcher.search("test query", ["google", "duckduckgo", "wikipedia", "arxiv"])
        
        self.assertEqual(len(results), 4)
        sources = [r.source for r in results]
        self.assertIn("Google", sources)
        self.assertIn("DuckDuckGo", sources)
        self.assertIn("Wikipedia", sources)
        self.assertIn("arXiv", sources)
        
        # Test with specific engines
        results = await self.searcher.search("test query", ["google", "duckduckgo"])
        
        self.assertEqual(len(results), 2)
        sources = [r.source for r in results]
        self.assertIn("Google", sources)
        self.assertIn("DuckDuckGo", sources)
        self.assertNotIn("Wikipedia", sources)
        self.assertNotIn("arXiv", sources)
        
        # Test with exception in one engine
        mock_google.side_effect = Exception("Search failed")
        results = await self.searcher.search("test query", ["google", "duckduckgo"])
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].source, "DuckDuckGo")
    
    def test_merge_results(self):
        """Test merge_results method."""
        # Create test results
        results = [
            SearchResult(
                title="Google Result",
                url="https://example.com/google",
                snippet="Google snippet",
                source="Google"
            ),
            SearchResult(
                title="DuckDuckGo Result",
                url="https://example.com/ddg",
                snippet="DuckDuckGo snippet",
                source="DuckDuckGo"
            ),
            SearchResult(
                title="Wikipedia Result",
                url="https://example.com/wiki",
                snippet="Wikipedia snippet",
                source="Wikipedia"
            ),
            SearchResult(
                title="arXiv Result",
                url="https://example.com/arxiv",
                snippet="arXiv snippet",
                source="arXiv",
                metadata={"type": "academic_paper"}
            )
        ]
        
        # Test alternate strategy
        merged = self.searcher.merge_results(results, "alternate")
        self.assertEqual(len(merged), 4)
        
        # Test relevance strategy
        merged = self.searcher.merge_results(results, "relevance")
        self.assertEqual(len(merged), 4)
        self.assertEqual(merged[0].source, "arXiv")  # Academic papers should be first
        
        # Test date strategy
        results[0].date = "2023-01-02"
        results[1].date = "2023-01-01"
        merged = self.searcher.merge_results(results, "date")
        self.assertEqual(len(merged), 4)
        self.assertEqual(merged[0].date, "2023-01-02")  # Most recent first
        self.assertEqual(merged[1].date, "2023-01-01")
        
        # Test with empty results
        merged = self.searcher.merge_results([])
        self.assertEqual(len(merged), 0)

class TestSearchCache(unittest.TestCase):
    """Test cases for the SearchCache class."""
    
    @patch("shandu.search.search.os.path.exists")
    @patch("shandu.search.search.open")
    @patch("shandu.search.search.json.load")
    @patch("shandu.search.search.time.time")
    def test_get_cache_hit(self, mock_time, mock_json_load, mock_open, mock_exists):
        """Test get method with cache hit."""
        # Mock file exists
        mock_exists.return_value = True
        
        # Mock time
        mock_time.return_value = 1000
        
        # Mock JSON data
        mock_json_load.return_value = {
            "timestamp": 900,  # Within TTL
            "results": [
                {
                    "title": "Cached Result",
                    "url": "https://example.com/cached",
                    "snippet": "Cached snippet",
                    "source": "Cached Source"
                }
            ]
        }
        
        # Create cache with TTL of 200 seconds
        cache = SearchCache(ttl=200)
        
        # Get cached results
        results = cache.get("test query", "test-engine")
        
        # Check if results were retrieved from cache
        self.assertIsNotNone(results)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "Cached Result")
        self.assertEqual(results[0]["url"], "https://example.com/cached")
        self.assertEqual(results[0]["snippet"], "Cached snippet")
        self.assertEqual(results[0]["source"], "Cached Source")
    
    @patch("shandu.search.search.os.path.exists")
    def test_get_cache_miss(self, mock_exists):
        """Test get method with cache miss."""
        # Mock file does not exist
        mock_exists.return_value = False
        
        # Create cache
        cache = SearchCache()
        
        # Get non-existent cached results
        results = cache.get("test query", "test-engine")
        
        # Check if results are None
        self.assertIsNone(results)

if __name__ == "__main__":
    unittest.main()
