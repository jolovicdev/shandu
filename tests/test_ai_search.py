import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime
from shandu.search.ai_search import AISearcher, AISearchResult


class TestAISearchResult(unittest.TestCase):
    """Test the AISearchResult class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.timestamp = datetime(2023, 1, 1, 12, 0, 0)
        self.result = AISearchResult(
            query="test query",
            summary="This is a test summary.",
            sources=[{"url": "https://example.com", "title": "Example", "snippet": "This is an example"}],
            timestamp=self.timestamp
        )
    
    def test_initialization(self):
        """Test that AISearchResult initializes with correct values."""
        self.assertEqual(self.result.query, "test query")
        self.assertEqual(self.result.summary, "This is a test summary.")
        self.assertEqual(len(self.result.sources), 1)
        self.assertEqual(self.result.sources[0]["url"], "https://example.com")
        self.assertEqual(self.result.timestamp, self.timestamp)
    
    def test_to_dict(self):
        """Test that to_dict returns the correct dictionary."""
        result_dict = self.result.to_dict()
        self.assertEqual(result_dict["query"], "test query")
        self.assertEqual(result_dict["summary"], "This is a test summary.")
        self.assertEqual(result_dict["sources"], [{"url": "https://example.com", "title": "Example", "snippet": "This is an example"}])
        self.assertEqual(result_dict["timestamp"], self.timestamp.isoformat())
    
    def test_to_markdown(self):
        """Test that to_markdown returns a non-empty string with expected content."""
        markdown = self.result.to_markdown()
        self.assertIsInstance(markdown, str)
        self.assertGreater(len(markdown), 0)
        # Check that key sections are included
        self.assertIn("# Search Results: test query", markdown)
        self.assertIn("This is a test summary.", markdown)
        self.assertIn("## Sources", markdown)
        self.assertIn("https://example.com", markdown)
        self.assertIn("This is an example", markdown)


class TestAISearcher(unittest.TestCase):
    """Test the AISearcher class."""
    
    @patch('shandu.search.ai_search.ChatOpenAI')
    @patch('shandu.search.ai_search.UnifiedSearcher')
    @patch('shandu.search.ai_search.config')
    def test_initialization(self, mock_config, mock_unified_searcher, mock_chat_openai):
        """Test that AISearcher initializes with correct parameters."""
        # Setup mocks
        mock_config.get.side_effect = lambda section, key, default=None: {
            ("api", "base_url"): "https://api.example.com",
            ("api", "api_key"): "test_key",
            ("api", "model"): "test_model",
            ("api", "temperature"): 0
        }.get((section, key), default)
        mock_chat_openai.return_value = MagicMock()
        mock_unified_searcher.return_value = MagicMock()
        
        # Initialize searcher
        searcher = AISearcher(max_results=15)
        
        # Assertions
        self.assertIsNotNone(searcher.llm)
        self.assertIsNotNone(searcher.searcher)
        self.assertEqual(searcher.max_results, 15)
        
        # Verify mocks were called correctly
        mock_chat_openai.assert_called_once()
        mock_unified_searcher.assert_called_once_with(max_results=15)


if __name__ == '__main__':
    unittest.main()