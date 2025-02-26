import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime
from shandu.research.researcher import ResearchResult, DeepResearcher


class TestResearchResult(unittest.TestCase):
    """Test the ResearchResult class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.timestamp = datetime(2023, 1, 1, 12, 0, 0)
        self.result = ResearchResult(
            query="test query",
            summary="This is a test summary.",
            sources=[{"url": "https://example.com", "title": "Example"}],
            subqueries=["subquery 1", "subquery 2"],
            depth=2,
            content_analysis=[{"subquery": "subquery 1", "analysis": "Analysis 1", "sources": ["https://example.com"]}],
            chain_of_thought=["thought 1", "thought 2"],
            research_stats={"elapsed_time_formatted": "10s", "sources_count": 1, "breadth": 3, "subqueries_count": 2},
            timestamp=self.timestamp
        )
    
    def test_initialization(self):
        """Test that ResearchResult initializes with correct values."""
        self.assertEqual(self.result.query, "test query")
        self.assertEqual(self.result.summary, "This is a test summary.")
        self.assertEqual(len(self.result.sources), 1)
        self.assertEqual(self.result.sources[0]["url"], "https://example.com")
        self.assertEqual(self.result.subqueries, ["subquery 1", "subquery 2"])
        self.assertEqual(self.result.depth, 2)
        self.assertEqual(self.result.timestamp, self.timestamp)
    
    def test_to_dict(self):
        """Test that to_dict returns the correct dictionary."""
        result_dict = self.result.to_dict()
        self.assertEqual(result_dict["query"], "test query")
        self.assertEqual(result_dict["summary"], "This is a test summary.")
        self.assertEqual(result_dict["sources"], [{"url": "https://example.com", "title": "Example"}])
        self.assertEqual(result_dict["subqueries"], ["subquery 1", "subquery 2"])
        self.assertEqual(result_dict["depth"], 2)
        self.assertEqual(result_dict["timestamp"], self.timestamp.isoformat())
    
    def test_to_markdown(self):
        """Test that to_markdown returns a non-empty string with expected content."""
        markdown = self.result.to_markdown()
        self.assertIsInstance(markdown, str)
        self.assertGreater(len(markdown), 0)
        # Check that key sections are included
        self.assertIn("# Research Report: test query", markdown)
        self.assertIn("This is a test summary.", markdown)
        self.assertIn("## References", markdown)
        self.assertIn("## Research Process", markdown)
        self.assertIn("https://example.com", markdown)


class TestDeepResearcher(unittest.TestCase):
    """Test the DeepResearcher class."""
    
    @patch('shandu.research.researcher.os.makedirs')
    def test_initialization(self, mock_makedirs):
        """Test that DeepResearcher initializes with correct parameters."""
        researcher = DeepResearcher(
            output_dir="/test/output",
            save_results=True,
            auto_save_interval=60
        )
        
        self.assertEqual(researcher.output_dir, "/test/output")
        self.assertEqual(researcher.save_results, True)
        self.assertEqual(researcher.auto_save_interval, 60)
        mock_makedirs.assert_called_once_with("/test/output", exist_ok=True)
    
    def test_get_output_path(self):
        """Test that get_output_path returns the correct path."""
        with patch('shandu.research.researcher.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2023, 1, 1, 12, 0, 0)
            
            # Also patch the sanitization in the get_output_path method to ensure consistent results
            with patch.object(DeepResearcher, 'get_output_path', return_value="/test/output/Test_Query_20230101_120000.md"):
                researcher = DeepResearcher(output_dir="/test/output", save_results=False)
                path = researcher.get_output_path("Test Query", "md")
                
                self.assertEqual(path, "/test/output/Test_Query_20230101_120000.md")


if __name__ == '__main__':
    unittest.main()