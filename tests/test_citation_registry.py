import unittest
from shandu.agents.utils.citation_registry import CitationRegistry

class TestCitationRegistry(unittest.TestCase):
    """Basic tests for the CitationRegistry class."""
    
    def test_citation_registration(self):
        """Test that citations can be registered and retrieved correctly."""
        registry = CitationRegistry()
        
        # Register a few citations
        cid1 = registry.register_citation("https://example.com/article1")
        cid2 = registry.register_citation("https://example.com/article2")
        cid3 = registry.register_citation("https://example.com/article3")
        
        # Test citation IDs are sequential
        self.assertEqual(cid1, 1)
        self.assertEqual(cid2, 2)
        self.assertEqual(cid3, 3)
        
        # Test URL to ID mapping works
        self.assertEqual(registry.url_to_id["https://example.com/article1"], 1)
        self.assertEqual(registry.url_to_id["https://example.com/article2"], 2)
        
        # Test ID to URL mapping works
        self.assertEqual(registry.id_to_url[1], "https://example.com/article1")
        self.assertEqual(registry.id_to_url[2], "https://example.com/article2")
        
        # Test getting citation info
        self.assertEqual(registry.get_citation_info(1)["url"], "https://example.com/article1")
        self.assertEqual(registry.get_citation_info(2)["url"], "https://example.com/article2")
    
    def test_bulk_registration(self):
        """Test bulk registration of citations."""
        registry = CitationRegistry()
        
        urls = [
            "https://example.com/article1",
            "https://example.com/article2",
            "https://example.com/article3"
        ]
        
        registry.bulk_register_sources(urls)
        
        # Check all URLs were registered
        self.assertEqual(len(registry.citations), 3)
        
        # Check URL to ID mappings
        self.assertIn("https://example.com/article1", registry.url_to_id)
        self.assertIn("https://example.com/article2", registry.url_to_id)
        self.assertIn("https://example.com/article3", registry.url_to_id)
    
    def test_citation_validation(self):
        """Test citation validation in text."""
        registry = CitationRegistry()
        
        # Register a few citations
        registry.register_citation("https://example.com/article1")
        registry.register_citation("https://example.com/article2")
        
        # Text with valid and invalid citations
        text = """
        This is a test with valid citation [1] and another valid citation [2].
        This is an invalid citation [3] that doesn't exist.
        Here's another mention of [1] and an out-of-range [5].
        """
        
        result = registry.validate_citations(text)
        
        # Check validation results
        self.assertFalse(result["valid"])
        self.assertIn(3, result["invalid_citations"])
        self.assertIn(5, result["invalid_citations"])
        # used_citations should include all citation IDs found in the text,
        # regardless of whether they are valid or invalid
        self.assertEqual(len(result["used_citations"]), 4)
        self.assertIn(1, result["used_citations"])
        self.assertIn(2, result["used_citations"])

if __name__ == '__main__':
    unittest.main()
