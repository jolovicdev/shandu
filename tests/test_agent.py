import unittest
from unittest.mock import MagicMock, patch
from shandu.agents.agent import ResearchAgent


class TestResearchAgent(unittest.TestCase):
    """Test the ResearchAgent class."""
    
    @patch('shandu.agents.agent.ChatOpenAI')
    @patch('shandu.agents.agent.UnifiedSearcher')
    @patch('shandu.agents.agent.WebScraper')
    @patch('shandu.agents.agent.initialize_agent')
    def test_agent_initialization(self, mock_initialize_agent, mock_web_scraper, mock_unified_searcher, mock_chat_openai):
        """Test that agent initializes with correct parameters."""
        # Setup mocks
        mock_chat_openai.return_value = MagicMock()
        mock_unified_searcher.return_value = MagicMock()
        mock_web_scraper.return_value = MagicMock()
        mock_initialize_agent.return_value = MagicMock()
        
        # Initialize agent
        agent = ResearchAgent(
            max_depth=2,
            breadth=3,
            temperature=0.1
        )
        
        # Assertions
        self.assertEqual(agent.max_depth, 2)
        self.assertEqual(agent.breadth, 3)
        self.assertIsNotNone(agent.searcher)
        self.assertIsNotNone(agent.scraper)
        self.assertIsNotNone(agent.llm)
        self.assertIsNotNone(agent.agent)
        
        # Verify mocks were called correctly
        mock_chat_openai.assert_called_once()
        mock_unified_searcher.assert_called_once()
        mock_web_scraper.assert_called_once()
        mock_initialize_agent.assert_called_once()
    
    @patch('shandu.agents.agent.ChatOpenAI')
    @patch('shandu.agents.agent.initialize_agent')
    def test_setup_tools(self, mock_initialize_agent, mock_chat_openai):
        """Test the tool setup method."""
        # Setup mocks
        mock_chat_openai.return_value = MagicMock()
        mock_initialize_agent.return_value = MagicMock()
        
        # Create agent and call method
        agent = ResearchAgent()
        tools = agent._setup_tools()
        
        # Assertions
        self.assertEqual(len(tools), 5)  # Check if all tools are created
        tool_names = [tool.name for tool in tools]
        self.assertIn("search", tool_names)
        self.assertIn("ddg_results", tool_names)
        self.assertIn("ddg_search", tool_names)
        self.assertIn("reflect", tool_names)
        self.assertIn("generate_queries", tool_names)


if __name__ == '__main__':
    unittest.main()