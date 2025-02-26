"""
Tests for the configuration module.
"""
import os
import tempfile
import unittest
from unittest.mock import patch
import json
from shandu.config import Config

class TestConfig(unittest.TestCase):
    """Test cases for the Config class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test config
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_path = os.path.join(self.temp_dir.name, "config.json")
        
        # Create a test config with default values
        self.test_config = {
            "api": {
                "base_url": "https://test-api.example.com",
                "api_key": "test-api-key",
                "model": "test-model"
            },
            "search": {
                "engines": ["test-engine1", "test-engine2"],
                "user_agent": "Test User Agent"
            }
        }
        
        # Write test config to file
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, "w") as f:
            json.dump(self.test_config, f)
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    @patch.dict(os.environ, {}, clear=True)
    def test_load_config_from_file(self):
        """Test loading configuration from file."""
        # We need to patch as an instance attribute, not a class attribute
        with patch('shandu.config.os.path.expanduser', return_value=self.config_path):
            config = Config()
            
            # Check if config was loaded correctly
            self.assertEqual(config.get("api", "base_url"), "https://test-api.example.com")
            self.assertEqual(config.get("api", "api_key"), "test-api-key")
            self.assertEqual(config.get("api", "model"), "test-model")
            self.assertEqual(config.get("search", "engines"), ["test-engine1", "test-engine2"])
            self.assertEqual(config.get("search", "user_agent"), "Test User Agent")
    
    @patch.dict(os.environ, {
        "OPENAI_API_BASE": "https://env-api.example.com",
        "OPENAI_API_KEY": "env-api-key",
        "OPENAI_MODEL_NAME": "env-model",
        "SHANDU_PROXY": "http://proxy.example.com",
        "USER_AGENT": "Env User Agent"
    })
    def test_load_config_from_env(self):
        """Test loading configuration from environment variables."""
        with patch('shandu.config.os.path.expanduser', return_value=self.config_path):
            config = Config()
            
            # Check if environment variables override file config
            self.assertEqual(config.get("api", "base_url"), "https://env-api.example.com")
            self.assertEqual(config.get("api", "api_key"), "env-api-key")
            self.assertEqual(config.get("api", "model"), "env-model")
            self.assertEqual(config.get("scraper", "proxy"), "http://proxy.example.com")
            self.assertEqual(config.get("search", "user_agent"), "Env User Agent")
    
    def test_set_and_save(self):
        """Test setting and saving configuration."""
        with patch('shandu.config.os.path.expanduser', return_value=self.config_path):
            config = Config()
            
            # Set new values
            config.set("api", "model", "new-model")
            config.set("search", "max_results", 20)
            
            # Save config
            config.save()
            
            # Load config again to check if values were saved
            with patch('shandu.config.os.path.expanduser', return_value=self.config_path):
                new_config = Config()
                
                self.assertEqual(new_config.get("api", "model"), "new-model")
                self.assertEqual(new_config.get("search", "max_results"), 20)
    
    def test_get_with_default(self):
        """Test getting configuration with default value."""
        with patch('shandu.config.os.path.expanduser', return_value=self.config_path):
            config = Config()
            
            # Get existing value
            self.assertEqual(config.get("api", "model"), "test-model")
            
            # Get non-existing value with default
            self.assertEqual(config.get("api", "non_existing", "default-value"), "default-value")
            
            # Get non-existing section with default
            self.assertEqual(config.get("non_existing", "key", "default-value"), "default-value")

if __name__ == "__main__":
    unittest.main()
