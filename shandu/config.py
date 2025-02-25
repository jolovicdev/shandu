"""
Configuration module for Shandu deep research system.
Handles loading and managing configuration settings.
"""
import os
import json
from typing import Dict, Any, Optional
from pathlib import Path
import datetime

DEFAULT_CONFIG = {
    "api": {
        "base_url": "https://api.openai.com/v1",
        "api_key": "",
        "model": "gpt-4",
        "temperature": 0
    },
    "search": {
        "engines": ["duckduckgo", "google"],
        "max_results": 10,
        "region": "wt-wt",
        "safesearch": "moderate",
        "user_agent": "Research 1.0"
    },
    "research": {
        "default_depth": 2,
        "default_breadth": 4,
        "max_depth": 5,
        "max_breadth": 10,
        "max_urls_per_query": 3
    },
    "scraper": {
        "timeout": 30,
        "max_retries": 3,
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "proxy": None
    },
    "display": {
        "verbose": False,
        "show_progress": True,
        "show_chain_of_thought": True
    }
}

class Config:
    """Configuration manager for Shandu."""
    
    def __init__(self):
        self._config = DEFAULT_CONFIG.copy()
        self._config_path = os.path.expanduser("~/.shandu/config.json")
        self._load_config()
        self._load_env_vars()
        
    def _load_config(self):
        """Load configuration from file."""
        config_path = Path(self._config_path)
        if config_path.exists():
            try:
                with open(config_path) as f:
                    file_config = json.load(f)
                    self._update_nested_dict(self._config, file_config)
            except Exception as e:
                print(f"Error loading config file: {e}")
    
    def _load_env_vars(self):
        """Load configuration from environment variables."""
        if os.environ.get("OPENAI_API_BASE"):
            self._config["api"]["base_url"] = os.environ["OPENAI_API_BASE"]
        if os.environ.get("OPENAI_API_KEY"):
            self._config["api"]["api_key"] = os.environ["OPENAI_API_KEY"]
        if os.environ.get("OPENAI_MODEL_NAME"):
            self._config["api"]["model"] = os.environ["OPENAI_MODEL_NAME"]
        
        if os.environ.get("SHANDU_PROXY"):
            self._config["scraper"]["proxy"] = os.environ["SHANDU_PROXY"]
            
        if os.environ.get("USER_AGENT"):
            self._config["search"]["user_agent"] = os.environ["USER_AGENT"]
    
    def _update_nested_dict(self, d: Dict, u: Dict):
        """Update nested dictionary with another dictionary."""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._update_nested_dict(d[k], v)
            else:
                d[k] = v
    
    def save(self):
        """Save configuration to file."""
        config_path = Path(self._config_path)
        config_path.parent.mkdir(exist_ok=True, parents=True)
        with open(config_path, "w") as f:
            json.dump(self._config, f, indent=2)
    
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        try:
            return self._config[section][key]
        except KeyError:
            return default
    
    def set(self, section: str, key: str, value: Any):
        """Set configuration value."""
        if section not in self._config:
            self._config[section] = {}
        self._config[section][key] = value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section."""
        return self._config.get(section, {}).copy()
    
    def get_all(self) -> Dict[str, Any]:
        """Get entire configuration."""
        return self._config.copy()

config = Config()

def get_current_date() -> str:
    """Get current date in ISO format."""
    return datetime.datetime.now().strftime("%Y-%m-%d")

def get_current_datetime() -> str:
    """Get current date and time in human-readable format."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_user_agent() -> str:
    """Get user agent string with fake user agent."""
    configured_agent = config.get("search", "user_agent", None)
    if configured_agent and configured_agent != "Research 1.0":
        return configured_agent
    
    try:
        from fake_useragent import UserAgent
        ua = UserAgent()
        return ua.random
    except ImportError:
        import random
        fake_user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0"
        ]
        return random.choice(fake_user_agents)
    except Exception as e:
        print(f"Error generating user agent: {e}")
        return "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
