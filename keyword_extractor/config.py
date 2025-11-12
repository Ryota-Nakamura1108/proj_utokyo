"""Configuration module for keyword extractor.

Handles environment variable loading and configuration management.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load .env file from project root
_project_root = Path(__file__).parent.parent.parent
_env_file = _project_root / ".env"

if _env_file.exists():
    load_dotenv(_env_file)
    print(f"✅ Loaded environment variables from {_env_file}")
else:
    print(f"⚠️  .env file not found at {_env_file}")


class Config:
    """Configuration class for keyword extractor."""

    def __init__(self):
        """Initialize configuration from environment variables."""
        self.openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
        self.openalex_email: Optional[str] = os.getenv("OPENALEX_EMAIL", "ysato@memorylab.jp")
        self.default_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.max_keywords: int = int(os.getenv("MAX_KEYWORDS", "10"))

    def validate(self) -> bool:
        """Validate that required configuration is present.

        Returns:
            bool: True if valid, False otherwise
        """
        if not self.openai_api_key:
            print("❌ Error: OPENAI_API_KEY not found in environment variables")
            return False

        return True


# Global config instance
config = Config()
