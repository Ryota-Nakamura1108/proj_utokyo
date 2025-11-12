"""Configuration module for common_module.

Handles environment variable loading and configuration management
for all submodules (keyword_extractor, company_search, openalex_loader, etc).
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv


class Config:
    """Configuration class for common_module."""

    def __init__(self):
        """Initialize configuration from environment variables."""
        _project_root = Path(__file__).resolve().parent.parent.parent
        _env_file = _project_root / ".env"
        if _env_file.exists():
            load_dotenv(_env_file)
            print(f"✅ Loaded environment variables from {_env_file}")
        else:
            print(f"⚠️  .env file not found at {_env_file}")

        # OpenAI Configuration
        self.openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
        self.default_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        # OpenAlex Configuration
        self.openalex_email: Optional[str] = os.getenv(
            "OPENALEX_EMAIL", "rnakamura@memorylab.jp"
        )

        # Keyword Extractor Configuration
        self.max_keywords: int = int(os.getenv("MAX_KEYWORDS", "10"))
        self.years_back: int = int(os.getenv("YEARS_BACK", "10"))
        self.min_citations: int = int(os.getenv("MIN_CITATIONS", "0"))

        # Company Search Configuration
        self.company_search_model: str = os.getenv(
            "COMPANY_SEARCH_MODEL", self.default_model
        )
        self.company_max_keywords: int = int(os.getenv("COMPANY_MAX_KEYWORDS", "10"))

        # OpenAlex Loader Configuration
        self.default_institution_id: str = os.getenv(
            "OPENALEX_INSTITUTION_ID", "https://openalex.org/I74801974"
        )  # University of Tokyo
        self.default_batch_size: int = int(os.getenv("BATCH_SIZE", "20000"))

        # AWS Configuration (for openalex_loader S3 operations)
        self.aws_access_key_id: Optional[str] = os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key: Optional[str] = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.aws_region: str = os.getenv("AWS_REGION", "us-east-1")
        self.s3_bucket: Optional[str] = os.getenv("S3_BUCKET")
        self.s3_prefix: str = os.getenv("S3_PREFIX", "papers")


# Global config instance
config = Config()

# Directory paths
RESEARCHER_LIST_DIR = Path(__file__).resolve().parent.parent / "output" / "researcher_list"
RESEARCHER_LIST_DIR.mkdir(parents=True, exist_ok=True)

# Institution IDs
UNIV_TOKYO_ID = "I74801974"  # University of Tokyo OpenAlex ID