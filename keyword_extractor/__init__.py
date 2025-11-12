"""Keyword Extractor Module for Researchers.

This module provides functionality to extract keywords for researchers
based on their OpenAlex ID or name.
"""

from .core import extract_keywords, extract_keywords_by_name, extract_keywords_by_id

__version__ = "0.1.0"

__all__ = [
    "extract_keywords",
    "extract_keywords_by_name",
    "extract_keywords_by_id",
]
