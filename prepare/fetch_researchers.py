#!/usr/bin/env python
"""Fetch researchers list from OpenAlex API.

東京大学の研究者リストをOpenAlex APIから取得します。

Usage:
    # Basic usage (default: min_works=5)
    python fetch_researchers.py

    # With custom min_works
    python fetch_researchers.py --min-works 10

    # Force refresh cache
    python fetch_researchers.py --force-refresh

    # Custom institution ID
    python fetch_researchers.py --institution-id I136199984
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    from prepare.researcher_manager import main
    sys.exit(main())
