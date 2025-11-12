"""Entry point for running keyword_extractor as a module.

This allows the module to be run with: python -m keyword_extractor
or with uv: uv run keyword_extractor
"""

from .cli import main

if __name__ == "__main__":
    exit(main())
