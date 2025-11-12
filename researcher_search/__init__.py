"""Central Researcher identification system using OpenAlex API and coauthorship networks.

This module implements a comprehensive system for identifying central researchers
in a given field based on a list of representative papers. It uses the OpenAlex
API to gather paper and author data, constructs coauthorship networks, and 
calculates various centrality and citation metrics to rank researchers.

Main components:
- OpenAlex API client with rate limiting and batching
- Coauthorship network construction with edge weighting
- Multiple centrality metrics (PageRank, betweenness, degree, k-core)
- Citation-based metrics (global and local h-index)
- Composite scoring system (CRS - Central Researcher Score)
"""

from .core import CentralResearcher, InputPaper, ResearcherRanking, CentralResearcherConfig

__all__ = [
    "CentralResearcher",
    "InputPaper", 
    "ResearcherRanking",
    "CentralResearcherConfig",
]

__version__ = "1.0.0"