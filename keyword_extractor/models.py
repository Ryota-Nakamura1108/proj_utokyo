"""Data models for keyword extractor."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass
class Paper:
    """Represents a research paper."""

    id: str
    title: str
    abstract: Optional[str] = None
    publication_year: Optional[int] = None
    cited_by_count: int = 0
    author_position: Optional[str] = None  # "first", "middle", "last"
    journal_name: Optional[str] = None
    topics: List[str] = field(default_factory=list)
    doi: Optional[str] = None
    is_corresponding: bool = False

    def to_embedding_text(self) -> str:
        """Convert paper to text for embedding generation."""
        parts = [f"Title: {self.title}"]

        if self.abstract:
            parts.append(f"Abstract: {self.abstract}")

        if self.topics:
            parts.append(f"Topics: {', '.join(self.topics)}")

        if self.journal_name:
            parts.append(f"Journal: {self.journal_name}")

        if self.publication_year:
            parts.append(f"Year: {self.publication_year}")

        return "\n".join(parts)


@dataclass
class Author:
    """Represents a researcher/author."""

    id: str
    display_name: str
    orcid: Optional[str] = None
    h_index: Optional[int] = None
    works_count: int = 0
    cited_by_count: int = 0


@dataclass
class Cluster:
    """Represents a cluster of papers."""

    cluster_id: int
    papers: List[Paper] = field(default_factory=list)
    theme: Optional[str] = None
    representative_papers: List[Paper] = field(default_factory=list)
    centroid: Optional[List[float]] = None


@dataclass
class Keyword:
    """Represents an extracted keyword."""

    keyword: str
    relevance_score: float
    cluster_ids: List[int] = field(default_factory=list)
    frequency: float = 0.0
    trend: Optional[str] = None  # "increasing", "stable", "decreasing"


@dataclass
class ExtractionResult:
    """Final keyword extraction result."""

    researcher_id: str
    researcher_name: str
    analysis_period: Dict[str, int]
    statistics: Dict[str, Any]
    keywords: List[Keyword]
    summary: Optional[str] = None
    clusters: Optional[List[Cluster]] = None
    method: str = "embedding_clustering_llm"
    status: str = "success"
    processing_time: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "researcher_id": self.researcher_id,
            "researcher_name": self.researcher_name,
            "analysis_period": self.analysis_period,
            "statistics": self.statistics,
            "keywords": [
                {
                    "keyword": kw.keyword,
                    "relevance_score": kw.relevance_score,
                    "cluster_ids": kw.cluster_ids,
                    "frequency": kw.frequency,
                    "trend": kw.trend,
                }
                for kw in self.keywords
            ],
            "summary": self.summary,
            "method": self.method,
            "status": self.status,
            "processing_time": self.processing_time,
            "timestamp": self.timestamp,
        }

        if self.clusters:
            result["clusters"] = [
                {
                    "cluster_id": c.cluster_id,
                    "theme": c.theme,
                    "size": len(c.papers),
                    "representative_papers": [
                        {
                            "title": p.title,
                            "year": p.publication_year,
                            "citations": p.cited_by_count,
                            "journal": p.journal_name,
                        }
                        for p in c.representative_papers
                    ],
                }
                for c in self.clusters
            ]

        return result


@dataclass
class FilterConfig:
    """Configuration for paper filtering."""

    years_back: int = 10
    author_positions: List[str] = field(default_factory=lambda: ["first", "last"])
    min_citations: int = 0
    max_papers: int = 200
    paper_types: List[str] = field(default_factory=lambda: ["article", "review"])
