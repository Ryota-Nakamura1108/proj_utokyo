"""Data models for the Central Researcher identification system."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from enum import Enum


class AuthorPosition(Enum):
    """Author position in publication."""
    FIRST = "first"
    MIDDLE = "middle" 
    LAST = "last"


@dataclass
class InputPaper:
    """Input paper specification."""
    doi: Optional[str] = None
    openalex_id: Optional[str] = None
    pmid: Optional[str] = None
    title: Optional[str] = None
    
    def __post_init__(self):
        if not any([self.doi, self.openalex_id, self.pmid, self.title]):
            raise ValueError("At least one identifier must be provided")


@dataclass
class CentralResearcherConfig:
    """Configuration for central researcher analysis."""
    papers: List[InputPaper]
    year_range: Optional[List[int]] = None  # [start_year, end_year]
    decay_lambda: float = 0.15
    first_last_bonus: float = 0.25
    corr_author_bonus: float = 0.25
    min_in_corpus_works: int = 2
    tau_shrinkage: int = 5
    email: str = "ysato@memorylab.jp"
    # Topic similarity features
    topic_similarity_enabled: bool = True
    topic_similarity_threshold: float = 0.1
    # Consortium suppression features  
    consortium_suppression_enabled: bool = True
    consortium_threshold: int = 50
    # Alphabetical authorship detection features
    alphabetical_detection_enabled: bool = True
    alphabetical_threshold: float = 0.8
    # Self-citation exclusion features
    exclude_self_citations: bool = False
    # Simmelian tie strengthening features
    # simmelian_strengthening_enabled: bool = True
    # Robust normalization features
    robust_normalization_enabled: bool = True
    normalization_method: str = "robust_z"  # "robust_z", "rankit", "standard"


@dataclass
class Authorship:
    """Author information for a work."""
    author_id: Optional[str] = None
    orcid: Optional[str] = None
    raw_name: str = ""
    author_position: Optional[AuthorPosition] = None
    is_corresponding: bool = False
    institution_ids: List[str] = field(default_factory=list)


@dataclass
class Topic:
    """Topic information from OpenAlex."""
    topic_id: str
    display_name: str = ""
    score: float = 0.0


@dataclass
class WorkRaw:
    """Raw work data from OpenAlex."""
    work_id: str
    doi: Optional[str] = None
    title: str = ""
    year: Optional[int] = None
    cited_by_count: int = 0
    referenced_work_ids: List[str] = field(default_factory=list)
    authorships: List[Authorship] = field(default_factory=list)
    topics: List[Topic] = field(default_factory=list)
    primary_topic: Optional[Topic] = None
    is_authors_truncated: bool = False


@dataclass
class SummaryStats:
    """Author summary statistics from OpenAlex."""
    h_index: Optional[int] = None
    i10_index: Optional[int] = None
    two_yr_mean_citedness: Optional[float] = None


@dataclass 
class InstitutionMaster:
    """Master institution data from OpenAlex."""
    institution_id: str
    display_name: str = ""
    country_code: Optional[str] = None
    type: Optional[str] = None
    ror: Optional[str] = None
    homepage_url: Optional[str] = None
    works_count: int = 0


@dataclass
class AuthorMaster:
    """Master author data from OpenAlex."""
    author_id: str
    display_name: str = ""
    orcid: Optional[str] = None
    summary_stats: Optional[SummaryStats] = None
    works_count: int = 0
    last_known_institution_ids: List[str] = field(default_factory=list)


@dataclass
class GraphEdge:
    """Graph edge between authors."""
    src_author_id: str
    dst_author_id: str
    w_freq: float = 0.0  # frequency-based weight
    w_recency: float = 0.0  # recency-weighted sum
    w_role: float = 0.0  # role contribution sum
    w_total: float = 0.0  # final total weight


@dataclass
class AuthorFeatures:
    """Computed features for an author."""
    author_id: str
    n_in_corpus_works: int = 0
    leadership_rate: float = 0.0
    
    # Centrality metrics
    deg_w: float = 0.0
    pagerank: float = 0.0
    betweenness: float = 0.0
    kcore: int = 0
    
    # Citation metrics
    h_index_global: Optional[int] = None
    h_index_local: int = 0
    h_index_local_inclusive: int = 0
    h_index_local_exclusive: int = 0
    i10_index_global: Optional[int] = None
    mean_2yr_citedness: Optional[float] = None
    
    # Composite scores
    centrality_score: float = 0.0
    citation_score: float = 0.0
    leadership_score: float = 0.0
    crs_raw: float = 0.0
    crs_final: float = 0.0


@dataclass
class ResearcherRanking:
    """Final researcher ranking output."""
    rank: int
    author_display_name: str
    author_id: str
    orcid: Optional[str] = None
    crs_final: float = 0.0
    crs_raw: float = 0.0
    centrality_deg: float = 0.0
    centrality_pagerank: float = 0.0
    centrality_betweenness: float = 0.0
    h_index_global: Optional[int] = None
    h_index_local: int = 0
    h_index_local_inclusive: int = 0
    h_index_local_exclusive: int = 0
    leadership_rate: float = 0.0
    n_in_corpus_works: int = 0


@dataclass
class LocalCitations:
    """Local citation counts within the corpus."""
    work_citations: Dict[str, int] = field(default_factory=dict)  # work_id -> in-corpus citation count