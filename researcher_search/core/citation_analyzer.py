"""Citation metrics calculation including global and local h-index."""

import logging
import math
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple
import numpy as np

from ..core.models import WorkRaw, AuthorMaster, LocalCitations
from .normalization import RobustNormalizer, NormalizationMethod

logger = logging.getLogger(__name__)


class CitationAnalyzer:
    """Analyzes citation metrics for researchers."""
    
    def __init__(self, exclude_self_citations: bool = False,
                 normalization_method: NormalizationMethod = NormalizationMethod.ROBUST_Z):
        """Initialize citation analyzer.
        
        Args:
            exclude_self_citations: If True, exclude self-citations from local h-index calculation
            normalization_method: Method to use for normalizing citation metrics
        """
        self.exclude_self_citations = exclude_self_citations
        self.normalization_method = normalization_method
        self.normalizer = RobustNormalizer(normalization_method)
        self.work_author_sets = {}  # Cache of author sets per work
    
    def calculate_all_citation_metrics(self, 
                                     works: List[WorkRaw],
                                     authors: Dict[str, AuthorMaster], 
                                     local_citations: LocalCitations) -> Dict[str, Dict[str, float]]:
        """Calculate all citation metrics for authors."""
        logger.info("Calculating citation metrics for all authors")
        
        # Pre-calculate author sets for self-citation detection
        if self.exclude_self_citations:
            self._precompute_author_sets(works)
            logger.info("Pre-computed author sets for self-citation exclusion")
        
        # Group works by author
        author_works = self._group_works_by_author(works)
        
        citation_metrics = {}
        
        for author_id, author_work_list in author_works.items():
            metrics = self._calculate_author_citation_metrics(
                author_id, author_work_list, authors, local_citations, works
            )
            citation_metrics[author_id] = metrics
        
        return citation_metrics
    
    def _group_works_by_author(self, works: List[WorkRaw]) -> Dict[str, List[WorkRaw]]:
        """Group works by author ID."""
        author_works = defaultdict(list)
        
        for work in works:
            for authorship in work.authorships:
                if authorship.author_id:
                    author_works[authorship.author_id].append(work)
        
        return author_works
    
    def _calculate_author_citation_metrics(self, 
                                         author_id: str,
                                         works: List[WorkRaw],
                                         authors: Dict[str, AuthorMaster],
                                         local_citations: LocalCitations,
                                         all_works: List[WorkRaw]) -> Dict[str, float]:
        """Calculate citation metrics for a single author."""
        metrics = {}
        
        # Get global h-index from OpenAlex
        author_data = authors.get(author_id)
        if author_data and author_data.summary_stats:
            metrics['h_index_global'] = author_data.summary_stats.h_index or 0
            metrics['i10_index_global'] = author_data.summary_stats.i10_index or 0
            metrics['mean_2yr_citedness'] = author_data.summary_stats.two_yr_mean_citedness or 0.0
        else:
            # Fallback: estimate global h-index from available data
            metrics['h_index_global'] = self._estimate_global_h_index(works)
            metrics['i10_index_global'] = 0
            metrics['mean_2yr_citedness'] = 0.0
        
        # Calculate local h-index (in-corpus) - both inclusive and exclusive
        metrics['h_index_local_inclusive'] = self._calculate_local_h_index(works, local_citations, exclude_self_citations=False)
        
        if self.exclude_self_citations:
            metrics['h_index_local_exclusive'] = self._calculate_local_h_index_exclusive(author_id, works, local_citations, all_works)
            # Set primary h_index_local to exclusive version when enabled
            metrics['h_index_local'] = metrics['h_index_local_exclusive']
        else:
            metrics['h_index_local_exclusive'] = metrics['h_index_local_inclusive']
            metrics['h_index_local'] = metrics['h_index_local_inclusive']
        
        # Calculate additional metrics
        metrics['n_in_corpus_works'] = len(works)
        metrics['mean_global_citations'] = self._calculate_mean_global_citations(works)
        metrics['total_local_citations'] = self._calculate_total_local_citations(works, local_citations)
        
        return metrics
    
    def _estimate_global_h_index(self, works: List[WorkRaw]) -> int:
        """Estimate global h-index from available citation data."""
        if not works:
            return 0
        
        # Get citation counts and sort in descending order
        citations = [work.cited_by_count for work in works]
        citations.sort(reverse=True)
        
        h_index = 0
        for i, citation_count in enumerate(citations, 1):
            if citation_count >= i:
                h_index = i
            else:
                break
        
        return h_index
    
    def _calculate_local_h_index(self, works: List[WorkRaw], local_citations: LocalCitations, exclude_self_citations: bool = False) -> int:
        """Calculate h-index based on in-corpus citations only."""
        if not works:
            return 0
        
        # Get local citation counts for this author's works
        local_citation_counts = []
        for work in works:
            local_count = local_citations.work_citations.get(work.work_id, 0)
            local_citation_counts.append(local_count)
        
        # Sort in descending order
        local_citation_counts.sort(reverse=True)
        
        # Calculate h-index
        h_index = 0
        for i, citation_count in enumerate(local_citation_counts, 1):
            if citation_count >= i:
                h_index = i
            else:
                break
        
        return h_index
    
    def _calculate_local_h_index_exclusive(self, author_id: str, works: List[WorkRaw], 
                                         local_citations: LocalCitations, all_works: List[WorkRaw]) -> int:
        """Calculate h-index excluding self-citations from local citation counts.
        
        Args:
            author_id: ID of the author whose h-index is being calculated
            works: List of works by this author
            local_citations: Original local citations (including self-citations)
            all_works: All works in corpus for self-citation detection
        """
        if not works:
            return 0
        
        # Get local citation counts excluding self-citations
        local_citation_counts = []
        for work in works:
            # Get original citation count
            original_count = local_citations.work_citations.get(work.work_id, 0)
            
            # Calculate self-citation count for this work
            self_citation_count = self._count_self_citations(work.work_id, author_id, all_works)
            
            # Subtract self-citations (but don't go below 0)
            exclusive_count = max(0, original_count - self_citation_count)
            local_citation_counts.append(exclusive_count)
        
        # Sort in descending order
        local_citation_counts.sort(reverse=True)
        
        # Calculate h-index
        h_index = 0
        for i, citation_count in enumerate(local_citation_counts, 1):
            if citation_count >= i:
                h_index = i
            else:
                break
        
        return h_index
    
    def _calculate_mean_global_citations(self, works: List[WorkRaw]) -> float:
        """Calculate mean global citations (log-transformed)."""
        if not works:
            return 0.0
        
        # Use log1p transformation to handle skewed citation distributions
        log_citations = [math.log1p(work.cited_by_count) for work in works]
        return sum(log_citations) / len(log_citations)
    
    def _calculate_total_local_citations(self, works: List[WorkRaw], local_citations: LocalCitations) -> int:
        """Calculate total local citations received by author's works."""
        total = 0
        for work in works:
            total += local_citations.work_citations.get(work.work_id, 0)
        return total
    
    def calculate_leadership_metrics(self, works: List[WorkRaw]) -> Dict[str, Dict[str, float]]:
        """Calculate leadership-related metrics for authors."""
        logger.debug("Calculating leadership metrics")
        
        author_leadership = defaultdict(lambda: {
            'first_author_count': 0,
            'last_author_count': 0,
            'corresponding_count': 0,
            'total_works': 0,
            'leadership_rate': 0.0
        })
        
        # Count leadership positions for each author
        for work in works:
            for authorship in work.authorships:
                if not authorship.author_id:
                    continue
                
                author_id = authorship.author_id
                author_leadership[author_id]['total_works'] += 1
                
                # Check for first/last author positions
                if authorship.author_position:
                    if authorship.author_position.value == 'first':
                        author_leadership[author_id]['first_author_count'] += 1
                    elif authorship.author_position.value == 'last':
                        author_leadership[author_id]['last_author_count'] += 1
                
                # Check for corresponding author
                if authorship.is_corresponding:
                    author_leadership[author_id]['corresponding_count'] += 1
        
        # Calculate leadership rates
        for author_id, metrics in author_leadership.items():
            total_works = metrics['total_works']
            if total_works > 0:
                leadership_positions = (
                    metrics['first_author_count'] + 
                    metrics['last_author_count'] + 
                    metrics['corresponding_count']
                )
                metrics['leadership_rate'] = leadership_positions / total_works
            
        return dict(author_leadership)
    
    def normalize_citation_metrics(self, citation_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Normalize citation metrics across all authors using robust methods."""
        if not citation_metrics:
            return {}
        
        # Use the unified normalization utility with method selection per metric
        method_overrides = {}
        
        # Collect values to analyze distribution characteristics
        metric_values = defaultdict(list)
        for author_metrics in citation_metrics.values():
            for metric_name, value in author_metrics.items():
                if isinstance(value, (int, float)) and not math.isnan(value):
                    metric_values[metric_name].append(value)
        
        # Recommend specialized methods for known heavy-tailed metrics
        for metric_name, values in metric_values.items():
            if 'h_index' in metric_name or 'cited_by_count' in metric_name:
                # Citation metrics are typically heavy-tailed
                if len(values) > 10:  # Only if we have enough data
                    recommended = RobustNormalizer.recommend_normalization_method(values)
                    if recommended == NormalizationMethod.RANKIT:
                        method_overrides[metric_name] = NormalizationMethod.RANKIT
                        logger.debug(f"Using rankit normalization for heavy-tailed {metric_name}")
        
        # Apply unified normalization
        normalized_metrics = self.normalizer.normalize_metrics_dict(
            citation_metrics, method_overrides
        )
        
        return normalized_metrics
    
    def get_citation_statistics(self, citation_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Calculate descriptive statistics for citation metrics."""
        if not citation_metrics:
            return {}
        
        # Collect all values for each metric
        metric_values = defaultdict(list)
        
        for author_metrics in citation_metrics.values():
            for metric_name, value in author_metrics.items():
                if isinstance(value, (int, float)) and not math.isnan(value):
                    metric_values[metric_name].append(value)
        
        # Calculate statistics
        stats = {}
        for metric_name, values in metric_values.items():
            if values:
                stats[metric_name] = {
                    'count': len(values),
                    'mean': np.mean(values),
                    'median': np.median(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'q25': np.percentile(values, 25),
                    'q75': np.percentile(values, 75)
                }
        
        return stats
    
    def _precompute_author_sets(self, works: List[WorkRaw]):
        """Pre-compute author sets for each work to optimize self-citation detection."""
        logger.debug("Pre-computing author sets for self-citation detection")
        
        for work in works:
            # Extract all author IDs from this work
            author_set = set()
            for authorship in work.authorships:
                if authorship.author_id:
                    author_set.add(authorship.author_id)
            
            self.work_author_sets[work.work_id] = author_set
        
        logger.debug(f"Pre-computed author sets for {len(works)} works")
    
    def _count_self_citations(self, cited_work_id: str, author_id: str, all_works: List[WorkRaw]) -> int:
        """Count how many times the cited work is cited by works involving the same author.
        
        Args:
            cited_work_id: ID of the work being cited
            author_id: ID of the author whose self-citations we're counting
            all_works: All works in the corpus
            
        Returns:
            Number of self-citations to this work by this author
        """
        if not self.work_author_sets:
            logger.warning("Author sets not pre-computed. Self-citation detection may be slow.")
            return 0
        
        # Get author set for the cited work
        cited_work_authors = self.work_author_sets.get(cited_work_id, set())
        
        if author_id not in cited_work_authors:
            # Author is not on the cited work, so no self-citations possible
            return 0
        
        self_citation_count = 0
        
        # Check all works in corpus for citations to the cited work
        for citing_work in all_works:
            # Skip the cited work itself
            if citing_work.work_id == cited_work_id:
                continue
            
            # Check if this work cites the target work
            if cited_work_id in citing_work.referenced_work_ids:
                # Check if there's author overlap (indicating self-citation)
                citing_work_authors = self.work_author_sets.get(citing_work.work_id, set())
                
                # If the author appears in both works, it's a self-citation
                if author_id in citing_work_authors:
                    self_citation_count += 1
        
        return self_citation_count