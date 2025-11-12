"""Composite scoring system (CRS) for central researcher identification."""

import logging
import math
from typing import Dict, List, Tuple
import numpy as np

from ..core.models import AuthorFeatures, ResearcherRanking

logger = logging.getLogger(__name__)


class CentralResearcherScorer:
    """Composite scoring system for ranking central researchers."""
    
    def __init__(self, 
                 tau_shrinkage: int = 5,
                 min_in_corpus_works: int = 2,
                 centrality_weight: float = 0.50,
                 citation_weight: float = 0.35, 
                 leadership_weight: float = 0.15):
        """Initialize scorer with weighting parameters.
        
        Args:
            tau_shrinkage: Sample size shrinkage parameter
            min_in_corpus_works: Minimum works required for ranking
            centrality_weight: Weight for centrality score component
            citation_weight: Weight for citation score component  
            leadership_weight: Weight for leadership score component
        """
        self.tau_shrinkage = tau_shrinkage
        self.min_in_corpus_works = min_in_corpus_works
        self.centrality_weight = centrality_weight
        self.citation_weight = citation_weight
        self.leadership_weight = leadership_weight
        
        # Validate weights sum to 1
        total_weight = centrality_weight + citation_weight + leadership_weight
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
    
    def calculate_composite_scores(self,
                                 centralities: Dict[str, Dict[str, float]],
                                 citations: Dict[str, Dict[str, float]],
                                 leadership: Dict[str, Dict[str, float]],
                                 kcore_values: Dict[str, int]) -> Dict[str, AuthorFeatures]:
        """Calculate composite CRS scores for all authors."""
        logger.info("Calculating composite CRS scores")

        # Collect all author IDs
        all_authors = set()
        # centralities is Dict[metric_name, Dict[author_id, value]]
        # Extract author IDs from each metric's dict
        for metric_dict in centralities.values():
            all_authors.update(metric_dict.keys())
        all_authors.update(citations.keys() if citations else [])
        all_authors.update(leadership.keys() if leadership else [])
        
        author_features = {}
        
        for author_id in all_authors:
            features = self._calculate_author_scores(
                author_id, centralities, citations, leadership, kcore_values
            )
            
            # Only include authors meeting minimum criteria
            if features.n_in_corpus_works >= self.min_in_corpus_works:
                author_features[author_id] = features
        
        logger.info(f"Calculated scores for {len(author_features)} qualifying authors")
        return author_features
    
    def _calculate_author_scores(self,
                               author_id: str,
                               centralities: Dict[str, Dict[str, float]],
                               citations: Dict[str, Dict[str, float]],
                               leadership: Dict[str, Dict[str, float]],
                               kcore_values: Dict[str, int]) -> AuthorFeatures:
        """Calculate all scores for a single author."""
        features = AuthorFeatures(author_id=author_id)

        # Extract basic metrics
        # centralities is Dict[metric_name, Dict[author_id, value]]
        citation_data = citations.get(author_id, {})
        leadership_data = leadership.get(author_id, {})

        # Centrality metrics - extract from each metric's dict
        features.deg_w = centralities.get('degree_weighted', {}).get(author_id, 0.0)
        features.pagerank = centralities.get('pagerank', {}).get(author_id, 0.0)
        features.betweenness = centralities.get('betweenness', {}).get(author_id, 0.0)
        features.kcore = kcore_values.get(author_id, 0)
        
        # Citation metrics
        features.h_index_global = citation_data.get('h_index_global', 0)
        features.h_index_local = citation_data.get('h_index_local', 0)
        features.h_index_local_inclusive = citation_data.get('h_index_local_inclusive', 0)
        features.h_index_local_exclusive = citation_data.get('h_index_local_exclusive', 0)
        features.i10_index_global = citation_data.get('i10_index_global', 0)
        features.mean_2yr_citedness = citation_data.get('mean_2yr_citedness', 0.0)
        
        # Leadership metrics
        features.leadership_rate = leadership_data.get('leadership_rate', 0.0)
        features.n_in_corpus_works = leadership_data.get('total_works', 0)
        
        # Calculate composite sub-scores
        features.centrality_score = self._calculate_centrality_score(features)
        features.citation_score = self._calculate_citation_score(features)
        features.leadership_score = self._calculate_leadership_score(features)
        
        # Calculate raw CRS
        features.crs_raw = (
            self.centrality_weight * features.centrality_score +
            self.citation_weight * features.citation_score + 
            self.leadership_weight * features.leadership_score
        )
        
        # Apply sample size shrinkage
        features.crs_final = self._apply_sample_size_shrinkage(
            features.crs_raw, features.n_in_corpus_works
        )
        
        return features
    
    def _calculate_centrality_score(self, features: AuthorFeatures) -> float:
        """Calculate centrality sub-score.
        
        Formula: 0.4 * PageRank + 0.4 * WeightedDegree + 0.2 * Betweenness
        """
        score = (
            0.4 * features.pagerank +
            0.4 * features.deg_w + 
            0.2 * features.betweenness
        )
        return max(0.0, min(1.0, score))  # Clamp to [0, 1]
    
    def _calculate_citation_score(self, features: AuthorFeatures) -> float:
        """Calculate citation sub-score.
        
        Formula: 0.6 * h_index_global + 0.4 * h_index_local
        Both h-indices should be normalized beforehand.
        """
        # For h-index normalization, we assume they've been pre-normalized
        # by the citation analyzer to 0-1 scale
        global_component = 0.6 * (features.h_index_global or 0)
        local_component = 0.4 * features.h_index_local
        
        score = global_component + local_component
        return max(0.0, min(1.0, score))  # Clamp to [0, 1]
    
    def _calculate_leadership_score(self, features: AuthorFeatures) -> float:
        """Calculate leadership sub-score.
        
        Formula: 0.5 * leadership_rate + 0.5 * kcore_normalized
        """
        # Normalize k-core by assuming max possible value
        # This should ideally be done globally, but we'll use a reasonable max
        max_kcore_estimate = 20  # Reasonable upper bound for most networks
        kcore_normalized = min(1.0, features.kcore / max_kcore_estimate)
        
        score = (
            0.5 * features.leadership_rate +
            0.5 * kcore_normalized
        )
        return max(0.0, min(1.0, score))  # Clamp to [0, 1]
    
    def _apply_sample_size_shrinkage(self, raw_score: float, n_works: int) -> float:
        """Apply sample size shrinkage to prevent over-confidence in small samples.
        
        Formula: CRS_raw * (1 - exp(-n_works/tau))
        """
        if n_works <= 0:
            return 0.0
        
        shrinkage_factor = 1.0 - math.exp(-n_works / self.tau_shrinkage)
        return raw_score * shrinkage_factor
    
    def create_rankings(self, 
                       author_features: Dict[str, AuthorFeatures],
                       authors_master: Dict[str, 'AuthorMaster']) -> List[ResearcherRanking]:
        """Create final researcher rankings sorted by CRS score."""
        logger.info(f"Creating rankings for {len(author_features)} researchers")
        
        # Sort by CRS final score (descending)
        sorted_authors = sorted(
            author_features.items(),
            key=lambda x: x[1].crs_final,
            reverse=True
        )
        
        rankings = []
        for rank, (author_id, features) in enumerate(sorted_authors, 1):
            author_data = authors_master.get(author_id, None)
            
            ranking = ResearcherRanking(
                rank=rank,
                author_id=author_id,
                author_display_name=author_data.display_name if author_data else "Unknown",
                orcid=author_data.orcid if author_data else None,
                crs_final=features.crs_final,
                crs_raw=features.crs_raw,
                centrality_deg=features.deg_w,
                centrality_pagerank=features.pagerank,
                centrality_betweenness=features.betweenness,
                h_index_global=features.h_index_global,
                h_index_local=features.h_index_local,
                h_index_local_inclusive=features.h_index_local_inclusive,
                h_index_local_exclusive=features.h_index_local_exclusive,
                leadership_rate=features.leadership_rate,
                n_in_corpus_works=features.n_in_corpus_works
            )
            rankings.append(ranking)
        
        return rankings
    
    def get_score_statistics(self, author_features: Dict[str, AuthorFeatures]) -> Dict[str, Dict[str, float]]:
        """Calculate descriptive statistics for all score components."""
        if not author_features:
            return {}
        
        # Collect score components
        scores = {
            'crs_final': [f.crs_final for f in author_features.values()],
            'crs_raw': [f.crs_raw for f in author_features.values()],
            'centrality_score': [f.centrality_score for f in author_features.values()],
            'citation_score': [f.citation_score for f in author_features.values()],
            'leadership_score': [f.leadership_score for f in author_features.values()],
            'n_works': [f.n_in_corpus_works for f in author_features.values()]
        }
        
        # Calculate statistics
        stats = {}
        for score_name, values in scores.items():
            if values:
                stats[score_name] = {
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
    
    def normalize_h_indices_globally(self, 
                                   citation_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Normalize h-indices globally across all authors for fair comparison."""
        if not citation_metrics:
            return {}
        
        # Collect all h-index values
        global_h_values = []
        local_h_values = []
        
        for metrics in citation_metrics.values():
            if metrics.get('h_index_global') is not None:
                global_h_values.append(metrics['h_index_global'])
            if metrics.get('h_index_local') is not None:
                local_h_values.append(metrics['h_index_local'])
        
        # Calculate normalization parameters
        global_max = max(global_h_values) if global_h_values else 1
        local_max = max(local_h_values) if local_h_values else 1
        
        # Ensure we don't divide by zero
        global_max = max(global_max, 1)
        local_max = max(local_max, 1)
        
        # Normalize
        normalized = {}
        for author_id, metrics in citation_metrics.items():
            normalized_metrics = metrics.copy()
            
            if metrics.get('h_index_global') is not None:
                normalized_metrics['h_index_global'] = metrics['h_index_global'] / global_max
            
            if metrics.get('h_index_local') is not None:
                normalized_metrics['h_index_local'] = metrics['h_index_local'] / local_max
                
            normalized[author_id] = normalized_metrics
        
        return normalized