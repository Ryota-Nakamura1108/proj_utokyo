"""Centrality metrics calculation for coauthorship networks."""

import logging
from typing import Dict, List, Optional
import networkx as nx
import numpy as np
from scipy import stats

from .normalization import RobustNormalizer, NormalizationMethod

logger = logging.getLogger(__name__)


class CentralityAnalyzer:
    """Calculates various centrality metrics for coauthorship networks."""
    
    def __init__(self, seed: int = 42, 
                 normalization_method: NormalizationMethod = NormalizationMethod.ROBUST_Z):
        """Initialize analyzer with random seed and normalization method.
        
        Args:
            seed: Random seed for reproducibility
            normalization_method: Method to use for normalizing centrality values
        """
        self.seed = seed
        self.normalization_method = normalization_method
        self.normalizer = RobustNormalizer(normalization_method)
        np.random.seed(seed)
    
    def calculate_all_centralities(self, graph: nx.Graph) -> Dict[str, Dict[str, float]]:
        """Calculate all centrality metrics for the network."""
        logger.info("Calculating centrality metrics")
        
        if graph.number_of_nodes() == 0:
            return {}
        
        centralities = {}
        
        # Weighted degree (strength)
        logger.debug("Calculating weighted degree centrality")
        centralities['degree_weighted'] = self._calculate_weighted_degree(graph)
        
        # PageRank (weighted)
        logger.debug("Calculating PageRank centrality")
        centralities['pagerank'] = self._calculate_pagerank(graph)
        
        # Betweenness centrality (weighted)
        logger.debug("Calculating betweenness centrality")
        centralities['betweenness'] = self._calculate_betweenness(graph)
        
        # K-core decomposition
        logger.debug("Calculating k-core values")
        centralities['kcore'] = self._calculate_kcore(graph)
        
        # Normalize all centralities
        logger.debug("Normalizing centrality scores")
        normalized_centralities = self._normalize_centralities(centralities)
        
        return normalized_centralities
    
    def _calculate_weighted_degree(self, graph: nx.Graph) -> Dict[str, float]:
        """Calculate weighted degree centrality (node strength)."""
        degree_centrality = {}
        
        for node in graph.nodes():
            strength = sum(graph[node][neighbor].get('weight', 1.0) 
                          for neighbor in graph.neighbors(node))
            degree_centrality[node] = strength
        
        return degree_centrality
    
    def _calculate_pagerank(self, graph: nx.Graph, alpha: float = 0.85, 
                          max_iter: int = 1000, tol: float = 1e-6) -> Dict[str, float]:
        """Calculate PageRank centrality for weighted undirected graph."""
        try:
            # For undirected graphs, NetworkX converts to bidirectional
            # We'll use the weight parameter to incorporate edge weights
            pagerank = nx.pagerank(
                graph, 
                alpha=alpha,
                max_iter=max_iter,
                tol=tol,
                weight='weight'
            )
            return pagerank
        except nx.NetworkXError as e:
            logger.error(f"PageRank calculation failed: {e}")
            # Fallback to uniform distribution
            n_nodes = graph.number_of_nodes()
            return {node: 1.0/n_nodes for node in graph.nodes()}
    
    def _calculate_betweenness(self, graph: nx.Graph, k: Optional[int] = None) -> Dict[str, float]:
        """Calculate betweenness centrality for weighted graph."""
        try:
            # For large graphs, sample k nodes for approximation
            if graph.number_of_nodes() > 1000 and k is None:
                k = min(1000, graph.number_of_nodes())
                logger.debug(f"Using approximation with k={k} for betweenness centrality")
            
            # Convert weights to distances (shorter path = higher weight)
            # We'll create a copy with distance weights
            distance_graph = graph.copy()
            for u, v, data in distance_graph.edges(data=True):
                weight = data.get('weight', 1.0)
                # Convert weight to distance: higher weight = shorter distance
                distance = 1.0 / max(weight, 1e-8)  # Avoid division by zero
                distance_graph[u][v]['distance'] = distance
            
            betweenness = nx.betweenness_centrality(
                distance_graph,
                weight='distance',
                k=k,
                seed=self.seed
            )
            return betweenness
        except Exception as e:
            logger.error(f"Betweenness centrality calculation failed: {e}")
            # Fallback to zeros
            return {node: 0.0 for node in graph.nodes()}
    
    def _calculate_kcore(self, graph: nx.Graph) -> Dict[str, int]:
        """Calculate k-core decomposition."""
        try:
            # K-core is calculated on unweighted graph
            kcore = nx.core_number(graph)
            return kcore
        except Exception as e:
            logger.error(f"K-core calculation failed: {e}")
            return {node: 1 for node in graph.nodes()}
    
    def _normalize_centralities(self, centralities: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Normalize all centrality measures to 0-1 range using robust methods."""
        normalized = {}
        
        # Analyze distribution characteristics and recommend methods per metric
        method_recommendations = {}
        
        for metric_name, metric_values in centralities.items():
            if not metric_values:
                normalized[metric_name] = {}
                continue
            
            values = list(metric_values.values())
            
            if metric_name == 'kcore':
                # For k-core (discrete), use simple max normalization
                max_val = max(values) if values else 1
                normalized[metric_name] = {
                    node: value / max_val for node, value in metric_values.items()
                }
            else:
                # For continuous metrics, use configurable robust normalization
                # Analyze distribution to recommend method if auto-detection is desired
                if self.normalization_method == NormalizationMethod.ROBUST_Z:
                    # Can override with auto-detection
                    recommended_method = RobustNormalizer.recommend_normalization_method(values)
                    method_recommendations[metric_name] = recommended_method
                else:
                    recommended_method = self.normalization_method
                
                # Use the robust normalizer utility
                normalized_dict = self.normalizer.normalize_dict_values(metric_values, recommended_method)
                normalized[metric_name] = normalized_dict
                
                logger.debug(f"Normalized {metric_name} using {recommended_method.value}")
        
        if method_recommendations:
            logger.info(f"Auto-selected normalization methods: {[(k, v.value) for k, v in method_recommendations.items()]}")
        
        return normalized
    
    def get_normalization_statistics(self, centralities: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Get detailed normalization statistics for each centrality metric.
        
        Returns:
            Dictionary with distribution statistics per metric
        """
        stats_dict = {}
        
        for metric_name, metric_values in centralities.items():
            if not metric_values:
                continue
            
            values = list(metric_values.values())
            stats_dict[metric_name] = self.normalizer.calculate_normalization_stats(values)
        
        return stats_dict
    
    def calculate_centrality_statistics(self, centralities: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Calculate statistics for each centrality metric."""
        stats_dict = {}
        
        for metric_name, metric_values in centralities.items():
            if not metric_values:
                continue
            
            values = list(metric_values.values())
            
            stats_dict[metric_name] = {
                'mean': np.mean(values),
                'median': np.median(values), 
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'q25': np.percentile(values, 25),
                'q75': np.percentile(values, 75)
            }
        
        return stats_dict
    
    def identify_central_nodes(self, centralities: Dict[str, Dict[str, float]], 
                             top_k: int = 10) -> Dict[str, List[str]]:
        """Identify top-k central nodes for each metric."""
        top_nodes = {}
        
        for metric_name, metric_values in centralities.items():
            if not metric_values:
                top_nodes[metric_name] = []
                continue
            
            # Sort by centrality value (descending)
            sorted_nodes = sorted(metric_values.items(), 
                                key=lambda x: x[1], reverse=True)
            
            top_nodes[metric_name] = [node for node, _ in sorted_nodes[:top_k]]
        
        return top_nodes
    
    def get_node_centralities(self, node_id: str, 
                            centralities: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Get all centrality values for a specific node."""
        node_centralities = {}
        
        for metric_name, metric_values in centralities.items():
            node_centralities[metric_name] = metric_values.get(node_id, 0.0)
        
        return node_centralities