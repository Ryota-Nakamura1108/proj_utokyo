"""Explainability utilities for Central Researcher Analysis."""

import logging
import math
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, Counter
from dataclasses import dataclass
import networkx as nx

from .models import WorkRaw, AuthorMaster, GraphEdge, ResearcherRanking, AuthorFeatures

logger = logging.getLogger(__name__)


@dataclass
class EdgeContribution:
    """Contribution of a single collaboration edge to author's CRS."""
    collaborator_id: str
    collaborator_name: str
    edge_weight: float
    shared_papers_count: int
    collaboration_strength: str  # "weak", "moderate", "strong"
    contribution_to_centrality: float
    top_shared_papers: List[str]  # Work IDs of most important shared papers


@dataclass
class PaperContribution:
    """Contribution of a single paper to author's CRS."""
    work_id: str
    title: str
    year: Optional[int]
    author_role: str  # "first", "middle", "last", "corresponding"
    coauthor_count: int
    time_decay_factor: float
    topic_similarity_score: float
    consortium_factor: float
    alphabetical_detected: bool
    contribution_to_leadership: float
    contribution_to_centrality: float
    total_contribution: float


@dataclass
class CRSDecomposition:
    """Detailed decomposition of CRS score components."""
    author_id: str
    author_name: str
    crs_final: float
    crs_raw: float
    # Component scores
    centrality_score: float
    citation_score: float
    leadership_score: float
    # Sub-component contributions
    pagerank_contribution: float
    degree_contribution: float
    betweenness_contribution: float
    h_global_contribution: float
    h_local_contribution: float
    leadership_rate_contribution: float
    kcore_contribution: float
    # Sample size shrinkage
    sample_size_factor: float


class ExplainabilityAnalyzer:
    """Provides detailed explanations for CRS scores and rankings."""
    
    def __init__(self):
        """Initialize explainability analyzer."""
        self.top_edges_cache = {}
        self.top_papers_cache = {}
        self.crs_decomposition_cache = {}
    
    def analyze_top_edges(self, 
                         author_id: str,
                         graph: nx.Graph,
                         edge_data: Dict[str, GraphEdge],
                         authors: Dict[str, AuthorMaster],
                         top_k: int = 3) -> List[EdgeContribution]:
        """Analyze top collaboration edges for an author.
        
        Args:
            author_id: ID of the author to analyze
            graph: Coauthorship network graph
            edge_data: Detailed edge information
            authors: Author master data
            top_k: Number of top edges to return
            
        Returns:
            List of top edge contributions sorted by impact
        """
        if author_id in self.top_edges_cache:
            return self.top_edges_cache[author_id][:top_k]
        
        edge_contributions = []
        
        # Get all edges for this author
        if author_id not in graph:
            return []
        
        neighbors = list(graph.neighbors(author_id))
        
        for neighbor_id in neighbors:
            if not graph.has_edge(author_id, neighbor_id):
                continue
            
            # Get edge weight from graph
            edge_weight = graph[author_id][neighbor_id].get('weight', 0.0)
            
            # Find corresponding edge data
            edge_key_1 = f"{author_id}_{neighbor_id}"
            edge_key_2 = f"{neighbor_id}_{author_id}"
            
            edge_info = edge_data.get(edge_key_1) or edge_data.get(edge_key_2)
            
            if not edge_info:
                continue
            
            # Get collaborator information
            collaborator_data = authors.get(neighbor_id)
            collaborator_name = collaborator_data.display_name if collaborator_data else "Unknown"
            
            # Determine collaboration strength
            if edge_weight > 2.0:
                strength = "strong"
            elif edge_weight > 0.5:
                strength = "moderate"
            else:
                strength = "weak"
            
            # Calculate contribution to centrality (simplified)
            contribution = edge_weight * 0.1  # Rough approximation
            
            edge_contribution = EdgeContribution(
                collaborator_id=neighbor_id,
                collaborator_name=collaborator_name,
                edge_weight=edge_weight,
                shared_papers_count=int(edge_info.w_freq) if edge_info else 1,
                collaboration_strength=strength,
                contribution_to_centrality=contribution,
                top_shared_papers=[]  # Would need additional analysis
            )
            
            edge_contributions.append(edge_contribution)
        
        # Sort by edge weight (most important collaborations first)
        edge_contributions.sort(key=lambda x: x.edge_weight, reverse=True)
        
        # Cache results
        self.top_edges_cache[author_id] = edge_contributions
        
        return edge_contributions[:top_k]
    
    def analyze_top_papers(self,
                          author_id: str,
                          works: List[WorkRaw],
                          author_features: AuthorFeatures,
                          top_k: int = 3) -> List[PaperContribution]:
        """Analyze top paper contributions for an author.
        
        Args:
            author_id: ID of the author to analyze
            works: List of all works in corpus
            author_features: Computed features for the author
            top_k: Number of top papers to return
            
        Returns:
            List of top paper contributions sorted by impact
        """
        if author_id in self.top_papers_cache:
            return self.top_papers_cache[author_id][:top_k]
        
        paper_contributions = []
        author_papers = []
        
        # Find papers by this author
        for work in works:
            for authorship in work.authorships:
                if authorship.author_id == author_id:
                    author_papers.append((work, authorship))
                    break
        
        current_year = 2024  # Could be configurable
        
        for work, authorship in author_papers:
            # Determine author role
            role_map = {
                "first": "first",
                "last": "last", 
                "middle": "middle"
            }
            author_role = role_map.get(authorship.author_position.value if authorship.author_position else "middle", "middle")
            if authorship.is_corresponding:
                author_role += "+corresponding"
            
            # Calculate time decay factor
            years_ago = max(0, current_year - (work.year or current_year))
            time_decay = math.exp(-0.15 * years_ago)  # Using default lambda
            
            # Calculate various factors
            coauthor_count = len(work.authorships)
            
            # Topic similarity (simplified - would need field topics)
            topic_similarity = 1.0  # Default if no topic analysis
            
            # Consortium factor
            consortium_factor = min(1.0, 50 / coauthor_count) if coauthor_count > 50 else 1.0
            
            # Alphabetical detection (simplified)
            alphabetical_detected = False  # Would need actual detection
            
            # Estimate contributions (simplified)
            role_bonus = 0.25 if author_role in ["first", "last"] and not alphabetical_detected else 0.0
            corresp_bonus = 0.25 if authorship.is_corresponding else 0.0
            
            base_contribution = (1.0 + role_bonus + corresp_bonus) / max(1, coauthor_count - 1)
            total_contribution = base_contribution * time_decay * consortium_factor * topic_similarity
            
            paper_contribution = PaperContribution(
                work_id=work.work_id,
                title=work.title[:100] + "..." if len(work.title) > 100 else work.title,
                year=work.year,
                author_role=author_role,
                coauthor_count=coauthor_count,
                time_decay_factor=time_decay,
                topic_similarity_score=topic_similarity,
                consortium_factor=consortium_factor,
                alphabetical_detected=alphabetical_detected,
                contribution_to_leadership=role_bonus + corresp_bonus,
                contribution_to_centrality=total_contribution * 0.5,  # Rough centrality contribution
                total_contribution=total_contribution
            )
            
            paper_contributions.append(paper_contribution)
        
        # Sort by total contribution (most impactful first)
        paper_contributions.sort(key=lambda x: x.total_contribution, reverse=True)
        
        # Cache results
        self.top_papers_cache[author_id] = paper_contributions
        
        return paper_contributions[:top_k]
    
    def decompose_crs_score(self,
                           author_id: str,
                           author_features: AuthorFeatures,
                           authors: Dict[str, AuthorMaster]) -> CRSDecomposition:
        """Provide detailed CRS score decomposition.
        
        Args:
            author_id: ID of the author to analyze
            author_features: Computed features for the author
            authors: Author master data
            
        Returns:
            Detailed CRS decomposition
        """
        if author_id in self.crs_decomposition_cache:
            return self.crs_decomposition_cache[author_id]
        
        author_data = authors.get(author_id)
        author_name = author_data.display_name if author_data else "Unknown"
        
        # Calculate component weights (from scorer defaults)
        centrality_weight = 0.50
        citation_weight = 0.35
        leadership_weight = 0.15
        
        # Sub-component weights within centrality (from scorer)
        pagerank_weight = 0.40
        degree_weight = 0.40
        betweenness_weight = 0.20
        
        # Calculate contributions
        pagerank_contribution = author_features.pagerank * pagerank_weight * centrality_weight
        degree_contribution = author_features.deg_w * degree_weight * centrality_weight
        betweenness_contribution = author_features.betweenness * betweenness_weight * centrality_weight
        
        # Citation contributions (within citation score)
        h_global_contribution = (author_features.h_index_global or 0) * 0.6 * citation_weight
        h_local_contribution = author_features.h_index_local * 0.4 * citation_weight
        
        # Leadership contributions
        leadership_rate_contribution = author_features.leadership_rate * 0.5 * leadership_weight
        kcore_contribution = (author_features.kcore / 20.0) * 0.5 * leadership_weight  # Assume max kcore = 20
        
        # Sample size shrinkage factor
        tau_shrinkage = 5  # Default from scorer
        sample_size_factor = 1.0 - math.exp(-author_features.n_in_corpus_works / tau_shrinkage)
        
        decomposition = CRSDecomposition(
            author_id=author_id,
            author_name=author_name,
            crs_final=author_features.crs_final,
            crs_raw=author_features.crs_raw,
            centrality_score=author_features.centrality_score,
            citation_score=author_features.citation_score,
            leadership_score=author_features.leadership_score,
            pagerank_contribution=pagerank_contribution,
            degree_contribution=degree_contribution,
            betweenness_contribution=betweenness_contribution,
            h_global_contribution=h_global_contribution,
            h_local_contribution=h_local_contribution,
            leadership_rate_contribution=leadership_rate_contribution,
            kcore_contribution=kcore_contribution,
            sample_size_factor=sample_size_factor
        )
        
        # Cache result
        self.crs_decomposition_cache[author_id] = decomposition
        
        return decomposition
    
    def generate_author_explanation(self,
                                  author_id: str,
                                  graph: nx.Graph,
                                  edge_data: Dict[str, GraphEdge],
                                  works: List[WorkRaw],
                                  author_features: AuthorFeatures,
                                  authors: Dict[str, AuthorMaster],
                                  ranking_info: Optional[ResearcherRanking] = None) -> Dict[str, Any]:
        """Generate comprehensive explanation for an author's ranking.
        
        Args:
            author_id: ID of the author to explain
            graph: Coauthorship network
            edge_data: Edge information
            works: All works in corpus
            author_features: Author's computed features
            authors: Author master data
            ranking_info: Optional ranking information
            
        Returns:
            Dictionary with comprehensive explanation data
        """
        # Get all analysis components
        top_edges = self.analyze_top_edges(author_id, graph, edge_data, authors, top_k=3)
        top_papers = self.analyze_top_papers(author_id, works, author_features, top_k=3)
        crs_decomposition = self.decompose_crs_score(author_id, author_features, authors)
        
        author_data = authors.get(author_id)
        
        explanation = {
            "author_info": {
                "author_id": author_id,
                "name": author_data.display_name if author_data else "Unknown",
                "orcid": author_data.orcid if author_data else None,
                "rank": ranking_info.rank if ranking_info else None,
                "crs_score": author_features.crs_final
            },
            "score_breakdown": {
                "crs_final": crs_decomposition.crs_final,
                "crs_raw": crs_decomposition.crs_raw,
                "sample_size_factor": crs_decomposition.sample_size_factor,
                "centrality_score": crs_decomposition.centrality_score,
                "citation_score": crs_decomposition.citation_score,
                "leadership_score": crs_decomposition.leadership_score
            },
            "centrality_breakdown": {
                "pagerank_contribution": crs_decomposition.pagerank_contribution,
                "degree_contribution": crs_decomposition.degree_contribution,
                "betweenness_contribution": crs_decomposition.betweenness_contribution
            },
            "citation_breakdown": {
                "h_global_contribution": crs_decomposition.h_global_contribution,
                "h_local_contribution": crs_decomposition.h_local_contribution,
                "h_index_global": author_features.h_index_global,
                "h_index_local": author_features.h_index_local
            },
            "leadership_breakdown": {
                "leadership_rate_contribution": crs_decomposition.leadership_rate_contribution,
                "kcore_contribution": crs_decomposition.kcore_contribution,
                "leadership_rate": author_features.leadership_rate,
                "papers_count": author_features.n_in_corpus_works
            },
            "top_collaborations": [
                {
                    "collaborator_name": edge.collaborator_name,
                    "collaboration_strength": edge.collaboration_strength,
                    "shared_papers": edge.shared_papers_count,
                    "edge_weight": round(edge.edge_weight, 4),
                    "centrality_contribution": round(edge.contribution_to_centrality, 4)
                }
                for edge in top_edges
            ],
            "top_papers": [
                {
                    "title": paper.title,
                    "year": paper.year,
                    "author_role": paper.author_role,
                    "coauthors": paper.coauthor_count,
                    "time_decay": round(paper.time_decay_factor, 3),
                    "total_contribution": round(paper.total_contribution, 4)
                }
                for paper in top_papers
            ]
        }
        
        return explanation
    
    def export_explanations_to_dict(self,
                                   rankings: List[ResearcherRanking],
                                   graph: nx.Graph,
                                   edge_data: Dict[str, GraphEdge],
                                   works: List[WorkRaw],
                                   author_features: Dict[str, AuthorFeatures],
                                   authors: Dict[str, AuthorMaster],
                                   top_n: int = 10) -> Dict[str, Dict[str, Any]]:
        """Export explanations for top N researchers.
        
        Args:
            rankings: List of researcher rankings
            graph: Coauthorship network
            edge_data: Edge information
            works: All works in corpus
            author_features: Author features dict
            authors: Author master data
            top_n: Number of top researchers to explain
            
        Returns:
            Dictionary with explanations for top researchers
        """
        explanations = {}
        
        for ranking in rankings[:top_n]:
            author_id = ranking.author_id
            features = author_features.get(author_id)
            
            if features:
                explanation = self.generate_author_explanation(
                    author_id=author_id,
                    graph=graph,
                    edge_data=edge_data,
                    works=works,
                    author_features=features,
                    authors=authors,
                    ranking_info=ranking
                )
                explanations[author_id] = explanation
        
        return explanations
    
    def clear_cache(self):
        """Clear all cached analysis results."""
        self.top_edges_cache.clear()
        self.top_papers_cache.clear() 
        self.crs_decomposition_cache.clear()
        logger.debug("Cleared explainability analysis cache")