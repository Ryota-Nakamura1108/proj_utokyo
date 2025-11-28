"""Coauthorship network construction with edge weighting."""

import math
import logging
from collections import defaultdict, Counter
from itertools import combinations
from typing import Dict, List, Tuple, Set, Optional
import networkx as nx
from datetime import datetime
import re

from ..core.models import WorkRaw, Authorship, GraphEdge, AuthorPosition, LocalCitations, InstitutionMaster

logger = logging.getLogger(__name__)


class NetworkBuilder:
    """Builds weighted coauthorship networks from publication data."""
    
    def __init__(self, decay_lambda: float = 0.15, 
                 first_last_bonus: float = 0.25,
                 corr_author_bonus: float = 0.25,
                 current_year: Optional[int] = None,
                 topic_similarity_enabled: bool = True,
                 topic_similarity_threshold: float = 0.1,
                 consortium_suppression_enabled: bool = True,
                 consortium_threshold: int = 50,
                 alphabetical_detection_enabled: bool = True,
                 alphabetical_threshold: float = 0.8):
        """Initialize network builder with weighting parameters."""
        self.decay_lambda = decay_lambda
        self.first_last_bonus = first_last_bonus
        self.corr_author_bonus = corr_author_bonus
        self.current_year = current_year or datetime.now().year
        self.topic_similarity_enabled = topic_similarity_enabled
        self.topic_similarity_threshold = topic_similarity_threshold
        self.consortium_suppression_enabled = consortium_suppression_enabled
        self.consortium_threshold = consortium_threshold
        self.alphabetical_detection_enabled = alphabetical_detection_enabled
        self.alphabetical_threshold = alphabetical_threshold
        # self.simmelian_strengthening_enabled = simmelian_strengthening_enabled (REMOVED)
        self.field_topics = None  # Will be set by build_field_topic_profile
        self.alphabetical_works = set()  # Will store work IDs with alphabetical authorship

    def build_coauthorship_network(self, works: List[WorkRaw]) -> Tuple[nx.Graph, Dict[str, GraphEdge], LocalCitations]:
        """Build weighted coauthorship network from works."""
        logger.info(f"Building coauthorship network from {len(works)} works")
        
        # Build field topic profile for similarity calculations
        if self.topic_similarity_enabled:
            self.field_topics = self._build_field_topic_profile(works)
            logger.info(f"Built field topic profile with {len(self.field_topics)} topics")
        
        # Detect alphabetical authorship patterns
        if self.alphabetical_detection_enabled:
            self._detect_alphabetical_authorship(works)
            if self.alphabetical_works:
                logger.info(f"Detected {len(self.alphabetical_works)} works with alphabetical authorship (first/last bonuses disabled)")
        
        # Create graph and edge data
        graph = nx.Graph()
        edge_data = defaultdict(lambda: GraphEdge("", ""))
        
        # Calculate local citations first
        local_citations = self._calculate_local_citations(works)
        
        # Process each work to add coauthorship edges
        for work in works:
            self._process_work_for_network(work, graph, edge_data)
        
        # --- Simmelian tie strengthening block REMOVED ---
        # if self.simmelian_strengthening_enabled:
        #     self._apply_simmelian_strengthening(graph)
        #     logger.info("Applied Simmelian tie strengthening based on triangle counts")
        # --------------------------------------------------
        
        # Convert edge data to final format
        final_edges = {}
        for (src, dst), edge in edge_data.items():
            edge.src_author_id = src
            edge.dst_author_id = dst
            edge.w_total = edge.w_freq + edge.w_recency + edge.w_role
            final_edges[f"{src}_{dst}"] = edge
        
        logger.info(f"Created network with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")


        return graph, final_edges, local_citations
    

    def _process_work_for_network(self, work: WorkRaw, graph: nx.Graph, edge_data: Dict):
        """Process a single work to add coauthorship edges."""
        # Get valid authors (those with author IDs or temporary IDs)
        valid_authors = [auth for auth in work.authorships if auth.author_id]
        n_authors = len(valid_authors)
        
        if n_authors < 2:
            return  # No coauthorship possible
        
        # Calculate time decay
        decay_factor = self._calculate_time_decay(work.year) if work.year else 1.0
        
        # Process all author pairs
        for auth1, auth2 in combinations(valid_authors, 2):
            self._add_coauthorship_edge(
                auth1, auth2, work, n_authors, decay_factor, graph, edge_data
            )
    
    def _add_coauthorship_edge(self, auth1: Authorship, auth2: Authorship, 
                             work: WorkRaw, n_authors: int, decay_factor: float,
                             graph: nx.Graph, edge_data: Dict):
        """Add or update coauthorship edge between two authors."""
        id1, id2 = auth1.author_id, auth2.author_id
        
        # Ensure consistent ordering for undirected graph
        if id1 > id2:
            id1, id2 = id2, id1
            auth1, auth2 = auth2, auth1
        
        # Calculate role weights
        role1 = self._calculate_role_weight(auth1, work)
        role2 = self._calculate_role_weight(auth2, work)
        
        # Calculate edge contribution from this work
        # Using formula: (r(i,p) + r(j,p))/2 * 1/(n_p-1) * d(p) * g(topic_similarity) * f(n_p)
        role_avg = (role1 + role2) / 2.0
        normalization = 1.0 / (n_authors - 1)
        
        # Apply topic similarity weighting if enabled
        topic_weight = 1.0
        if self.topic_similarity_enabled and self.field_topics:
            topic_similarity = self._calculate_topic_similarity(work, self.field_topics)
            topic_weight = self._get_topic_weight_factor(topic_similarity)
        
        # Apply consortium suppression if enabled
        consortium_weight = 1.0
        if self.consortium_suppression_enabled:
            consortium_weight = self._get_consortium_weight_factor(n_authors)
        
        edge_weight = role_avg * normalization * decay_factor * topic_weight * consortium_weight
        
        # Add nodes to graph
        graph.add_node(id1)
        graph.add_node(id2)
        
        # Update edge data
        edge_key = (id1, id2)
        edge = edge_data[edge_key]
        edge.w_freq += 1.0  # Frequency count
        edge.w_recency += decay_factor  # Time-decayed sum
        edge.w_role += edge_weight  # Role-weighted contribution
        
        # Update graph edge weight
        if graph.has_edge(id1, id2):
            graph[id1][id2]['weight'] += edge_weight
        else:
            graph.add_edge(id1, id2, weight=edge_weight)
    
    def _calculate_role_weight(self, authorship: Authorship, work: WorkRaw) -> float:
        """Calculate role-based weight for an author."""
        weight = 1.0
        
        # Add bonus for first/last author (skip if alphabetical authorship detected)
        is_alphabetical = work.work_id in self.alphabetical_works
        if not is_alphabetical and authorship.author_position in [AuthorPosition.FIRST, AuthorPosition.LAST]:
            weight += self.first_last_bonus
        
        # Add bonus for corresponding author (always applies)
        if authorship.is_corresponding:
            weight += self.corr_author_bonus
        
        return weight
    
    def _calculate_time_decay(self, year: int) -> float:
        """Calculate time decay factor."""
        if not year:
            return 1.0
        
        years_ago = max(0, self.current_year - year)
        return math.exp(-self.decay_lambda * years_ago)
    

    def _calculate_local_citations(self, works: List[WorkRaw]) -> LocalCitations:
        """Calculate in-corpus citation counts."""
        logger.debug("Calculating local citations within corpus")
        
        # Create mapping of work ID to work for quick lookup
        work_map = {work.work_id: work for work in works}
        work_ids = set(work_map.keys())
        
        # Count citations within corpus
        citation_counts = Counter()
        
        for work in works:
            for ref_work_id in work.referenced_work_ids:
                if ref_work_id in work_ids:
                    citation_counts[ref_work_id] += 1
        
        return LocalCitations(work_citations=dict(citation_counts))
    
    def generate_author_missing_ids(self, works: List[WorkRaw]) -> Dict[str, str]:
        """Generate temporary IDs for authors without OpenAlex IDs."""
        logger.debug("Generating temporary IDs for authors without OpenAlex IDs")
        
        temp_id_map = {}
        temp_id_counter = 1
        
        for work in works:
            for authorship in work.authorships:
                if not authorship.author_id:
                    # Create temporary ID based on available information
                    key = self._create_author_key(authorship, work.work_id)
                    if key not in temp_id_map:
                        temp_id = f"TEMP_A{temp_id_counter:06d}"
                        temp_id_map[key] = temp_id
                        authorship.author_id = temp_id  # Assign the temp ID
                        temp_id_counter += 1
                        logger.debug(f"Created temporary ID {temp_id} for {authorship.raw_name}")
                    else:
                        authorship.author_id = temp_id_map[key]
        
        return temp_id_map
    
    def _create_author_key(self, authorship: Authorship, work_id: str) -> str:
        """Create a key for author identification."""
        if authorship.orcid:
            return f"orcid:{authorship.orcid}"
        if authorship.raw_name:
            name = authorship.raw_name.lower().strip()
            return f"name:{name}"
        else:
            logger.warning(f"著者名が None のため、{work_id} の著者キーをフォールバックします。")
            
            return f"name:unknown_author_{work_id}_{authorship.author_position}"
    
    def get_network_statistics(self, graph: nx.Graph) -> Dict[str, float]:
        """Calculate basic network statistics."""
        if graph.number_of_nodes() == 0:
            return {}
        
        stats = {
            'num_nodes': graph.number_of_nodes(),
            'num_edges': graph.number_of_edges(),
            'density': nx.density(graph),
            'avg_clustering': nx.average_clustering(graph),
        }
        
        # Add connected components info
        if nx.is_connected(graph):
            stats['is_connected'] = True
            stats['diameter'] = nx.diameter(graph)
            stats['avg_shortest_path'] = nx.average_shortest_path_length(graph)
        else:
            stats['is_connected'] = False
            stats['num_components'] = nx.number_connected_components(graph)
            stats['largest_component_size'] = len(max(nx.connected_components(graph), key=len))
        
        return stats
    
    def _build_field_topic_profile(self, works: List[WorkRaw]) -> Dict[str, float]:
        """Build representative topic profile for the field from primary topics."""
        topic_counts = Counter()
        total_works = 0
        
        for work in works:
            if work.primary_topic:
                topic_counts[work.primary_topic.topic_id] += 1
                total_works += 1
        
        if total_works == 0:
            logger.warning("No primary topics found in works")
            return {}
        
        # Create field topic profile with normalized scores
        field_topics = {}
        for topic_id, count in topic_counts.items():
            field_topics[topic_id] = count / total_works
        
        logger.debug(f"Field topic profile: {len(field_topics)} topics, "
                    f"top topic appears in {max(topic_counts.values())}/{total_works} works")
        
        return field_topics
    
    def _calculate_topic_similarity(self, work: WorkRaw, field_topics: Dict[str, float]) -> float:
        """Calculate topic similarity between a work and the field profile.
        
        Uses formula: S_topic(work, field) = Σ_t∈(T_work ∩ T_field) min(s_work,t, s_field,t)
        """
        if not work.topics:
            return 0.0
        
        # Create work topic dict for fast lookup
        work_topics = {topic.topic_id: topic.score for topic in work.topics}
        
        similarity = 0.0
        for topic_id in field_topics:
            if topic_id in work_topics:
                # Take minimum of work score and field score
                similarity += min(work_topics[topic_id], field_topics[topic_id])
        
        return similarity
    
    def _get_topic_weight_factor(self, topic_similarity: float) -> float:
        """Convert topic similarity to edge weight multiplier.
        
        Uses step function: g(x) = 1.0 if x >= threshold, else 0.2
        This reduces edge contributions from off-topic papers.
        """
        if topic_similarity >= self.topic_similarity_threshold:
            return 1.0
        else:
            return 0.2  # Heavily discount off-topic edges
    
    def _get_consortium_weight_factor(self, n_authors: int) -> float:
        """Apply consortium suppression for very large collaborations.
        
        Uses formula: f(n) = min(1, threshold/n) for n > threshold
        This reduces the influence of massive consortiums.
        """
        if n_authors > self.consortium_threshold:
            return min(1.0, self.consortium_threshold / n_authors)
        else:
            return 1.0
    
    def _detect_alphabetical_authorship(self, works: List[WorkRaw]):
        """Detect works with alphabetical authorship ordering.
        
        Analyzes author last names to determine if authors are listed alphabetically.
        If alphabetical ordering is detected, first/last author bonuses are disabled.
        """
        alphabetical_count = 0
        analyzable_works = 0
        
        for work in works:
            if len(work.authorships) < 3:
                continue  # Need at least 3 authors for reliable detection
            
            # Extract surnames from author names
            surnames = []
            for authorship in work.authorships:
                if authorship.raw_name:
                    surname = self._extract_surname(authorship.raw_name)
                    if surname:
                        surnames.append(surname.lower())
            
            if len(surnames) < 3:
                continue  # Not enough extractable surnames
            
            analyzable_works += 1
            
            # Calculate alphabetical ordering ratio
            alphabetical_ratio = self._calculate_alphabetical_ratio(surnames)
            
            # If ratio exceeds threshold, mark as alphabetical
            if alphabetical_ratio >= self.alphabetical_threshold:
                self.alphabetical_works.add(work.work_id)
                alphabetical_count += 1
                logger.debug(f"Work {work.work_id} marked as alphabetical (ratio: {alphabetical_ratio:.2f})")
        
        if analyzable_works > 0:
            alphabetical_percentage = (alphabetical_count / analyzable_works) * 100
            logger.info(f"Alphabetical authorship analysis: {alphabetical_count}/{analyzable_works} "
                       f"({alphabetical_percentage:.1f}%) works have alphabetical ordering")
    
    def _extract_surname(self, full_name: str) -> Optional[str]:
        """Extract surname from full author name.
        
        Handles various name formats:
        - "LastName, FirstName" 
        - "FirstName LastName"
        - "FirstName MiddleName LastName"
        """
        if not full_name:
            return None
        
        name = full_name.strip()
        
        # Handle comma-separated format: "LastName, FirstName"
        if ',' in name:
            parts = name.split(',')
            return parts[0].strip()
        
        # Handle space-separated format: assume last word is surname
        parts = name.split()
        if len(parts) > 0:
            return parts[-1].strip()
        
        return None
    
    def _calculate_alphabetical_ratio(self, surnames: List[str]) -> float:
        """Calculate what fraction of surname pairs are in alphabetical order.
        
        Returns ratio between 0.0 and 1.0 indicating alphabetical ordering strength.
        """
        if len(surnames) < 2:
            return 0.0
        
        total_pairs = 0
        alphabetical_pairs = 0
        
        # Check all consecutive pairs
        for i in range(len(surnames) - 1):
            surname1 = surnames[i]
            surname2 = surnames[i + 1]
            
            # Skip pairs with identical surnames
            if surname1 != surname2:
                total_pairs += 1
                if surname1 <= surname2:  # Alphabetical order
                    alphabetical_pairs += 1
        
        if total_pairs == 0:
            return 0.0
        
        return alphabetical_pairs / total_pairs

    # --- _apply_simmelian_strengthening method REMOVED ---
    
    # --- _fast_count_triangles_per_edge method REMOVED ---