"""Representative paper extraction from clusters."""

import logging
from typing import List
import numpy as np

from .models import Cluster, Paper

logger = logging.getLogger(__name__)


class RepresentativeExtractor:
    """Extracts representative papers from clusters."""

    def __init__(self, strategy: str = "weighted", max_representatives: int = 3):
        """Initialize the extractor.

        Args:
            strategy: Extraction strategy ('medoid' or 'weighted')
            max_representatives: Maximum number of representatives per cluster
        """
        self.strategy = strategy
        self.max_representatives = max_representatives

    def extract_representatives(
        self,
        clusters: List[Cluster],
        embeddings_by_cluster: dict,
    ) -> List[Cluster]:
        """Extract representative papers for each cluster.

        Args:
            clusters: List of Cluster objects
            embeddings_by_cluster: Dict mapping cluster_id to list of embeddings

        Returns:
            Updated clusters with representative papers
        """
        logger.info(
            f"Extracting representatives using '{self.strategy}' strategy..."
        )

        for cluster in clusters:
            cluster_embeddings = embeddings_by_cluster.get(cluster.cluster_id, [])

            if not cluster_embeddings or not cluster.papers:
                continue

            if self.strategy == "medoid":
                representatives = self._medoid_selection(
                    cluster.papers, cluster_embeddings
                )
            elif self.strategy == "weighted":
                representatives = self._weighted_selection(
                    cluster.papers, cluster_embeddings
                )
            else:
                logger.warning(
                    f"Unknown strategy '{self.strategy}', using medoid"
                )
                representatives = self._medoid_selection(
                    cluster.papers, cluster_embeddings
                )

            cluster.representative_papers = representatives

        logger.info("Representative extraction complete")
        return clusters

    def _medoid_selection(
        self, papers: List[Paper], embeddings: List[np.ndarray]
    ) -> List[Paper]:
        """Select medoid (most central paper) as representative.

        Args:
            papers: Papers in cluster
            embeddings: Embeddings for papers

        Returns:
            List containing the medoid paper
        """
        if not papers or not embeddings:
            return []

        # Calculate centroid
        centroid = np.mean(embeddings, axis=0)

        # Find closest paper to centroid
        distances = [
            np.linalg.norm(emb - centroid) for emb in embeddings
        ]
        medoid_idx = np.argmin(distances)

        return [papers[medoid_idx]]

    def _weighted_selection(
        self, papers: List[Paper], embeddings: List[np.ndarray]
    ) -> List[Paper]:
        """Select representatives using weighted score (centrality + citations).

        Args:
            papers: Papers in cluster
            embeddings: Embeddings for papers

        Returns:
            List of top representatives (up to max_representatives)
        """
        if not papers or not embeddings:
            return []

        # Calculate centroid
        centroid = np.mean(embeddings, axis=0)

        # Calculate distances to centroid
        distances = np.array(
            [np.linalg.norm(emb - centroid) for emb in embeddings]
        )

        # Normalize distance scores (closer = higher score)
        if distances.max() > distances.min():
            distance_scores = 1 - (distances - distances.min()) / (
                distances.max() - distances.min()
            )
        else:
            distance_scores = np.ones(len(distances))

        # Calculate citation scores (log scale)
        citation_counts = np.array([p.cited_by_count for p in papers])
        citation_scores = np.log1p(citation_counts)

        if citation_scores.max() > 0:
            citation_scores = citation_scores / citation_scores.max()

        # Combined weighted score
        # 60% centrality, 40% citations
        weights = 0.6 * distance_scores + 0.4 * citation_scores

        # Get top-N papers
        top_indices = np.argsort(weights)[
            -min(self.max_representatives, len(papers)) :
        ][::-1]

        representatives = [papers[i] for i in top_indices]

        return representatives
