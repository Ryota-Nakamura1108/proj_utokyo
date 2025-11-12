"""Clustering engine for grouping papers."""

import logging
from typing import List, Tuple
import numpy as np
from sklearn.cluster import KMeans

from .models import Paper, Cluster

logger = logging.getLogger(__name__)


class ClusteringEngine:
    """Clusters papers based on embeddings."""

    def __init__(self, method: str = "kmeans", random_state: int = 42):
        """Initialize the clustering engine.

        Args:
            method: Clustering method ('kmeans' or 'hdbscan')
            random_state: Random state for reproducibility
        """
        self.method = method
        self.random_state = random_state

    def cluster_papers(
        self,
        papers: List[Paper],
        embeddings: List[np.ndarray],
    ) -> Tuple[List[Cluster], np.ndarray]:
        """Cluster papers based on their embeddings.

        Args:
            papers: List of papers to cluster
            embeddings: List of embedding vectors

        Returns:
            Tuple of (list of Cluster objects, cluster labels array)
        """
        if len(papers) != len(embeddings):
            raise ValueError("Number of papers and embeddings must match")

        if len(papers) < 5:
            # Too few papers to cluster meaningfully
            logger.warning(
                f"Only {len(papers)} papers - skipping clustering"
            )
            return self._create_single_cluster(papers, embeddings)

        logger.info(f"Clustering {len(papers)} papers using {self.method}...")

        # Convert embeddings to numpy array
        X = np.array(embeddings)

        # Determine optimal number of clusters
        n_clusters = self._determine_n_clusters(len(papers))
        logger.info(f"Using {n_clusters} clusters")

        # Perform clustering
        if self.method == "kmeans":
            labels = self._cluster_kmeans(X, n_clusters)
        else:
            logger.warning(f"Unsupported method '{self.method}', using kmeans")
            labels = self._cluster_kmeans(X, n_clusters)

        # Create Cluster objects
        clusters = self._create_clusters(papers, embeddings, labels)

        logger.info(f"Clustering complete: created {len(clusters)} clusters")

        return clusters, labels

    def _determine_n_clusters(self, n_papers: int) -> int:
        """Determine optimal number of clusters based on number of papers.

        Args:
            n_papers: Number of papers

        Returns:
            Number of clusters to use
        """
        if n_papers < 10:
            return min(3, n_papers)
        elif n_papers < 30:
            return 5
        elif n_papers < 60:
            return 7
        else:
            return min(10, int(np.sqrt(n_papers)))

    def _cluster_kmeans(
        self, X: np.ndarray, n_clusters: int
    ) -> np.ndarray:
        """Perform K-Means clustering.

        Args:
            X: Embedding matrix (n_samples, n_features)
            n_clusters: Number of clusters

        Returns:
            Array of cluster labels
        """
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=10,
        )
        labels = kmeans.fit_predict(X)
        return labels

    def _create_clusters(
        self,
        papers: List[Paper],
        embeddings: List[np.ndarray],
        labels: np.ndarray,
    ) -> List[Cluster]:
        """Create Cluster objects from clustering results.

        Args:
            papers: List of papers
            embeddings: List of embeddings
            labels: Cluster labels

        Returns:
            List of Cluster objects
        """
        clusters = []
        unique_labels = np.unique(labels)

        for label in unique_labels:
            # Get papers in this cluster
            cluster_indices = np.where(labels == label)[0]
            cluster_papers = [papers[i] for i in cluster_indices]
            cluster_embeddings = [embeddings[i] for i in cluster_indices]

            # Calculate centroid
            centroid = np.mean(cluster_embeddings, axis=0)

            cluster = Cluster(
                cluster_id=int(label),
                papers=cluster_papers,
                centroid=centroid.tolist(),
            )

            clusters.append(cluster)

        # Sort by cluster size (descending)
        clusters.sort(key=lambda c: len(c.papers), reverse=True)

        return clusters

    def _create_single_cluster(
        self, papers: List[Paper], embeddings: List[np.ndarray]
    ) -> Tuple[List[Cluster], np.ndarray]:
        """Create a single cluster containing all papers.

        Args:
            papers: List of papers
            embeddings: List of embeddings

        Returns:
            Tuple of (single cluster list, labels array)
        """
        centroid = np.mean(embeddings, axis=0) if embeddings else None

        cluster = Cluster(
            cluster_id=0,
            papers=papers,
            centroid=centroid.tolist() if centroid is not None else None,
        )

        labels = np.zeros(len(papers), dtype=int)

        return [cluster], labels
