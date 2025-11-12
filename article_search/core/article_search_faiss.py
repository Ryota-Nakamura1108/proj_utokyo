"""Search papers using FAISS index for fast similarity search"""
from .openai_services import OpenAIService

from pydantic import BaseModel, Field
from typing import List, Optional, Any
from datetime import datetime
import numpy as np
import faiss
import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class SearchResult(BaseModel):
    """Pydantic model for a single search result"""
    rank: int
    similarity_score: float
    id: str = ""
    title: str = ""
    doi: str = ""
    abstract: str = ""
    authorships: Optional[List[Any]] = []
    concepts: Optional[List[Any]] = []
    fwci: Optional[float] = None
    cited_by_count: Optional[int] = 0
    publication_date: str = ""
    primary_location: Optional[dict] = {}
    counts_by_year: Optional[List[Any]] = []
    referenced_works: Optional[List[str]] = []
    related_works: Optional[List[str]] = []
    type: str = ""


class SearchResults(BaseModel):
    """Pydantic model for all search results"""
    query: str
    timestamp: str
    top_k: int
    results: List[SearchResult]


class PaperSearchEngineFAISS:
    def __init__(self, index_dir: str):
        """
        Initialize FAISS-based search engine.

        Args:
            index_dir: Directory containing papers.index and papers_metadata.pkl
        """
        self.openai_service = OpenAIService()
        self.index_dir = index_dir

        # Paths to FAISS index and metadata
        self.index_path = os.path.join(index_dir, "papers.index")
        self.metadata_path = os.path.join(index_dir, "papers_metadata.pkl")

        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        print(f"📂 Loading FAISS index from '{index_dir}'...")

        # Load FAISS index
        self.index = faiss.read_index(self.index_path)
        print(f"✓ Loaded FAISS index with {self.index.ntotal} vectors")

        # Load metadata
        with open(self.metadata_path, "rb") as f:
            self.metadata = pickle.load(f)
        print(f"✓ Loaded metadata for {len(self.metadata)} papers")

        if self.index.ntotal != len(self.metadata):
            raise ValueError(
                f"Index/metadata mismatch: {self.index.ntotal} vectors vs {len(self.metadata)} metadata entries"
            )

    def search(
        self,
        query: str,
        top_k: int = 10,
        model: str = "text-embedding-3-small",
        similarity_threshold: float = 0.55,
    ) -> SearchResults:
        """
        Search papers using FAISS index.

        Args:
            query: Search query text
            top_k: Number of top results to return
            model: OpenAI embedding model to use
            similarity_threshold: Minimum cosine similarity score

        Returns:
            SearchResults object with top-k results above threshold
        """
        print(f"🔍 Searching for: '{query}' (FAISS search)")

        # Get query embedding
        query_embedding = self.openai_service.embedder(model=model, texts=query)
        query_vector = np.array(query_embedding, dtype=np.float32).reshape(1, -1)

        # Normalize query vector for cosine similarity
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            raise ValueError("Query embedding norm is zero.")
        query_normalized = query_vector / query_norm

        # FAISS search (IndexFlatIP expects normalized vectors for cosine similarity)
        # Search for more candidates to ensure we have enough above threshold
        search_k = min(top_k * 10, self.index.ntotal)
        similarities, indices = self.index.search(query_normalized, search_k)

        # Flatten results (FAISS returns 2D arrays)
        similarities = similarities[0]
        indices = indices[0]

        print(f"✓ FAISS search completed: {len(indices)} candidates retrieved")

        # Filter by similarity threshold and collect metadata
        candidate_results = []
        for sim, idx in zip(similarities, indices):
            if sim > similarity_threshold and idx < len(self.metadata):
                candidate_results.append((float(sim), self.metadata[idx]))

        print(f"📊 Matches above threshold ({similarity_threshold}): {len(candidate_results)}")

        if not candidate_results:
            print("⚠️ No papers found above similarity threshold.")
            return SearchResults(
                query=query,
                timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
                top_k=0,
                results=[],
            )

        # Already sorted by FAISS, just take top_k
        top_k_actual = min(top_k, len(candidate_results))
        top_results = candidate_results[:top_k_actual]

        # Build SearchResult objects
        results = []
        for rank, (sim, meta) in enumerate(top_results, 1):
            # Preserve full authorship structure with author IDs
            authorships = []
            if meta.get("authorships"):
                for authorship in meta["authorships"]:
                    if isinstance(authorship, dict):
                        authorships.append(authorship)

            result = SearchResult(
                rank=rank,
                similarity_score=sim,
                id=meta.get("id") or "",
                title=meta.get("title") or "",
                doi=meta.get("doi") or "",
                abstract=meta.get("abstract") or "",
                authorships=authorships,
                concepts=meta.get("concepts") or [],
                fwci=meta.get("fwci"),
                cited_by_count=meta.get("cited_by_count") or 0,
                publication_date=meta.get("publication_date") or "",
                primary_location=meta.get("primary_location") or {},
                counts_by_year=meta.get("counts_by_year") or [],
                referenced_works=meta.get("referenced_works") or [],
                related_works=meta.get("related_works") or [],
                type=meta.get("type") or "",
            )
            results.append(result)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return SearchResults(
            query=query,
            timestamp=timestamp,
            top_k=top_k_actual,
            results=results,
        )


def main():
    """Example usage"""
    INDEX_DIR = os.path.join(BASE_DIR, "../../faiss_index")

    search_engine = PaperSearchEngineFAISS(index_dir=INDEX_DIR)
    query = "quantum computing"

    results = search_engine.search(query=query, top_k=10, similarity_threshold=0)

    # Print brief summary
    print(f"\n✅ Top {len(results.results)} results above threshold:\n")
    for r in results.results[:3]:
        print(f"🏆 Rank {r.rank} | Score: {r.similarity_score:.3f} | {r.title[:80]}")


if __name__ == "__main__":
    main()
