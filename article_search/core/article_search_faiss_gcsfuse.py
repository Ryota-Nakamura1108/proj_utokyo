"""Search papers using FAISS index with GCS FUSE-mounted SQLite for Cloud Run

Architecture:
- FAISS index: Downloaded to /tmp on startup (~4.7GB in memory)
- SQLite database: Accessed via GCS FUSE mount (18GB on GCS, lazy-loaded)
- Memory usage: ~5-8GB total (FAISS + query cache only)
"""
from .openai_services import OpenAIService

from pydantic import BaseModel, Field
from typing import List, Optional, Any
from datetime import datetime
import numpy as np
import faiss
import sqlite3
import os
import json
from google.cloud import storage


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# GCS Configuration
GCS_BUCKET = "proj-utokyo-vectors"
INDEX_PATH_GCS = "faiss_index/papers.index"
SQLITE_PATH_GCS = "faiss_index/papers_metadata.db"

# Local paths
LOCAL_INDEX = "/tmp/papers.index"
# SQLite accessed via GCS FUSE mount (configured in Cloud Run)
GCS_FUSE_MOUNT = "/mnt/gcs"  # Cloud Run volume mount point
SQLITE_PATH_FUSE = os.path.join(GCS_FUSE_MOUNT, SQLITE_PATH_GCS)


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


def download_from_gcs(bucket_name: str, source_path: str, local_path: str):
    """
    Download file from GCS to local path.

    Args:
        bucket_name: GCS bucket name
        source_path: Path in GCS bucket
        local_path: Local file path to save to
    """
    if os.path.exists(local_path):
        file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
        print(f"✓ File already exists: {local_path} ({file_size_mb:.1f} MB)")
        return

    print(f"📥 Downloading gs://{bucket_name}/{source_path} → {local_path}")

    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(source_path)

        blob.download_to_filename(local_path)

        file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
        print(f"✓ Download completed: {file_size_mb:.1f} MB")
    except Exception as e:
        print(f"❌ Download failed: {e}")
        raise


def load_faiss_index(local_index_path: str) -> faiss.Index:
    """
    Load FAISS index from file.

    Args:
        local_index_path: Path to local FAISS index file

    Returns:
        FAISS Index object
    """
    print(f"📂 Loading FAISS index from: {local_index_path}")

    index = faiss.read_index(local_index_path)

    print(f"✓ FAISS index loaded:")
    print(f"  Total vectors: {index.ntotal:,}")
    print(f"  Dimension: {index.d}")

    # Optimize for multi-threading
    faiss.omp_set_num_threads(4)

    return index


def open_sqlite_connection_fuse(sqlite_fuse_path: str) -> sqlite3.Connection:
    """
    Open SQLite connection via GCS FUSE mount (no download required).

    Args:
        sqlite_fuse_path: Path to SQLite database on GCS FUSE mount

    Returns:
        SQLite connection object
    """
    print(f"🗄️  Connecting to SQLite via GCS FUSE: {sqlite_fuse_path}")

    if not os.path.exists(sqlite_fuse_path):
        raise FileNotFoundError(
            f"SQLite database not found at GCS FUSE mount: {sqlite_fuse_path}\n"
            f"Ensure Cloud Run volume mount is configured correctly."
        )

    # Open connection (read-only for safety and performance)
    # check_same_thread=False: Allow connection to be used across threads (safe for read-only)
    conn = sqlite3.connect(f"file:{sqlite_fuse_path}?mode=ro", uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row  # Enable column name access

    # Verify database
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM papers")
    count = cursor.fetchone()[0]

    db_size_mb = os.path.getsize(sqlite_fuse_path) / (1024 * 1024)
    print(f"✓ SQLite connection established (via GCS FUSE):")
    print(f"  Total papers: {count:,}")
    print(f"  Database size: {db_size_mb:.1f} MB (on GCS, not in memory)")

    return conn


class PaperSearchEngineFAISSGCSFuse:
    def __init__(
        self,
        bucket_name: str = GCS_BUCKET,
        index_path_gcs: str = INDEX_PATH_GCS,
        sqlite_path_fuse: str = SQLITE_PATH_FUSE,
        local_index: str = LOCAL_INDEX,
        auto_download_index: bool = True
    ):
        """
        Initialize FAISS-based search engine with GCS FUSE-mounted SQLite.

        Architecture:
        - FAISS index: Downloaded to /tmp on startup (~4.7GB in memory)
        - SQLite database: Accessed via GCS FUSE mount (18GB on GCS, lazy-loaded)
        - Memory usage: ~5-8GB total

        Args:
            bucket_name: GCS bucket name
            index_path_gcs: Path to FAISS index in GCS
            sqlite_path_fuse: Path to SQLite on GCS FUSE mount
            local_index: Local path for FAISS index
            auto_download_index: Automatically download FAISS index on init
        """
        self.openai_service = OpenAIService()
        self.bucket_name = bucket_name
        self.index_path_gcs = index_path_gcs
        self.sqlite_path_fuse = sqlite_path_fuse
        self.local_index = local_index

        print("=" * 60)
        print("🚀 Initializing FAISS Search Engine (GCS FUSE SQLite)")
        print("=" * 60)

        # Download FAISS index only (not SQLite)
        if auto_download_index:
            download_from_gcs(bucket_name, index_path_gcs, local_index)

        # Load FAISS index to memory
        self.index = load_faiss_index(local_index)

        # Connect to SQLite via GCS FUSE (no download)
        self.db_conn = open_sqlite_connection_fuse(sqlite_path_fuse)

        print("=" * 60)
        print(f"✅ Search engine ready! ({self.index.ntotal:,} papers)")
        print(f"   Memory: FAISS index only (~5GB)")
        print(f"   Metadata: SQLite on GCS FUSE (on-demand reads)")
        print("=" * 60)

    def _fetch_metadata_by_indices(self, indices: List[int]) -> List[dict]:
        """
        Fetch metadata from GCS FUSE SQLite for specific FAISS indices.

        Args:
            indices: List of FAISS vector indices

        Returns:
            List of metadata dictionaries
        """
        if not indices:
            return []

        # Build SQL query with placeholders
        placeholders = ','.join('?' * len(indices))
        query = f"SELECT * FROM papers WHERE ROWID IN ({placeholders})"

        cursor = self.db_conn.cursor()
        # SQLite ROWID is 1-indexed, FAISS indices are 0-indexed
        rowids = [idx + 1 for idx in indices]
        cursor.execute(query, rowids)

        results = []
        for row in cursor.fetchall():
            # Convert sqlite3.Row to dict
            meta = dict(row)
            results.append(meta)

        return results

    def search(
        self,
        query: str,
        top_k: int = 10,
        model: str = "text-embedding-3-small",
        similarity_threshold: float = 0.55,
    ) -> SearchResults:
        """
        Search papers using FAISS index + GCS FUSE SQLite lazy-loading.

        Architecture:
        1. Generate query embedding
        2. FAISS vector search (memory) → get candidate indices
        3. Filter by similarity threshold
        4. SQLite query (GCS FUSE) → fetch metadata for candidates only

        Args:
            query: Search query text
            top_k: Number of top results to return
            model: OpenAI embedding model to use
            similarity_threshold: Minimum cosine similarity score

        Returns:
            SearchResults object with top-k results above threshold
        """
        print(f"🔍 Searching for: '{query}' (FAISS + GCS FUSE SQLite)")

        # Step 1: Get query embedding
        query_embedding = self.openai_service.embedder(model=model, texts=query)
        query_vector = np.array(query_embedding, dtype=np.float32).reshape(1, -1)

        # Normalize query vector for cosine similarity
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            raise ValueError("Query embedding norm is zero.")
        query_normalized = query_vector / query_norm

        # Step 2: FAISS search (memory-resident vectors)
        # Search for more candidates to ensure we have enough above threshold
        search_k = min(top_k * 10, self.index.ntotal)
        similarities, indices = self.index.search(query_normalized, search_k)

        # Flatten results (FAISS returns 2D arrays)
        similarities = similarities[0]
        indices = indices[0]

        print(f"✓ FAISS search completed: {len(indices)} candidates retrieved")

        # Step 3: Filter by similarity threshold
        candidate_indices = []
        candidate_similarities = []
        for sim, idx in zip(similarities, indices):
            if sim > similarity_threshold and idx < self.index.ntotal:
                candidate_indices.append(int(idx))
                candidate_similarities.append(float(sim))

        print(f"📊 Matches above threshold ({similarity_threshold}): {len(candidate_indices)}")

        if not candidate_indices:
            print("⚠️ No papers found above similarity threshold.")
            return SearchResults(
                query=query,
                timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
                top_k=0,
                results=[],
            )

        # Step 4: Fetch metadata from GCS FUSE SQLite (only for candidates)
        print(f"🗄️  Fetching metadata for {len(candidate_indices)} papers from GCS FUSE SQLite...")
        metadata_list = self._fetch_metadata_by_indices(candidate_indices)

        if len(metadata_list) != len(candidate_indices):
            print(f"⚠️ Warning: Retrieved {len(metadata_list)} metadata entries for {len(candidate_indices)} indices")

        # Step 5: Combine results and take top_k
        top_k_actual = min(top_k, len(metadata_list))
        results = []

        for rank in range(top_k_actual):
            sim = candidate_similarities[rank]
            meta = metadata_list[rank]

            # Parse JSON fields
            authorships = json.loads(meta.get("authorships", "[]")) if meta.get("authorships") else []
            concepts_json = meta.get("concepts")
            concepts = json.loads(concepts_json) if concepts_json else []
            primary_location = json.loads(meta.get("primary_location", "{}")) if meta.get("primary_location") else {}
            counts_by_year = json.loads(meta.get("counts_by_year", "[]")) if meta.get("counts_by_year") else []
            referenced_works = json.loads(meta.get("referenced_works", "[]")) if meta.get("referenced_works") else []
            related_works = json.loads(meta.get("related_works", "[]")) if meta.get("related_works") else []

            result = SearchResult(
                rank=rank + 1,
                similarity_score=sim,
                id=meta.get("id", ""),
                title=meta.get("title", ""),
                doi=meta.get("doi") if meta.get("doi") is not None else "",
                abstract=meta.get("abstract", ""),
                authorships=authorships,
                concepts=concepts,
                fwci=float(meta.get("fwci")) if meta.get("fwci") else None,
                cited_by_count=int(meta.get("cited_by_count", 0)),
                publication_date=meta.get("publication_date", ""),
                primary_location=primary_location,
                counts_by_year=counts_by_year,
                referenced_works=referenced_works,
                related_works=related_works,
                type=meta.get("type", ""),
            )
            results.append(result)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"✅ Search completed: {len(results)} results returned")

        return SearchResults(
            query=query,
            timestamp=timestamp,
            top_k=top_k_actual,
            results=results,
        )

    def close(self):
        """Close SQLite connection"""
        if hasattr(self, 'db_conn'):
            self.db_conn.close()
            print("🗄️  SQLite connection closed")


def main():
    """Example usage (for local testing with GCS FUSE)"""
    # Initialize search engine
    search_engine = PaperSearchEngineFAISSGCSFuse()

    try:
        # Test search
        query = "quantum computing"
        results = search_engine.search(query=query, top_k=10, similarity_threshold=0.55)

        # Print brief summary
        print(f"\n✅ Top {len(results.results)} results:\n")
        for r in results.results[:3]:
            print(f"🏆 Rank {r.rank} | Score: {r.similarity_score:.3f} | {r.title[:80]}")
    finally:
        search_engine.close()


if __name__ == "__main__":
    main()
