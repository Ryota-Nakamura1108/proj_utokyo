"""Search papers using split embeddings (10K per file) with parallel processing"""
from .openai_services import OpenAIService

from pydantic import BaseModel, Field
from typing import List, Optional, Any
from datetime import datetime
import numpy as np
import h5py
import pickle
import pandas as pd
import os
import glob
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed

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


class PaperSearchEngine:
    def __init__(self, vectors_dir: str, max_workers: int = 16):
        """
        Initialize search engine for multiple split embedding files.
        Args:
            vectors_dir: Directory containing .h5 and .pkl split files
            max_workers: Number of parallel workers (default: 16, optimized for 32GB memory, ~36sec processing)
        """
        self.openai_service = OpenAIService()
        self.vectors_dir = vectors_dir
        self.max_workers = max_workers

        # Collect all embedding/metadata file pairs
        self.embedding_files = sorted(glob.glob(os.path.join(vectors_dir, "paper_embeddings_part_*.h5")))
        self.metadata_files = sorted(glob.glob(os.path.join(vectors_dir, "paper_metadata_part_*.pkl")))

        if not self.embedding_files or not self.metadata_files:
            raise FileNotFoundError(f"No split embedding or metadata files found in {vectors_dir}")

        if len(self.embedding_files) != len(self.metadata_files):
            raise ValueError("Embedding and metadata file counts do not match")

        print(f"📂 Found {len(self.embedding_files)} embedding chunks in '{vectors_dir}' (using {max_workers} workers)")

    def _process_chunk(self, chunk_idx: int, emb_path: str, meta_path: str,
                      query_normalized: np.ndarray, similarity_threshold: float, top_k: int):
        """Process a single chunk file (used in parallel processing)"""
        chunk_matches = []

        try:
            # Load and process embeddings in batches
            with h5py.File(emb_path, "r") as h5f:
                emb_dataset = h5f["embeddings"]
                num_embeddings = emb_dataset.shape[0]

                batch_size = 500  # Reduced from 1000 to minimize memory usage
                for batch_start in range(0, num_embeddings, batch_size):
                    batch_end = min(batch_start + batch_size, num_embeddings)
                    embeddings_batch = emb_dataset[batch_start:batch_end]

                    # Normalize embeddings
                    norms = np.linalg.norm(embeddings_batch, axis=1, keepdims=True)
                    embeddings_normalized = embeddings_batch / norms

                    # Compute cosine similarities
                    sims = embeddings_normalized @ query_normalized

                    # Filter by threshold
                    mask = sims > similarity_threshold
                    if np.any(mask):
                        filtered_indices = np.where(mask)[0]
                        for idx in filtered_indices:
                            global_idx = batch_start + idx
                            chunk_matches.append((float(sims[idx]), global_idx))

                    del embeddings_batch, norms, embeddings_normalized, sims, mask

            # Load metadata for matches
            results = []
            if chunk_matches:
                with open(meta_path, "rb") as f:
                    metadata = pickle.load(f)

                for sim, idx in chunk_matches:
                    results.append((sim, metadata[idx]))

                del metadata

            return (chunk_idx, results, len(chunk_matches))

        except Exception as e:
            print(f"⚠️  Error processing chunk {chunk_idx}: {e}")
            return (chunk_idx, [], 0)
        finally:
            gc.collect()

    def search(
        self,
        query: str,
        top_k: int = 10,
        model: str = "text-embedding-3-small",
        similarity_threshold: float = 0.55,
    ) -> SearchResults:
        """
        Search papers across all embedding chunks using parallel processing.
        """
        print(f"🔍 Searching for: '{query}' (parallel: {self.max_workers} workers)")

        # Get query embedding
        query_embedding = self.openai_service.embedder(model=model, texts=query)
        query_vector = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            raise ValueError("Query embedding norm is zero.")
        query_normalized = query_vector / query_norm

        # Parallel processing of all chunks
        candidate_results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all chunks for parallel processing
            futures = {}
            for i, (emb_path, meta_path) in enumerate(zip(self.embedding_files, self.metadata_files)):
                future = executor.submit(
                    self._process_chunk,
                    i + 1,  # chunk_idx (1-indexed for display)
                    emb_path,
                    meta_path,
                    query_normalized,
                    similarity_threshold,
                    top_k
                )
                futures[future] = (i + 1, emb_path)

            # Collect results as they complete
            total_matches = 0
            for future in as_completed(futures):
                chunk_idx, emb_path = futures[future]
                try:
                    chunk_idx, results, match_count = future.result()
                    if results:
                        candidate_results.extend(results)
                        total_matches += match_count
                        print(f"✓ Chunk {chunk_idx}/{len(self.embedding_files)}: {match_count} matches")
                    else:
                        print(f"  Chunk {chunk_idx}/{len(self.embedding_files)}: 0 matches")
                except Exception as e:
                    print(f"⚠️  Error in chunk {chunk_idx}: {e}")

        print(f"\n📊 Total matches across all chunks: {total_matches}")

        if not candidate_results:
            print("⚠️ No papers found above similarity threshold.")
            return SearchResults(
                query=query,
                timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
                top_k=0,
                results=[],
            )

        # Sort by similarity (descending)
        candidate_results.sort(key=lambda x: x[0], reverse=True)

        # Keep top-k
        top_k_actual = min(top_k, len(candidate_results))
        top_results = candidate_results[:top_k_actual]

        results = []
        for rank, (sim, meta) in enumerate(top_results, 1):
            # Preserve full authorship structure with author IDs (not just names)
            authorships = []

            if meta.get("authorships"):
                for authorship in meta["authorships"]:
                    if isinstance(authorship, dict):
                        # Keep the full authorship dict to preserve author IDs
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

    def save_to_csv(self, search_results: SearchResults, output_dir: str = "search_results"):
        """Save results to CSV"""
        os.makedirs(output_dir, exist_ok=True)

        clean_query = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in search_results.query)
        clean_query = clean_query.replace(" ", "_")[:50]
        filename = f"{clean_query}_{search_results.timestamp}.csv"
        filepath = os.path.join(output_dir, filename)

        rows = []
        for r in search_results.results:
            rows.append({
                "rank": r.rank,
                "similarity_score": r.similarity_score,
                "id": r.id,
                "title": r.title,
                "doi": r.doi,
                "abstract": r.abstract,
                "authors": ", ".join(r.authorships),
                "fwci": r.fwci,
                "cited_by_count": r.cited_by_count,
                "publication_date": r.publication_date,
                "type": r.type,
                "concepts": str(r.concepts),
                "referenced_works_count": len(r.referenced_works or []),
                "related_works_count": len(r.related_works or []),
            })

        pd.DataFrame(rows).to_csv(filepath, index=False, encoding="utf-8-sig")
        print(f"💾 Results saved to: {filepath}")
        return filepath


def main():
    """Example usage"""
    VECTOR_DIR = os.path.join(BASE_DIR, "../vectors")  # contains paper_embeddings_part_XXXX.h5 / pkl pairs
    OUTPUT_DIR = os.path.join(BASE_DIR, "../search_results")

    search_engine = PaperSearchEngine(vectors_dir=VECTOR_DIR)
    query = "quantum computing"

    results = search_engine.search(query=query, top_k=10, similarity_threshold=0)

    # Print brief summary
    print(f"\n✅ Top {len(results.results)} results above threshold:\n")
    for r in results.results[:3]:
        print(f"🏆 Rank {r.rank} | Score: {r.similarity_score:.3f} | {r.title[:80]}")

    # Save to CSV
    search_engine.save_to_csv(results, output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    main()
