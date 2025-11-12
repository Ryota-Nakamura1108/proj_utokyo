"""Convert split H5 embeddings to a single FAISS index for fast similarity search"""
import numpy as np
import h5py
import pickle
import faiss
import glob
import os
from tqdm import tqdm

def create_faiss_index(vectors_dir: str, output_dir: str):
    """
    Convert 82 split H5 files into a single FAISS index.

    Args:
        vectors_dir: Directory containing paper_embeddings_part_*.h5 and paper_metadata_part_*.pkl
        output_dir: Directory to save FAISS index and consolidated metadata
    """
    print("🚀 Starting FAISS index creation...")

    # Collect all embedding/metadata file pairs
    embedding_files = sorted(glob.glob(os.path.join(vectors_dir, "paper_embeddings_part_*.h5")))
    metadata_files = sorted(glob.glob(os.path.join(vectors_dir, "paper_metadata_part_*.pkl")))

    if not embedding_files or not metadata_files:
        raise FileNotFoundError(f"No embedding or metadata files found in {vectors_dir}")

    if len(embedding_files) != len(metadata_files):
        raise ValueError(f"Mismatch: {len(embedding_files)} embeddings vs {len(metadata_files)} metadata files")

    print(f"📂 Found {len(embedding_files)} file pairs")

    # Get embedding dimension from first file
    with h5py.File(embedding_files[0], "r") as h5f:
        first_embedding = h5f["embeddings"][0]
        dimension = len(first_embedding)

    print(f"📏 Embedding dimension: {dimension}")

    # Create FAISS index (Inner Product for cosine similarity)
    # We use IndexFlatIP because our embeddings are normalized
    index = faiss.IndexFlatIP(dimension)

    # Consolidated metadata list
    all_metadata = []
    total_vectors = 0

    # Process each file pair
    for i, (emb_path, meta_path) in enumerate(tqdm(zip(embedding_files, metadata_files),
                                                     total=len(embedding_files),
                                                     desc="Processing files"), 1):
        # Load embeddings
        with h5py.File(emb_path, "r") as h5f:
            embeddings = h5f["embeddings"][:]

            # Normalize embeddings for cosine similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings_normalized = embeddings / norms

            # Add to FAISS index
            index.add(embeddings_normalized.astype(np.float32))

            num_embeddings = embeddings.shape[0]
            total_vectors += num_embeddings

        # Load metadata
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)
            all_metadata.extend(metadata)

        print(f"  ✓ Processed chunk {i}/{len(embedding_files)}: {num_embeddings} vectors")

    print(f"\n✅ Index created with {total_vectors} vectors")
    print(f"✅ Metadata collected for {len(all_metadata)} papers")

    # Save FAISS index
    os.makedirs(output_dir, exist_ok=True)
    index_path = os.path.join(output_dir, "papers.index")
    faiss.write_index(index, index_path)
    print(f"💾 FAISS index saved to: {index_path}")

    # Save consolidated metadata
    metadata_path = os.path.join(output_dir, "papers_metadata.pkl")
    with open(metadata_path, "wb") as f:
        pickle.dump(all_metadata, f)
    print(f"💾 Metadata saved to: {metadata_path}")

    # Print size info
    index_size_mb = os.path.getsize(index_path) / (1024 * 1024)
    metadata_size_mb = os.path.getsize(metadata_path) / (1024 * 1024)
    print(f"\n📊 File sizes:")
    print(f"  - FAISS index: {index_size_mb:.1f} MB")
    print(f"  - Metadata: {metadata_size_mb:.1f} MB")
    print(f"  - Total: {index_size_mb + metadata_size_mb:.1f} MB")

    return index_path, metadata_path


if __name__ == "__main__":
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    VECTORS_DIR = os.path.join(BASE_DIR, "vectors")
    OUTPUT_DIR = os.path.join(BASE_DIR, "../faiss_index")

    print(f"Input directory: {VECTORS_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")

    # Create FAISS index
    index_path, metadata_path = create_faiss_index(VECTORS_DIR, OUTPUT_DIR)

    print("\n🎉 FAISS index creation complete!")
    print("\n📝 Next steps:")
    print("1. Upload to GCS:")
    print(f"   gsutil cp {index_path} gs://proj-utokyo-vectors/faiss/")
    print(f"   gsutil cp {metadata_path} gs://proj-utokyo-vectors/faiss/")
    print("2. Update article_search.py to use FAISS")
    print("3. Deploy to Cloud Run")
