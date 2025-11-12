"""Split large vector file into 10K chunks"""
import h5py
import pickle
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def split_vector_file(
    embedding_path: str,
    metadata_path: str,
    output_dir: str,
    chunk_size: int = 10000,
    start_index: int = 51
):
    """
    Split a large vector file into smaller chunks.

    Args:
        embedding_path: Path to large .h5 file
        metadata_path: Path to large .pkl file
        output_dir: Output directory
        chunk_size: Papers per chunk (default 10000)
        start_index: Starting index for output files
    """
    print(f"Loading {embedding_path}...")

    # Load embeddings
    with h5py.File(embedding_path, "r") as h5f:
        embeddings = h5f["embeddings"][:]

    # Load metadata
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    total_papers = len(embeddings)
    print(f"Total papers: {total_papers:,}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Metadata count: {len(metadata)}")

    if len(embeddings) != len(metadata):
        raise ValueError("Embedding and metadata counts do not match!")

    # Calculate number of chunks
    num_chunks = (total_papers + chunk_size - 1) // chunk_size
    print(f"\nSplitting into {num_chunks} chunks of {chunk_size:,} papers each...")

    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, total_papers)
        file_index = start_index + i

        # Extract chunk
        chunk_embeddings = embeddings[start:end]
        chunk_metadata = metadata[start:end]

        # Save embedding
        emb_path = os.path.join(output_dir, f"paper_embeddings_part_{file_index:04d}.h5")
        with h5py.File(emb_path, "w") as h5f:
            h5f.create_dataset(
                "embeddings",
                data=chunk_embeddings,
                compression="gzip",
                compression_opts=4,
            )

        # Save metadata
        meta_path = os.path.join(output_dir, f"paper_metadata_part_{file_index:04d}.pkl")
        with open(meta_path, "wb") as f:
            pickle.dump(chunk_metadata, f)

        print(f"✓ Saved part_{file_index:04d}: {len(chunk_metadata):,} papers")

    print(f"\n✅ Split complete! Created {num_chunks} chunks")
    print(f"Files: part_{start_index:04d} to part_{start_index + num_chunks - 1:04d}")


if __name__ == "__main__":
    vectors_dir = os.path.join(BASE_DIR, "../vectors")

    embedding_file = os.path.join(vectors_dir, "paper_embeddings_part_0051.h5")
    metadata_file = os.path.join(vectors_dir, "paper_metadata_part_0051.pkl")

    if not os.path.exists(embedding_file):
        print(f"❌ File not found: {embedding_file}")
        exit(1)

    if not os.path.exists(metadata_file):
        print(f"❌ File not found: {metadata_file}")
        exit(1)

    split_vector_file(
        embedding_path=embedding_file,
        metadata_path=metadata_file,
        output_dir=vectors_dir,
        chunk_size=10000,
        start_index=51
    )

    print("\n📝 Next steps:")
    print("1. Delete the original part_0051 files:")
    print(f"   rm {embedding_file}")
    print(f"   rm {metadata_file}")
    print("2. Verify the new files are correct")
