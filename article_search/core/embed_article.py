"""Build embeddings for approximately 400K papers (split every 10K)"""
from openai_services import OpenAIService

import os
import json
import numpy as np
import h5py
import pickle
from tqdm import tqdm
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 

class EmbedPapers:
    def __init__(self):
        self.openai_service = OpenAIService()

    def get_all_json_file_paths(self, dir: str):
        """Get all JSON file paths in a directory"""
        json_files = []
        for root, _, files in os.walk(dir):
            for file in files:
                if file.endswith(".json"):
                    json_files.append(os.path.join(root, file))
        return json_files

    def load_json(self, path: str):
        """Load JSON file"""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def build_embeddings_and_save(
        self,
        dir_path: str,
        output_dir: str,
        model: str = "text-embedding-3-small",
        batch_size: int = 30,
        chunk_size: int = 10000,  # 🔸 1万件ごとに分割保存
    ):
        """
        Read JSON files, embed titles/abstracts, save to multiple HDF5 + metadata files.

        Args:
            dir_path: Directory containing JSON files
            output_dir: Output directory for embeddings
            model: OpenAI embedding model
            batch_size: Batch size for API calls
            chunk_size: Number of papers per saved file
        """
        os.makedirs(output_dir, exist_ok=True)

        json_paths = self.get_all_json_file_paths(dir_path)
        print(os.path.abspath(dir_path))
        print(f"Found {len(json_paths)} JSON files")

        all_embeddings = []
        all_metadata = []
        paper_count = 0
        file_index = 1  # ファイル番号（part_0001などに使う）

        for jpath in tqdm(json_paths, desc="Processing JSON files"):
            try:
                papers = self.load_json(jpath)
                if not isinstance(papers, list):
                    continue

                texts = []
                batch_metadata = []

                for paper in papers:
                    title = paper.get("title", "")
                    abstract_index = paper.get("abstract_inverted_index", {})
                    abstract = self._reconstruct_abstract(abstract_index)
                    if not abstract:
                        abstract = paper.get("abstract", "")

                    text = f"{title}\n{abstract}".strip()
                    if not text:
                        continue

                    texts.append(text)
                    batch_metadata.append({
                        "id": paper.get("id", ""),
                        "title": title,
                        "doi": paper.get("doi", ""),
                        "abstract": abstract,
                        "authorships": paper.get("authorships", []),
                        "concepts": paper.get("concepts", []),
                        "fwci": paper.get("fwci", ""),
                        "cited_by_count": paper.get("cited_by_count", ""),
                        "publication_date": paper.get("publication_date", ""),
                        "primary_location": paper.get("primary_location", ""),
                        "counts_by_year": paper.get("counts_by_year", []),
                        "referenced_works": paper.get("referenced_works", []),
                        "related_works": paper.get("related_works", []),
                        "type": paper.get("type", ""),
                    })

                # --- 埋め込み処理 (100件ずつ) ---
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    batch_meta = batch_metadata[i:i + batch_size]

                    retries = 3
                    for attempt in range(retries):
                        try:
                            vectors = self.openai_service.batch_embedder(
                                model=model,
                                texts=batch_texts
                            )
                            all_embeddings.extend(vectors)
                            all_metadata.extend(batch_meta)
                            paper_count += len(vectors)
                            break
                        except Exception as e:
                            if attempt < retries - 1:
                                wait_time = 2 ** attempt
                                print(f"\n⚠️ API error, retrying in {wait_time}s: {e}")
                                time.sleep(wait_time)
                            else:
                                print(f"\n❌ Failed after {retries} attempts: {e}")
                                raise

                    time.sleep(0.1)

                    # --- 🔸1万件たまったら保存 ---
                    if len(all_embeddings) >= chunk_size:
                        self._save_chunk(
                            all_embeddings, all_metadata, output_dir, file_index
                        )
                        file_index += 1
                        all_embeddings.clear()
                        all_metadata.clear()

            except Exception as e:
                print(f"\n❌ Error processing {jpath}: {e}")
                continue

        # --- 最後の余りを保存 ---
        if all_embeddings:
            self._save_chunk(all_embeddings, all_metadata, output_dir, file_index)

        print(f"\n✅ Completed! Total papers processed: {paper_count}")

    def _save_chunk(self, embeddings, metadata, output_dir, index):
        """Save one chunk of embeddings and metadata"""
        emb_array = np.array(embeddings, dtype=np.float32)

        embedding_path = os.path.join(
            output_dir, f"paper_embeddings_part_{index:04d}.h5"
        )
        metadata_path = os.path.join(
            output_dir, f"paper_metadata_part_{index:04d}.pkl"
        )

        with h5py.File(embedding_path, "w") as h5f:
            h5f.create_dataset(
                "embeddings",
                data=emb_array,
                compression="gzip",
                compression_opts=4,
            )

        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)

        print(
            f"\n💾 Saved part {index:04d}: {len(metadata)} papers → "
            f"{os.path.basename(embedding_path)}"
        )

    def _reconstruct_abstract(self, inverted_index: dict) -> str:
        """Reconstruct abstract from inverted index"""
        if not inverted_index:
            return ""
        try:
            word_positions = []
            for word, positions in inverted_index.items():
                for pos in positions:
                    word_positions.append((pos, word))
            word_positions.sort(key=lambda x: x[0])
            return " ".join([word for _, word in word_positions])
        except Exception:
            return ""


if __name__ == "__main__":
    dir_path = os.path.join(BASE_DIR, "../../papers")   # JSONファイル群のディレクトリ
    output_dir = os.path.join(BASE_DIR, "../vectors")

    embedder = EmbedPapers()
    embedder.build_embeddings_and_save(
        dir_path=dir_path,
        output_dir=output_dir,
        model="text-embedding-3-small",
        batch_size=30,
        chunk_size=10000,  # 🔸ここで分割サイズを設定
    )
