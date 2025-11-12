"""Retry failed JSON files only"""
from embed_article import EmbedPapers
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# エラーログから抽出した失敗したファイル
FAILED_FILES = [
    "_0035.json",
    "_0019.json",
    "_0039.json",
    "_0042.json",
    "_0015.json",
    "_0003.json",
    "_0008.json",
    "_0032.json",
    "_0024.json",
    "_0004.json",
    "_0028.json",
    "_0029.json",
    "_0013.json",
    "_0005.json",
    "_0025.json",
    "_0033.json",
    "_0009.json",
]


class RetryFailedEmbeddings(EmbedPapers):
    def build_embeddings_for_failed_files(
        self,
        dir_path: str,
        failed_files: list,
        output_dir: str,
        model: str = "text-embedding-3-small",
        batch_size: int = 30,
    ):
        """
        Process only the failed JSON files.

        Args:
            dir_path: Directory containing JSON files
            failed_files: List of failed file names
            output_dir: Output directory for embeddings
            model: OpenAI embedding model
            batch_size: Batch size for API calls
        """
        os.makedirs(output_dir, exist_ok=True)

        # Build full paths for failed files
        failed_paths = [
            os.path.join(dir_path, fname) for fname in failed_files
        ]

        print(f"Found {len(failed_paths)} failed files to retry")

        # 既存の最後のファイルインデックスを取得
        existing_files = [f for f in os.listdir(output_dir) if f.startswith("paper_embeddings_part_")]
        if existing_files:
            last_index = max([int(f.split("_")[-1].replace(".h5", "")) for f in existing_files])
            file_index = last_index + 1
        else:
            file_index = 51  # part_0050の次から始める

        all_embeddings = []
        all_metadata = []

        for jpath in failed_paths:
            if not os.path.exists(jpath):
                print(f"⚠️ File not found: {jpath}")
                continue

            try:
                print(f"\n📄 Processing: {os.path.basename(jpath)}")
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

                # Batch processing
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
                            print(f"✓ Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
                            break
                        except Exception as e:
                            if attempt < retries - 1:
                                import time
                                wait_time = 2 ** attempt
                                print(f"⚠️ API error, retrying in {wait_time}s: {e}")
                                time.sleep(wait_time)
                            else:
                                print(f"❌ Failed after {retries} attempts: {e}")
                                raise

                print(f"✅ Completed: {os.path.basename(jpath)} ({len(batch_metadata)} papers)")

            except Exception as e:
                print(f"❌ Error processing {jpath}: {e}")
                continue

        # Save all collected embeddings
        if all_embeddings:
            self._save_chunk(all_embeddings, all_metadata, output_dir, file_index)
            print(f"\n✅ Saved {len(all_metadata)} papers from failed files")
        else:
            print("\n⚠️ No embeddings to save")


if __name__ == "__main__":
    dir_path = os.path.join(BASE_DIR, "../../papers")
    output_dir = os.path.join(BASE_DIR, "../vectors")

    embedder = RetryFailedEmbeddings()
    embedder.build_embeddings_for_failed_files(
        dir_path=dir_path,
        failed_files=FAILED_FILES,
        output_dir=output_dir,
        model="text-embedding-3-small",
        batch_size=30,
    )
