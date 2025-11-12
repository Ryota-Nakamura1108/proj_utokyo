#!/usr/bin/env python3
"""東京大学所属著者の確認スクリプト。

このスクリプトは、authors_from_tokyo_univ.txtに記載された著者IDを読み込み、
各著者のlast_known_institutionが東京大学(I74801974)かどうかを確認します。
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from central_researcher.prepare import OpenAlexService, UNIV_TOKYO_ID

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 定数（config.pyから読み込み）
AUTHORS_FILE = Path(__file__).parent.parent / "papers" / "authors_from_tokyo_univ.txt"
OUTPUT_FILE = Path(__file__).parent.parent / "papers" / "verified_tokyo_univ_authors.txt"
REPORT_FILE = Path(__file__).parent.parent / "papers" / "verification_report.txt"
CACHE_FILE = Path(__file__).parent.parent / "papers" / "verification_cache.json"


def load_cache(cache_file: Path) -> Dict[str, Any]:
    """キャッシュファイルを読み込む。

    Args:
        cache_file: キャッシュファイルのパス

    Returns:
        キャッシュデータの辞書。キャッシュがない場合は空の辞書
    """
    if not cache_file.exists():
        logger.info("No cache file found. Starting from scratch.")
        return {}

    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            cache = json.load(f)
        logger.info(f"Loaded cache with {len(cache.get('processed', {}))} processed authors")
        return cache
    except Exception as e:
        logger.warning(f"Failed to load cache: {e}. Starting from scratch.")
        return {}


def save_cache(cache_file: Path, cache_data: Dict[str, Any]):
    """キャッシュをファイルに保存する。

    Args:
        cache_file: キャッシュファイルのパス
        cache_data: 保存するキャッシュデータ
    """
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        logger.debug(f"Cache saved with {len(cache_data.get('processed', {}))} processed authors")
    except Exception as e:
        logger.error(f"Failed to save cache: {e}")


def load_author_ids(file_path: Path) -> List[str]:
    """著者IDファイルを読み込む。

    Args:
        file_path: 著者IDが記載されたテキストファイルのパス

    Returns:
        著者IDのリスト
    """
    logger.info(f"Loading author IDs from: {file_path}")

    if not file_path.exists():
        raise FileNotFoundError(f"Author file not found: {file_path}")

    author_ids = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            author_id = line.strip()
            if author_id and not author_id.startswith("#"):
                author_ids.append(author_id)

    logger.info(f"Loaded {len(author_ids)} author IDs")
    return author_ids


def check_tokyo_affiliation(author_data: Dict[str, Any]) -> bool:
    """著者の所属機関に東京大学が含まれているか確認する。

    Args:
        author_data: OpenAlexから取得した著者データ

    Returns:
        東京大学に所属している場合True、それ以外False
    """
    last_known_institutions = author_data.get("last_known_institutions", [])

    for inst in last_known_institutions:
        if isinstance(inst, dict):
            inst_id = inst.get("id", "")
        else:
            inst_id = str(inst)

        # URLプレフィックスを除去
        clean_inst_id = inst_id.replace("https://openalex.org/", "")

        if clean_inst_id == UNIV_TOKYO_ID:
            return True

    return False


async def verify_authors(author_ids: List[str], cache_file: Path = CACHE_FILE) -> Dict[str, Any]:
    """著者の東京大学所属を確認する。

    Args:
        author_ids: 確認する著者IDのリスト
        cache_file: キャッシュファイルのパス

    Returns:
        確認結果を含む辞書
        - verified: 東京大学所属が確認できた著者IDリスト
        - not_verified: 東京大学所属が確認できなかった著者IDリスト
        - not_found: OpenAlexで見つからなかった著者IDリスト
        - details: 各著者の詳細情報
    """
    # キャッシュを読み込み
    cache = load_cache(cache_file)
    processed = cache.get("processed", {})

    logger.info(f"Starting verification for {len(author_ids)} authors")
    logger.info(f"Already processed: {len(processed)} authors")
    logger.info(f"Remaining: {len([aid for aid in author_ids if aid not in processed])} authors")

    verified = []
    not_verified = []
    not_found = []
    details = []

    # キャッシュから既に処理済みの結果を復元
    for author_id in author_ids:
        if author_id in processed:
            detail = processed[author_id]
            details.append(detail)

            if detail["is_tokyo_univ"]:
                verified.append(author_id)
            elif detail["name"] in ["Not Found", "Error"]:
                not_found.append(author_id)
            else:
                not_verified.append(author_id)

    async with OpenAlexService() as service:
        # 1件ずつ著者データを取得（403エラーを避けるため）
        total_count = 0
        for idx, author_id in enumerate(author_ids, 1):
            # すでに処理済みの場合はスキップ
            if author_id in processed:
                continue

            total_count += 1

            if total_count % 100 == 0 or total_count == 1:
                logger.info(f"Processing {total_count}/{len(author_ids) - len(processed)} remaining authors (total: {idx}/{len(author_ids)})")

            try:
                # 個別APIエンドポイントを直接使用（バッチクエリで403エラーが出るため）
                params = {
                    "select": "id,display_name,orcid,summary_stats,works_count,last_known_institutions"
                }
                author = await service._make_request(f"/authors/{author_id}", params)

                # 実際の著者IDを抽出（マージされている可能性があるため）
                actual_author_id = author.get("id", "").replace("https://openalex.org/", "")
                is_tokyo = check_tokyo_affiliation(author)

                # 所属機関名のリスト
                institution_names = []
                for inst in author.get("last_known_institutions", []):
                    if isinstance(inst, dict):
                        institution_names.append(inst.get("display_name", "Unknown"))
                    else:
                        institution_names.append(str(inst))

                detail = {
                    "author_id": author_id,
                    "actual_id": actual_author_id,
                    "name": author.get("display_name", "Unknown"),
                    "is_tokyo_univ": is_tokyo,
                    "institutions": institution_names,
                    "works_count": author.get("works_count", 0)
                }
                details.append(detail)

                if is_tokyo:
                    verified.append(author_id)
                    if total_count % 100 == 0:
                        logger.info(
                            f"✓ {author_id} → {actual_author_id} ({detail['name']}): Tokyo University confirmed"
                        )
                else:
                    not_verified.append(author_id)
                    if total_count % 100 == 0:
                        logger.warning(
                            f"✗ {author_id} → {actual_author_id} ({detail['name']}): "
                            f"Not affiliated with Tokyo University. "
                            f"Current: {', '.join(institution_names)}"
                        )

            except Exception as e:
                logger.error(f"✗ {author_id}: Error - {str(e)}")
                not_found.append(author_id)
                detail = {
                    "author_id": author_id,
                    "actual_id": author_id,
                    "name": "Error",
                    "is_tokyo_univ": False,
                    "institutions": [],
                    "works_count": 0
                }
                details.append(detail)

            # キャッシュに追加
            processed[author_id] = detail

            # 10件ごとにキャッシュを保存（途中で中断しても再開できるように）
            if total_count % 10 == 0:
                save_cache(cache_file, {"processed": processed})

            # レート制限を守るため、少し待機
            await asyncio.sleep(0.11)

    # 最終キャッシュを保存
    save_cache(cache_file, {"processed": processed})

    logger.info(f"Verification complete:")
    logger.info(f"  - Verified: {len(verified)}")
    logger.info(f"  - Not verified: {len(not_verified)}")
    logger.info(f"  - Not found: {len(not_found)}")

    return {
        "verified": verified,
        "not_verified": not_verified,
        "not_found": not_found,
        "details": details
    }


def save_results(results: Dict[str, Any], output_file: Path, report_file: Path):
    """確認結果をファイルに保存する。

    Args:
        results: verify_authorsの戻り値
        output_file: 確認済み著者IDを保存するファイル
        report_file: 詳細レポートを保存するファイル
    """
    # 確認済み著者IDを保存
    logger.info(f"Saving verified author IDs to: {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        for author_id in results["verified"]:
            f.write(f"{author_id}\n")

    # 詳細レポートを保存
    logger.info(f"Saving detailed report to: {report_file}")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("東京大学所属著者確認レポート\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"総著者数: {len(results['details'])}\n")
        f.write(f"確認済み: {len(results['verified'])} ({len(results['verified'])/len(results['details'])*100:.1f}%)\n")
        f.write(f"未確認: {len(results['not_verified'])} ({len(results['not_verified'])/len(results['details'])*100:.1f}%)\n")
        f.write(f"見つからず: {len(results['not_found'])} ({len(results['not_found'])/len(results['details'])*100:.1f}%)\n")
        f.write("\n" + "=" * 80 + "\n\n")

        # 確認済み著者
        f.write("【確認済み著者】\n")
        f.write("-" * 80 + "\n")
        for detail in results["details"]:
            if detail["is_tokyo_univ"]:
                f.write(f"ID: {detail['author_id']}\n")
                f.write(f"名前: {detail['name']}\n")
                f.write(f"論文数: {detail['works_count']}\n")
                f.write(f"所属: {', '.join(detail['institutions'])}\n")
                f.write("-" * 80 + "\n")

        f.write("\n")

        # 未確認著者（他機関所属）
        if results["not_verified"]:
            f.write("【未確認著者（他機関所属）】\n")
            f.write("-" * 80 + "\n")
            for detail in results["details"]:
                if not detail["is_tokyo_univ"] and detail["name"] != "Not Found" and detail["name"] != "Error":
                    f.write(f"ID: {detail['author_id']}\n")
                    f.write(f"名前: {detail['name']}\n")
                    f.write(f"論文数: {detail['works_count']}\n")
                    f.write(f"所属: {', '.join(detail['institutions']) if detail['institutions'] else '不明'}\n")
                    f.write("-" * 80 + "\n")
            f.write("\n")

        # 見つからなかった著者
        if results["not_found"]:
            f.write("【OpenAlexで見つからなかった著者】\n")
            f.write("-" * 80 + "\n")
            for author_id in results["not_found"]:
                f.write(f"{author_id}\n")
            f.write("-" * 80 + "\n")

    logger.info("Results saved successfully")


async def main():
    """メイン処理。"""
    try:
        # 著者IDを読み込み
        author_ids = load_author_ids(AUTHORS_FILE)

        # 所属を確認（キャッシュ機能付き）
        results = await verify_authors(author_ids, CACHE_FILE)

        # 結果を保存
        save_results(results, OUTPUT_FILE, REPORT_FILE)

        # サマリー表示
        print("\n" + "=" * 80)
        print("確認完了")
        print("=" * 80)
        print(f"総著者数: {len(results['details'])}")
        print(f"✓ 東京大学所属確認: {len(results['verified'])} 名")
        print(f"✗ 他機関所属: {len(results['not_verified'])} 名")
        print(f"✗ 見つからず: {len(results['not_found'])} 名")
        print()
        print(f"確認済み著者ID: {OUTPUT_FILE}")
        print(f"詳細レポート: {REPORT_FILE}")
        print(f"キャッシュファイル: {CACHE_FILE}")
        print("=" * 80)
        print("\n※途中で中断した場合も、再実行すれば自動的に途中から再開されます。")

    except KeyboardInterrupt:
        logger.info("\n処理が中断されました。次回実行時に途中から再開します。")
        print("\n" + "=" * 80)
        print("処理が中断されました")
        print("=" * 80)
        print(f"進捗はキャッシュに保存されています: {CACHE_FILE}")
        print("再度スクリプトを実行すれば、途中から再開できます。")
        print("=" * 80)
    except Exception as e:
        logger.exception(f"Error in main: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
