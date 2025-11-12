#!/usr/bin/env python3
"""東京大学所属著者の確認スクリプト（サンプル版）。

このスクリプトは、authors_from_tokyo_univ.txtから少数のサンプルを取得し、
各著者のlast_known_institutionが東京大学(I74801974)かどうかを確認します。
"""

import asyncio
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

# 定数
AUTHORS_FILE = Path(__file__).parent.parent / "papers" / "authors_from_tokyo_univ.txt"
SAMPLE_SIZE = 100  # サンプル数


def load_sample_author_ids(file_path: Path, sample_size: int) -> List[str]:
    """著者IDファイルからサンプルを読み込む。

    Args:
        file_path: 著者IDが記載されたテキストファイルのパス
        sample_size: サンプル数

    Returns:
        著者IDのリスト
    """
    logger.info(f"Loading sample author IDs from: {file_path}")

    if not file_path.exists():
        raise FileNotFoundError(f"Author file not found: {file_path}")

    author_ids = []
    with open(file_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= sample_size:
                break
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


async def verify_authors(author_ids: List[str]) -> Dict[str, Any]:
    """著者の東京大学所属を確認する。

    Args:
        author_ids: 確認する著者IDのリスト

    Returns:
        確認結果を含む辞書
    """
    logger.info(f"Starting verification for {len(author_ids)} authors")

    verified = []
    not_verified = []
    not_found = []
    details = []

    async with OpenAlexService() as service:
        # 1件ずつ著者データを取得（バッチクエリで403エラーが出るため）
        for idx, author_id in enumerate(author_ids, 1):
            if idx % 10 == 0 or idx == 1:
                logger.info(f"Processing {idx}/{len(author_ids)} authors")

            try:
                # 個別APIエンドポイントを直接使用
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
                    if idx % 10 == 0:
                        logger.info(
                            f"✓ {author_id} → {actual_author_id} ({detail['name']}): Tokyo University confirmed"
                        )
                else:
                    not_verified.append(author_id)
                    if idx % 10 == 0:
                        logger.warning(
                            f"✗ {author_id} → {actual_author_id} ({detail['name']}): "
                            f"Not affiliated with Tokyo University. "
                            f"Current: {', '.join(institution_names)}"
                        )

                # レート制限を守るため、少し待機
                await asyncio.sleep(0.11)

            except Exception as e:
                logger.error(f"✗ {author_id}: Error - {str(e)}")
                not_found.append(author_id)
                details.append({
                    "author_id": author_id,
                    "actual_id": author_id,
                    "name": "Error",
                    "is_tokyo_univ": False,
                    "institutions": [],
                    "works_count": 0
                })

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


async def main():
    """メイン処理。"""
    try:
        # サンプル著者IDを読み込み
        author_ids = load_sample_author_ids(AUTHORS_FILE, SAMPLE_SIZE)

        # 所属を確認
        results = await verify_authors(author_ids)

        # サマリー表示
        print("\n" + "=" * 80)
        print(f"サンプル確認完了（{SAMPLE_SIZE}件）")
        print("=" * 80)
        print(f"総著者数: {len(results['details'])}")
        print(f"✓ 東京大学所属確認: {len(results['verified'])} 名 ({len(results['verified'])/len(results['details'])*100:.1f}%)")
        print(f"✗ 他機関所属: {len(results['not_verified'])} 名 ({len(results['not_verified'])/len(results['details'])*100:.1f}%)")
        print(f"✗ 見つからず: {len(results['not_found'])} 名 ({len(results['not_found'])/len(results['details'])*100:.1f}%)")
        print("=" * 80)

        # 詳細表示
        if results['verified']:
            print("\n【確認済み著者（最初の5件）】")
            for detail in results['details'][:5]:
                if detail['is_tokyo_univ']:
                    print(f"  - {detail['author_id']}: {detail['name']} ({detail['works_count']} works)")

        if results['not_verified']:
            print("\n【未確認著者（他機関所属、最初の5件）】")
            count = 0
            for detail in results['details']:
                if not detail['is_tokyo_univ'] and detail['name'] not in ["Not Found", "Error"]:
                    print(f"  - {detail['author_id']}: {detail['name']} @ {', '.join(detail['institutions'])}")
                    count += 1
                    if count >= 5:
                        break

    except Exception as e:
        logger.exception(f"Error in main: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
