"""Researcher list management for University of Tokyo.

東京大学の研究者リストを取得・管理するモジュール。
既存のResearcherListManagerと同じ機能を提供します（後方互換性維持）。
"""

import json
import asyncio
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

from .openalex_service import OpenAlexService
from .config import config, RESEARCHER_LIST_DIR, UNIV_TOKYO_ID

logger = logging.getLogger(__name__)


def sanitize_filename(name: str) -> str:
    """ファイル名として安全な文字列に変換

    Args:
        name: 元の名前

    Returns:
        サニタイズされた名前
    """
    # 使用できない文字を置換
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    # 連続するスペースやアンダースコアを1つに
    name = re.sub(r'[\s_]+', '_', name)
    # 前後の空白とアンダースコアを削除
    name = name.strip('_ ')
    # 長すぎる場合は切り詰め
    if len(name) > 200:
        name = name[:200]
    return name


class ResearcherManager:
    """東京大学の研究者リスト管理

    OpenAlexServiceを使用して研究者リストを取得し、キャッシュします。

    Examples:
        # 非同期版
        manager = ResearcherManager()
        researchers = await manager.fetch_researchers(min_works=10)

        # 同期版
        researchers = ResearcherManager.fetch_researchers_sync(min_works=10)
    """

    def __init__(
        self,
        institution_id: str = UNIV_TOKYO_ID,
        cache_dir: Optional[str] = None,
    ):
        """初期化

        Args:
            institution_id: OpenAlex機関ID（デフォルト: 東京大学）
            cache_dir: キャッシュディレクトリ
        """
        self.institution_id = institution_id

        if cache_dir is None:
            cache_dir = str(RESEARCHER_LIST_DIR)

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.cache_file = self.cache_dir / f"{self.institution_id}_researchers.json"

    async def fetch_researchers(
        self,
        min_works: int = 5,
        force_refresh: bool = False,
        cache_days: int = 30
    ) -> List[Dict[str, Any]]:
        """研究者リストを取得（非同期版）

        Args:
            min_works: 最小論文数
            force_refresh: キャッシュを無視して再取得
            cache_days: キャッシュ有効期限（日数）

        Returns:
            研究者辞書のリスト（id, name, works_count を含む）

        Examples:
            manager = ResearcherManager()
            researchers = await manager.fetch_researchers(min_works=10)
        """
        logger.info(f"Fetching researchers for institution: {self.institution_id}")

        # OpenAlexServiceを使用（内部でキャッシュを処理）
        async with OpenAlexService() as service:
            researchers = await service.fetch_researchers_by_institution(
                institution_id=self.institution_id,
                min_works=min_works,
                cache_dir=self.cache_dir,
                force_refresh=force_refresh,
                cache_days=cache_days
            )

        logger.info(f"Retrieved {len(researchers)} researchers")
        return researchers

    @staticmethod
    def fetch_researchers_sync(
        institution_id: str = UNIV_TOKYO_ID,
        min_works: int = 5,
        force_refresh: bool = False,
        cache_days: int = 30,
    ) -> List[Dict[str, Any]]:
        """研究者リストを取得（同期版）

        Args:
            institution_id: OpenAlex機関ID
            min_works: 最小論文数
            force_refresh: キャッシュを無視して再取得
            cache_days: キャッシュ有効期限（日数）

        Returns:
            研究者辞書のリスト

        Examples:
            # CLIから使用
            researchers = ResearcherManager.fetch_researchers_sync(min_works=10)
        """
        manager = ResearcherManager(institution_id=institution_id)
        return asyncio.run(manager.fetch_researchers(
            min_works=min_works,
            force_refresh=force_refresh,
            cache_days=cache_days
        ))

    def load_cached_list(self) -> Optional[List[Dict[str, Any]]]:
        """キャッシュから研究者リストを読み込み

        Returns:
            研究者リスト、またはキャッシュが存在しない場合はNone
        """
        if not self.cache_file.exists():
            return None

        with open(self.cache_file, "r", encoding="utf-8") as f:
            cached_data = json.load(f)

        return cached_data.get("researchers", [])

    def save_custom_list(
        self,
        researchers: List[Dict[str, Any]],
        output_path: Optional[str] = None
    ) -> str:
        """カスタム研究者リストを保存

        Args:
            researchers: 研究者辞書のリスト
            output_path: 保存先パス（Noneの場合はデフォルト位置）

        Returns:
            保存されたファイルパス
        """
        if output_path is None:
            output_path = self.cache_dir / "custom_researchers.json"
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({
                "institution_id": self.institution_id,
                "total_researchers": len(researchers),
                "researchers": researchers
            }, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(researchers)} researchers to {output_path}")
        return str(output_path)

    def save_researchers_names_to_file(
        self,
        researchers: List[Dict[str, Any]],
        output_path: Optional[str] = None
    ) -> str:
        """研究者の名前をテキストファイルに保存

        Args:
            researchers: 研究者辞書のリスト
            output_path: 保存先ファイルパス（Noneの場合はデフォルト位置）

        Returns:
            保存されたファイルパス
        """
        if output_path is None:
            output_path = self.cache_dir / "researchers_names.txt"
        else:
            output_path = Path(output_path)

        # ディレクトリを作成
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 全研究者の名前をファイルに書き込む
        with open(output_path, "w", encoding="utf-8") as f:
            for researcher in researchers:
                name = researcher.get("name", "Unknown")
                f.write(f"{name}\n")

        logger.info(f"Saved {len(researchers)} researcher names to {output_path}")
        return str(output_path)

    def save_researchers_ids_to_file(
        self,
        researchers: List[Dict[str, Any]],
        output_path: Optional[str] = None
    ) -> str:
        """研究者のOpenAlex IDをテキストファイルに保存

        Args:
            researchers: 研究者辞書のリスト
            output_path: 保存先ファイルパス（Noneの場合はデフォルト位置）

        Returns:
            保存されたファイルパス
        """
        if output_path is None:
            output_path = self.cache_dir / "researchers_ids.txt"
        else:
            output_path = Path(output_path)

        # ディレクトリを作成
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 全研究者のIDをファイルに書き込む
        with open(output_path, "w", encoding="utf-8") as f:
            for researcher in researchers:
                openalex_id = researcher.get("id", "Unknown")
                f.write(f"{openalex_id}\n")

        logger.info(f"Saved {len(researchers)} researcher IDs to {output_path}")
        return str(output_path)


# 後方互換性のためのエイリアス
ResearcherListManager = ResearcherManager


def main():
    """CLI entry point for fetching researcher list."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Fetch University of Tokyo researchers from OpenAlex API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (default: min_works=5)
  python -m prepare.researcher_manager

  # With custom min_works
  python -m prepare.researcher_manager --min-works 10

  # Force refresh cache
  python -m prepare.researcher_manager --force-refresh

  # Custom institution ID
  python -m prepare.researcher_manager --institution-id I136199984

  # Save to custom location
  python -m prepare.researcher_manager --output custom_researchers.json

Note:
  Researchers are automatically cached to output/researcher_list/
  Default institution: University of Tokyo (I74801974)
        """
    )

    parser.add_argument(
        "--institution-id",
        type=str,
        default=UNIV_TOKYO_ID,
        help=f"OpenAlex institution ID (default: {UNIV_TOKYO_ID} - University of Tokyo)"
    )
    parser.add_argument(
        "--min-works",
        type=int,
        default=5,
        help="Minimum number of works per researcher (default: 5)"
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force refresh researcher cache (ignore existing cache)"
    )
    parser.add_argument(
        "--cache-days",
        type=int,
        default=30,
        help="Cache validity in days (default: 30)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Custom output file path (optional)"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    try:
        logger.info("=" * 80)
        logger.info("Fetching researchers from OpenAlex API")
        logger.info("=" * 80)
        logger.info(f"Institution ID: {args.institution_id}")
        logger.info(f"Min works: {args.min_works}")
        logger.info(f"Force refresh: {args.force_refresh}")
        logger.info(f"Cache validity: {args.cache_days} days")
        logger.info("")

        # Fetch researchers
        manager = ResearcherManager(institution_id=args.institution_id)
        researchers = asyncio.run(manager.fetch_researchers(
            min_works=args.min_works,
            force_refresh=args.force_refresh,
            cache_days=args.cache_days
        ))

        # Display summary
        logger.info("")
        logger.info("=" * 80)
        logger.info("✅ Successfully fetched researchers")
        logger.info("=" * 80)
        logger.info(f"Total researchers: {len(researchers)}")

        if researchers:
            # Calculate statistics
            total_works = sum(r.get("works_count", 0) for r in researchers)
            total_citations = sum(r.get("cited_by_count", 0) for r in researchers)
            avg_works = total_works / len(researchers)
            avg_citations = total_citations / len(researchers)

            logger.info(f"Total works: {total_works:,}")
            logger.info(f"Total citations: {total_citations:,}")
            logger.info(f"Average works per researcher: {avg_works:.1f}")
            logger.info(f"Average citations per researcher: {avg_citations:.1f}")

            # Show top 5 researchers
            logger.info("")
            logger.info("Top 5 researchers by works count:")
            sorted_researchers = sorted(
                researchers,
                key=lambda r: r.get("works_count", 0),
                reverse=True
            )
            for i, r in enumerate(sorted_researchers[:5], 1):
                logger.info(
                    f"  {i}. {r['name']:<30} | Works: {r['works_count']:>6,} | "
                    f"Citations: {r['cited_by_count']:>8,}"
                )

        # Save to custom location if specified
        if args.output:
            output_path = manager.save_custom_list(researchers, args.output)
            logger.info("")
            logger.info(f"📁 Saved to custom location: {output_path}")
        else:
            logger.info("")
            logger.info(f"📁 Cached to: {manager.cache_file}")

        # Save researchers names to text file
        names_file = manager.save_researchers_names_to_file(researchers)
        logger.info(f"📄 Saved researcher names to: {names_file}")

        # Save researchers IDs to text file
        ids_file = manager.save_researchers_ids_to_file(researchers)
        logger.info(f"📄 Saved researcher IDs to: {ids_file}")

        logger.info("")
        logger.info("✨ Done!")
        return 0

    except KeyboardInterrupt:
        logger.warning("\n⚠️  Interrupted by user")
        return 130
    except Exception as e:
        logger.exception(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())