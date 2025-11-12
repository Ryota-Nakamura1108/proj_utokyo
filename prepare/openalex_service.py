"""OpenAlex API統合サービス。

このモジュールはOpenAlex APIへのアクセスを提供する共通ユーティリティです。
全てのプロジェクト（central_researcher、visualize_map、keyword_extractorなど）から
利用されることを想定しています。

主な機能:
- Works（論文）の取得: ID/DOI/タイトル検索、バッチ処理、切り捨て著者の自動再取得
- Authors（著者）の取得: バッチ処理、統計情報、名前検索
- Institutions（機関）の取得: バッチ処理
- 機関所属研究者のリスト取得: カーソルページネーション、キャッシュ機能
- レート制限: 10リクエスト/秒、100,000リクエスト/日
- リトライロジック: 429エラー時の自動リトライ

使用例:
    # 基本的な使用
    >>> from common_module.openalex_services import OpenAlexService
    >>>
    >>> async with OpenAlexService(email="user@example.com") as service:
    ...     # Works取得
    ...     works = await service.get_works_by_ids(["W2741809807", "W2963436116"])
    ...
    ...     # Authors取得
    ...     authors = await service.get_authors_by_ids(["A2208157607"])
    ...
    ...     # タイトル検索
    ...     works = await service.search_works_by_title("deep learning", year=2020)
    ...
    ...     # 機関の研究者リスト取得
    ...     researchers = await service.fetch_researchers_by_institution(
    ...         institution_id="I74801974",  # 東京大学
    ...         min_works=5
    ...     )

    # 同期的な使用
    >>> from common_module.openalex_services import OpenAlexServiceSync
    >>>
    >>> service = OpenAlexServiceSync(email="user@example.com")
    >>> works = service.get_works_by_ids(["W2741809807"])

Architecture:
    - OpenAlexService: 非同期APIクライアント（メインクラス）
    - OpenAlexServiceSync: 同期ラッパー（後方互換性用）
    - レート制限とリトライロジックを内蔵
    - 柔軟な戻り値型（辞書として返し、各プロジェクトで独自モデルに変換可能）

注意事項:
    - email引数を設定することでOpenAlex Polite Poolにアクセス可能（推奨）
    - Polite Pool使用時はレート制限が100,000リクエスト/日に緩和
    - 大量のリクエストを行う場合は必ずemailを設定すること
    - 機関メールアドレス（例: @university.edu）の使用を推奨（Gmail等より信頼性が高い）
    - 403エラーが発生する場合は、リクエスト間隔を増やすか、日次クォータを確認すること
    - 安全のため、5リクエスト/秒程度に抑えることを推奨（10リクエスト/秒が上限）
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path
import json

import httpx

from .config import config

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


class OpenAlexService:
    """OpenAlex API統合サービス（非同期版）。

    OpenAlex APIへのアクセスを提供するメインクラスです。
    レート制限、リトライロジック、バッチ処理を内蔵しています。

    Attributes:
        BASE_URL: OpenAlex APIのベースURL
        RATE_LIMIT_PER_SEC: 秒あたりのリクエスト制限（デフォルト: 10）
        RATE_LIMIT_PER_DAY: 日あたりのリクエスト制限（デフォルト: 100,000）
        MAX_BATCH_SIZE: バッチ処理の最大サイズ（デフォルト: 100）

    Examples:
        >>> async with OpenAlexService(email="user@example.com") as service:
        ...     works = await service.get_works_by_ids(["W2741809807"])
        ...     print(works[0]["title"])
    """

    BASE_URL = "https://api.openalex.org"
    RATE_LIMIT_PER_SEC = 10
    RATE_LIMIT_PER_DAY = 100000
    MAX_BATCH_SIZE = 100

    def __init__(self, email: Optional[str] = None):
        """サービスを初期化します。

        Args:
            email: OpenAlex Polite Pool用のメールアドレス。
                  指定しない場合は環境変数OPENALEX_EMAILから取得。
        """
        self.email = email or config.openalex_email
        self.session: Optional[httpx.AsyncClient] = None
        self.request_count = 0
        self.daily_request_count = 0
        self.last_request_time = 0.0

    async def __aenter__(self):
        """非同期コンテキストマネージャーのエントリー。"""
        self.session = httpx.AsyncClient(timeout=30.0)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """非同期コンテキストマネージャーの終了。"""
        if self.session:
            await self.session.aclose()

    async def _rate_limit(self):
        """レート制限を適用します。"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        min_interval = 1.0 / self.RATE_LIMIT_PER_SEC

        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            await asyncio.sleep(sleep_time)

        self.last_request_time = time.time()
        self.request_count += 1
        self.daily_request_count += 1

        if self.daily_request_count >= self.RATE_LIMIT_PER_DAY:
            raise RuntimeError(f"Daily rate limit ({self.RATE_LIMIT_PER_DAY}) exceeded")

    async def _make_request(
        self,
        endpoint: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """OpenAlex APIへのレート制限付きリクエストを行います。

        Args:
            endpoint: APIエンドポイント（例: "/works", "/authors"）
            params: クエリパラメータ

        Returns:
            APIレスポンス（JSON辞書）

        Raises:
            httpx.HTTPStatusError: HTTPエラーが発生した場合
            RuntimeError: 3回のリトライ後も失敗した場合
        """
        if not self.session:
            raise RuntimeError("Session not initialized. Use 'async with' context manager.")

        await self._rate_limit()

        # Polite Pool用にメールを追加
        if self.email:
            params["mailto"] = self.email

        url = f"{self.BASE_URL}/{endpoint.lstrip('/')}"

        for attempt in range(3):  # 最大3回リトライ
            try:
                response = await self.session.get(url, params=params)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:  # レート制限
                    wait_time = 2 ** attempt
                    logger.warning(
                        f"Rate limited (429), waiting {wait_time}s before retry {attempt + 1}"
                    )
                    await asyncio.sleep(wait_time)
                    continue
                elif e.response.status_code == 403:  # Forbidden（レート制限の可能性）
                    wait_time = (2 ** attempt) * 2  # 403の場合はより長く待機
                    logger.warning(
                        f"Access forbidden (403), waiting {wait_time}s before retry {attempt + 1}. "
                        f"This may indicate rate limiting or quota exceeded."
                    )
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise
            except Exception as e:
                if attempt == 2:  # 最後の試行
                    raise
                await asyncio.sleep(2 ** attempt)

        raise RuntimeError("Failed to make request after 3 attempts")

    # ==================== Works（論文）関連メソッド ====================

    async def get_works_by_ids(
        self,
        work_ids: List[str],
        refetch_truncated: bool = True
    ) -> List[Dict[str, Any]]:
        """Work IDまたはDOIのリストから論文を取得します。

        バッチ処理（100件/バッチ）を使用して効率的に取得します。

        Args:
            work_ids: OpenAlex Work IDまたはDOIのリスト
                     例: ["W2741809807", "10.1038/nature12373"]
            refetch_truncated: 著者が切り捨てられている論文を再取得するか（デフォルト: True）

        Returns:
            論文データの辞書リスト。各辞書には以下のキーが含まれます:
                - id: OpenAlex Work ID
                - doi: DOI
                - title: タイトル
                - publication_year: 出版年
                - authorships: 著者情報のリスト
                - referenced_works: 引用論文のIDリスト
                - cited_by_count: 被引用数
                - topics: トピックリスト
                - primary_topic: 主要トピック
                - is_authors_truncated: 著者が切り捨てられているか

        Examples:
            >>> works = await service.get_works_by_ids(["W2741809807"])
            >>> print(works[0]["title"])
        """
        all_works = []

        # 100件ごとにバッチ処理
        for i in range(0, len(work_ids), self.MAX_BATCH_SIZE):
            batch = work_ids[i:i + self.MAX_BATCH_SIZE]
            works_batch = await self._get_works_batch(batch, refetch_truncated)
            all_works.extend(works_batch)

        return all_works

    async def _get_works_batch(
        self,
        identifiers: List[str],
        refetch_truncated: bool = True
    ) -> List[Dict[str, Any]]:
        """Worksのバッチを取得します（内部メソッド）。"""
        # DOIとOpenAlex IDを分離
        dois = []
        openalex_ids = []

        for identifier in identifiers:
            if identifier.startswith("W") or identifier.startswith("https://openalex.org/W"):
                clean_id = identifier.replace("https://openalex.org/", "")
                openalex_ids.append(clean_id)
            elif "doi.org" in identifier or identifier.count("/") >= 1:
                clean_doi = identifier.replace("https://doi.org/", "").lower()
                dois.append(clean_doi)
            else:
                openalex_ids.append(identifier)

        works = []

        # DOIバッチ取得
        if dois:
            doi_filter = "|".join(dois)
            works.extend(
                await self._fetch_works_with_filter(
                    f"doi:{doi_filter}",
                    refetch_truncated
                )
            )

        # OpenAlex IDバッチ取得
        if openalex_ids:
            id_filter = "|".join(openalex_ids)
            works.extend(
                await self._fetch_works_with_filter(
                    f"openalex:{id_filter}",
                    refetch_truncated
                )
            )

        return works

    async def _fetch_works_with_filter(
        self,
        filter_str: str,
        refetch_truncated: bool = True
    ) -> List[Dict[str, Any]]:
        """フィルタ条件でWorksを取得します（内部メソッド）。"""
        params = {
            "filter": filter_str,
            "select": "id,doi,display_name,publication_year,authorships,referenced_works,cited_by_count,topics,primary_topic,is_authors_truncated",
            "per-page": self.MAX_BATCH_SIZE
        }

        response = await self._make_request("/works", params)
        works = response.get("results", [])

        # 切り捨てられた著者の再取得
        if refetch_truncated:
            truncated_works = [w for w in works if w.get("is_authors_truncated", False)]
            if truncated_works:
                logger.info(
                    f"Re-fetching {len(truncated_works)} works with truncated authors"
                )
                complete_works = await self._refetch_truncated_works(truncated_works)

                # 切り捨てられたWorksを完全版で置換
                work_lookup = {w["id"]: w for w in complete_works}
                for i, work in enumerate(works):
                    if work["id"] in work_lookup:
                        works[i] = work_lookup[work["id"]]

        return works

    async def _refetch_truncated_works(
        self,
        truncated_works: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """切り捨てられた著者を持つWorksを個別に再取得します（内部メソッド）。"""
        complete_works = []

        for work in truncated_works:
            try:
                work_id = work["id"].replace("https://openalex.org/", "")
                params = {
                    "select": "id,doi,display_name,publication_year,authorships,referenced_works,cited_by_count,topics,primary_topic"
                }

                response = await self._make_request(f"/works/{work_id}", params)
                complete_works.append(response)

            except Exception as e:
                logger.warning(f"Failed to re-fetch work {work['id']}: {e}")
                complete_works.append(work)  # 失敗時は元のデータを使用

        return complete_works

    async def search_works_by_title(
        self,
        title: str,
        year: Optional[int] = None,
        refetch_truncated: bool = True
    ) -> List[Dict[str, Any]]:
        """タイトルでWorksを検索します。

        Args:
            title: 検索するタイトル
            year: 出版年でフィルタ（オプション）
            refetch_truncated: 著者が切り捨てられている論文を再取得するか

        Returns:
            検索結果の論文データリスト（最大10件）

        Examples:
            >>> works = await service.search_works_by_title("deep learning", year=2020)
            >>> for work in works[:3]:
            ...     print(work["title"])
        """
        params = {
            "search": title,
            "select": "id,doi,display_name,publication_year,authorships,referenced_works,cited_by_count,topics,primary_topic,is_authors_truncated",
            "per-page": 10
        }

        if year:
            params["filter"] = f"publication_year:{year}"

        response = await self._make_request("/works", params)
        works = response.get("results", [])

        # 切り捨てられた著者の再取得
        if refetch_truncated:
            truncated_works = [w for w in works if w.get("is_authors_truncated", False)]
            if truncated_works:
                logger.info(
                    f"Re-fetching {len(truncated_works)} search results with truncated authors"
                )
                complete_works = await self._refetch_truncated_works(truncated_works)

                work_lookup = {w["id"]: w for w in complete_works}
                for i, work in enumerate(works):
                    if work["id"] in work_lookup:
                        works[i] = work_lookup[work["id"]]

        return works

    # ==================== Authors（著者）関連メソッド ====================

    async def get_authors_by_ids(
        self,
        author_ids: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Author IDのリストから著者データを取得します。

        バッチ処理（100件/バッチ）を使用して効率的に取得します。

        Args:
            author_ids: OpenAlex Author IDのリスト
                       例: ["A2208157607", "A2964268087"]

        Returns:
            Author IDをキーとする著者データ辞書。各著者データには以下のキーが含まれます:
                - id: OpenAlex Author ID
                - display_name: 表示名
                - orcid: ORCID ID（存在する場合）
                - summary_stats: 統計情報（h-index, i10-index, 2yr_mean_citedness）
                - works_count: 論文数
                - last_known_institutions: 最後の所属機関リスト

        Examples:
            >>> authors = await service.get_authors_by_ids(["A2208157607"])
            >>> print(authors["A2208157607"]["display_name"])
        """
        all_authors = {}

        # 100件ごとにバッチ処理
        for i in range(0, len(author_ids), self.MAX_BATCH_SIZE):
            batch = author_ids[i:i + self.MAX_BATCH_SIZE]
            authors_batch = await self._get_authors_batch(batch)
            all_authors.update(authors_batch)

        return all_authors

    async def _get_authors_batch(
        self,
        author_ids: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Authorsのバッチを取得します（内部メソッド）。"""
        # Author IDをクリーンアップ
        clean_ids = []
        for author_id in author_ids:
            if author_id:
                clean_id = author_id.replace("https://openalex.org/", "")
                clean_ids.append(clean_id)

        if not clean_ids:
            return {}

        id_filter = "|".join(clean_ids)
        params = {
            "filter": f"openalex:{id_filter}",
            "select": "id,display_name,orcid,summary_stats,works_count,last_known_institutions",
            "per-page": self.MAX_BATCH_SIZE,
            "sort": "id.asc"
        }

        response = await self._make_request("/authors", params)
        authors = {}

        for author_data in response.get("results", []):
            author_id = author_data.get("id", "").replace("https://openalex.org/", "")

            # Clean author ID in the data itself
            author_data["id"] = author_id

            # Clean ORCID URL
            if "orcid" in author_data and author_data["orcid"]:
                author_data["orcid"] = author_data["orcid"].replace("https://orcid.org/", "")

            # Clean institution IDs in last_known_institutions
            if "last_known_institutions" in author_data:
                cleaned_institutions = []
                for inst in author_data["last_known_institutions"]:
                    if isinstance(inst, dict):
                        cleaned_inst = inst.copy()
                        if "id" in cleaned_inst:
                            cleaned_inst["id"] = cleaned_inst["id"].replace("https://openalex.org/", "")
                        cleaned_institutions.append(cleaned_inst)
                    elif isinstance(inst, str):
                        cleaned_institutions.append(inst.replace("https://openalex.org/", ""))
                author_data["last_known_institutions"] = cleaned_institutions

            authors[author_id] = author_data

        return authors

    async def search_author_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """名前で著者を検索します。

        Args:
            name: 検索する著者名

        Returns:
            最も関連性の高い著者データ（見つからない場合はNone）

        Examples:
            >>> author = await service.search_author_by_name("Geoffrey Hinton")
            >>> if author:
            ...     print(author["display_name"])
        """
        logger.info(f"Searching for author: {name}")

        params = {"search": name}

        try:
            response = await self._make_request("/authors", params)

            if not response.get("results"):
                logger.warning(f"No author found for name: {name}")
                return None

            # 最も関連性の高い結果を返す
            author_data = response["results"][0]
            logger.info(
                f"Found author: {author_data.get('display_name')} "
                f"(ID: {author_data.get('id')})"
            )
            return author_data

        except Exception as e:
            logger.error(f"Error searching author: {e}")
            return None

    async def fetch_author_papers(
        self,
        author_id: str,
        years_back: Optional[int] = None,
        author_positions: Optional[List[str]] = None,
        min_citations: int = 0,
        max_papers: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """著者の論文を取得します（フィルタ付き）。

        Args:
            author_id: OpenAlex Author ID
            years_back: 過去何年分の論文を取得するか（Noneの場合は全期間）
            author_positions: 著者位置フィルタ（例: ["first", "last"]）
            min_citations: 最小引用数フィルタ
            max_papers: 取得する論文の最大数

        Returns:
            論文データのリスト

        Examples:
            >>> papers = await service.fetch_author_papers(
            ...     "A2208157607",
            ...     years_back=10,
            ...     author_positions=["first", "last"],
            ...     min_citations=5
            ... )
        """
        clean_id = author_id.replace("https://openalex.org/", "")

        # フィルタ条件を構築
        filters = [f"author.id:{clean_id}"]

        if years_back:
            current_year = datetime.now().year
            start_year = current_year - years_back
            filters.append(f"publication_year:{start_year}-{current_year}")

        if author_positions:
            pos_filter = "|".join(author_positions)
            filters.append(f"authorships.author_position:{pos_filter}")

        if min_citations > 0:
            filters.append(f"cited_by_count:>{min_citations}")

        filter_str = ",".join(filters)

        params = {
            "filter": filter_str,
            "select": "id,doi,display_name,publication_year,authorships,cited_by_count,topics,primary_topic,abstract_inverted_index",
            "per-page": 200,
            "sort": "cited_by_count:desc"
        }

        all_papers = []
        page = 1

        while True:
            params["page"] = page
            response = await self._make_request("/works", params)

            results = response.get("results", [])
            if not results:
                break

            all_papers.extend(results)

            # max_papersに達したら終了
            if max_papers and len(all_papers) >= max_papers:
                all_papers = all_papers[:max_papers]
                break

            # メタデータで次ページがあるか確認
            meta = response.get("meta", {})
            if page >= meta.get("count", 0) // params["per-page"] + 1:
                break

            page += 1

        logger.info(f"Fetched {len(all_papers)} papers for author {clean_id}")
        return all_papers

    # ==================== Institutions（機関）関連メソッド ====================

    async def get_institutions_by_ids(
        self,
        institution_ids: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Institution IDのリストから機関データを取得します。

        バッチ処理（100件/バッチ）を使用して効率的に取得します。

        Args:
            institution_ids: OpenAlex Institution IDのリスト
                           例: ["I74801974", "I136199984"]

        Returns:
            Institution IDをキーとする機関データ辞書。各機関データには以下のキーが含まれます:
                - id: OpenAlex Institution ID
                - display_name: 表示名
                - country_code: 国コード
                - type: 機関タイプ
                - ror: ROR ID
                - homepage_url: ホームページURL
                - works_count: 論文数

        Examples:
            >>> institutions = await service.get_institutions_by_ids(["I74801974"])
            >>> print(institutions["I74801974"]["display_name"])
        """
        all_institutions = {}

        # 100件ごとにバッチ処理
        for i in range(0, len(institution_ids), self.MAX_BATCH_SIZE):
            batch = institution_ids[i:i + self.MAX_BATCH_SIZE]
            institutions_batch = await self._get_institutions_batch(batch)
            all_institutions.update(institutions_batch)

        return all_institutions

    async def _get_institutions_batch(
        self,
        institution_ids: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Institutionsのバッチを取得します（内部メソッド）。"""
        # Institution IDをクリーンアップ
        filter_ids = "|".join([
            inst_id.replace("https://openalex.org/", "")
            for inst_id in institution_ids
        ])

        params = {
            "filter": f"openalex:{filter_ids}",
            "select": "id,display_name,country_code,type,ror,homepage_url,works_count",
            "per-page": self.MAX_BATCH_SIZE
        }

        response = await self._make_request("/institutions", params)

        institutions = {}
        for inst_data in response.get("results", []):
            inst_id = inst_data.get("id", "").replace("https://openalex.org/", "")
            institutions[inst_id] = inst_data

        return institutions

    # ==================== 機関所属研究者リスト取得 ====================

    async def fetch_researchers_by_institution(
        self,
        institution_id: str,
        min_works: int = 5,
        cache_dir: Optional[Path] = None,
        force_refresh: bool = False,
        cache_days: int = 30
    ) -> List[Dict[str, Any]]:
        """機関に所属する研究者リストを取得します（カーソルページネーション、キャッシュ付き）。

        大量の研究者を効率的に取得するため、カーソルベースのページネーションを使用します。
        結果はキャッシュされ、指定日数以内であればキャッシュから読み込まれます。

        Args:
            institution_id: OpenAlex Institution ID（例: "I74801974"）
            min_works: 最小論文数フィルタ
            cache_dir: キャッシュディレクトリ（Noneの場合は/tmp/openalex_cache）
            force_refresh: キャッシュを無視して強制的に再取得
            cache_days: キャッシュの有効期限（日数）

        Returns:
            研究者データのリスト。各辞書には以下のキーが含まれます:
                - id: Author ID
                - name: 表示名
                - works_count: 論文数
                - cited_by_count: 被引用数
                - institutions: 所属機関名のリスト

        Examples:
            >>> # 東京大学の研究者リストを取得
            >>> researchers = await service.fetch_researchers_by_institution(
            ...     institution_id="I74801974",
            ...     min_works=10
            ... )
            >>> print(f"Found {len(researchers)} researchers")
            >>> for r in researchers[:5]:
            ...     print(f"{r['name']}: {r['works_count']} works")
        """
        # Institution IDをクリーンアップ
        clean_id = institution_id.replace("https://openalex.org/", "")
        if not clean_id.startswith("I"):
            clean_id = "I" + clean_id

        # キャッシュディレクトリの設定
        if cache_dir is None:
            cache_dir = Path("/tmp/openalex_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)

        cache_file = cache_dir / f"{clean_id}_researchers.json"

        # キャッシュチェック
        if not force_refresh and cache_file.exists():
            logger.info(f"Loading researchers from cache: {cache_file}")
            with open(cache_file, "r", encoding="utf-8") as f:
                cached_data = json.load(f)

            # キャッシュの有効期限チェック
            cached_at = datetime.fromisoformat(cached_data["cached_at"])
            age_days = (datetime.now() - cached_at).days

            if age_days < cache_days:
                logger.info(f"Using cached data (age: {age_days} days)")
                return cached_data["researchers"]
            else:
                logger.info(f"Cache expired (age: {age_days} days), fetching new data")

        # OpenAlexから取得
        logger.info(
            f"Fetching researchers from OpenAlex (institution: {clean_id}, min_works: {min_works})"
        )

        researchers = []
        per_page = 200
        next_cursor = "*"

        while next_cursor:
            params = {
                "filter": f"last_known_institutions.id:{clean_id},works_count:>{min_works}",
                "per-page": per_page,
                "cursor": next_cursor,
                "select": "id,display_name,works_count,cited_by_count,last_known_institutions"
            }

            log_msg = (
                "Initial fetch" if next_cursor == "*"
                else f"Fetching with cursor: {next_cursor[:10]}..."
            )
            logger.info(log_msg)

            try:
                response = await self._make_request("/authors", params)

                results = response.get("results", [])

                # 研究者データを整形
                for author in results:
                    researcher = {
                        "id": author["id"].replace("https://openalex.org/", ""),
                        "name": author.get("display_name", "Unknown"),
                        "works_count": author.get("works_count", 0),
                        "cited_by_count": author.get("cited_by_count", 0),
                        "institutions": [
                            inst.get("display_name", "Unknown")
                            for inst in author.get("last_known_institutions", [])
                        ]
                    }
                    researchers.append(researcher)

                logger.info(
                    f"  Fetched {len(results)} researchers (total: {len(researchers)})"
                )

                # 次のカーソルを取得
                meta = response.get("meta", {})
                next_cursor = meta.get("next_cursor")

                if not next_cursor:
                    logger.info("全データの取得が完了しました。")
                    break

                # レート制限（既に_make_requestで適用されているが、念のため）
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.exception(f"Error fetching researchers: {e}")
                break

        logger.info(f"Total researchers fetched: {len(researchers)}")

        # 論文数でソート（降順）
        researchers.sort(key=lambda x: x["works_count"], reverse=True)

        # キャッシュに保存
        cache_data = {
            "institution_id": clean_id,
            "cached_at": datetime.now().isoformat(),
            "min_works": min_works,
            "total_researchers": len(researchers),
            "researchers": researchers
        }

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Cached {len(researchers)} researchers to {cache_file}")

        return researchers


class OpenAlexServiceSync:
    """OpenAlex API統合サービス（同期版）。

    非同期APIを内部で使用し、同期的なインターフェースを提供します。
    後方互換性や、同期コードから使用する場合に便利です。

    Examples:
        >>> service = OpenAlexServiceSync(email="user@example.com")
        >>> works = service.get_works_by_ids(["W2741809807"])
        >>> print(works[0]["title"])
    """

    def __init__(self, email: Optional[str] = None):
        """サービスを初期化します。

        Args:
            email: OpenAlex Polite Pool用のメールアドレス
        """
        self.email = email

    def get_works_by_ids(
        self,
        work_ids: List[str],
        refetch_truncated: bool = True
    ) -> List[Dict[str, Any]]:
        """Work IDまたはDOIのリストから論文を取得します（同期版）。"""
        async def _fetch():
            async with OpenAlexService(email=self.email) as service:
                return await service.get_works_by_ids(work_ids, refetch_truncated)

        return asyncio.run(_fetch())

    def search_works_by_title(
        self,
        title: str,
        year: Optional[int] = None,
        refetch_truncated: bool = True
    ) -> List[Dict[str, Any]]:
        """タイトルでWorksを検索します（同期版）。"""
        async def _fetch():
            async with OpenAlexService(email=self.email) as service:
                return await service.search_works_by_title(title, year, refetch_truncated)

        return asyncio.run(_fetch())

    def get_authors_by_ids(self, author_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Author IDのリストから著者データを取得します（同期版）。"""
        async def _fetch():
            async with OpenAlexService(email=self.email) as service:
                return await service.get_authors_by_ids(author_ids)

        return asyncio.run(_fetch())

    def search_author_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """名前で著者を検索します（同期版）。"""
        async def _fetch():
            async with OpenAlexService(email=self.email) as service:
                return await service.search_author_by_name(name)

        return asyncio.run(_fetch())

    def fetch_author_papers(
        self,
        author_id: str,
        years_back: Optional[int] = None,
        author_positions: Optional[List[str]] = None,
        min_citations: int = 0,
        max_papers: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """著者の論文を取得します（同期版）。"""
        async def _fetch():
            async with OpenAlexService(email=self.email) as service:
                return await service.fetch_author_papers(
                    author_id, years_back, author_positions,
                    min_citations, max_papers
                )

        return asyncio.run(_fetch())

    def get_institutions_by_ids(
        self,
        institution_ids: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Institution IDのリストから機関データを取得します（同期版）。"""
        async def _fetch():
            async with OpenAlexService(email=self.email) as service:
                return await service.get_institutions_by_ids(institution_ids)

        return asyncio.run(_fetch())

    def fetch_researchers_by_institution(
        self,
        institution_id: str,
        min_works: int = 5,
        cache_dir: Optional[Path] = None,
        force_refresh: bool = False,
        cache_days: int = 30
    ) -> List[Dict[str, Any]]:
        """機関に所属する研究者リストを取得します（同期版）。"""
        async def _fetch():
            async with OpenAlexService(email=self.email) as service:
                return await service.fetch_researchers_by_institution(
                    institution_id, min_works, cache_dir,
                    force_refresh, cache_days
                )

        return asyncio.run(_fetch())


# 後方互換性のためのエイリアス
OpenAlexClient = OpenAlexService