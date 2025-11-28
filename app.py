import asyncio
import os
import logging
import sys
from typing import List, Optional, Any, Dict
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import traceback

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from article_search.core.article_search_faiss_gcsfuse import PaperSearchEngineFAISSGCSFuse, SearchResults, SearchResult
    # Initialize search engine globally on startup (downloads FAISS index once)
    logger.info("Initializing PaperSearchEngine globally...")
    PaperSearchEngine = PaperSearchEngineFAISSGCSFuse()
    logger.info("✅ PaperSearchEngine initialized successfully")
except ImportError as e:
    logger.error(f"エラー: {e}")
    logger.error(f"このスクリプトと同じディレクトリ ({BASE_DIR}) に配置してください。")
    PaperSearchEngine = None
    SearchResults = None
    SearchResult = None
except Exception as e:
    logger.error(f"PaperSearchEngine initialization failed: {e}")
    PaperSearchEngine = None
    SearchResults = None
    SearchResult = None

try:
    from researcher_search.core.central_researcher import CentralResearcher
    from researcher_search.core.models import (
        CentralResearcherConfig, InputPaper, WorkRaw,
        Authorship, Topic, AuthorPosition, ResearcherRanking
    )
except ImportError as e:
    logger.error(f"エラー: 'central_researcher' モジュールのインポートに失敗しました: {e}")
    CentralResearcher = None

try:
    sys.path.insert(0, str(Path(BASE_DIR) / "kaken_package" / "src"))
    from kaken_info.scraper import get_research_field_data
except ImportError as e:
    logger.warning(f"KAKENHI package のインポートに失敗しました: {e}")
    get_research_field_data = None

try:
    sys.path.insert(0, str(Path(BASE_DIR) / "keyword_extractor"))
    from keyword_extractor.core import KeywordExtractor
except ImportError as e:
    logger.warning(f"Keyword extractor のインポートに失敗しました: {e}")
    KeywordExtractor = None

try:
    sys.path.insert(0, str(Path(__file__).parent))
    from mainpage_generator import MainHTMLReportGenerator
except ImportError as e:
    logger.error(f"MainHTMLReportGenerator のインポートに失敗しました: {e}")
    MainHTMLReportGenerator = None

try:
    sys.path.insert(0, str(Path(__file__).parent))
    from researcher_page_generator import ResearcherPageGenerator
except ImportError as e:
    logger.error(f"ResearcherPageGenerator のインポートに失敗しました: {e}")
    ResearcherPageGenerator = None

try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    logger.warning("google-cloud-storage がインストールされていません。Cloud Storage機能は無効です。")
    GCS_AVAILABLE = False

app = Flask(__name__)

# CORS設定 - Firebase HostingとCloud Runからのアクセスを許可
CORS(app, resources={
    r"/api/*": {
        "origins": [
            "https://proj-utokyo.web.app",
            "https://proj-utokyo.firebaseapp.com",
            "https://central-researcher-api-1008514239787.asia-northeast1.run.app",
            "http://localhost:5000",
            "http://127.0.0.1:5000"
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": False
    }
})

# Configuration
FAISS_INDEX_DIR = os.environ.get('FAISS_INDEX_DIR', os.path.join(BASE_DIR, "./faiss_index"))
OUTPUT_DIR = Path(BASE_DIR) / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Cloud Storage設定
GCS_BUCKET_NAME = os.environ.get('GCS_BUCKET_NAME', '')
DOWNLOAD_ON_STARTUP = os.environ.get('DOWNLOAD_VECTORS_ON_STARTUP', 'true').lower() == 'true'


def download_vectors_from_gcs():
    """Cloud Storageからベクトルデータをダウンロードする"""
    if not GCS_AVAILABLE:
        logger.warning("Cloud Storage機能が利用できません")
        return False

    if not GCS_BUCKET_NAME:
        logger.info("GCS_BUCKET_NAME が設定されていません。ローカルのベクトルデータを使用します。")
        return False

    # ベクトルディレクトリが既にデータを持っている場合はスキップ
    if os.path.isdir(VECTOR_DIR):
        existing_files = [f for f in os.listdir(VECTOR_DIR) if f.endswith(('.h5', '.pkl'))]
        if len(existing_files) > 0:
            logger.info(f"ベクトルディレクトリに既に {len(existing_files)} 個のファイルが存在します。ダウンロードをスキップします。")
            return True

    logger.info(f"Cloud Storageからベクトルデータをダウンロード中: gs://{GCS_BUCKET_NAME}/vectors/")

    try:
        # ベクトルディレクトリを作成
        os.makedirs(VECTOR_DIR, exist_ok=True)

        # Cloud Storageクライアントを初期化
        storage_client = storage.Client()
        bucket = storage_client.bucket(GCS_BUCKET_NAME)

        # vectors/ プレフィックスのファイルをリスト
        blobs = list(bucket.list_blobs(prefix='vectors/'))

        if not blobs:
            logger.error(f"バケット gs://{GCS_BUCKET_NAME}/vectors/ にファイルが見つかりません")
            return False

        logger.info(f"{len(blobs)} 個のファイルをダウンロード中...")

        # 各ファイルをダウンロード
        downloaded_count = 0
        for blob in blobs:
            # vectors/ プレフィックスを除いたファイル名を取得
            if blob.name == 'vectors/':
                continue

            filename = blob.name.replace('vectors/', '')
            local_path = os.path.join(VECTOR_DIR, filename)

            logger.info(f"  ダウンロード中: {filename}")
            blob.download_to_filename(local_path)
            downloaded_count += 1

        logger.info(f"✓ {downloaded_count} 個のファイルをダウンロードしました")
        return True

    except Exception as e:
        logger.error(f"Cloud Storageからのダウンロードに失敗しました: {e}")
        logger.error(traceback.format_exc())
        return False


# アプリケーション起動時にベクトルデータをダウンロード
if DOWNLOAD_ON_STARTUP and GCS_BUCKET_NAME:
    logger.info("起動時ベクトルダウンロードを開始...")
    download_vectors_from_gcs()


def _parse_year(date_str: str) -> Optional[int]:
    """'YYYY-MM-DD' 形式の文字列から年を抽出する"""
    if not date_str:
        return None
    try:
        return int(date_str.split('-')[0])
    except (ValueError, IndexError):
        logger.warning(f"日付形式のパースに失敗: {date_str}")
        return None


def _convert_authorships(search_result_authors: List[Any]) -> List[Authorship]:
    """SearchResult.authorships を WorkRaw.authorships (List[Authorship]) に変換する。"""
    work_authorships = []
    if not search_result_authors:
        return []

    if isinstance(search_result_authors[0], str):
        logger.warning("Detected List[str] authorships. Converting, but author_ids will be missing.")
        for i, name in enumerate(search_result_authors):
            if not name: continue
            pos = AuthorPosition.MIDDLE
            if i == 0: pos = AuthorPosition.FIRST
            elif i == len(search_result_authors) - 1: pos = AuthorPosition.LAST

            work_authorships.append(
                Authorship(
                    author_id=None,
                    raw_name=name,
                    author_position=pos,
                )
            )

    elif isinstance(search_result_authors[0], dict):
        if "display_name" in search_result_authors[0]:
            logger.info("生の著者辞書リスト(id, display_name)を検出しました。")
            for i, auth_data in enumerate(search_result_authors):
                author_id = auth_data.get("id") or None
                display_name = auth_data.get("display_name")

                if display_name is None:
                    display_name = ""

                pos = AuthorPosition.MIDDLE
                if i == 0: pos = AuthorPosition.FIRST
                elif i == len(search_result_authors) - 1: pos = AuthorPosition.LAST

                work_authorships.append(
                    Authorship(
                        author_id=author_id,
                        raw_name=display_name,
                        author_position=pos,
                        is_corresponding=False,
                        institution_ids=[]
                    )
                )

        elif "author" in search_result_authors[0]:
            logger.info("生の著者辞書リスト(複雑なOpenAlex形式)を検出しました。")
            for auth_data in search_result_authors:
                author_info = auth_data.get("author", {})
                inst_ids = [inst.get("id") for inst in auth_data.get("institutions", []) if inst.get("id")]
                pos_str = auth_data.get("author_position")
                pos = AuthorPosition.MIDDLE
                if pos_str == "first": pos = AuthorPosition.FIRST
                elif pos_str == "last": pos = AuthorPosition.LAST

                raw_name = auth_data.get("raw_author_name") or author_info.get("display_name")

                work_authorships.append(
                    Authorship(
                        author_id=author_info.get("id"),
                        orcid=author_info.get("orcid"),
                        raw_name=raw_name if raw_name else "",
                        author_position=pos,
                        is_corresponding=auth_data.get("is_corresponding", False),
                        institution_ids=inst_ids
                    )
                )

    return work_authorships


def _convert_concepts(search_result_concepts: List[Any]) -> List[Topic]:
    """SearchResult.concepts (List[Dict]) を List[Topic] に変換する。"""
    topics = []
    if not search_result_concepts:
        return []

    for concept in search_result_concepts:
        if isinstance(concept, dict):
            topic_id = concept.get("id")
            if topic_id:
                topics.append(
                    Topic(
                        topic_id=topic_id,
                        display_name=concept.get("display_name"),
                        score=concept.get("score", 0.0)
                    )
                )
    return topics


async def collect_researcher_data(
    rankings: List[ResearcherRanking],
    top_n: int = 20
) -> Dict[str, Dict[str, Any]]:
    """Collect additional data for top N researchers using kaken_package and keyword_extractor."""
    researcher_data = {}

    for i, ranking in enumerate(rankings[:top_n], 1):
        logger.info(f"Processing researcher {i}/{top_n}: {ranking.author_display_name}")

        data = {
            "author_id": ranking.author_id,
            "author_name": ranking.author_display_name,
            "kakenhi_info": None,
            "keywords": [],
            "summary": None,
            "statistics": {}
        }

        if get_research_field_data:
            try:
                logger.info(f"  Fetching KAKENHI data for {ranking.author_display_name}...")
                kakenhi_df = get_research_field_data(ranking.author_display_name, "東京大学")
                if kakenhi_df is not None and not kakenhi_df.empty:
                    data["kakenhi_info"] = kakenhi_df.to_dict('records')
                    logger.info(f"Found {len(kakenhi_df)} KAKENHI grants")
                else:
                    logger.info(f"No KAKENHI data found")
            except Exception as e:
                logger.error(f"  Error fetching KAKENHI data: {e}")

        if KeywordExtractor:
            try:
                logger.info(f"  Extracting keywords for {ranking.author_display_name}...")
                extractor = KeywordExtractor()

                try:
                    keywords_result = await extractor.extract_by_id(
                        ranking.author_id,
                        years_back=10,
                        max_keywords=10
                    )
                except:
                    keywords_result = await extractor.extract_by_name(
                        ranking.author_display_name,
                        years_back=10,
                        max_keywords=10
                    )

                if keywords_result.get("status") != "error":
                    data["keywords"] = keywords_result.get("keywords", [])
                    data["summary"] = keywords_result.get("summary", "")
                    data["statistics"] = keywords_result.get("statistics", {})
                    logger.info(f"  Extracted {len(data['keywords'])} keywords")
                else:
                    logger.info(f"  No keywords extracted: {keywords_result.get('error')}")
            except Exception as e:
                logger.error(f"  Error extracting keywords: {e}")

        researcher_data[ranking.author_id] = data

    return researcher_data


def convert_search_results_to_workraw_list(search_results: SearchResults) -> List[WorkRaw]:
    """PaperSearchEngine の SearchResults を CentralResearcher が使用する WorkRaw のリストに変換する。"""
    workraw_list = []
    for res in search_results.results:
        work_authorships = _convert_authorships(res.authorships)
        work_topics = _convert_concepts(res.concepts)
        primary_topic = max(work_topics, key=lambda t: t.score) if work_topics else None

        work = WorkRaw(
            work_id=res.id,
            doi=res.doi,
            title=res.title,
            year=_parse_year(res.publication_date),
            cited_by_count=res.cited_by_count or 0,
            referenced_work_ids=res.referenced_works or [],
            authorships=work_authorships,
            topics=work_topics,
            primary_topic=primary_topic,
            is_authors_truncated=False
        )
        workraw_list.append(work)

    logger.info(f"{len(search_results.results)} 件の SearchResult を {len(workraw_list)} 件の WorkRaw に変換しました。")
    return workraw_list


async def run_combined_analysis(
    query: str,
    email: str,
    filter_institution: str = "The University of Tokyo",
    top_k: int = 100,
    similarity_threshold: float = 0.55
) -> Optional[Dict[str, Any]]:
    """分析を実行し、結果を辞書形式で返す。"""

    logger.info(f"--- ステップ 1: 論文検索 (Query: '{query}') ---")
    try:
        if not PaperSearchEngine:
            raise Exception("PaperSearchEngine がインポートされていません")

        # Use globally initialized search engine (no re-initialization)
        search_results: SearchResults = PaperSearchEngine.search(
            query=query,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )

        if not search_results.results:
            logger.warning("論文検索でヒットしませんでした。処理を終了します。")
            return None
        logger.info(f"{len(search_results.results)} 件の論文がヒットしました。")

    except Exception as e:
        logger.error(f"論文検索ステップでエラーが発生しました: {e}")
        raise

    logger.info(f"--- ステップ 2: Central Researcher 分析準備 ---")

    if not CentralResearcher:
        raise Exception("CentralResearcher がインポートされていません")

    config = CentralResearcherConfig(
        papers=[InputPaper(title="dummy")],
        email=email,
    )

    analyzer = CentralResearcher(config)
    workraw_list = convert_search_results_to_workraw_list(search_results)

    if not workraw_list:
        logger.warning("WorkRaw への変換に失敗しました。処理を終了します。")
        return None

    analyzer.works = workraw_list

    logger.info(f"--- ステップ 3: Central Researcher 分析実行 ---")
    try:
        async with analyzer.client:
            await analyzer._fetch_author_data()
            await analyzer._fetch_institution_data()
            await analyzer._build_network()
            await analyzer._calculate_centralities()
            await analyzer._calculate_citations()
            await analyzer._calculate_scores()

            logger.info("--- 分析完了 ---")

            all_rankings = analyzer.rankings
            if not all_rankings:
                logger.warning("分析は完了しましたが、ランキング結果がありません。")
                return None

            logger.info(f"--- ステップ 4: 機関フィルタリング ---")
            if filter_institution:
                logger.info(f"フィルタ対象: '{filter_institution}'")

                # デバッグ: 機関データの状況を確認
                logger.info(f"DEBUG: 全研究者数 = {len(all_rankings)}")
                logger.info(f"DEBUG: 機関マスターデータ数 = {len(analyzer.institutions)}")
                logger.info(f"DEBUG: 著者マスターデータ数 = {len(analyzer.authors)}")

                # デバッグ: サンプル機関名を表示
                sample_institutions = list(analyzer.institutions.values())[:10]
                logger.info(f"DEBUG: サンプル機関名 (最初の10件):")
                for inst in sample_institutions:
                    logger.info(f"  - {inst.display_name}")

                # デバッグ: 著者の所属機関データを確認
                authors_with_institutions = sum(1 for author in analyzer.authors.values() if author.last_known_institution_ids)
                logger.info(f"DEBUG: 所属機関データを持つ著者数 = {authors_with_institutions}")

                rankings: List[ResearcherRanking] = analyzer.filter_by_institution(
                    institution_names=[filter_institution],
                    fuzzy_match=True
                )
                logger.info(f"フィルタリング完了: '{filter_institution}' の研究者数 = {len(rankings)}")

                if not rankings:
                    logger.warning(f"'{filter_institution}' に所属する研究者は見つかりませんでした。")
                    return None
            else:
                logger.info("機関フィルタなし - 全研究者を対象")
                rankings = all_rankings

            # ステップ 5: 個人ページ作成機能は別途実装予定のため、一旦スキップ
            logger.info(f"--- ステップ 5: スキップ (個人ページ作成は別途実装予定) ---")
            # researcher_data_dict = await collect_researcher_data(rankings, top_n=10)
            researcher_data_dict = {}  # 空の辞書として返す

            # Convert rankings to dict format
            rankings_list = []
            for r in rankings[:20]:
                author_master = analyzer.authors.get(r.author_id)  # authors is a dict, not list
                institution_names = []
                if author_master and author_master.last_known_institution_ids:
                    for inst_id in author_master.last_known_institution_ids:
                        inst = analyzer.institutions.get(inst_id)  # institutions is a dict, not list
                        if inst:
                            institution_names.append(inst.display_name)

                rankings_list.append({
                    'rank': r.rank,
                    'author_id': r.author_id,
                    'author_name': r.author_display_name,
                    'crs_score': float(r.crs_final),
                    'h_index': int(r.h_index_global) if r.h_index_global else 0,
                    'papers_in_corpus': r.n_in_corpus_works,
                    'leadership_rate': float(r.leadership_rate) if r.leadership_rate else 0.0,
                    'institutions': institution_names
                })

            network_stats = analyzer.get_network_statistics()

            return {
                'query': query,
                'filter_institution': filter_institution,
                'total_papers': len(search_results.results),
                'total_researchers': len(rankings),
                'rankings': rankings_list,
                'researcher_data': researcher_data_dict,
                'network_statistics': network_stats,
                'timestamp': datetime.now().isoformat()
            }

    except Exception as e:
        logger.error(f"Central Researcher 分析ステップでエラーが発生しました: {e}")
        raise


@app.route('/', methods=['GET'])
def index():
    """Root endpoint - Serve front page"""
    try:
        index_path = os.path.join(BASE_DIR, 'public', 'index.html')
        return send_file(index_path)
    except Exception as e:
        logger.error(f"Error serving index.html: {e}")
        # Fallback to API information
        return jsonify({
            'service': 'Central Researcher API',
            'version': '1.0.0',
            'status': 'running',
            'endpoints': {
                'health': '/api/health',
                'analyze': '/api/analyze (POST)',
                'researcher_details': '/api/researchers/<researcher_id> (GET)'
            },
            'documentation': 'Access API endpoints under /api/'
        })


@app.route('/results.html', methods=['GET'])
def results():
    """Serve results page"""
    try:
        results_path = os.path.join(BASE_DIR, 'public', 'results.html')
        return send_file(results_path)
    except Exception as e:
        logger.error(f"Error serving results.html: {e}")
        return jsonify({'error': 'Results page not found'}), 404


@app.route('/debug/files', methods=['GET'])
def debug_files():
    """Debug endpoint to list files in /app"""
    import glob
    files = {
        'base_dir': BASE_DIR,
        'public_exists': os.path.exists(os.path.join(BASE_DIR, 'public')),
        'public_files': glob.glob(os.path.join(BASE_DIR, 'public', '*')) if os.path.exists(os.path.join(BASE_DIR, 'public')) else [],
        'app_files': glob.glob(os.path.join(BASE_DIR, '*'))[:20]
    }
    return jsonify(files)


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'modules': {
            'PaperSearchEngine': PaperSearchEngine is not None,
            'CentralResearcher': CentralResearcher is not None,
            'KAKENHI': get_research_field_data is not None,
            'KeywordExtractor': KeywordExtractor is not None
        }
    })


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """
    Main analysis endpoint

    Request body:
    {
        "query": str,
        "email": str,
        "filter_institution": str (optional, default: "The University of Tokyo"),
        "top_k": int (optional, default: 100),
        "similarity_threshold": float (optional, default: 0.55)
    }

    Returns:
    {
        "status": "success" | "error",
        "data": {...} | null,
        "error": str | null
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                'status': 'error',
                'error': 'Request body is required'
            }), 400

        query = data.get('query')
        email = data.get('email')

        if not query:
            return jsonify({
                'status': 'error',
                'error': 'query parameter is required'
            }), 400

        if not email:
            return jsonify({
                'status': 'error',
                'error': 'email parameter is required'
            }), 400

        filter_institution = data.get('filter_institution', 'The University of Tokyo')
        # 空文字列の場合はフィルタを無効化
        if filter_institution == "":
            filter_institution = None
        top_k = data.get('top_k', 100)
        similarity_threshold = data.get('similarity_threshold', 0.55)

        logger.info(f"Received analysis request: query={query}, institution={filter_institution}")

        result = asyncio.run(run_combined_analysis(
            query=query,
            email=email,
            filter_institution=filter_institution,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        ))

        if result:
            return jsonify({
                'status': 'success',
                'data': result
            })
        else:
            return jsonify({
                'status': 'error',
                'error': 'Analysis returned no results'
            }), 404

    except Exception as e:
        logger.error(f"Error in analyze endpoint: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@app.route('/api/researchers/<researcher_id>', methods=['GET'])
def get_researcher_details(researcher_id: str):
    """
    Get detailed information about a specific researcher

    Query parameters:
    - author_name: str (required)
    """
    try:
        author_name = request.args.get('author_name')

        if not author_name:
            return jsonify({
                'status': 'error',
                'error': 'author_name parameter is required'
            }), 400

        logger.info(f"Fetching researcher details: {researcher_id}, {author_name}")

        # Collect data for single researcher
        mock_ranking = type('obj', (object,), {
            'author_id': researcher_id,
            'author_display_name': author_name
        })

        researcher_data = asyncio.run(collect_researcher_data([mock_ranking], top_n=1))

        if researcher_id in researcher_data:
            return jsonify({
                'status': 'success',
                'data': researcher_data[researcher_id]
            })
        else:
            return jsonify({
                'status': 'error',
                'error': 'Researcher data not found'
            }), 404

    except Exception as e:
        logger.error(f"Error in get_researcher_details endpoint: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'status': 'error',
        'error': 'Endpoint not found'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'status': 'error',
        'error': 'Internal server error'
    }), 500


if __name__ == '__main__':
    if not os.path.isdir(VECTOR_DIR):
        logger.warning(f"Warning: Vector directory not found: {VECTOR_DIR}")

    logger.info("Starting Flask API server...")
    logger.info(f"Vector directory: {VECTOR_DIR}")
    logger.info(f"Output directory: {OUTPUT_DIR}")

    # ポートを環境変数から取得（Cloud Run対応）
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'

    app.run(host='0.0.0.0', port=port, debug=debug)
