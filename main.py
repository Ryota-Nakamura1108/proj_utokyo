import asyncio
import os
import logging
import sys
from typing import List, Optional, Any, Dict, Tuple
from datetime import datetime
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from article_search.core.article_search import PaperSearchEngine, SearchResults, SearchResult
except ImportError as e:
    print(f"エラー: {e}")
    print(f"このスクリプトと同じディレクトリ ({BASE_DIR}) に配置してください。")
    sys.exit(1)

try:
    from researcher_search.core.central_researcher import CentralResearcher
    from researcher_search.core.models import (
        CentralResearcherConfig, InputPaper, WorkRaw,
        Authorship, Topic, AuthorPosition, ResearcherRanking
    )
except ImportError as e:
    print(f"エラー: 'central_researcher' モジュールのインポートに失敗しました: {e}")
    print(f"'{BASE_DIR}' に 'central_researcher' フォルダが正しく配置されているか確認してください。")
    sys.exit(1)

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
    logger.error("このスクリプトと同じディレクトリに main_html_generator.py を配置してください。")
    MainHTMLReportGenerator = None

try:
    sys.path.insert(0, str(Path(__file__).parent))
    from researcher_page_generator import ResearcherPageGenerator
except ImportError as e:
    logger.error(f"ResearcherPageGenerator のインポートに失敗しました: {e}")
    logger.error("このスクリプトと同じディレクトリに researcher_page_generator.py を配置してください。")
    ResearcherPageGenerator = None


def _parse_year(date_str: str) -> Optional[int]:
    """'YYYY-MM-DD' 形式の文字列から年を抽出する"""
    if not date_str:
        return None
    try:
        # YYYY-MM-DD 形式を想定
        return int(date_str.split('-')[0])
    except (ValueError, IndexError):
        logger.warning(f"日付形式のパースに失敗: {date_str}")
        return None

def _convert_authorships(search_result_authors: List[Any]) -> List[Authorship]:
    """
    SearchResult.authorships を WorkRaw.authorships (List[Authorship]) に変換する。
    """
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
    """
    Collect additional data for top N researchers using kaken_package and keyword_extractor.

    Args:
        rankings: List of researcher rankings (sorted by CRS score)
        top_n: Number of top researchers to process (default: 20)

    Returns:
        Dictionary mapping author_id to researcher data (only for top N researchers)
    """
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
                    data["kakenhi_info"] = kakenhi_df
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

def generate_main_html_report(
    researcher: CentralResearcher,
    output_path: str,
    field_name: str,
    rankings: List[ResearcherRanking],
    filtered_researchers_count: Optional[int] = None,
) -> str:
    """
    Generate a main HTML report using MainHTMLReportGenerator.

    Args:
        researcher: CentralResearcher instance
        output_path: Output HTML file path
        field_name: Research field name
        rankings: List of researcher rankings
        filtered_researchers_count: Total count of filtered researchers (optional)

    Returns:
        Path to generated HTML file
    """
    if not MainHTMLReportGenerator:
        logger.error("MainHTMLReportGenerator が利用できないため、HTMLレポートを生成できません。")
        return ""

    generator = MainHTMLReportGenerator(rankings_list=rankings[:20])

    graph = researcher.network
    edge_data = researcher.edge_data
    works = researcher.works
    authors = researcher.authors
    institutions = researcher.institutions
    network_stats = researcher.get_network_statistics()

    html_file = generator.generate_report(
        rankings=rankings[:20],
        graph=graph,
        edge_data=edge_data,
        works=works,
        authors=authors,
        institutions=institutions,
        network_stats=network_stats,
        field_name=field_name,
        output_path=output_path,
        language="en",
        researcher_data=None,
        filtered_researchers_count=filtered_researchers_count
    )

    logger.info(f"Generated main HTML report: {html_file}")
    return html_file


def convert_search_results_to_workraw_list(search_results: SearchResults) -> List[WorkRaw]:
    """
    PaperSearchEngine の SearchResults を CentralResearcher が
    使用する WorkRaw のリストに変換する。
    """
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
    vector_dir: str, 
    email: str, 
    filter_institution: str = "The University of Tokyo"
) -> Optional[Tuple[CentralResearcher, List[ResearcherRanking], Dict[str, Dict[str, Any]]]]:
    """
    分析を実行し、分析インスタンス、ランキング、追加データを返す。
    （HTML生成ロジックを分離）
    """

    logger.info(f"--- ステップ 1: 論文検索 (Query: '{query}') ---")
    try:
        search_engine = PaperSearchEngine(vectors_dir=vector_dir)
        search_results: SearchResults = search_engine.search(
            query=query,
            top_k=100,
            similarity_threshold=0.55
        )

        if not search_results.results:
            logger.warning("論文検索でヒットしませんでした。処理を終了します。")
            return None
        logger.info(f"{len(search_results.results)} 件の論文がヒットしました。")

    except Exception as e:
        logger.error(f"論文検索ステップでエラーが発生しました: {e}")
        return None

    logger.info(f"--- ステップ 2: Central Researcher 分析準備 ---")

    config = CentralResearcherConfig(
        papers=[InputPaper(title="dummy")],
        email=email,
        # min_in_corpus_works=2,
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
            
            logger.info(f"--- ステップ 4: 機関フィルタリング ('{filter_institution}') ---")
            rankings: List[ResearcherRanking] = analyzer.filter_by_institution(
                institution_names=[filter_institution],
                fuzzy_match=True
            )
            logger.info(f"フィルタリング完了: '{filter_institution}' の研究者数 = {len(rankings)}")

            if not rankings:
                logger.warning(f"'{filter_institution}' に所属する研究者は見つかりませんでした。")
                return None

            print("\n" + "="*80)
            print(f"🏆 TOP 10 CENTRAL RESEARCHERS - {filter_institution}")
            print("="*80)

            for ranking in rankings[:1]:
                h_index = int(ranking.h_index_global) if ranking.h_index_global else 0
                print(f"{ranking.rank:2d}. {ranking.author_display_name:<40} "
                      f"CRS: {ranking.crs_final:.4f} | "
                      f"h-index: {h_index:2d} | "
                      f"Papers: {ranking.n_in_corpus_works:2d}")

            logger.info(f"--- ステップ 5: Top 10 研究者の追加データ収集 (KAKENHI & Keywords) ---")
            researcher_data_dict = await collect_researcher_data(rankings, top_n=10)
            logger.info(f"分析とデータ収集が完了。HTMLレポート生成のためにデータを返します。")

            return (analyzer, rankings, researcher_data_dict)

    except Exception as e:
        logger.error(f"Central Researcher 分析ステップでエラーが発生しました: {e}")
        return None


def main():
    VECTOR_DIR = os.path.join(BASE_DIR, "./article_search/vectors")
    SEARCH_QUERY = "quantum computing"
    YOUR_EMAIL = "rnakamura@memorylab.jp"
    FILTER_INSTITUTION = "The University of Tokyo"

    if not os.path.isdir(VECTOR_DIR):
        logger.error(f"ベクトルディレクトリが見つかりません: {VECTOR_DIR}")
        logger.error("VECTOR_DIR のパス設定を確認してください。")
        return

    if YOUR_EMAIL == "rnakamura@memorylab.jp" or YOUR_EMAIL == "your_email@example.com":
        logger.warning("="*50)
        logger.warning("警告: YOUR_EMAIL をご自身のメールアドレスに変更してください。")
        logger.warning("OpenAlex API の利用に必要です。")
        logger.warning("="*50)

    print("\n🧠 Central Researcher Analysis System - Article Search Combined")
    print("=" * 80)
    print(f"Query: {SEARCH_QUERY}")
    print(f"Filter Institution: {FILTER_INSTITUTION}")
    print("=" * 80)

    try:
        analysis_results = asyncio.run(run_combined_analysis(
            query=SEARCH_QUERY,
            vector_dir=VECTOR_DIR,
            email=YOUR_EMAIL,
            filter_institution=FILTER_INSTITUTION
        ))

        if analysis_results:
            analyzer_instance, rankings, researcher_data_dict = analysis_results
            logger.info("分析成功。レポート生成を開始します。")

            output_dir = Path(BASE_DIR) / "output"
            institution_safe = FILTER_INSTITUTION.replace(" ", "_").replace("/", "_")
            html_filename = f"{SEARCH_QUERY}.html"
            html_file_path = str(output_dir / html_filename)

            logger.info(f"--- ステップ 6: Top 10 研究者の個別ページ生成 ---")
            researcher_pages_dir = output_dir / "researcher_pages"
            researcher_pages_dir.mkdir(parents=True, exist_ok=True)

            if ResearcherPageGenerator:
                page_generator = ResearcherPageGenerator(main_report_filename=html_filename)

                for ranking in rankings[:10]:
                    if ranking.author_id in researcher_data_dict:
                        researcher_data = researcher_data_dict[ranking.author_id]
                        page_path = researcher_pages_dir / f"{ranking.author_id}.html"
                        
                        page_generator.generate_page(researcher_data, page_path, ranking)
            else:
                logger.error("ResearcherPageGenerator がインポートされていないため、個別ページを生成できません。")

            logger.info(f"--- ステップ 7: メインHTML レポート生成 ---")

            field_name = f"{FILTER_INSTITUTION} - {SEARCH_QUERY}"

            html_file = generate_main_html_report(
                researcher=analyzer_instance,
                output_path=html_file_path,
                field_name=field_name,
                rankings=rankings,
                filtered_researchers_count=len(rankings)
            )

            if html_file:
                logger.info(f"HTMLレポート生成完了: {html_file_path}")
                logger.info(f"ブラウザで開く: file://{Path(html_file_path).absolute()}")
            else:
                logger.error("メインHTMLレポートの生成に失敗しました。")

            # --- ステップ 8: CSVエクスポート ---
            logger.info(f"--- ステップ 8: CSVエクスポート ---")
            csv_filename = f"{institution_safe}.csv"
            output_csv_path = output_dir / csv_filename

            try:
                import pandas as pd
                ranking_data = []
                for r in rankings[:1]:
                    # 機関情報の取得
                    author_master = next((a for a in analyzer_instance.authors if a.author_id == r.author_id), None)
                    institution_names = []
                    if author_master and author_master.last_known_institution_ids:
                        for inst_id in author_master.last_known_institution_ids:
                            inst = next((i for i in analyzer_instance.institutions if i.institution_id == inst_id), None)
                            if inst:
                                institution_names.append(inst.display_name)

                    ranking_data.append({
                        'Rank': r.rank,
                        'Author ID': r.author_id,
                        'Author Name': r.author_display_name,
                        'CRS Score': r.crs_final,
                        'H-Index': r.h_index_global or 0,
                        'Papers in Corpus': r.n_in_corpus_works,
                        'Leadership Rate': r.leadership_rate,
                        'Institution': ', '.join(institution_names) if institution_names else 'N/A'
                    })

                df = pd.DataFrame(ranking_data)
                df.to_csv(output_csv_path, index=False, encoding='utf-8')
                logger.info(f"CSVファイルを保存しました: {output_csv_path}")

            except Exception as e:
                logger.error(f"CSV保存に失敗しました: {e}")

            print("\n✨ Analysis completed successfully!")
            print(f"📁 Check the 'output' directory for results")
            print(f"📄 Main HTML report: {html_filename}")
            print(f"📄 Individual researcher pages in 'output/researcher_pages'")
        else:
            logger.warning("分析がデータを返しませんでした。")

    except Exception as e:
        logger.error(f"メイン処理で予期せぬエラーが発生しました: {e}", exc_info=True)


if __name__ == "__main__":
    main()