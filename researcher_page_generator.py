import logging
import sys
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(BASE_DIR)
sys.path.append(PROJ_ROOT)

from openai_model import OpenAIBase, LLM

# キーワード説明キャッシュのインポート
try:
    from keyword_description_cache import KeywordDescriptionCache, get_keyword_cache
except ImportError:
    # フォールバック: キャッシュが利用できない場合
    KeywordDescriptionCache = None
    get_keyword_cache = None

try:
    from researcher_search.core.models import ResearcherRanking
except ImportError:
    print(f"警告: 'ResearcherRanking' のインポートに失敗しました。型ヒントを 'Any' にフォールバックします。")
    print(f"'{BASE_DIR}' に 'researcher_search' フォルダが正しく配置されているか確認してください。")
    ResearcherRanking = Any

logger = logging.getLogger(__name__)

class ResearcherPageGenerator:
    """
    研究者個人のプロファイルHTMLページを生成するクラス。
    """

    def __init__(
        self,
        main_report_filename: str,
        keyword_cache_file: Optional[Path] = None,
        use_cache_only: bool = True
    ):
        """
        ジェネレータを初期化します。

        Args:
            main_report_filename: メインレポートへの「戻る」リンクに使用するファイル名 (例: "The_University_of_Tokyo.html")
            keyword_cache_file: キーワード説明キャッシュファイルのパス（Noneの場合はデフォルト）
            use_cache_only: Trueの場合、キャッシュにない説明はスキップ（LLM呼び出しなし）
        """
        self.main_report_filename = main_report_filename
        self.use_cache_only = use_cache_only

        # キーワード説明キャッシュの初期化
        self._keyword_cache: Optional[KeywordDescriptionCache] = None
        if get_keyword_cache is not None:
            self._keyword_cache = get_keyword_cache(keyword_cache_file)
            self._keyword_cache.load()
            logger.info(f"Keyword description cache loaded: {self._keyword_cache.stats()}")
        else:
            logger.warning("Keyword description cache not available")

        logger.debug(f"ResearcherPageGenerator initialized (Back-link: {main_report_filename})")

    def generate_page(
        self,
        researcher_data: Dict[str, Any],
        output_path: Path,
        ranking: Optional[Any] = None,
    ) -> None:
        """
        個別のHTMLページを生成してファイルに書き出す。
        (元の generate_researcher_page 関数のロジック)
        キャッシュデータのみで動作する簡易版（rankingはオプショナル）
        """
        # Support both "author_name" and "researcher_name" keys
        author_name = researcher_data.get("author_name", researcher_data.get("researcher_name", "Unknown"))
        kakenhi_info = researcher_data.get("kakenhi_info")
        keywords = researcher_data.get("keywords", [])
        summary = researcher_data.get("summary", "")
        statistics = researcher_data.get("statistics", {})

        # LLM Settings
        llm = LLM(base="openai", use_model="gpt4o-mini")

        # Generate KAKENHI section HTML
        kakenhi_html = ""
        if kakenhi_info is not None and not kakenhi_info.empty:
            kakenhi_html = "<div class='kakenhi-section'>"
            for idx, row in kakenhi_info.iterrows():
                kakenhi_html += f"""
                <div class='grant-item'>
                    <h4>研究期間: {row.get('研究期間 (年度)', 'N/A')}</h4>
                    <p><strong>配分額:</strong> {row.get('配分額*注記', 'N/A')}</p>
                    <p><strong>研究分野:</strong> {row.get('研究分野', 'N/A')}</p>
                    <p><strong>キーワード:</strong> {row.get('キーワード', 'N/A')}</p>
                    <p><strong>研究概要:</strong> {row.get('研究概要', row.get('研究開始時の研究の概要', 'N/A'))}</p>
                </div>
                """
            kakenhi_html += "</div>"
        else:
            kakenhi_html = "<p class='no-data'>KAKENHI情報が見つかりませんでした。</p>"

        # Generate keywords HTML
        keywords_html = ""
        if keywords:
            keywords_html = "<ul class='keywords-list'>"
            for i, kw in enumerate(keywords):
                # Extract 'keyword' value if kw is a dict, otherwise use kw as-is
                if isinstance(kw, dict):
                    keyword_text = kw.get('keyword', str(kw))
                else:
                    keyword_text = kw

                # キャッシュからキーワード説明を取得（O(1)の高速ルックアップ）
                keyword_explanation = None
                if self._keyword_cache is not None:
                    keyword_explanation = self._keyword_cache.get(keyword_text)

                if keyword_explanation is None:
                    if self.use_cache_only:
                        # キャッシュのみモード: 説明なしで表示
                        keyword_explanation = "説明準備中"
                    else:
                        # フォールバック: LLMで説明を生成（低速）
                        try:
                            if i > 0:
                                time.sleep(0.5)  # Rate limiting

                            keyword_explanation = llm.get_response(
                                f"次のキーワードの説明を日本語で教えてください: {keyword_text}",
                                temperature=0.3
                            )[0]

                            # キャッシュに追加
                            if self._keyword_cache is not None:
                                self._keyword_cache.set(keyword_text, keyword_explanation)
                        except Exception as e:
                            logger.warning(f"Failed to get explanation for keyword '{keyword_text}': {e}")
                            keyword_explanation = "説明を取得できませんでした"

                keywords_html += f"""
                    <li class="keyword-item">
                        {keyword_text}
                        <span class="keyword-tooltip">{keyword_explanation}</span>
                    </li>
                """
            keywords_html += "</ul>"
        else:
            keywords_html = "<p class='no-data'>キーワードが抽出されませんでした。</p>"

        # Generate summary HTML
        summary_html = summary if summary else "<p class='no-data'>研究サマリーが生成されませんでした。</p>"

        # Generate statistics HTML from cache data
        stats_html = ""
        if statistics:
            total_papers = statistics.get('total_papers', 0)
            first_author = statistics.get('papers_as_first_author', 0)
            last_author = statistics.get('papers_as_last_author', 0)
            total_citations = statistics.get('total_citations', 0)
            avg_citations = statistics.get('average_citations', 0)
            year_range = statistics.get('year_range', {})

            stats_html = f"""
            <div class="researcher-stats">
                <div class="stat-card">
                    <div class="stat-label">Total Papers</div>
                    <div class="stat-value">{total_papers}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">First Author</div>
                    <div class="stat-value">{first_author}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Last Author</div>
                    <div class="stat-value">{last_author}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Total Citations</div>
                    <div class="stat-value">{total_citations}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Avg Citations</div>
                    <div class="stat-value">{avg_citations:.1f}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Year Range</div>
                    <div class="stat-value">{year_range.get('start', 'N/A')}-{year_range.get('end', 'N/A')}</div>
                </div>
            </div>
            """

        # Generate complete HTML
        html_content = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{author_name} - Researcher Profile</title>
    <style>
        {self._get_css()}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{author_name}</h1>
            {stats_html}
            <a href="../{self.main_report_filename}" class="back-link">← Back to Rankings</a>
        </header>

        <section class="summary-section">
            <h2>📝 Researcher Summary</h2>
            <div class="summary-content">
                {summary_html}
            </div>
        </section>

        <section class="keywords-section">
            <h2>🔑 Research Keywords</h2>
            <div class="keywords-content">
                {keywords_html}
            </div>
        </section>

        <section class="kakenhi-section-wrapper">
            <h2>💰 KAKENHI Grants Information</h2>
            {kakenhi_html}
        </section>

        <footer>
            <p>Generated by Central Researcher Analysis System</p>
        </footer>
    </div>
</body>
</html>"""

        # Write to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info(f"Generated researcher page: {output_path}")

    def _get_css(self) -> str:
        """
        Return CSS styles for researcher individual pages.
        (元の get_researcher_page_css 関数の内容)
        """
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f8f9fa;
        }

        .container {
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
        }

        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }

        header h1 {
            font-size: 2.5em;
            margin-bottom: 20px;
        }

        .researcher-stats {
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
            margin-bottom: 20px;
        }

        .stat-card {
            background: rgba(255, 255, 255, 0.2);
            padding: 15px 20px;
            border-radius: 8px;
            text-align: center;
            min-width: 120px;
        }

        .stat-label {
            font-size: 0.85em;
            opacity: 0.9;
            margin-bottom: 5px;
        }

        .stat-value {
            font-size: 1.8em;
            font-weight: bold;
        }

        .back-link {
            display: inline-block;
            color: white;
            text-decoration: none;
            padding: 10px 20px;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 5px;
            transition: background 0.3s;
        }

        .back-link:hover {
            background: rgba(255, 255, 255, 0.3);
        }

        section {
            background: white;
            padding: 25px;
            margin-bottom: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }

        h2 {
            font-size: 1.8em;
            color: #4a5568;
            margin-bottom: 20px;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }

        .summary-content {
            font-size: 1.05em;
            line-height: 1.8;
            color: #2d3748;
        }

        .statistics {
            margin-top: 15px;
            padding: 15px;
            background: #f7fafc;
            border-left: 4px solid #667eea;
            border-radius: 5px;
        }

        .keywords-content {
        }

        .keywords-list {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            list-style: none;
            padding: 0;
            margin: 0;
        }

        .keyword-item {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 10px 20px;
            border-radius: 25px;
            font-size: 0.95em;
            font-weight: 500;

            position: relative;
            transition: all 0.3s ease;
        }

        .keyword-item:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }

        .keyword-tooltip {
            visibility: hidden; 
            opacity: 0;
            transition: opacity 0.3s ease;

            width: 250px;
            background-color: #333;
            color: #fff;
            text-align: left; 
            padding: 10px;
            border-radius: 6px;
            font-size: 0.85em;
            font-weight: normal;
            line-height: 1.5;

            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            transform: translateX(-50%);
            
            pointer-events: none;
        }

        .keyword-tooltip::after {
            content: "";
            position: absolute;
            top: 100%;
            left: 50%;
            margin-left: -5px;
            border-width: 5px;
            border-style: solid;
            border-color: #333 transparent transparent transparent;
        }

        .keyword-item:hover .keyword-tooltip {
            visibility: visible;
            opacity: 1;
        }

        .kakenhi-section {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        .grant-item {
            padding: 20px;
            background: #f7fafc;
            border-left: 5px solid #667eea;
            border-radius: 5px;
        }

        .grant-item h4 {
            color: #667eea;
            margin-bottom: 15px;
            font-size: 1.2em;
        }

        .grant-item p {
            margin-bottom: 10px;
            line-height: 1.6;
        }

        .grant-item strong {
            color: #4a5568;
        }

        .no-data {
            color: #718096;
            font-style: italic;
            padding: 20px;
            background: #f7fafc;
            border-radius: 5px;
            text-align: center;
        }

        footer {
            text-align: center;
            padding: 20px;
            color: #718096;
            font-size: 0.9em;
        }

        @media (max-width: 768px) {
            header h1 {
                font-size: 2em;
            }

            .researcher-stats {
                justify-content: center;
            }

            .container {
                padding: 10px;
            }
        }
    """