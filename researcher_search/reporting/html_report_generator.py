"""HTML Report Generator for Central Researcher Analysis."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import networkx as nx
from collections import Counter
# import country_converter as coco # 削除
import pandas as pd

from ..core.models import ResearcherRanking, WorkRaw, AuthorMaster, GraphEdge
from ..core.explainability import ExplainabilityAnalyzer

logger = logging.getLogger(__name__)


csv_dir = "data/csv"
if not Path(csv_dir).exists():
    pass
else: 
    with open(csv_dir) as f:
        researcher_data = pd.read_csv(f)


class HTMLReportGenerator:
    """Generates comprehensive HTML reports with visualizations."""
    
    def __init__(self):
        """Initialize the HTML report generator."""
        self.explainability = ExplainabilityAnalyzer()
        self.i18n = self._load_i18n_strings()
    
    def _load_i18n_strings(self) -> Dict[str, Dict[str, str]]:
        """Load internationalization strings."""
        return {
            "en": {
                "title": "Central Researcher Analysis Report",
                "field": "Research Field",
                "generated_on": "Generated on",
                "summary_stats": "📊 Summary Statistics",
                "total_researchers": "Total Researchers",
                "total_papers": "Total Papers",
                "total_collaborations": "Total Collaborations",
                "network_density": "Network Density",
                "avg_clustering": "Average Clustering",
                "UTokyo_researchers": "UTokyo Researchers",
                "detailed_rankings": "📋 Detailed Rankings",
                "top_central_researchers": "🏆 Top Central Researchers",
                "search_placeholder": "🔍 Search by researcher name or institution...",
                "rank": "Rank",
                "researcher": "Researcher",
                "institution": "Institution",
                "crs_score": "CRS Score",
                "h_index": "H-Index",
                "papers": "Papers",
                "leadership_rate": "Leadership Rate",
                "researcher_summary": "Researcher Summary",
                "kakenhi_info": "KAKENHI Info",
                "keyword": "Keywords",  
                "researcher_summary": "Researcher Summary",
                "kakenhi_info": "KAKENHI Grants Info",
                "keyword": "Keywords",
                "institution_distribution": "🌍 Institution Distribution (CRS-Weighted)",
                "research_topics_count": "📚 Research Topics by Count",
                "research_topics_crs": "📚 Research Topics by CRS Impact",
                "collaboration_network": "🔗 Collaboration Network",
                "timeline_analysis": "📈 Publication Timeline",
                "zoom_in": "🔍 Zoom In",
                "zoom_out": "🔍 Zoom Out",
                "reset_view": "↻ Reset View",
                "powered_by": "Powered by Central Researcher Analysis System",
                "methodology_title": "📊 Methodology: Central Researcher Score (CRS)",
                "methodology_intro": "The Central Researcher Score (CRS) is a composite metric that measures a researcher's influence and centrality within the research network. It combines three key components:",
                "centrality_title": "🌐 Network Centrality (70%)",
                "centrality_desc": "Position and connectivity within the collaboration network:",
                "degree_centrality": "Degree (20%): Number of direct collaborators",
                "pagerank": "PageRank (40%): Influence based on collaborators' importance",
                "betweenness": "Betweenness (10%): Role as bridge between research groups",
                "citation_title": "📚 Citation Impact (15%)",
                "citation_desc": "Research impact and scholarly influence:",
                "h_index_desc": "H-Index (100%): Citation-based research impact measure",
                "leadership_title": "👑 Leadership Score (15%)",
                "leadership_desc": "Research leadership and initiative:",
                "leadership_rate_desc": "Leadership Rate (50%): Ratio of papers as first/last/corresponding author",
                "network_overview": "Network Overview",
                "researchers": "researchers",
                "collaborations": "collaborations",
                "institutions": "Institutions",
                "showing_all": "Showing all",
                "showing_of": "Showing",
                "of": "of",
                "crs_tooltip": "Central Researcher Score: A composite metric combining centrality (network position), citation impact (H-index), and leadership roles. Higher scores indicate greater research influence and collaboration centrality.",
                "h_index_tooltip": "H-Index measures citation impact: the maximum number h where a researcher has h papers with at least h citations each. Higher values indicate greater citation influence.",
                "papers_tooltip": "Number of papers by this researcher included in the analyzed corpus. This represents their publication activity within the specific research domain.",
                "leadership_tooltip": "Ratio of papers where the researcher held leadership positions (first author, last author, or corresponding author). Values >1.0 indicate multiple leadership roles per paper on average.",
                "crs_chart_label": "Central Researcher Scores (CRS)",
                # "countries_explanation": "Countries ranked by median CRS scores with statistical variance indicators", # 削除
                # "institutions_explanation": "Institutions ranked by median CRS scores with statistical variance indicators", # 削除
                "network_statistics": "📈 Network Statistics",
                "research_topics_count_title": "📊 Research Topics by Paper Count",
                "research_topics_crs_title": "📊 Research Topics by CRS Impact",
                "timeline_title": "📈 Publication Timeline",
                "topics_count_explanation": "Topics ranked by number of papers",
                "topics_crs_explanation": "Topics ranked by total CRS impact of contributing researchers",
                "timeline_explanation": "Timeline shows publication impact weighted by citations and recency",
                "coloring_mode": "Coloring Mode",
                "color_by_institution": "Color by Institution",
                "color_by_country": "Color by Country",
                "color_by_crs": "Color by CRS Score",
                "cluster_info": "Cluster Info",
                "cluster_details": "Click on a colored cluster area to see institution composition"
            },
            "ja": {
                "title": "中心研究者分析レポート",
                "field": "研究分野",
                "generated_on": "生成日時",
                "summary_stats": "📊 要約統計",
                "total_researchers": "総研究者数",
                "total_papers": "総論文数",
                "total_collaborations": "総共同研究数",
                "network_density": "ネットワーク密度",
                "avg_clustering": "平均クラスタリング係数",
                "UTokyo_researchers": "東京大学中心研究者",
                "detailed_rankings": "📋 詳細ランキング",
                "top_central_researchers": "🏆 上位中心研究者",
                "search_placeholder": "🔍 研究者名または所属機関で検索...",
                "rank": "順位",
                "researcher": "研究者",
                "institution": "所属機関",
                "crs_score": "CRSスコア",
                "h_index": "標準化H指数",
                "papers": "論文数",
                "leadership_rate": "リーダーシップ率",
                "researcher_summary": "研究者サマリー",
                "kakenhi_info": "科研費",
                "keyword": "キーワード",  
                "researcher_summary": "研究者サマリー",
                "kakenhi_info": "科研費情報",
                "keyword": "キーワード",
                "institution_distribution": "🌍 機関別分布（CRS重み付き）",
                "research_topics_count": "📚 研究トピック（論文数順）",
                "research_topics_crs": "📚 研究トピック（CRS影響力順）",
                "collaboration_network": "🔗 共同研究ネットワーク",
                "timeline_analysis": "📈 出版タイムライン",
                "zoom_in": "🔍 拡大",
                "zoom_out": "🔍 縮小",
                "reset_view": "↻ リセット",
                "powered_by": "中心研究者分析システム",
                "methodology_title": "📊 手法: 中心研究者スコア（CRS）",
                "methodology_intro": "中心研究者スコア（CRS）は、研究ネットワーク内での研究者の影響力と中心性を測定する複合指標です。3つの主要要素を組み合わせています：",
                "centrality_title": "🌐 ネットワーク中心性（70%）",
                "centrality_desc": "共同研究ネットワーク内での位置と接続性：",
                "degree_centrality": "次数（20%）：直接的な共同研究者数",
                "pagerank": "PageRank（40%）：共同研究者の重要性に基づく影響力",
                "betweenness": "媒介中心性（10%）：研究グループ間の橋渡し役割",
                "citation_title": "📚 引用影響力（15%）",
                "citation_desc": "研究インパクトと学術的影響力：",
                "h_index_desc": "標準化H指数（100%）：引用ベースの研究影響力指標",
                "leadership_title": "👑 リーダーシップスコア（15%）",
                "leadership_desc": "研究リーダーシップとイニシアティブ：",
                "leadership_rate_desc": "リーダーシップ率（50%）：筆頭・責任・責任著者として論文の比率",
                "network_overview": "ネットワーク概要",
                "researchers": "研究者",
                "collaborations": "共同研究",
                "institutions": "所属機関",
                "showing_all": "全",
                "showing_of": "表示中",
                "of": "/",
                "crs_tooltip": "中心研究者スコア：ネットワーク中心性（ネットワーク位置）、引用影響力（標準化H指数）、リーダーシップ役割を組み合わせた複合指標。高いスコアは大きな研究影響力と共同研究での中心性を示します。",
                "h_index_tooltip": "標準化H指数は引用影響力を測定：研究者がh編以上の引用を受けたh編の論文を持つ最大数hを正規化した指標。高い値ほど大きな引用影響力を示します。",
                "papers_tooltip": "分析対象コーパスに含まれる当該研究者の論文数。特定研究領域での出版活動を表します。",
                "leadership_tooltip": "研究者がリーダーシップ地位（筆頭著者、責任著者、責任著者）を担った論文の比率。1.0を超える値は平均して論文あたり複数のリーダーシップ役割を示します。",
                "crs_chart_label": "中心研究者スコア（CRS）",
                "network_statistics": "📈 ネットワーク統計",
                "research_topics_count_title": "📊 研究トピック（論文数順）",
                "research_topics_crs_title": "📊 研究トピック（CRS影響力順）",
                "timeline_title": "📈 出版タイムライン",
                "topics_count_explanation": "論文数による研究トピックランキング",
                "topics_crs_explanation": "研究者のCRS影響力合計による研究トピックランキング",
                "timeline_explanation": "引用数と最新性で重み付けした出版影響力タイムライン",
                "coloring_mode": "色分けモード",
                "color_by_institution": "所属機関別",
                "color_by_country": "国別",
                "color_by_crs": "CRSスコア別",
                "cluster_info": "クラスタ情報",
                "cluster_details": "色付きクラスタエリアをクリックして所属機関構成を表示"
            }
        }
    
    def generate_report(self,
                       rankings: List[ResearcherRanking],
                       graph: nx.Graph,
                       edge_data: Dict[str, GraphEdge],
                       works: List[WorkRaw],
                       authors: Dict[str, AuthorMaster],
                       institutions: Dict[str, Any],
                       network_stats: Dict[str, Any],
                       field_name: str = "Research Field",
                       output_path: str = "report.html",
                       language: str = "en",
                       researcher_data: Optional[pd.DataFrame] = None,
                       filtered_researchers_count: Optional[int] = None) -> str:
        """Generate comprehensive HTML report.
        
        Args:
            rankings: List of researcher rankings
            graph: NetworkX collaboration graph
            edge_data: Edge information
            works: List of works in corpus
            authors: Author master data
            institutions: Institution data
            network_stats: Network statistics
            field_name: Name of the research field
            output_path: Output file path
            language: Language for report ('en' or 'ja')
            
        Returns:
            Path to generated HTML file
        """
        logger.info(f"Generating HTML report for {field_name} (language: {language})")
        
        # Get language-specific strings
        lang_strings = self.i18n.get(language, self.i18n["en"])
        
        # Prepare visualization data
        network_data = self._prepare_network_data(graph, rankings[:20], authors, institutions)
        ranking_data = self._prepare_ranking_data(rankings[:20], authors, institutions, researcher_data)
        topic_data = self._prepare_topic_data(works, rankings, authors, lang_strings)
        timeline_data = self._prepare_timeline_data(works, lang_strings)
        stats_data = self._prepare_statistics_data(rankings, works, authors, network_stats)
        # institution_data = self._prepare_institution_data(authors, institutions, rankings) # 削除
        
        # Generate HTML content
        html_content = self._generate_html_template(
            field_name=field_name,
            language=language,
            lang_strings=lang_strings,
            network_data=network_data,
            ranking_data=ranking_data,
            topic_data=topic_data,
            timeline_data=timeline_data,
            stats_data=stats_data,
            # institution_data=institution_data, 削除
            total_papers=len(works),
            total_authors=len(authors),
            total_rankings=len(rankings),
            filtered_researchers_count=filtered_researchers_count if filtered_researchers_count is not None else len(rankings)
        )
        
        # Write to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        logger.info(f"HTML report generated: {output_file}")
        return str(output_file)
    
    def _prepare_network_data(self, graph: nx.Graph, top_rankings: List[ResearcherRanking], 
                             authors: Dict[str, AuthorMaster],
                             institutions: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare network visualization data."""
        nodes = []
        links = []
        
        # Create ranking lookup for colors
        ranking_lookup = {r.author_id: i for i, r in enumerate(top_rankings)}
        
        # Collect all institutions and countries to create color mapping
        all_institutions = set()
        all_countries = set()
        for node_id in graph.nodes():
            author_data = authors.get(node_id)
            if author_data and author_data.last_known_institution_ids:
                for inst_id in author_data.last_known_institution_ids:
                    if inst_id in institutions:
                        all_institutions.add(institutions[inst_id].display_name)
                        all_countries.add(institutions[inst_id].country_code or "Unknown")
        
        # Convert to sorted lists for consistent colors
        institution_list = sorted(list(all_institutions))
        country_list = sorted(list(all_countries))
        
        # Add nodes
        for node_id in graph.nodes():
            author_data = authors.get(node_id, None)
            name = author_data.display_name if author_data else "Unknown"
            
            # Get institution and country information
            primary_institution = "Unknown"
            primary_country = "Unknown"
            institution_id = None
            if author_data and author_data.last_known_institution_ids:
                # Use the first institution as primary
                inst_id = author_data.last_known_institution_ids[0]
                if inst_id in institutions:
                    primary_institution = institutions[inst_id].display_name
                    primary_country = institutions[inst_id].country_code or "Unknown"
                    institution_id = inst_id
            
            # Get institution and country indices for consistent coloring
            institution_index = 0
            country_index = 0
            if primary_institution in institution_list:
                institution_index = institution_list.index(primary_institution)
            if primary_country in country_list:
                country_index = country_list.index(primary_country)
            
            # Get CRS score for this author
            author_crs = 0.0
            for r in top_rankings:
                if r.author_id == node_id:
                    author_crs = r.crs_final
                    break
            
            # Determine node size based on ranking
            if node_id in ranking_lookup:
                rank = ranking_lookup[node_id]
                size = max(20 - rank, 8)  # Top researchers get larger nodes
                color_intensity = max(1.0 - rank * 0.05, 0.3)  # Fade out lower ranks
                group = 1  # Top researchers
            else:
                size = 5
                color_intensity = 0.1
                group = 2  # Other researchers
            
            nodes.append({
                "id": node_id,
                "name": name,
                "size": size,
                "group": group,
                "rank": ranking_lookup.get(node_id, None),
                "color_intensity": color_intensity,
                "institution": primary_institution,
                "institution_index": institution_index,
                "country": primary_country,
                "country_index": country_index,
                "crs_score": author_crs
            })
        
        # Add edges (sample for performance - reduce for better performance)
        edge_count = 0
        max_edges = 200  # Reduced limit for better performance
        
        # Sort edges by weight to keep the most important ones
        edges_with_weights = []
        for source, target in graph.edges():
            weight = graph[source][target].get('weight', 1.0)
            edges_with_weights.append((source, target, weight))
        
        # Sort by weight (descending) and take top edges
        edges_with_weights.sort(key=lambda x: x[2], reverse=True)
        
        for source, target, weight in edges_with_weights[:max_edges]:
            links.append({
                "source": source,
                "target": target,
                "weight": weight,
                "width": min(weight * 1.5, 8)  # Reduced scaling for better visibility
            })
            edge_count += 1
        
        return {
            "nodes": nodes,
            "links": links,
            "institutions": institution_list,
            "countries": country_list,
            "total_nodes": len(nodes),
            "total_edges": len(links),
            "sampled": edge_count >= max_edges,
            "max_crs": max([n["crs_score"] for n in nodes] + [0.0])
        }
    
    def _prepare_ranking_data(self, rankings: List[ResearcherRanking], 
                            authors: Dict[str, AuthorMaster],
                            institutions: Dict[str, Any],
                            researcher_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Prepare ranking visualization data with institution information."""
        names = []
        institutions_list = []
        summaries = []
        kakenhi_infos = []
        keywords = [] 
        
        for r in rankings:
            names.append(r.author_display_name[:30])
            
            # Get institution information for this author
            author_data = authors.get(r.author_id)
            if author_data and author_data.last_known_institution_ids:
                inst_id = author_data.last_known_institution_ids[0]
                if inst_id in institutions:
                    institution_name = institutions[inst_id].display_name[:40] + ("..." if len(institutions[inst_id].display_name) > 40 else "")
                    institutions_list.append(institution_name)
                else:
                    institutions_list.append("Unknown")
            else:
                institutions_list.append("Unknown")
        
            # Get extra info from DataFrame
            summary = "N/A"
            kakenhi = "N/A"
            keyword = "N/A"
            
            # researcher_data が渡されていて、author_id が存在する場合
            if researcher_data is not None and not researcher_data.empty and r.author_id in researcher_data.index:
                author_info = researcher_data.loc[r.author_id]
                summary = author_info.get('researcher_summary', 'N/A')
                kakenhi = author_info.get('kakenhi_info', 'N/A')
                keyword = author_info.get('keyword', 'N/A')

            summaries.append(summary)
            kakenhi_infos.append(kakenhi)
            keywords.append(keyword)

        return {
            "names": names,
            "institutions": institutions_list,
            "crs_scores": [r.crs_final for r in rankings],
            "h_indices": [r.h_index_global or 0 for r in rankings],
            "paper_counts": [r.n_in_corpus_works for r in rankings],
            "leadership_rates": [r.leadership_rate for r in rankings],
            "summaries": summaries,
            "kakenhi_infos": kakenhi_infos,
            "keywords": keywords
        }
    
    def _prepare_topic_data(self, works: List[WorkRaw], 
                          rankings: List[ResearcherRanking],
                          authors: Dict[str, AuthorMaster],
                          lang_strings: Dict[str, str]) -> Dict[str, Any]:
        """Prepare topic distribution data with both count and CRS weighting."""
        topic_counter = Counter()
        topic_crs_weights = Counter()
        
        # Create CRS lookup
        crs_lookup = {r.author_id: r.crs_final for r in rankings}
        
        for work in works:
            # Calculate CRS weight for this work (average of authors' CRS scores)
            work_crs_weight = 0.0
            author_count = 0
            
            for authorship in work.authorships:
                author_id = authorship.author_id
                if author_id in crs_lookup:
                    work_crs_weight += crs_lookup[author_id]
                    author_count += 1
            
            if author_count > 0:
                work_crs_weight /= author_count  # Average CRS of authors
            else:
                work_crs_weight = 0.0
            
            # Count topics (paper count version)
            if work.primary_topic:
                topic_counter[work.primary_topic.display_name] += 1
                topic_crs_weights[work.primary_topic.display_name] += work_crs_weight
                
            for topic in work.topics:
                topic_counter[topic.display_name] += 1
                topic_crs_weights[topic.display_name] += work_crs_weight
        
        # Get top 15 topics by count
        top_topics_by_count = topic_counter.most_common(15)
        top_topics_by_crs = topic_crs_weights.most_common(15)
        
        return {
            "labels": [topic for topic, count in top_topics_by_count],
            "counts": [count for topic, count in top_topics_by_count],
            "crs_labels": [topic for topic, weight in top_topics_by_crs],
            "crs_weights": [weight for topic, weight in top_topics_by_crs],
            "total_topics": len(topic_counter),
            "explanation_count": lang_strings['topics_count_explanation'],
            "explanation_crs": lang_strings['topics_crs_explanation']
        }
    
    def _prepare_timeline_data(self, works: List[WorkRaw], lang_strings: Dict[str, str]) -> Dict[str, Any]:
        """Prepare timeline visualization data weighted by citation impact."""
        year_counts = Counter()
        year_citations = Counter()
        year_total_impact = Counter()
        
        for work in works:
            if work.year:
                year_counts[work.year] += 1
                year_citations[work.year] += work.cited_by_count
                
                # Calculate impact score (combination of citations + recency)
                current_year = 2024  # Could be configurable
                years_since = max(1, current_year - work.year)
                recency_factor = max(0.1, 1 / years_since)  # Recent papers get higher weight
                impact_score = (work.cited_by_count + 1) * recency_factor  # +1 to avoid zero
                year_total_impact[work.year] += impact_score
        
        # Sort by year
        years = sorted(year_counts.keys())
        counts = [year_counts[year] for year in years]
        citations = [year_citations[year] for year in years]
        impacts = [year_total_impact[year] for year in years]
        
        return {
            "years": years,
            "counts": counts,
            "citations": citations,
            "impact_scores": impacts,
            "total_years": len(years),
            "year_range": [min(years), max(years)] if years else [2020, 2024],
            "explanation": lang_strings['timeline_explanation']
        }
    
    def _prepare_statistics_data(self, rankings: List[ResearcherRanking], 
                               works: List[WorkRaw], authors: Dict[str, AuthorMaster],
                               network_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare general statistics data."""
        h_indices = [r.h_index_global for r in rankings if r.h_index_global is not None]
        crs_scores = [r.crs_final for r in rankings]
        paper_counts = [r.n_in_corpus_works for r in rankings]
        
        return {
            "network": {
                "nodes": network_stats.get("num_nodes", 0),
                "edges": network_stats.get("num_edges", 0), 
                "density": network_stats.get("density", 0),
                "clustering": network_stats.get("avg_clustering", 0),
                "components": network_stats.get("num_components", 0),
                "largest_component": network_stats.get("largest_component_size", 0)
            },
            "rankings": {
                "total": len(rankings),
                "avg_crs": sum(crs_scores) / len(crs_scores) if crs_scores else 0,
                "max_crs": max(crs_scores) if crs_scores else 0,
                "avg_h_index": sum(h_indices) / len(h_indices) if h_indices else 0,
                "max_h_index": max(h_indices) if h_indices else 0,
                "avg_papers": sum(paper_counts) / len(paper_counts) if paper_counts else 0
            },
            "corpus": {
                "total_papers": len(works),
                "total_authors": len(authors),
                "year_span": max(w.year for w in works if w.year) - min(w.year for w in works if w.year) + 1 if any(w.year for w in works) else 0
            }
        }
    
    # _prepare_institution_data メソッド全体を削除
    
    def _generate_html_template(self, **data) -> str:
        """Generate the complete HTML report template."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lang_strings = data.get('lang_strings', self.i18n['en'])
        language = data.get('language', 'en')
        
        return f"""<!DOCTYPE html>
<html lang="{language}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{lang_strings['title']} - {data['field_name']}</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        {self._get_css_styles()}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{lang_strings['title']}</h1>
            <h2>{data['field_name']}</h2>
            <p class="timestamp">{lang_strings['generated_on']}: {timestamp}</p>
            <div class="summary">
                <div class="stat-card">
                    <div class="stat-value">{data['total_papers']}</div>
                    <div class="stat-label">{lang_strings['total_papers']}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{data['total_authors']}</div>
                    <div class="stat-label">{lang_strings['total_researchers']}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{data.get('filtered_researchers_count', data['total_rankings'])}</div>
                    <div class="stat-label">{lang_strings['UTokyo_researchers']}</div>
                </div>
            </div>
        </header>

        <section class="network-section">
            <h2>{lang_strings['collaboration_network']}</h2>
            <div class="chart-container">
                <div class="network-controls">
                    <div class="coloring-controls">
                        <label for="color-mode">{lang_strings['coloring_mode']}:</label>
                        <select id="color-mode" class="color-mode-select">
                            <option value="institution">{lang_strings['color_by_institution']}</option>
                            <option value="country">{lang_strings['color_by_country']}</option>
                            <option value="crs">{lang_strings['color_by_crs']}</option>
                        </select>
                    </div>
                    <div class="cluster-info">
                        <label>{lang_strings.get('cluster_info', 'Cluster Info')}: </label>
                        <span id="cluster-details"></span>
                    </div>
                    <div class="zoom-controls">
                        <button id="zoom-in" class="zoom-btn">{lang_strings['zoom_in']}</button>
                        <button id="zoom-out" class="zoom-btn">{lang_strings['zoom_out']}</button>
                        <button id="zoom-reset" class="zoom-btn">{lang_strings['reset_view']}</button>
                    </div>
                </div>
                <div id="network-chart"></div>
                <div class="chart-info">
                    <p><strong>{lang_strings['network_overview']}:</strong> {data['network_data']['total_nodes']} {lang_strings['researchers']}, {data['network_data']['total_edges']} {lang_strings['collaborations']}</p>
                    {f"<p><em>Note: Showing sample of {data['network_data']['total_edges']} edges for performance</em></p>" if data['network_data']['sampled'] else ""}
                    <p><strong>Density:</strong> {data['stats_data']['network']['density']:.4f} | 
                       <strong>Clustering:</strong> {data['stats_data']['network']['clustering']:.4f}</p>
                </div>
            </div>
        </section>

        <section class="rankings-section">
            <h2>{lang_strings['top_central_researchers']}</h2>
            <div class="chart-container">
                <canvas id="ranking-chart"></canvas>
            </div>
        </section>

        <section class="analysis-section">
            <div class="chart-container">
                <h3>{lang_strings['research_topics_count_title']} <span class="info-icon" title="{data['topic_data']['explanation_count']}">ℹ️</span></h3>
                <canvas id="topic-count-chart"></canvas>
            </div>
        </section>
        
        <section class="analysis-section">
            <div class="chart-container">
                <h3>{lang_strings['research_topics_crs_title']} <span class="info-icon" title="{data['topic_data']['explanation_crs']}">ℹ️</span></h3>
                <canvas id="topic-crs-chart"></canvas>
            </div>
        </section>
        
        <section class="analysis-section">
            <div class="chart-container">
                <h3>{lang_strings['timeline_title']} <span class="info-icon" title="{data['timeline_data']['explanation']}">ℹ️</span></h3>
                <canvas id="timeline-chart"></canvas>
            </div>
        </section>
        
        <section class="methodology">
            <h2>{lang_strings['methodology_title']}</h2>
            <div class="methodology-content">
                <p>{lang_strings['methodology_intro']}</p>
                
                <div class="component-grid">
                    <div class="component-item">
                        <h3>{lang_strings['centrality_title']}</h3>
                        <p>{lang_strings['centrality_desc']}</p>
                        <ul>
                            <li><strong>{lang_strings['pagerank']}</strong></li>
                            <li><strong>{lang_strings['degree_centrality']}</strong></li>
                            <li><strong>{lang_strings['betweenness']}</strong></li>
                        </ul>
                    </div>
                    
                    <div class="component-item">
                        <h3>{lang_strings['citation_title']}</h3>
                        <p>{lang_strings['citation_desc']}</p>
                        <ul>
                            <li><strong>{lang_strings['h_index_desc']}</strong></li>
                        </ul>
                    </div>
                    
                    <div class="component-item">
                        <h3>{lang_strings['leadership_title']}</h3>
                        <p>{lang_strings['leadership_desc']}</p>
                        <ul>
                            <li><strong>{lang_strings['leadership_rate_desc']}</strong></li>
                        </ul>
                    </div>
                </div>
                
                <p class="note"><strong>Note:</strong> {lang_strings['methodology_intro']}</p>
            </div>
        </section>

        <section class="detailed-rankings">
            <h2>{lang_strings['detailed_rankings']}</h2>
            <div class="search-container">
                <input type="text" id="researcher-search" placeholder="{lang_strings['search_placeholder']}" class="search-input">
                <div class="search-stats">
                    <span id="results-count"></span>
                </div>
            </div>
            <div class="ranking-table">
                {self._generate_ranking_table(data['ranking_data'], lang_strings)}
            </div>
        </section>

        <section class="network-stats">
            <h2>{lang_strings['network_statistics']}</h2>
            <div class="stats-grid">
                <div class="stat-item">
                    <span class="stat-label">Total Nodes:</span>
                    <span class="stat-value">{data['stats_data']['network']['nodes']}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Total Edges:</span>
                    <span class="stat-value">{data['stats_data']['network']['edges']}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Network Density:</span>
                    <span class="stat-value">{data['stats_data']['network']['density']:.4f}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Average Clustering:</span>
                    <span class="stat-value">{data['stats_data']['network']['clustering']:.4f}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Connected Components:</span>
                    <span class="stat-value">{data['stats_data']['network']['components']}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Largest Component Size:</span>
                    <span class="stat-value">{data['stats_data']['network']['largest_component']}</span>
                </div>
            </div>
        </section>
    </div>

    <script>
        // Data for visualizations
        const networkData = {json.dumps(data['network_data'])};
        const rankingData = {json.dumps(data['ranking_data'])};
        const topicData = {json.dumps(data['topic_data'])};
        const timelineData = {json.dumps(data['timeline_data'])};
        const langStrings = {json.dumps(lang_strings)};

        {self._get_javascript_code()}
    </script>
</body>
</html>"""
    
    def _generate_ranking_table(self, ranking_data: Dict[str, Any], lang_strings: Dict[str, str]) -> str:
        """Generate HTML table for detailed rankings."""
        rows = []
        for i, (name, crs, h_index, papers, leadership, summary, kakenhi, keyword) in enumerate(zip(
            ranking_data['names'],
            ranking_data['crs_scores'], 
            ranking_data['h_indices'],
            ranking_data['paper_counts'],
            ranking_data['leadership_rates'],
            ranking_data["summaries"],
            ranking_data['kakenhi_infos'],
            ranking_data['keywords'],
        )):
            rows.append(f"""
                <tr>
                    <td>{i+1}</td>
                    <td>{name}</td>
                    <td>{crs:.4f}</td>
                    <td>{h_index:.2f}</td>
                    <td>{papers}</td>
                    <td>{leadership:.2f}</td>
                    <td>{summary}</td>
                    <td>{kakenhi}</td>
                    <td>{keyword}</td>
                </tr>
            """)
        
        return f"""
            <table>
                <thead>
                    <tr>
                        <th>{lang_strings['rank']}</th>
                        <th>{lang_strings['researcher']}</th>
                        <th>{lang_strings['crs_score']} <span class="info-icon" title="{lang_strings['crs_tooltip']}">ℹ️</span></th>
                        <th>{lang_strings['h_index']} <span class="info-icon" title="{lang_strings['h_index_tooltip']}">ℹ️</span></th>
                        <th>{lang_strings['papers']} <span class="info-icon" title="{lang_strings['papers_tooltip']}">ℹ️</span></th>
                        <th>{lang_strings['leadership_rate']} <span class="info-icon" title="{lang_strings['leadership_tooltip']}">ℹ️</span></th>
                        <th>{lang_strings['researcher_summary']}</th>
                        <th>{lang_strings['kakenhi_info']}</th> 
                        <th>{lang_strings['keyword']}</th>   
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        """
    
    def _get_css_styles(self) -> str:
        """Get CSS styles for the report."""
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
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }
        
        h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        h2 {
            font-size: 1.8em;
            margin-bottom: 20px;
            color: #4a5568;
        }
        
        h3 {
            font-size: 1.3em;
            margin-bottom: 15px;
            color: #2d3748;
        }
        
        .timestamp {
            font-size: 0.9em;
            opacity: 0.8;
            margin-bottom: 20px;
        }
        
        .summary {
            display: flex;
            justify-content: center;
            gap: 30px;
            flex-wrap: wrap;
        }
        
        .stat-card {
            background: rgba(255,255,255,0.2);
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            min-width: 120px;
        }
        
        .stat-card .stat-value {
            font-size: 2em;
            font-weight: bold;
            display: block;
        }
        
        .stat-card .stat-label {
            font-size: 0.9em;
            opacity: 0.9;
        }
        
        section {
            background: white;
            margin-bottom: 30px;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .chart-container {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        
        .analysis-section .chart-container {
            background: transparent;
            box-shadow: none;
            margin-bottom: 0;
            height: 400px; /* 明示的に高さを制限 */
            position: relative;
        }
        
        .analysis-section canvas {
            max-height: 350px !important;
            width: 100% !important;
        }
        
        .analysis-section {
            background: white;
            margin-bottom: 30px;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .chart-info {
            margin-top: 15px;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 5px;
            font-size: 0.9em;
        }
        
        #network-chart {
            width: 100%;
            height: 600px;
            border: 1px solid #e2e8f0;
            border-radius: 5px;
        }
        
        canvas {
            max-width: 100%;
            height: 400px;
        }
        
        .ranking-table table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }
        
        .ranking-table th,
        .ranking-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
        }
        
        .ranking-table th {
            background: #f7fafc;
            font-weight: 600;
            color: #4a5568;
        }
        
        .ranking-table tr:hover {
            background: #f7fafc;
        }
        
        /* Methodology section styles */
        .methodology {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 25px;
            margin-bottom: 25px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .methodology-content p {
            margin-bottom: 15px;
            line-height: 1.7;
        }
        
        .component-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        
        .component-item {
            background: #f8f9fa;
            border-left: 4px solid #4299e1;
            padding: 20px;
            border-radius: 6px;
        }
        
        .component-item h3 {
            margin-bottom: 10px;
            color: #2d3748;
        }
        
        .component-item ul {
            margin: 10px 0 0 20px;
        }
        
        .component-item li {
            margin-bottom: 5px;
            line-height: 1.5;
        }
        
        .note {
            background: #edf2f7;
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid #38a169;
            margin-top: 20px;
            font-style: italic;
        }
        
        /* Search functionality styles */
        .search-container {
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 15px;
            flex-wrap: wrap;
        }
        
        .search-input {
            flex: 1;
            min-width: 250px;
            padding: 12px 16px;
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            font-size: 14px;
            background-color: #ffffff;
            transition: all 0.3s ease;
        }
        
        .search-input:focus {
            outline: none;
            border-color: #4299e1;
            box-shadow: 0 0 0 3px rgba(66, 153, 225, 0.1);
        }
        
        .search-stats {
            color: #718096;
            font-size: 14px;
            font-weight: 500;
        }
        
        .ranking-table tbody tr.hidden {
            display: none;
        }
        
        .ranking-table tbody tr.highlight {
            background-color: #fff3cd !important;
            border-left: 4px solid #ffc107;
        }
        
        /* Network controls styles */
        .network-controls {
            margin-bottom: 15px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 6px;
            border: 1px solid #e9ecef;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 15px;
        }
        
        .coloring-controls {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .coloring-controls label {
            font-weight: 500;
            color: #4a5568;
            font-size: 14px;
        }
        
        .color-mode-select {
            padding: 8px 12px;
            border: 2px solid #e2e8f0;
            border-radius: 4px;
            background: white;
            font-size: 14px;
            cursor: pointer;
            transition: border-color 0.2s ease;
        }
        
        .color-mode-select:focus {
            outline: none;
            border-color: #4299e1;
        }
        
        .zoom-controls {
            display: flex;
            gap: 5px;
        }
        
        .cluster-info {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .cluster-info label {
            font-weight: 500;
            color: #4a5568;
            font-size: 14px;
        }
        
        #cluster-details {
            font-size: 12px;
            color: #718096;
            font-style: italic;
        }
        
        .cluster-area {
            transition: all 0.3s ease;
        }
        
        .cluster-area:hover {
            filter: drop-shadow(2px 2px 4px rgba(0,0,0,0.2));
        }
        
        .zoom-btn {
            background: #4299e1;
            color: white;
            border: none;
            padding: 8px 16px;
            margin: 0 5px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.2s ease;
        }
        
        .zoom-btn:hover {
            background: #3182ce;
            transform: translateY(-1px);
        }
        
        .zoom-btn:active {
            transform: translateY(0);
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        
        .stat-item {
            display: flex;
            justify-content: space-between;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 5px;
        }
        
        .stat-item .stat-label {
            font-weight: 500;
        }
        
        .stat-item .stat-value {
            font-weight: bold;
            color: #667eea;
        }
        
        @media (max-width: 768px) {
            .summary {
                flex-direction: column;
                align-items: center;
            }
            
            .analysis-section {
                padding: 15px;
            }
            
            h1 {
                font-size: 2em;
            }
            
            .container {
                padding: 10px;
            }
        }
        
        /* Tooltip styles */
        .tooltip, .info-icon {
            cursor: help;
            color: #4299e1;
            font-size: 0.85em;
            margin-left: 5px;
            opacity: 0.7;
            transition: opacity 0.2s ease;
        }
        
        .info-icon:hover {
            opacity: 1;
        }
        
        .tooltip:hover::after, .info-icon:hover::after {
            content: attr(title);
            position: absolute;
            background: #2d3748;
            color: white;
            padding: 10px 14px;
            border-radius: 6px;
            font-size: 0.85em;
            white-space: normal;
            max-width: 320px;
            z-index: 1000;
            box-shadow: 0 4px 12px rgba(0,0,0,0.25);
            margin-top: -45px;
            margin-left: -160px;
            line-height: 1.4;
        }
        
        .tooltip:hover::before, .info-icon:hover::before {
            content: '';
            position: absolute;
            border: 6px solid transparent;
            border-top-color: #2d3748;
            margin-top: -29px;
            margin-left: -6px;
            z-index: 1001;
        }
        """
    
    def _get_javascript_code(self) -> str:
        """Get JavaScript code for interactive visualizations."""
        return """
        // Initialize all charts
        document.addEventListener('DOMContentLoaded', function() {
            createNetworkVisualization();
            createRankingChart();
            createTopicCountChart();
            createTopicCRSChart();
            createTimelineChart();
            // createCountryChart(); // 削除
            // createInstitutionChart(); // 削除
            initializeResearcherSearch();
        });

        function createNetworkVisualization() {
            const container = d3.select("#network-chart");
            const width = container.node().getBoundingClientRect().width;
            const height = 600;

            const svg = container.append("svg")
                .attr("width", width)
                .attr("height", height)
                .attr("style", "border: 1px solid #ddd; border-radius: 5px;");

            // Add zoom behavior
            const zoom = d3.zoom()
                .scaleExtent([0.1, 4])
                .on("zoom", (event) => {
                    g.attr("transform", event.transform);
                });

            svg.call(zoom);

            // Create group for zoomable content
            const g = svg.append("g");
            
            // Create groups for different visual elements
            const clustersGroup = g.append("g").attr("class", "clusters");
            const linksGroup = g.append("g").attr("class", "links");
            const nodesGroup = g.append("g").attr("class", "nodes");
            const labelsGroup = g.append("g").attr("class", "labels");

            // Optimize simulation for performance
            const simulation = d3.forceSimulation(networkData.nodes)
                .force("link", d3.forceLink(networkData.links).id(d => d.id)
                    .distance(d => d.group === 1 ? 100 : 80))
                .force("charge", d3.forceManyBody()
                    .strength(d => d.group === 1 ? -300 : -150))
                .force("center", d3.forceCenter(width / 2, height / 2))
                .force("collision", d3.forceCollide().radius(d => d.size + 5))
                .alpha(1)
                .alphaDecay(0.02);

            // Create color scales for different modes
            const institutionColors = d3.scaleOrdinal()
                .domain(networkData.institutions)
                .range(d3.schemeCategory10.concat(d3.schemeSet3));
                
            const countryColors = d3.scaleOrdinal()
                .domain(networkData.countries)
                .range(d3.schemeSet2.concat(d3.schemeSet1));
                
            const crsColorScale = d3.scaleSequential(d3.interpolateViridis)
                .domain([0, networkData.max_crs]);
            
            // Current coloring mode
            let currentColorMode = 'institution';
            
            // Function to get node color based on current mode
            const getNodeColor = (d) => {
                switch(currentColorMode) {
                    case 'institution':
                        return d.institution === "Unknown" ? "#cccccc" : institutionColors(d.institution);
                    case 'country':
                        return d.country === "Unknown" ? "#cccccc" : countryColors(d.country);
                    case 'crs':
                        return d.crs_score > 0 ? crsColorScale(d.crs_score) : "#cccccc";
                    default:
                        return "#cccccc";
                }
            };
            
            // Function to detect natural clusters using connectivity
            const detectClusters = () => {
                const clusters = [];
                const visited = new Set();
                
                // Create adjacency list from links
                const adjacencyList = new Map();
                networkData.nodes.forEach(node => {
                    adjacencyList.set(node.id, []);
                });
                
                networkData.links.forEach(link => {
                    adjacencyList.get(link.source.id || link.source).push(link.target.id || link.target);
                    adjacencyList.get(link.target.id || link.target).push(link.source.id || link.source);
                });
                
                // DFS to find connected components
                const dfs = (nodeId, cluster) => {
                    visited.add(nodeId);
                    const node = networkData.nodes.find(n => n.id === nodeId);
                    if (node) cluster.push(node);
                    
                    const neighbors = adjacencyList.get(nodeId) || [];
                    neighbors.forEach(neighborId => {
                        if (!visited.has(neighborId)) {
                            dfs(neighborId, cluster);
                        }
                    });
                };
                
                networkData.nodes.forEach(node => {
                    if (!visited.has(node.id)) {
                        const cluster = [];
                        dfs(node.id, cluster);
                        if (cluster.length >= 2) { // Only clusters with 2+ nodes
                            clusters.push(cluster);
                        }
                    }
                });
                
                return clusters;
            };

            // Add links with reduced opacity for performance
            const link = linksGroup
                .selectAll("line")
                .data(networkData.links)
                .enter().append("line")
                .attr("stroke", "#999")
                .attr("stroke-opacity", 0.4)
                .attr("stroke-width", d => Math.max(d.width * 0.8, 0.5))
                .attr("class", "network-link");

            // Add nodes
            const node = nodesGroup
                .selectAll("circle")
                .data(networkData.nodes)
                .enter().append("circle")
                .attr("r", d => d.size)
                .attr("fill", d => getNodeColor(d))
                .attr("fill-opacity", d => d.color_intensity)
                .attr("stroke", "#fff")
                .attr("stroke-width", d => d.group === 1 ? 2 : 1)
                .attr("class", "network-node")
                .call(d3.drag()
                    .on("start", dragstarted)
                    .on("drag", dragged)
                    .on("end", dragended));

            // Add labels only for top researchers to reduce clutter
            const labels = labelsGroup
                .selectAll("text")
                .data(networkData.nodes.filter(d => d.group === 1))
                .enter().append("text")
                .text(d => d.name.length > 15 ? d.name.substring(0, 15) + "..." : d.name)
                .attr("font-size", "11px")
                .attr("font-weight", "bold")
                .attr("fill", "#333")
                .attr("text-anchor", "middle")
                .attr("dy", -25)
                .attr("class", "network-label");

            // Add tooltip with better positioning
            const tooltip = d3.select("body").append("div")
                .attr("class", "tooltip")
                .style("position", "absolute")
                .style("padding", "10px")
                .style("background", "rgba(0, 0, 0, 0.8)")
                .style("color", "white")
                .style("border-radius", "5px")
                .style("font-size", "12px")
                .style("pointer-events", "none")
                .style("opacity", 0);

            node.on("mouseover", function(event, d) {
                tooltip.transition().duration(200).style("opacity", 1);
                tooltip.html(`<strong>${d.name}</strong><br/>Rank: ${d.rank !== null ? '#' + (d.rank + 1) : 'Not ranked'}<br/>Institution: ${d.institution}<br/>Country: ${d.country}<br/>CRS Score: ${d.crs_score.toFixed(4)}`)
                    .style("left", (event.pageX + 10) + "px")
                    .style("top", (event.pageY - 10) + "px");
            })
            .on("mouseout", function() {
                tooltip.transition().duration(200).style("opacity", 0);
            });

            // Function to update permanent cluster boundaries
            const updateClusterBoundaries = () => {
                const clusters = detectClusters();
                const clusterPaths = [];
                
                clusters.forEach((nodes, index) => {
                    // Calculate convex hull for cluster
                    const points = nodes.map(d => [d.x, d.y]);
                    const hull = d3.polygonHull(points);
                    
                    if (hull && hull.length >= 3) {
                        // Expand hull slightly for padding
                        const centroid = d3.polygonCentroid(hull);
                        const expandedHull = hull.map(point => {
                            const dx = point[0] - centroid[0];
                            const dy = point[1] - centroid[1];
                            return [centroid[0] + dx * 1.4, centroid[1] + dy * 1.4];
                        });
                        
                        // Calculate institution composition for this cluster
                        const institutionCounts = new Map();
                        nodes.forEach(node => {
                            const inst = node.institution || 'Unknown';
                            institutionCounts.set(inst, (institutionCounts.get(inst) || 0) + 1);
                        });
                        
                        const institutionComposition = Array.from(institutionCounts.entries())
                            .sort((a, b) => b[1] - a[1])
                            .map(([inst, count]) => `${inst} (${count})`)
                            .join(', ');
                        
                        clusterPaths.push({
                            key: `cluster-${index}`,
                            path: expandedHull,
                            nodes: nodes,
                            institutions: institutionComposition,
                            size: nodes.length
                        });
                    }
                });
                
                // Create color scale for clusters
                const clusterColorScale = d3.scaleOrdinal()
                    .domain(clusterPaths.map(d => d.key))
                    .range(d3.schemeCategory10.concat(d3.schemeSet3));
                
                // Update cluster areas with semi-transparent fills
                const clusterAreas = clustersGroup.selectAll('.cluster-area')
                    .data(clusterPaths, d => d.key);
                
                clusterAreas.enter()
                    .append('path')
                    .attr('class', 'cluster-area')
                    .attr('fill', d => clusterColorScale(d.key))
                    .attr('fill-opacity', 0.15)
                    .attr('stroke', d => clusterColorScale(d.key))
                    .attr('stroke-width', 2)
                    .attr('stroke-opacity', 0.4)
                    .attr('stroke-dasharray', '5,3')
                    .style('cursor', 'pointer')
                    .on('click', function(event, d) {
                        // Show cluster composition
                        const clusterNum = d.key.split('-')[1];
                        document.getElementById('cluster-details').textContent = 
                            `Cluster ${clusterNum}: ${d.size} ${langStrings['researchers']} - ${d.institutions}`;
                    })
                    .on('mouseover', function(event, d) {
                        d3.select(this)
                            .attr('fill-opacity', 0.25)
                            .attr('stroke-width', 3)
                            .attr('stroke-opacity', 0.7);
                    })
                    .on('mouseout', function(event, d) {
                        d3.select(this)
                            .attr('fill-opacity', 0.15)
                            .attr('stroke-width', 2)
                            .attr('stroke-opacity', 0.4);
                    });
                
                clusterAreas
                    .attr('d', d => d.path ? `M${d.path.join('L')}Z` : '')
                    .attr('fill', d => clusterColorScale(d.key))
                    .attr('stroke', d => clusterColorScale(d.key));
                
                clusterAreas.exit().remove();
                
                return clusterPaths.length;
            };
            
            // Function to update node colors
            const updateNodeColors = () => {
                node.attr('fill', d => getNodeColor(d));
            };
            
            // Function to update legend based on current mode
            const updateLegend = () => {
                d3.select('.legend').remove();
                createLegend();
            };

            // Optimize simulation tick for better performance
            let tickCounter = 0;
            simulation.on("tick", () => {
                tickCounter++;
                // Update positions less frequently for better performance
                if (tickCounter % 2 === 0) {
                    link
                        .attr("x1", d => d.source.x)
                        .attr("y1", d => d.source.y)
                        .attr("x2", d => d.target.x)
                        .attr("y2", d => d.target.y);

                    node
                        .attr("cx", d => d.x)
                        .attr("cy", d => d.y);

                    labels
                        .attr("x", d => d.x)
                        .attr("y", d => d.y);
                        
                    // Update cluster boundaries every few ticks for performance
                    if (tickCounter % 15 === 0) {
                        updateClusterBoundaries();
                    }
                }
            });

            // Color mode change handler
            document.getElementById('color-mode').addEventListener('change', function() {
                currentColorMode = this.value;
                updateNodeColors();
                updateLegend();
            });

            function dragstarted(event, d) {
                if (!event.active) simulation.alphaTarget(0.1).restart();
                d.fx = d.x;
                d.fy = d.y;
            }

            function dragged(event, d) {
                d.fx = event.x;
                d.fy = event.y;
            }

            function dragended(event, d) {
                if (!event.active) simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
            }
            
            // Create dynamic legend
            const createLegend = () => {
                const legend = svg.append("g")
                    .attr("class", "legend")
                    .attr("transform", "translate(20, 20)");
                
                let legendData = [];
                let legendTitle = '';
                
                switch(currentColorMode) {
                    case 'institution':
                        legendData = networkData.institutions.slice(0, 10);
                        legendTitle = 'Institutions';
                        break;
                    case 'country':
                        legendData = networkData.countries.slice(0, 10);
                        legendTitle = 'Countries';
                        break;
                    case 'crs':
                        // Create CRS scale legend
                        const crsSteps = 5;
                        const stepSize = networkData.max_crs / crsSteps;
                        legendData = Array.from({length: crsSteps}, (_, i) => {
                            const value = (i + 1) * stepSize;
                            return {
                                label: value.toFixed(3),
                                value: value
                            };
                        });
                        legendTitle = 'CRS Score';
                        break;
                }
                
                const legendItems = legend.selectAll(".legend-item")
                    .data(legendData)
                    .enter().append("g")
                    .attr("class", "legend-item")
                    .attr("transform", (d, i) => `translate(0, ${i * 20})`);
                
                legendItems.append("circle")
                    .attr("cx", 8)
                    .attr("cy", 8)
                    .attr("r", 6)
                    .attr("fill", d => {
                        switch(currentColorMode) {
                            case 'institution':
                                return institutionColors(d);
                            case 'country':
                                return countryColors(d);
                            case 'crs':
                                return crsColorScale(d.value);
                            default:
                                return '#cccccc';
                        }
                    })
                    .attr("stroke", "#fff")
                    .attr("stroke-width", 1);
                
                legendItems.append("text")
                    .attr("x", 20)
                    .attr("y", 8)
                    .attr("dy", "0.35em")
                    .style("font-size", "11px")
                    .style("font-family", "Arial, sans-serif")
                    .text(d => {
                        if (currentColorMode === 'crs') {
                            return d.label;
                        }
                        return d.length > 25 ? d.substring(0, 25) + "..." : d;
                    });
                
                // Add legend title
                legend.insert("text", ":first-child")
                    .attr("x", 0)
                    .attr("y", -5)
                    .style("font-size", "12px")
                    .style("font-weight", "bold")
                    .style("font-family", "Arial, sans-serif")
                    .text(legendTitle);
                
                // Add background for better visibility
                const bbox = legend.node().getBBox();
                legend.insert("rect", ":first-child")
                    .attr("x", bbox.x - 5)
                    .attr("y", bbox.y - 5)
                    .attr("width", bbox.width + 10)
                    .attr("height", bbox.height + 10)
                    .attr("fill", "rgba(255, 255, 255, 0.9)")
                    .attr("stroke", "#ccc")
                    .attr("stroke-width", 1)
                    .attr("rx", 3);
            };
            
            // Initialize legend
            createLegend();
            
            // Add zoom control button listeners
            document.getElementById('zoom-in').addEventListener('click', function() {
                svg.transition().duration(300).call(
                    zoom.scaleBy, 1.5
                );
            });
            
            document.getElementById('zoom-out').addEventListener('click', function() {
                svg.transition().duration(300).call(
                    zoom.scaleBy, 1 / 1.5
                );
            });
            
            document.getElementById('zoom-reset').addEventListener('click', function() {
                svg.transition().duration(750).call(
                    zoom.transform,
                    d3.zoomIdentity
                );
            });
            
            // Initialize cluster boundaries and set default message
            document.getElementById('cluster-details').textContent = langStrings['cluster_details'];
            
            setTimeout(() => {
                const clusterCount = updateClusterBoundaries();
                if (clusterCount > 0) {
                    document.getElementById('cluster-details').textContent = 
                        `${clusterCount} clusters detected. Click on colored areas to see composition.`;
                }
            }, 1500); // Give simulation time to settle
        }

        function createRankingChart() {
            const ctx = document.getElementById('ranking-chart').getContext('2d');
            new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: rankingData.names,
                    datasets: [{
                        label: 'CRS Score',
                        data: rankingData.crs_scores,
                        backgroundColor: 'rgba(102, 126, 234, 0.8)',
                        borderColor: 'rgba(102, 126, 234, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    aspectRatio: 2,
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'CRS Score'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Researchers'
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            display: false
                        },
                        title: {
                            display: true,
                            text: langStrings['crs_chart_label']
                        }
                    }
                }
            });
        }

        function createTopicCountChart() {
            const ctx = document.getElementById('topic-count-chart').getContext('2d');
            new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: topicData.labels,
                    datasets: [{
                        data: topicData.counts,
                        backgroundColor: [
                            '#667eea', '#764ba2', '#f093fb', '#f5576c',
                            '#4facfe', '#00f2fe', '#43e97b', '#38f9d7',
                            '#ffecd2', '#fcb69f', '#a8edea', '#fed6e3',
                            '#fdfcfb', '#e2d1c3', '#c1dfc4'
                        ]
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    aspectRatio: 1.5,
                    plugins: {
                        legend: {
                            position: 'right',
                            labels: {
                                boxWidth: 12,
                                font: {
                                    size: 10
                                }
                            }
                        }
                    }
                }
            });
        }

        function createTopicCRSChart() {
            const ctx = document.getElementById('topic-crs-chart').getContext('2d');
            new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: topicData.crs_labels,
                    datasets: [{
                        data: topicData.crs_weights,
                        backgroundColor: [
                            '#667eea', '#764ba2', '#f093fb', '#f5576c',
                            '#4facfe', '#00f2fe', '#43e97b', '#38f9d7',
                            '#ffecd2', '#fcb69f', '#a8edea', '#fed6e3',
                            '#fdfcfb', '#e2d1c3', '#c1dfc4'
                        ]
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    aspectRatio: 1.5,
                    plugins: {
                        legend: {
                            position: 'right',
                            labels: {
                                boxWidth: 12,
                                font: {
                                    size: 10
                                }
                            }
                        }
                    }
                }
            });
        }

        function createTimelineChart() {
            const ctx = document.getElementById('timeline-chart').getContext('2d');
            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: timelineData.years,
                    datasets: [{
                        label: 'Publication Impact Score',
                        data: timelineData.impact_scores,
                        borderColor: '#667eea',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        fill: true,
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    aspectRatio: 2,
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Impact Score (Citations × Recency)'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Year'
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            display: false
                        },
                        tooltip: {
                            callbacks: {
                                afterLabel: function(context) {
                                    const index = context.dataIndex;
                                    const publicationCount = timelineData.counts[index];
                                    const totalCitations = timelineData.citations[index];
                                    return [`Publications: ${publicationCount}`, `Total Citations: ${totalCitations}`];
                                }
                            }
                        }
                    }
                }
            });
        }

        // createCountryChart 関数を削除
        
        // createInstitutionChart 関数を削除
        
        // Researcher search functionality
        function initializeResearcherSearch() {
            const searchInput = document.getElementById('researcher-search');
            const resultsCount = document.getElementById('results-count');
            const tableBody = document.querySelector('.ranking-table tbody');
            const allRows = Array.from(tableBody.querySelectorAll('tr'));
            const totalRows = allRows.length;
            
            // Initialize results count
            updateResultsCount(totalRows, totalRows);
            
            searchInput.addEventListener('input', function() {
                const searchTerm = this.value.toLowerCase().trim();
                let visibleCount = 0;
                
                allRows.forEach(row => {
                    const researcherNameCell = row.cells[1]; // Second cell contains researcher name
                    const institutionCell = row.cells[2]; // Third cell contains institution name
                    
                    if (researcherNameCell && institutionCell) {
                        const researcherName = researcherNameCell.textContent.toLowerCase();
                        const institutionName = institutionCell.textContent.toLowerCase();
                        const isMatch = researcherName.includes(searchTerm) || institutionName.includes(searchTerm);
                        
                        if (isMatch || searchTerm === '') {
                            row.classList.remove('hidden');
                            if (searchTerm !== '' && isMatch) {
                                row.classList.add('highlight');
                            } else {
                                row.classList.remove('highlight');
                            }
                            visibleCount++;
                        } else {
                            row.classList.add('hidden');
                            row.classList.remove('highlight');
                        }
                    }
                });
                
                updateResultsCount(visibleCount, totalRows);
            });
            
            function updateResultsCount(visible, total) {
                if (visible === total) {
                    resultsCount.textContent = `${langStrings['showing_all']} ${total} ${langStrings['researchers']}`;
                } else {
                    resultsCount.textContent = `${langStrings['showing_of']} ${visible} ${langStrings['of']} ${total} ${langStrings['researchers']}`;
                }
            }
        }
        """