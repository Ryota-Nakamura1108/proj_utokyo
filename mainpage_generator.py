"""Custom HTML Report Generator with researcher page links."""

from typing import Dict, Any, List
from pathlib import Path
import sys

# Import from researcher_search
sys.path.insert(0, str(Path(__file__).parent / "researcher_search"))
from researcher_search.reporting.html_report_generator import HTMLReportGenerator
from researcher_search.core.models import ResearcherRanking


class MainHTMLReportGenerator(HTMLReportGenerator):
    """
    Custom HTML Report Generator that modifies the ranking table to include
    researcher page links instead of inline data.
    """

    def __init__(self, rankings_list: List[ResearcherRanking]):
        """
        Initialize the custom generator.

        Args:
            rankings_list: List of ResearcherRanking objects (used for getting author IDs)
        """
        super().__init__()
        self.rankings_list = rankings_list

    def _generate_ranking_table(self, ranking_data: Dict[str, Any], lang_strings: Dict[str, str]) -> str:
        """
        Generate HTML table for detailed rankings with researcher page links.

        This method overrides the parent method to replace the Summary, KAKENHI, and Keywords
        columns with a single "Researcher Page" column containing links to individual pages.

        Args:
            ranking_data: Dictionary containing ranking data
            lang_strings: Language-specific strings

        Returns:
            HTML table string
        """
        rows = []
        for i, (name, crs, h_index, papers, leadership) in enumerate(zip(
            ranking_data['names'],
            ranking_data['crs_scores'],
            ranking_data['h_indices'],
            ranking_data['paper_counts'],
            ranking_data['leadership_rates'],
        )):
            author_id = self.rankings_list[i].author_id if i < len(self.rankings_list) else "unknown"
            researcher_page_link = f'researcher_pages/{author_id}.html'

            rows.append(f"""
                <tr>
                    <td>{i+1}</td>
                    <td>{name}</td>
                    <td>{crs:.4f}</td>
                    <td>{h_index:.2f}</td>
                    <td>{papers}</td>
                    <td>{leadership:.2f}</td>
                    <td class="researcher-page-cell">
                        <a href="{researcher_page_link}" class="researcher-link" target="_blank">
                            <span class="link-icon">👤</span> View Profile
                        </a>
                    </td>
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
                        <th>Researcher Page</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        """

    def _get_css_styles(self) -> str:
        """
        Get CSS styles for the report with additional styles for researcher links.

        Returns:
            CSS string
        """
        original_css = super()._get_css_styles()

        custom_css = """
        /* Custom styles for researcher page links */
        .researcher-page-cell {
            text-align: center;
        }

        .researcher-link {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            padding: 8px 16px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            text-decoration: none;
            border-radius: 5px;
            font-size: 0.9em;
            font-weight: 500;
            transition: all 0.3s ease;
        }

        .researcher-link:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }

        .link-icon {
            font-size: 1.1em;
        }
        """

        return original_css + custom_css
