"""Export functionality for keyword extraction results.

Provides functions to export results to various formats including Excel.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import json
import logging

logger = logging.getLogger(__name__)


def export_to_excel(
    result: Dict[str, Any],
    output_path: str,
) -> str:
    """Export keyword extraction results to Excel file.

    Args:
        result: Dictionary containing extraction results
        output_path: Path to output Excel file

    Returns:
        Path to created Excel file
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for Excel export. Install with: uv add pandas openpyxl")

    logger.info(f"Exporting results to Excel: {output_path}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create DataFrame from keywords
    keywords_data = []
    for i, kw in enumerate(result.get("keywords", []), 1):
        if isinstance(kw, dict):
            keywords_data.append({
                "Rank": i,
                "Keyword": kw.get("keyword", ""),
                "Relevance Score": kw.get("relevance_score", 0.0),
            })
        else:
            keywords_data.append({
                "Rank": i,
                "Keyword": str(kw),
                "Relevance Score": 0.0,
            })

    # Create metadata DataFrame
    statistics = result.get("statistics", {})
    analysis_period = result.get("analysis_period", {})

    metadata = {
        "Field": [
            "Researcher Name",
            "Researcher ID",
            "Analysis Period",
            "Total Papers",
            "First Author Papers",
            "Last Author Papers",
            "Total Citations",
            "Average Citations",
            "Extraction Method",
            "Status",
        ],
        "Value": [
            result.get("researcher_name", ""),
            result.get("researcher_id", ""),
            f"{analysis_period.get('start_year', 'N/A')} - {analysis_period.get('end_year', 'N/A')}",
            statistics.get("total_papers", 0),
            statistics.get("papers_as_first_author", 0),
            statistics.get("papers_as_last_author", 0),
            statistics.get("total_citations", 0),
            f"{statistics.get('average_citations', 0):.1f}",
            result.get("method", ""),
            result.get("status", ""),
        ],
    }

    df_keywords = pd.DataFrame(keywords_data)
    df_metadata = pd.DataFrame(metadata)

    # Create summary DataFrame if summary exists
    summary_text = result.get("summary")
    if summary_text:
        df_summary = pd.DataFrame({
            "Researcher Summary": [summary_text]
        })

    # Write to Excel with multiple sheets
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_keywords.to_excel(writer, sheet_name='Keywords', index=False)
        df_metadata.to_excel(writer, sheet_name='Metadata', index=False)
        if summary_text:
            df_summary.to_excel(writer, sheet_name='Summary', index=False)

    logger.info(f"✅ Successfully exported to {output_path}")
    return str(output_path)


def export_to_json(
    result: Dict[str, Any],
    output_path: str,
) -> str:
    """Export keyword extraction results to JSON file.

    Args:
        result: Dictionary containing extraction results
        output_path: Path to output JSON file

    Returns:
        Path to created JSON file
    """
    logger.info(f"Exporting results to JSON: {output_path}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info(f"✅ Successfully exported to {output_path}")
    return str(output_path)


def export_results(
    result: Dict[str, Any],
    output_path: str,
    format: Optional[str] = None,
) -> str:
    """Export results to file. Format is auto-detected from file extension.

    Args:
        result: Dictionary containing extraction results
        output_path: Path to output file
        format: Force specific format ('excel', 'json'). If None, auto-detect.

    Returns:
        Path to created file
    """
    output_path = Path(output_path)

    # Auto-detect format from extension
    if format is None:
        ext = output_path.suffix.lower()
        if ext in ['.xlsx', '.xls']:
            format = 'excel'
        elif ext == '.json':
            format = 'json'
        else:
            # Default to JSON
            format = 'json'
            if not output_path.suffix:
                output_path = output_path.with_suffix('.json')

    if format == 'excel':
        return export_to_excel(result, str(output_path))
    elif format == 'json':
        return export_to_json(result, str(output_path))
    else:
        raise ValueError(f"Unsupported export format: {format}")
