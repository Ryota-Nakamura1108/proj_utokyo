"""Command-line interface for keyword extractor."""

import argparse
import logging
import sys
from typing import Optional

from .core import extract_keywords
from .export import export_results
from .config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main(args: Optional[list] = None) -> int:
    """Main CLI entry point.

    Args:
        args: Command line arguments (for testing)

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    parser = argparse.ArgumentParser(
        description="Extract keywords for researchers based on their OpenAlex profile",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract keywords by researcher name
  uv run keyword_extractor "Geoffrey Hinton"

  # Extract keywords by OpenAlex ID
  uv run keyword_extractor A1234567890

  # Export to Excel
  uv run keyword_extractor "Yoshua Bengio" --export results/bengio_keywords.xlsx

  # Analyze last 5 years with minimum 10 citations
  uv run keyword_extractor "Andrew Ng" --years-back 5 --min-citations 10

  # Include cluster information in output
  uv run keyword_extractor A1234567890 --include-clusters --export results/detailed.json
        """
    )

    parser.add_argument(
        "identifier",
        type=str,
        help="Researcher name or OpenAlex ID (e.g., 'Geoffrey Hinton' or 'A1234567890')"
    )

    parser.add_argument(
        "--export",
        type=str,
        metavar="PATH",
        help="Export results to file (Excel or JSON). Format auto-detected from extension."
    )

    parser.add_argument(
        "--max-keywords",
        type=int,
        default=10,
        metavar="N",
        help="Maximum number of keywords to extract (default: 10)"
    )

    parser.add_argument(
        "--years-back",
        type=int,
        default=10,
        metavar="N",
        help="Number of years back to analyze (default: 10)"
    )

    parser.add_argument(
        "--min-citations",
        type=int,
        default=0,
        metavar="N",
        help="Minimum citation count filter (default: 0)"
    )

    parser.add_argument(
        "--include-clusters",
        action="store_true",
        help="Include cluster information in output"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    parsed_args = parser.parse_args(args)

    # Set logging level
    if parsed_args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        # Reduce logging noise in normal mode
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)

    # Validate configuration
    if not config.validate():
        logger.error("Configuration validation failed")
        return 1

    print("=" * 80)
    print("🔍 KEYWORD EXTRACTOR FOR RESEARCHERS")
    print("=" * 80)
    print(f"Identifier: {parsed_args.identifier}")
    print(f"Max Keywords: {parsed_args.max_keywords}")
    print(f"Years Back: {parsed_args.years_back}")
    print(f"Min Citations: {parsed_args.min_citations}")
    print(f"Include Clusters: {parsed_args.include_clusters}")
    if parsed_args.export:
        print(f"Export Path: {parsed_args.export}")
    print("-" * 80)

    try:
        # Extract keywords
        print("\n📊 Extracting keywords...")
        print("This may take a few minutes depending on the number of papers...\n")

        result = extract_keywords(
            identifier=parsed_args.identifier,
            max_keywords=parsed_args.max_keywords,
            years_back=parsed_args.years_back,
            min_citations=parsed_args.min_citations,
            include_clusters=parsed_args.include_clusters,
        )

        # Check for errors
        if result.get("status") == "error":
            print(f"\n❌ Error: {result.get('error', 'Unknown error')}")
            return 1

        # Display results
        print("\n✅ Extraction completed!")
        print(f"\nResearcher: {result.get('researcher_name', 'Unknown')}")
        print(f"OpenAlex ID: {result.get('researcher_id', 'Unknown')}")

        # Display statistics
        stats = result.get('statistics', {})
        if stats:
            print(f"\n📈 Statistics:")
            print(f"  Total Papers Analyzed: {stats.get('total_papers', 0)}")
            print(f"  First Author Papers: {stats.get('papers_as_first_author', 0)}")
            print(f"  Last Author Papers: {stats.get('papers_as_last_author', 0)}")
            print(f"  Total Citations: {stats.get('total_citations', 0)}")
            year_range = stats.get('year_range')
            if year_range:
                print(f"  Year Range: {year_range['start']} - {year_range['end']}")

        # Display keywords
        keywords = result.get('keywords', [])
        print(f"\n🏷️  Top {len(keywords)} Keywords:")
        print("-" * 80)
        for i, kw in enumerate(keywords, 1):
            if isinstance(kw, dict):
                keyword = kw.get('keyword', '')
                score = kw.get('relevance_score', 0.0)
                cluster_ids = kw.get('cluster_ids', [])
                cluster_info = f" (clusters: {', '.join(map(str, cluster_ids))})" if cluster_ids else ""
                print(f"  {i:2d}. {keyword:<40} (relevance: {score:.3f}){cluster_info}")
            else:
                print(f"  {i:2d}. {kw}")

        # Display researcher summary
        summary = result.get('summary')
        if summary:
            print(f"\n📝 Researcher Summary:")
            print("-" * 80)
            # Format summary with word wrapping
            import textwrap
            wrapped_summary = textwrap.fill(summary, width=78, initial_indent='  ', subsequent_indent='  ')
            print(wrapped_summary)

        # Display cluster information if requested
        if parsed_args.include_clusters and result.get('clusters'):
            clusters = result.get('clusters', [])
            print(f"\n📊 Cluster Information ({len(clusters)} clusters):")
            print("-" * 80)
            for cluster in clusters:
                print(f"\nCluster {cluster['cluster_id'] + 1}: {cluster.get('theme', 'N/A')}")
                print(f"  Size: {cluster['size']} papers")
                if cluster.get('representative_papers'):
                    print(f"  Representative Papers:")
                    for i, paper in enumerate(cluster['representative_papers'][:2], 1):
                        print(f"    {i}. {paper.get('title', 'N/A')[:70]}...")
                        print(f"       ({paper.get('year', 'N/A')}, {paper.get('citations', 0)} citations)")

        # Display processing time
        proc_time = result.get('processing_time')
        if proc_time:
            print(f"\n⏱️  Processing Time: {proc_time:.2f} seconds")

        # Export if requested
        if parsed_args.export:
            print(f"\n💾 Exporting results to {parsed_args.export}...")
            output_file = export_results(result, parsed_args.export)
            print(f"✅ Results exported to: {output_file}")

        print("\n" + "=" * 80)
        print("✨ Process completed successfully!")
        return 0

    except KeyboardInterrupt:
        print("\n\n⏸️  Process interrupted by user")
        return 130

    except Exception as e:
        logger.exception("Fatal error during extraction")
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
