#!/usr/bin/env python3
"""Command-line interface for the Central Researcher identification system."""

import argparse
import asyncio
import logging
from pathlib import Path
from typing import List, Optional

from .analysis.research_analysis import load_papers_from_excel
from .core.central_researcher import CentralResearcher
from .core.models import CentralResearcherConfig, InputPaper, ResearcherRanking

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def construct_data_from_excel(
    excel_path: str, 
    min_works: int = 1,
    email: str = "ysato@memorylab.jp"
) -> CentralResearcherConfig:
    """Construct analysis configuration from Excel file.
    
    Args:
        excel_path: Path to Excel file containing paper data
        min_works: Minimum number of in-corpus works for inclusion
        email: Contact email for OpenAlex API
    
    Returns:
        CentralResearcherConfig object ready for analysis
    """
    excel_file = Path(excel_path)
    if not excel_file.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")
    
    logger.info(f"🔄 Loading papers from Excel file: {excel_file}")
    papers = load_papers_from_excel(str(excel_file))
    logger.info(f"✅ Loaded {len(papers)} papers")
    
    # Create analysis configuration
    config = CentralResearcherConfig(
        papers=papers,
        min_in_corpus_works=min_works,
        topic_similarity_threshold=0.3,
        email=email,
    )
    
    return config


async def analyze_central_researchers(
    config: CentralResearcherConfig,
    field_name: str,
    output_dir: Optional[str] = None,
) -> List[ResearcherRanking]:
    """Analyze central researchers from configuration data.
    
    Args:
        config: CentralResearcherConfig object with analysis parameters
        field_name: Name of the research field
        output_dir: Output directory for results
    
    Returns:
        List of researcher rankings
    """
    # Run analysis
    logger.info("🚀 Starting central researcher analysis...")
    researcher = CentralResearcher(config)
    rankings = await researcher.analyze()
    
    logger.info(f"✅ Analysis completed! Identified {len(rankings)} central researchers")
    
    # Set up output directory
    if output_dir is None:
        output_dir = "src/memory_ai_dev/central_researcher/test/out"
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Generate safe filename
    safe_field_name = field_name.replace(" ", "_").replace("/", "_").lower()
    
    # Generate English HTML report
    html_file = researcher.generate_html_report(
        output_path=str(output_path / f"{safe_field_name}_report.html"),
        field_name=field_name,
    )
    logger.info(f"📊 English HTML report generated: {html_file}")
    
    # Generate Japanese HTML report
    html_file_jp = researcher.generate_html_report(
        output_path=str(output_path / f"{safe_field_name}_report_jp.html"),
        field_name=field_name,
        language="ja",
    )
    logger.info(f"📊 Japanese HTML report generated: {html_file_jp}")
    
    # Export CSV results
    csv_file = output_path / f"{safe_field_name}_results.csv"
    researcher.export_results(str(csv_file))
    logger.info(f"📋 CSV results exported: {csv_file}")
    
    # Display summary
    print(f"\n🏆 TOP 10 CENTRAL RESEARCHERS IN {field_name.upper()}")
    print("=" * 80)
    
    for ranking in rankings[:10]:
        h_index = int(ranking.h_index_global) if ranking.h_index_global else 0
        institution_info = ""
        
        print(f"{ranking.rank:2d}. {ranking.author_display_name:<40} "
              f"CRS: {ranking.crs_final:.4f} | h-index: {h_index:2d} | "
              f"Papers: {ranking.n_in_corpus_works:2d}")
    
    return rankings


async def analyze_from_excel(
    excel_path: str, 
    field_name: str, 
    output_dir: Optional[str] = None,
    min_works: int = 1,
    email: str = "ysato@memorylab.jp"
) -> List[ResearcherRanking]:
    """Analyze central researchers from Excel file (legacy wrapper).
    
    Args:
        excel_path: Path to Excel file containing paper data
        field_name: Name of the research field
        output_dir: Output directory for results
        min_works: Minimum number of in-corpus works for inclusion
        email: Contact email for OpenAlex API
    
    Returns:
        List of researcher rankings
    """
    config = construct_data_from_excel(excel_path, min_works, email)
    return await analyze_central_researchers(config, field_name, output_dir)


def create_ai_ml_papers() -> List[InputPaper]:
    """Create a predefined set of AI/ML papers for demo purposes."""
    return [
        # Deep Learning foundational papers
        InputPaper(doi="10.1038/nature14539"),  # LeCun, Bengio, Hinton - Deep Learning
        InputPaper(doi="10.1162/neco.1997.9.8.1735"),  # Hochreiter & Schmidhuber - LSTM
        
        # Transformer architecture papers
        InputPaper(openalex_id="W2963073457"),  # Attention is All You Need
        InputPaper(title="BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"),
        
        # Computer vision breakthroughs
        InputPaper(doi="10.1145/3065386"),  # AlexNet
        InputPaper(doi="10.1109/CVPR.2016.90"),  # ResNet
        
        # Reinforcement learning milestones
        InputPaper(doi="10.1038/nature16961"),  # AlphaGo Nature paper
        InputPaper(doi="10.1126/science.aar6308"),  # AlphaGo Zero
        
        # Generative models
        InputPaper(doi="10.1145/3422622"),  # GANs
        InputPaper(title="Auto-Encoding Variational Bayes"),  # VAE
        
        # Recent large language models
        InputPaper(title="Language Models are Few-Shot Learners"),  # GPT-3
        InputPaper(title="Training language models to follow instructions with human feedback"),  # ChatGPT
    ]


async def demo_analysis(output_dir: Optional[str] = None) -> None:
    """Run a demo analysis with AI/ML papers."""
    papers = create_ai_ml_papers()
    
    config = CentralResearcherConfig(
        papers=papers,
        year_range=[2000, 2024],
        decay_lambda=0.15,
        first_last_bonus=0.25,
        corr_author_bonus=0.25,
        min_in_corpus_works=2,
        tau_shrinkage=5,
        email="ysato@memorylab.jp",
    )
    
    logger.info("🚀 Starting demo AI/ML analysis...")
    researcher = CentralResearcher(config)
    rankings = await researcher.analyze()
    
    logger.info(f"✅ Analysis completed! Identified {len(rankings)} central researchers")
    
    # Set up output directory
    if output_dir is None:
        output_dir = "src/memory_ai_dev/central_researcher/test/out"
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Generate reports
    html_file = researcher.generate_html_report(
        output_path=str(output_path / "ai_ml_demo_report.html"),
        field_name="AI/ML Research (Demo)",
    )
    logger.info(f"📊 English HTML report: {html_file}")
    
    html_file_jp = researcher.generate_html_report(
        output_path=str(output_path / "ai_ml_demo_report_jp.html"),
        field_name="AI/ML Research (Demo)",
        language="ja",
    )
    logger.info(f"📊 Japanese HTML report: {html_file_jp}")
    
    csv_file = output_path / "ai_ml_demo_results.csv"
    researcher.export_results(str(csv_file))
    logger.info(f"📋 CSV results: {csv_file}")
    
    # Display results
    print(f"\n🏆 TOP 10 CENTRAL RESEARCHERS IN AI/ML (DEMO)")
    print("=" * 80)
    
    for ranking in rankings[:10]:
        h_index = int(ranking.h_index_global) if ranking.h_index_global else 0
        print(f"{ranking.rank:2d}. {ranking.author_display_name:<40} "
              f"CRS: {ranking.crs_final:.4f} | h-index: {h_index:2d}")


async def main() -> None:
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Central Researcher Identification System - Identify key researchers in any field",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Run AI/ML demo analysis
  python -m memory_ai_dev.central_researcher.cli --demo
  
  # Analyze from Excel file
  python -m memory_ai_dev.central_researcher.cli --excel data.xlsx --field "Neuroscience"
  
  # Custom output directory and minimum works threshold
  python -m memory_ai_dev.central_researcher.cli --excel data.xlsx --field "AI Research" --output results/ --min-works 2
  
  # Specify contact email for API
  python -m memory_ai_dev.central_researcher.cli --excel data.xlsx --field "Medicine" --email yourname@institution.edu
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--demo", 
        action="store_true", 
        help="Run demo analysis with predefined AI/ML papers"
    )
    group.add_argument(
        "--excel", 
        type=str, 
        help="Path to Excel file with paper data (required columns: DOI, Title, or OpenAlex_ID)"
    )
    
    parser.add_argument(
        "--field", 
        type=str, 
        help="Research field name (required with --excel)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="src/memory_ai_dev/central_researcher/test/out",
        help="Output directory (default: src/memory_ai_dev/central_researcher/test/out)"
    )
    parser.add_argument(
        "--min-works", 
        type=int, 
        default=1,
        help="Minimum number of in-corpus works for inclusion (default: 1)"
    )
    parser.add_argument(
        "--email", 
        type=str, 
        default="ysato@memorylab.jp",
        help="Contact email for OpenAlex API (default: ysato@memorylab.jp)"
    )
    
    args = parser.parse_args()
    
    print("🧠 Central Researcher Identification System")
    print("=" * 60)
    
    try:
        if args.demo:
            print("\n🎯 Running AI/ML Demo Analysis")
            print("-" * 40)
            await demo_analysis(args.output)
            
        elif args.excel:
            if not args.field:
                parser.error("--field is required when using --excel")
            
            print(f"\n📊 Analyzing Data From: {args.excel}")
            print(f"🔬 Field: {args.field}")
            print(f"📁 Output: {args.output}")
            print(f"📧 Email: {args.email}")
            print(f"🔢 Min Works: {args.min_works}")
            print("-" * 60)
            
            await analyze_from_excel(
                excel_path=args.excel,
                field_name=args.field,
                output_dir=args.output,
                min_works=args.min_works,
                email=args.email
            )
        
        print(f"\n✨ Analysis completed successfully!")
        print(f"📁 Results saved to: {args.output}")
        print(f"🌐 Open the HTML reports in your browser for interactive exploration")
        
    except KeyboardInterrupt:
        print("\n⏸️ Analysis interrupted by user")
    except Exception as e:
        logger.exception(f"💥 Analysis failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())