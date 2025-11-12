"""Generic research analysis framework for Central Researcher system."""

import asyncio
import logging
import pandas as pd
import re
from pathlib import Path
from typing import List, Optional, Dict, Any

from ..core.models import CentralResearcherConfig, InputPaper
from ..core.central_researcher import CentralResearcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_papers_from_excel(file_path: str, 
                          title_column: str = "タイトル",
                          url_column: str = "URL",
                          authors_column: str = "著者") -> List[InputPaper]:
    """Load research papers from Excel file.
    
    Args:
        file_path: Path to Excel file
        title_column: Name of title column (default: "タイトル")
        url_column: Name of URL column (default: "URL")
        authors_column: Name of authors column (default: "著者")
    
    Returns:
        List of InputPaper objects with author information preserved
    """
    
    logger.info(f"📖 Loading papers from Excel file: {file_path}")
    
    # Read Excel file
    df = pd.read_excel(file_path)
    
    papers = []
    valid_dois = 0
    
    for idx, row in df.iterrows():
        title = row[title_column] if title_column in df.columns else ""
        url = row[url_column] if url_column in df.columns else ""
        authors_text = row[authors_column] if authors_column in df.columns else ""
        
        # Extract DOI from URL
        doi = None
        if pd.notna(url) and isinstance(url, str):
            if 'doi.org' in url:
                doi_match = re.search(r'doi\.org/(.+)$', url)
                if doi_match:
                    doi = doi_match.group(1)
                    valid_dois += 1
            elif url.startswith('10.'):
                doi = url
                valid_dois += 1
        
        # Create InputPaper - prefer DOI, fallback to title
        # Ensure we always have a valid identifier
        if doi:
            paper = InputPaper(doi=doi)
        elif title and pd.notna(title):
            paper = InputPaper(title=str(title))
        else:
            # Skip papers without any valid identifier
            logger.warning(f"Skipping paper at row {idx}: no valid identifier")
            continue
        
        papers.append(paper)
    
    logger.info(f"✅ Loaded {len(papers)} papers ({valid_dois} with DOIs, {len(papers)-valid_dois} title-only)")
    return papers


def load_papers_from_csv(file_path: str,
                        doi_column: str = "doi", 
                        title_column: str = "title",
                        openalex_column: Optional[str] = None,
                        pmid_column: Optional[str] = None) -> List[InputPaper]:
    """Load research papers from CSV file.
    
    Args:
        file_path: Path to CSV file
        doi_column: Name of DOI column
        title_column: Name of title column  
        openalex_column: Name of OpenAlex ID column (optional)
        pmid_column: Name of PMID column (optional)
    
    Returns:
        List of InputPaper objects
    """
    
    logger.info(f"📖 Loading papers from CSV file: {file_path}")
    
    df = pd.read_csv(file_path)
    papers = []
    
    for idx, row in df.iterrows():
        # Extract identifiers
        doi = row.get(doi_column) if pd.notna(row.get(doi_column)) else None
        title = row.get(title_column) if pd.notna(row.get(title_column)) else None
        openalex_id = row.get(openalex_column) if openalex_column and pd.notna(row.get(openalex_column)) else None
        pmid = row.get(pmid_column) if pmid_column and pd.notna(row.get(pmid_column)) else None
        
        # Create InputPaper with available identifiers
        paper = InputPaper(
            doi=doi,
            openalex_id=openalex_id,
            pmid=pmid,
            title=title
        )
        papers.append(paper)
    
    logger.info(f"✅ Loaded {len(papers)} papers from CSV")
    return papers


async def analyze_research_field(papers: List[InputPaper],
                               field_name: str = "Research Field",
                               config_overrides: Optional[Dict[str, Any]] = None,
                               email: str = "user@example.com",
                               output_dir: str = "src/memory_ai_dev/central_researcher/test/out") -> List['ResearcherRanking']:
    """Analyze central researchers in any research field.
    
    Args:
        papers: List of papers representing the field
        field_name: Name of the research field (for logging)
        config_overrides: Override default configuration parameters
        email: Email for OpenAlex API requests
        output_dir: Directory for output files
    
    Returns:
        List of ResearcherRanking objects sorted by CRS score
    """
    
    print(f"🧠 Central Researchers in {field_name}")
    print("=" * 70)
    
    # Create configuration with defaults and overrides
    config_params = {
        "papers": papers,
        "year_range": [2010, 2024],
        "decay_lambda": 0.12,
        "first_last_bonus": 0.30,
        "corr_author_bonus": 0.25,
        "min_in_corpus_works": 2,
        "tau_shrinkage": 4,
        "email": email
    }
    
    if config_overrides:
        config_params.update(config_overrides)
    
    config = CentralResearcherConfig(**config_params)
    
    logger.info(f"🚀 Starting Central Researcher Analysis for {field_name}")
    logger.info(f"📊 Input: {len(papers)} research papers")
    logger.info(f"⚙️  Configuration: years {config.year_range[0]}-{config.year_range[1]}, "
                f"min_works≥{config.min_in_corpus_works}")
    
    # Initialize and run analysis
    researcher = CentralResearcher(config)
    rankings = await researcher.analyze()
    
    logger.info(f"✅ Analysis completed! Identified {len(rankings)} central researchers")
    
    # Display results
    print(f"\n🎯 CENTRAL RESEARCHERS IN {field_name.upper()}")
    print("=" * 80)
    print(f"📈 Based on analysis of {len(papers)} representative papers")
    print(f"🔗 Network contains {len(rankings)} researchers")
    print()
    
    if len(rankings) == 0:
        print("⚠️  No researchers met the minimum criteria.")
        print(f"   Try reducing min_in_corpus_works (currently {config.min_in_corpus_works})")
        return rankings
    
    # Display top researchers
    print("🏆 TOP CENTRAL RESEARCHERS:")
    print("-" * 40)
    
    for i, ranking in enumerate(rankings[:15], 1):  # Show top 15
        # Format display
        name = ranking.author_display_name[:45]  # Truncate long names
        crs = ranking.crs_final
        h_global = ranking.h_index_global or 0
        h_local = ranking.h_index_local
        papers_count = ranking.n_in_corpus_works
        leadership = ranking.leadership_rate
        
        print(f"{i:2d}. {name:<45} CRS:{crs:6.4f} h:{h_global:2.0f}/{h_local:2.0f} "
              f"Papers:{papers_count:2d} Lead:{leadership:5.1%}")
    
    print()
    print("Legend: CRS=Central Researcher Score, h=h-index (global/local), Lead=Leadership Rate")
    
    # Export detailed results in multiple formats to specified directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    base_filename = field_name.lower().replace(' ', '_').replace('-', '_')
    
    # Export detailed CSV
    csv_file = output_path / f"{base_filename}_central_researchers_detailed.csv"
    researcher.export_results(str(csv_file), format='csv')
    logger.info(f"📊 Detailed CSV exported to: {csv_file}")
    
    # Export summary Excel
    excel_file = output_path / f"{base_filename}_central_researchers_summary.xlsx"
    researcher.export_results(str(excel_file), format='summary_excel')
    logger.info(f"📊 Summary Excel exported to: {excel_file}")
    
    # Export summary JSON
    json_file = output_path / f"{base_filename}_central_researchers_summary.json"
    researcher.export_results(str(json_file), format='summary_json')
    logger.info(f"📊 Summary JSON exported to: {json_file}")
    
    # Network analysis
    network_stats = researcher.get_network_statistics()
    print(f"\n📈 COLLABORATION NETWORK STATISTICS:")
    print("-" * 40)
    print(f"Researchers (nodes):     {network_stats.get('num_nodes', 0):4d}")
    print(f"Collaborations (edges):  {network_stats.get('num_edges', 0):4d}")
    print(f"Network density:         {network_stats.get('density', 0):7.4f}")
    print(f"Average clustering:      {network_stats.get('avg_clustering', 0):7.4f}")
    if 'is_connected' in network_stats:
        if network_stats['is_connected']:
            print(f"Network diameter:        {network_stats.get('diameter', 0):4d}")
        else:
            print(f"Connected components:    {network_stats.get('num_components', 0):4d}")
            print(f"Largest component:       {network_stats.get('largest_component_size', 0):4d}")
    
    # Spotlight on top researcher
    if rankings:
        top_researcher = rankings[0]
        details = researcher.get_researcher_details(top_researcher.author_id)
        
        print(f"\n🌟 SPOTLIGHT: TOP RESEARCHER")
        print("-" * 40)
        print(f"Name:                    {top_researcher.author_display_name}")
        print(f"CRS Score:               {top_researcher.crs_final:.4f}")
        print(f"Global h-index:          {top_researcher.h_index_global or 'N/A'}")
        print(f"Local h-index:           {top_researcher.h_index_local}")
        print(f"Leadership rate:         {top_researcher.leadership_rate:.1%}")
        print(f"Papers in corpus:        {top_researcher.n_in_corpus_works}")
        print(f"Centrality (PageRank):   {top_researcher.centrality_pagerank:.4f}")
        print(f"Centrality (Betweenness):{top_researcher.centrality_betweenness:.4f}")
        
        if details and details['author_data']:
            author_data = details['author_data']
            print(f"Total career works:      {author_data.works_count}")
            if author_data.orcid:
                print(f"ORCID:                   {author_data.orcid}")
        
        print(f"OpenAlex ID:             {top_researcher.author_id}")
    
    print(f"\n💡 FIELD INSIGHTS:")
    print("-" * 40)
    print(f"• {field_name} research shows {len(rankings)} active central researchers")
    print(f"• Network analysis reveals collaboration patterns in this field")
    print(f"• Top researchers demonstrate both high impact and leadership")
    print(f"• Results can guide collaboration, funding, and research direction decisions")
    
    return rankings


# ================================================================
# GENERIC ANALYSIS TEMPLATES
# ================================================================

def create_biomedical_config(papers: List[InputPaper], 
                           email: str = "user@example.com",
                           min_works: int = 2) -> CentralResearcherConfig:
    """Create standard configuration for biomedical research analysis.
    
    Optimized for biomedical research with moderate collaboration networks.
    """
    return CentralResearcherConfig(
        papers=papers,
        year_range=[2015, 2024],
        decay_lambda=0.12,  # ~6 year half-life for biomedical fields
        first_last_bonus=0.30,  # Higher bonus for authorship position in biomed
        corr_author_bonus=0.25,
        min_in_corpus_works=min_works,
        tau_shrinkage=4,  # Moderate shrinkage for established field
        email=email
    )


def create_cs_config(papers: List[InputPaper], 
                    email: str = "user@example.com",
                    min_works: int = 3) -> CentralResearcherConfig:
    """Create standard configuration for computer science research analysis.
    
    Optimized for CS with faster publication cycles and larger author lists.
    """
    return CentralResearcherConfig(
        papers=papers,
        year_range=[2018, 2024],  # Shorter window for fast-moving field
        decay_lambda=0.20,  # ~3.5 year half-life for fast-moving CS
        first_last_bonus=0.20,  # Lower bonus as CS has less strict authorship hierarchy
        corr_author_bonus=0.15,
        min_in_corpus_works=min_works,
        tau_shrinkage=5,
        email=email
    )


def create_physics_config(papers: List[InputPaper], 
                         email: str = "user@example.com",
                         min_works: int = 2) -> CentralResearcherConfig:
    """Create standard configuration for physics research analysis.
    
    Optimized for physics with very large collaboration networks.
    """
    return CentralResearcherConfig(
        papers=papers,
        year_range=[2010, 2024],  # Longer window for established field
        decay_lambda=0.08,  # ~8.5 year half-life for slower-moving physics
        first_last_bonus=0.15,  # Lower bonus due to large author lists
        corr_author_bonus=0.20,
        min_in_corpus_works=min_works,
        tau_shrinkage=6,  # Higher shrinkage for large networks
        email=email
    )


def create_custom_config(papers: List[InputPaper],
                        field_characteristics: Dict[str, Any],
                        email: str = "user@example.com") -> CentralResearcherConfig:
    """Create custom configuration based on field characteristics.
    
    Args:
        papers: List of input papers
        field_characteristics: Dictionary with field-specific parameters:
            - publication_speed: 'fast' | 'medium' | 'slow'
            - collaboration_size: 'small' | 'medium' | 'large' 
            - authorship_hierarchy: 'strict' | 'moderate' | 'loose'
            - field_maturity: 'emerging' | 'established' | 'mature'
        email: Contact email for API requests
    """
    # Set year range based on publication speed
    speed_map = {
        'fast': [2020, 2024],      # 4 years for fast fields (CS, AI)
        'medium': [2015, 2024],    # 9 years for medium fields (biomedical)
        'slow': [2010, 2024]       # 14 years for slow fields (physics)
    }
    year_range = speed_map.get(field_characteristics.get('publication_speed', 'medium'), [2015, 2024])
    
    # Set decay lambda based on publication speed  
    decay_map = {
        'fast': 0.25,    # ~2.8 year half-life
        'medium': 0.12,  # ~5.8 year half-life  
        'slow': 0.08     # ~8.7 year half-life
    }
    decay_lambda = decay_map.get(field_characteristics.get('publication_speed', 'medium'), 0.12)
    
    # Set bonuses based on authorship hierarchy
    hierarchy = field_characteristics.get('authorship_hierarchy', 'moderate')
    if hierarchy == 'strict':
        first_last_bonus = 0.35
        corr_author_bonus = 0.30
    elif hierarchy == 'loose':
        first_last_bonus = 0.15
        corr_author_bonus = 0.15
    else:  # moderate
        first_last_bonus = 0.25
        corr_author_bonus = 0.25
    
    # Set minimum works based on collaboration size
    collab_map = {
        'small': 2,    # Smaller networks, lower threshold
        'medium': 3,   # Medium networks  
        'large': 4     # Large networks, higher threshold
    }
    min_works = collab_map.get(field_characteristics.get('collaboration_size', 'medium'), 3)
    
    # Set shrinkage based on field maturity
    maturity_map = {
        'emerging': 3,    # Lower shrinkage for new fields
        'established': 5, # Standard shrinkage
        'mature': 6       # Higher shrinkage for mature fields
    }
    tau_shrinkage = maturity_map.get(field_characteristics.get('field_maturity', 'established'), 5)
    
    return CentralResearcherConfig(
        papers=papers,
        year_range=year_range,
        decay_lambda=decay_lambda,
        first_last_bonus=first_last_bonus,
        corr_author_bonus=corr_author_bonus,
        min_in_corpus_works=min_works,
        tau_shrinkage=tau_shrinkage,
        email=email
    )


async def analyze_any_research_field(papers: List[InputPaper],
                                   field_name: str = "Research Field",
                                   config: Optional[CentralResearcherConfig] = None,
                                   config_template: str = "biomedical",
                                   email: str = "user@example.com") -> List['ResearcherRanking']:
    """Analyze central researchers in any research field with template support.
    
    Args:
        papers: List of papers representing the field
        field_name: Name of the research field (for display)
        config: Custom configuration (overrides template)
        config_template: Template to use ('biomedical', 'cs', 'physics', 'custom')
        email: Email for OpenAlex API requests
    
    Returns:
        List of ResearcherRanking objects sorted by CRS score
    """
    
    if config is None:
        # Use template configurations
        if config_template == "biomedical":
            config = create_biomedical_config(papers, email)
        elif config_template == "cs":
            config = create_cs_config(papers, email)
        elif config_template == "physics":
            config = create_physics_config(papers, email)
        else:
            # Default to biomedical template
            config = create_biomedical_config(papers, email)
            logger.warning(f"Unknown template '{config_template}', using biomedical template")
    
    return await analyze_research_field(papers, field_name, None, email)


# Legacy functions for backward compatibility
def load_organoids_papers_from_excel(file_path: str) -> List[InputPaper]:
    """Legacy function name for backward compatibility."""
    return load_papers_from_excel(file_path, "タイトル", "URL")


async def analyze_organoids_researchers():
    """Legacy function for organoids-specific analysis."""
    data_path = "../test/data/human_cerebral_organoids.xlsx"
    
    papers = load_papers_from_excel(data_path, "タイトル", "URL")
    
    config_overrides = {
        "decay_lambda": 0.12,
        "first_last_bonus": 0.30,
        "min_in_corpus_works": 2,
        "tau_shrinkage": 4
    }
    
    return await analyze_research_field(
        papers=papers,
        field_name="Human Cerebral Organoids Research",
        config_overrides=config_overrides,
        email="ysato@memorylab.jp"
    )


async def quick_research_test(papers: List[InputPaper], field_name: str = "Research Field"):
    """Quick test with subset of papers."""
    
    print(f"🧪 Quick Test: {field_name} Analysis")
    print("=" * 50)
    
    # Use first 10 papers for faster testing
    test_papers = papers[:10] if len(papers) > 10 else papers
    
    config_overrides = {
        "min_in_corpus_works": 1,  # Lower threshold for test
        "tau_shrinkage": 2,
    }
    
    logger.info(f"🔬 Quick test with {len(test_papers)} papers")
    
    rankings = await analyze_research_field(
        papers=test_papers,
        field_name=field_name,
        config_overrides=config_overrides,
        email="test@memorylab.jp"
    )
    
    print(f"\n✅ Quick test completed: {len(rankings)} researchers identified")
    for i, ranking in enumerate(rankings[:5], 1):
        print(f"{i}. {ranking.author_display_name} (CRS: {ranking.crs_final:.4f})")
    
    return rankings


async def main():
    """Main execution function with generic interface."""
    
    print("🧠 Central Researcher Analysis - Generic Research Field System")
    print("=" * 75)
    print("Analyze any research field using paper lists")
    print()
    
    # Example: Load and analyze any research field
    # Users can modify this section for their specific field
    
    # Default: Use organoids data for demonstration
    data_file = Path("../test/data/human_cerebral_organoids.xlsx")
    if data_file.exists():
        papers = load_papers_from_excel(str(data_file))
        rankings = await analyze_research_field(
            papers=papers,
            field_name="Human Cerebral Organoids Research",
            email="ysato@memorylab.jp"
        )
        
        if rankings:
            print(f"\n🎉 Successfully identified {len(rankings)} central researchers!")
        else:
            print("⚠️ No researchers met the criteria - consider adjusting parameters")
    else:
        print("📄 No test data found. Please provide your own research paper list.")
        print("   Use load_papers_from_excel() or load_papers_from_csv() to load data")


if __name__ == "__main__":
    asyncio.run(main())