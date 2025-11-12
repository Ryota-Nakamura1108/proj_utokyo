"""Command Line Interface for the Central Researcher system."""

import asyncio
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import click
import pandas as pd

from ..core.central_researcher import CentralResearcher
from ..core.models import CentralResearcherConfig, InputPaper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def cli(verbose: bool):
    """Central Researcher identification system."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@cli.command()
@click.option('--input', '-i', required=True, type=click.Path(exists=True),
              help='Input file containing paper identifiers (JSON, CSV, or Excel)')
@click.option('--output', '-o', required=True, type=click.Path(),
              help='Output file for results')
@click.option('--format', '-f', default='csv', 
              type=click.Choice(['csv', 'parquet', 'excel', 'json', 'summary_excel', 'summary_json']),
              help='Output format (summary formats include only researcher name, CRS score, h-index)')
@click.option('--email', '-e', default='ysato@memorylab.jp',
              help='Email for OpenAlex API requests')
@click.option('--year-start', type=int, help='Start year for filtering papers')
@click.option('--year-end', type=int, help='End year for filtering papers')
@click.option('--decay-lambda', default=0.15, type=float,
              help='Time decay parameter for edge weights')
@click.option('--first-last-bonus', default=0.25, type=float,
              help='Bonus weight for first/last authors')
@click.option('--corr-author-bonus', default=0.25, type=float,
              help='Bonus weight for corresponding authors')
@click.option('--min-works', default=2, type=int,
              help='Minimum in-corpus works for ranking inclusion')
@click.option('--tau-shrinkage', default=5, type=int,
              help='Sample size shrinkage parameter')
@click.option('--output-dir', '-d', type=click.Path(),
              help='Output directory for automatic multi-format export')
def analyze(input: str, output: str, format: str, email: str,
           year_start: int, year_end: int, decay_lambda: float,
           first_last_bonus: float, corr_author_bonus: float,
           min_works: int, tau_shrinkage: int, output_dir: str):
    """Analyze central researchers from input papers."""
    
    try:
        # Load input papers
        papers = load_input_papers(input)
        click.echo(f"Loaded {len(papers)} papers from {input}")
        
        # Create configuration
        year_range = None
        if year_start and year_end:
            year_range = [year_start, year_end]
        
        config = CentralResearcherConfig(
            papers=papers,
            year_range=year_range,
            decay_lambda=decay_lambda,
            first_last_bonus=first_last_bonus,
            corr_author_bonus=corr_author_bonus,
            min_in_corpus_works=min_works,
            tau_shrinkage=tau_shrinkage,
            email=email
        )
        
        # Run analysis
        click.echo("Starting central researcher analysis...")
        researcher = CentralResearcher(config)
        
        rankings = asyncio.run(researcher.analyze())
        
        # Export results
        researcher.export_results(output, format)
        
        # Display summary
        click.echo(f"\nAnalysis complete! Results saved to {output}")
        click.echo(f"Identified {len(rankings)} central researchers")
        
        if rankings:
            click.echo("\nTop 10 researchers:")
            for i, ranking in enumerate(rankings[:10], 1):
                click.echo(f"{i:2d}. {ranking.author_display_name} "
                          f"(CRS: {ranking.crs_final:.4f})")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.option('--input', '-i', required=True, type=click.Path(exists=True),
              help='Paper identifiers file')
@click.option('--email', '-e', default='ysato@memorylab.jp',
              help='Email for OpenAlex API requests')
def validate(input: str, email: str):
    """Validate input paper identifiers."""
    
    try:
        papers = load_input_papers(input)
        click.echo(f"Loaded {len(papers)} papers")
        
        # Basic validation
        valid_papers = []
        invalid_papers = []
        
        for i, paper in enumerate(papers):
            if not any([paper.doi, paper.openalex_id, paper.pmid, paper.title]):
                invalid_papers.append((i, "No identifier provided"))
            elif paper.doi and not paper.doi.startswith('10.'):
                invalid_papers.append((i, f"Invalid DOI format: {paper.doi}"))
            elif paper.openalex_id and not paper.openalex_id.startswith('W'):
                invalid_papers.append((i, f"Invalid OpenAlex ID: {paper.openalex_id}"))
            else:
                valid_papers.append(paper)
        
        click.echo(f"Valid papers: {len(valid_papers)}")
        click.echo(f"Invalid papers: {len(invalid_papers)}")
        
        if invalid_papers:
            click.echo("\nInvalid papers:")
            for idx, error in invalid_papers[:10]:  # Show first 10
                click.echo(f"  Row {idx}: {error}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.argument('template_file', type=click.Path())
@click.option('--format', '-f', default='json', type=click.Choice(['json', 'csv', 'excel']),
              help='Template format')
def create_template(template_file: str, format: str):
    """Create a template input file."""
    
    sample_papers = [
        {
            "doi": "10.1038/nature12373",
            "title": "Sample Nature Paper",
            "comments": "Example with DOI"
        },
        {
            "openalex_id": "W2741809807", 
            "title": "Sample OpenAlex Paper",
            "comments": "Example with OpenAlex ID"
        },
        {
            "pmid": "28935993",
            "title": "Sample PubMed Paper", 
            "comments": "Example with PMID"
        },
        {
            "title": "Sample Title-Only Paper",
            "comments": "Example with title only (less reliable)"
        }
    ]
    
    try:
        if format == 'json':
            with open(template_file, 'w') as f:
                json.dump(sample_papers, f, indent=2)
        elif format == 'csv':
            df = pd.DataFrame(sample_papers)
            df.to_csv(template_file, index=False)
        elif format == 'excel':
            df = pd.DataFrame(sample_papers)
            df.to_excel(template_file, index=False)
        
        click.echo(f"Template created at {template_file}")
        click.echo(f"Edit this file with your paper identifiers, then run 'analyze' command")
        
    except Exception as e:
        click.echo(f"Error creating template: {e}", err=True)
        raise click.Abort()


def load_input_papers(file_path: str) -> List[InputPaper]:
    """Load input papers from various file formats."""
    file_path = Path(file_path)
    
    if file_path.suffix.lower() == '.json':
        return load_papers_from_json(file_path)
    elif file_path.suffix.lower() == '.csv':
        return load_papers_from_csv(file_path)
    elif file_path.suffix.lower() in ['.xlsx', '.xls']:
        from ..analysis.research_analysis import load_papers_from_excel
        return load_papers_from_excel(str(file_path))
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def load_papers_from_json(file_path: Path) -> List[InputPaper]:
    """Load papers from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    papers = []
    for item in data:
        paper = InputPaper(
            doi=item.get('doi'),
            openalex_id=item.get('openalex_id'),
            pmid=item.get('pmid'),
            title=item.get('title')
        )
        papers.append(paper)
    
    return papers


def load_papers_from_csv(file_path: Path) -> List[InputPaper]:
    """Load papers from CSV file."""
    df = pd.read_csv(file_path)
    return dataframe_to_papers(df)


def load_papers_from_excel(file_path: Path) -> List[InputPaper]:
    """Load papers from Excel file."""
    df = pd.read_excel(file_path)
    return dataframe_to_papers(df)


def dataframe_to_papers(df: pd.DataFrame) -> List[InputPaper]:
    """Convert DataFrame to InputPaper objects."""
    papers = []
    
    for _, row in df.iterrows():
        paper = InputPaper(
            doi=row.get('doi') if pd.notna(row.get('doi')) else None,
            openalex_id=row.get('openalex_id') if pd.notna(row.get('openalex_id')) else None,
            pmid=row.get('pmid') if pd.notna(row.get('pmid')) else None,
            title=row.get('title') if pd.notna(row.get('title')) else None
        )
        papers.append(paper)
    
    return papers


if __name__ == '__main__':
    cli()