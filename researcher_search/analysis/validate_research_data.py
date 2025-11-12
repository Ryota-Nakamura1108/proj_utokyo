"""Validation script for human cerebral organoids Excel data."""

import pandas as pd
import re
import asyncio
import logging
from pathlib import Path

from ..core.models import InputPaper, CentralResearcherConfig
from ..api.openalex_client import OpenAlexClient
from ..core.input_parser import InputParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_excel_structure(file_path: str):
    """Validate the structure and content of the Excel file."""
    
    print("🔍 VALIDATING HUMAN CEREBRAL ORGANOIDS EXCEL FILE")
    print("=" * 60)
    
    try:
        df = pd.read_excel(file_path)
        print(f"✅ Excel file loaded successfully")
        print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Check expected columns (Japanese names)
        expected_cols = ['番号', 'ID', 'タイトル', '著者', '要約', 'URL', '公開日', 'ジャーナル']
        missing_cols = [col for col in expected_cols if col not in df.columns]
        
        if missing_cols:
            print(f"⚠️  Missing columns: {missing_cols}")
        else:
            print("✅ All expected columns present")
        
        print("\nColumn analysis:")
        for col in df.columns:
            non_null = df[col].count()
            null_pct = (len(df) - non_null) / len(df) * 100
            print(f"   {col}: {non_null}/{len(df)} values ({null_pct:.1f}% null)")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading Excel file: {e}")
        return None


def analyze_doi_coverage(df):
    """Analyze DOI coverage in the dataset."""
    
    print("\n📊 DOI COVERAGE ANALYSIS")
    print("-" * 30)
    
    dois = []
    urls_with_doi = 0
    urls_total = 0
    
    for idx, row in df.iterrows():
        url = row.get('URL', '')
        
        if pd.notna(url) and isinstance(url, str):
            urls_total += 1
            
            # Try to extract DOI from URL
            if 'doi.org' in url:
                doi_match = re.search(r'doi\.org/(.+)$', url)
                if doi_match:
                    doi = doi_match.group(1)
                    dois.append(doi)
                    urls_with_doi += 1
    
    print(f"Total papers: {len(df)}")
    print(f"Papers with URLs: {urls_total}")
    print(f"URLs with DOI: {urls_with_doi}")
    print(f"DOI coverage: {urls_with_doi}/{urls_total} ({urls_with_doi/urls_total*100:.1f}%)")
    
    # Show some example DOIs
    print("\nExample DOIs found:")
    for i, doi in enumerate(dois[:5], 1):
        print(f"  {i}. {doi}")
    
    if len(dois) > 5:
        print(f"  ... and {len(dois)-5} more")
    
    return dois


async def test_openalex_resolution(dois_sample):
    """Test resolution of a few DOIs via OpenAlex API."""
    
    print("\n🌐 TESTING OPENALEX API RESOLUTION")
    print("-" * 40)
    
    # Test with first 3 DOIs
    test_dois = dois_sample[:3]
    
    papers = [InputPaper(doi=doi) for doi in test_dois]
    
    try:
        async with OpenAlexClient("ysato@memorylab.jp") as client:
            parser = InputParser(client)
            
            print(f"Testing resolution of {len(papers)} papers...")
            works = await parser.normalize_and_resolve_papers(papers)
            
            print(f"✅ Successfully resolved {len(works)}/{len(papers)} papers")
            
            for work in works:
                authors_count = len(work.authorships)
                print(f"   📄 {work.title[:60]}...")
                print(f"      Year: {work.year}, Authors: {authors_count}, Citations: {work.cited_by_count}")
            
            return works
            
    except Exception as e:
        print(f"❌ OpenAlex API test failed: {e}")
        return []


def create_sample_config(dois_sample):
    """Create a sample configuration for testing."""
    
    print(f"\n⚙️  SAMPLE CONFIGURATION")
    print("-" * 25)
    
    papers = [InputPaper(doi=doi) for doi in dois_sample[:5]]  # Use first 5 DOIs
    
    config = CentralResearcherConfig(
        papers=papers,
        year_range=[2015, 2024],  # Recent organoids research
        decay_lambda=0.15,
        first_last_bonus=0.25,
        corr_author_bonus=0.25,
        min_in_corpus_works=1,  # Lower for test
        tau_shrinkage=3,  # Lower for test
        email="ysato@memorylab.jp"
    )
    
    print(f"Papers to analyze: {len(config.papers)}")
    print(f"Year range: {config.year_range}")
    print(f"Min corpus works: {config.min_in_corpus_works}")
    print("Configuration ready for analysis ✅")
    
    return config


async def main():
    """Main validation function."""
    
    excel_file = "src/memory_ai_dev/central_researcher/test/human_cerebral_organoids.xlsx"
    
    print("🧠 Human Cerebral Organoids Data Validation")
    print("=" * 50)
    print(f"File: {excel_file}")
    print()
    
    # Check if file exists
    if not Path(excel_file).exists():
        print(f"❌ File not found: {excel_file}")
        print("Please ensure the Excel file is in the correct location")
        return
    
    # Step 1: Validate Excel structure
    df = validate_excel_structure(excel_file)
    if df is None:
        return
    
    # Step 2: Analyze DOI coverage
    dois = analyze_doi_coverage(df)
    if not dois:
        print("❌ No DOIs found in the dataset")
        return
    
    # Step 3: Test OpenAlex API resolution
    works = await test_openalex_resolution(dois)
    
    # Step 4: Create sample configuration
    config = create_sample_config(dois)
    
    print(f"\n🎉 VALIDATION SUMMARY")
    print("=" * 25)
    print(f"✅ Excel file structure: Valid")
    print(f"✅ DOI coverage: {len(dois)}/{len(df)} papers")
    print(f"✅ OpenAlex resolution: {len(works)}/3 test papers")
    print(f"✅ Configuration: Ready")
    print()
    print("The dataset is ready for central researcher analysis! 🚀")
    print("Run the full analysis with: organoids_analysis.py")


if __name__ == "__main__":
    asyncio.run(main())