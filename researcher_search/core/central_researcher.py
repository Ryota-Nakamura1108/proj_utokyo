"""Main orchestrator for the Central Researcher identification system."""

import logging
import asyncio
import re
from typing import List, Dict, Optional, Tuple, Any
import pandas as pd
from pathlib import Path
import json

from .models import (
    CentralResearcherConfig, InputPaper, ResearcherRanking, 
    WorkRaw, AuthorMaster, AuthorFeatures, InstitutionMaster
)
from .normalization import NormalizationMethod
from ..api.openalex_client import OpenAlexClient
from .input_parser import InputParser
from .network_builder import NetworkBuilder
from .centrality_analyzer import CentralityAnalyzer
from .citation_analyzer import CitationAnalyzer
from .scoring import CentralResearcherScorer
from .explainability import ExplainabilityAnalyzer
from ..reporting.html_report_generator import HTMLReportGenerator

logger = logging.getLogger(__name__)


class CentralResearcher:
    """Main orchestrator for central researcher identification."""
    
    def __init__(self, config: CentralResearcherConfig):
        """Initialize with configuration."""
        self.config = config
        
        # Parse normalization method from string
        normalization_method = NormalizationMethod.ROBUST_Z  # Default
        if hasattr(config, 'normalization_method'):
            method_map = {
                'robust_z': NormalizationMethod.ROBUST_Z,
                'rankit': NormalizationMethod.RANKIT,
                'standard': NormalizationMethod.STANDARD
            }
            normalization_method = method_map.get(config.normalization_method, NormalizationMethod.ROBUST_Z)
        
        # Initialize components
        self.client = OpenAlexClient(config.email)
        self.parser = InputParser(self.client)
        self.network_builder = NetworkBuilder(
            decay_lambda=config.decay_lambda,
            first_last_bonus=config.first_last_bonus,
            corr_author_bonus=config.corr_author_bonus,
            topic_similarity_enabled=config.topic_similarity_enabled,
            topic_similarity_threshold=config.topic_similarity_threshold,
            consortium_suppression_enabled=config.consortium_suppression_enabled,
            consortium_threshold=config.consortium_threshold,
            alphabetical_detection_enabled=config.alphabetical_detection_enabled,
            alphabetical_threshold=config.alphabetical_threshold,
            # simmelian_strengthening_enabled=config.simmelian_strengthening_enabled
        )
        self.centrality_analyzer = CentralityAnalyzer(normalization_method=normalization_method)
        self.citation_analyzer = CitationAnalyzer(
            exclude_self_citations=config.exclude_self_citations,
            normalization_method=normalization_method
        )
        self.scorer = CentralResearcherScorer(
            tau_shrinkage=config.tau_shrinkage,
            min_in_corpus_works=config.min_in_corpus_works
        )
        self.explainability = ExplainabilityAnalyzer()
        self.html_generator = HTMLReportGenerator()
        
        # Data storage
        self.works: List[WorkRaw] = []
        self.authors: Dict[str, AuthorMaster] = {}
        self.institutions: Dict[str, 'InstitutionMaster'] = {}
        self.network = None
        self.edge_data = {}
        self.author_features = {}
        self.centralities = {}
        self.citations = {}
        self.leadership = {}
        self.rankings: List[ResearcherRanking] = []
        
    async def analyze(self) -> List[ResearcherRanking]:
        """Run complete central researcher analysis."""
        logger.info("Starting central researcher analysis")
        
        async with self.client:
            try:
                # Step 1: Parse and resolve input papers
                await self._resolve_input_papers()
                
                # Step 2: Get author data
                await self._fetch_author_data()
                
                # Step 2.5: Get institution data
                await self._fetch_institution_data()
                
                # Step 3: Build coauthorship network
                await self._build_network()
                
                # Step 4: Calculate centrality metrics
                await self._calculate_centralities()
                
                # Step 5: Calculate citation metrics
                await self._calculate_citations()
                
                # Step 6: Calculate composite scores and rankings
                await self._calculate_scores()
                
                logger.info(f"Analysis complete. Generated {len(self.rankings)} researcher rankings")
                return self.rankings
                
            except Exception as e:
                logger.error(f"Analysis failed: {e}")
                raise
    
    async def _resolve_input_papers(self):
        """Resolve input paper identifiers to WorkRaw objects."""
        logger.info(f"Resolving {len(self.config.papers)} input papers")
        
        self.works = await self.parser.normalize_and_resolve_papers(self.config.papers)
        
        # Apply year filtering if specified
        if self.config.year_range:
            start_year, end_year = self.config.year_range
            filtered_works = []
            for work in self.works:
                if work.year and start_year <= work.year <= end_year:
                    filtered_works.append(work)
            
            logger.info(f"Filtered {len(self.works)} works to {len(filtered_works)} "
                       f"within year range {start_year}-{end_year}")
            self.works = filtered_works
        
        logger.info(f"Successfully resolved {len(self.works)} papers")
    
    async def _fetch_author_data(self):
        """Fetch author master data from OpenAlex."""
        # Generate missing author IDs
        temp_ids = self.network_builder.generate_author_missing_ids(self.works)
        logger.info(f"Generated {len(temp_ids)} temporary author IDs")
        
        # Collect all author IDs
        author_ids = set()
        for work in self.works:
            for authorship in work.authorships:
                if authorship.author_id:
                    author_ids.add(authorship.author_id)
        
        # Filter out temporary IDs for API calls (they don't exist in OpenAlex)
        real_author_ids = [aid for aid in author_ids if not aid.startswith("TEMP_")]
        
        logger.info(f"Fetching data for {len(real_author_ids)} authors from OpenAlex")
        
        self.authors = await self.client.get_authors_by_ids(real_author_ids)
        
        logger.info(f"Successfully fetched data for {len(self.authors)} authors")
    
    async def _fetch_institution_data(self):
        """Fetch institution master data from OpenAlex."""
        # Collect all institution IDs from authors
        institution_ids = set()
        for author_data in self.authors.values():
            if author_data.last_known_institution_ids:
                institution_ids.update(author_data.last_known_institution_ids)
        
        if not institution_ids:
            logger.info("No institution IDs found")
            return
        
        logger.info(f"Fetching data for {len(institution_ids)} institutions from OpenAlex")
        
        self.institutions = await self.client.get_institutions_by_ids(list(institution_ids))
        
        logger.info(f"Successfully fetched data for {len(self.institutions)} institutions")
    
    async def _build_network(self):
        """Build weighted coauthorship network."""
        logger.info("Building coauthorship network")

        self.network, edge_data, local_citations = self.network_builder.build_coauthorship_network(self.works)
        
        # Store network data for later use and explainability
        self.edge_data = edge_data
        self.local_citations = local_citations
        
        # Log network statistics
        stats = self.network_builder.get_network_statistics(self.network)
        logger.info(f"Network built: {stats}")
    
    async def _calculate_centralities(self):
        """Calculate all centrality metrics."""
        logger.info("Calculating centrality metrics")
        
        self.centralities = self.centrality_analyzer.calculate_all_centralities(self.network)
        
        # Log centrality statistics
        stats = self.centrality_analyzer.calculate_centrality_statistics(self.centralities)
        logger.debug(f"Centrality statistics: {stats}")
    
    async def _calculate_citations(self):
        """Calculate citation and leadership metrics."""
        logger.info("Calculating citation metrics")
        
        # Calculate citation metrics
        self.citations = self.citation_analyzer.calculate_all_citation_metrics(
            self.works, self.authors, self.local_citations
        )
        
        # Normalize h-indices globally
        self.citations = self.scorer.normalize_h_indices_globally(self.citations)
        
        # Calculate leadership metrics
        self.leadership = self.citation_analyzer.calculate_leadership_metrics(self.works)
        
        logger.info(f"Calculated metrics for {len(self.citations)} authors")
    
    async def _calculate_scores(self):
        """Calculate composite CRS scores and create rankings."""
        logger.info("Calculating composite scores and rankings")
        
        # Extract k-core values from centralities
        kcore_values = self.centralities.get('kcore', {})
        
        # Calculate composite scores
        author_features = self.scorer.calculate_composite_scores(
            self.centralities, self.citations, self.leadership, kcore_values
        )
        
        # Store author features for explainability
        self.author_features = author_features
        
        # Create final rankings
        self.rankings = self.scorer.create_rankings(author_features, self.authors)
        
        # Log score statistics
        stats = self.scorer.get_score_statistics(author_features)
        logger.debug(f"Score statistics: {stats}")
    
    def export_results(self, output_path: str, format: str = 'csv') -> None:
        """Export results to file.
        
        Args:
            output_path: Path to output file
            format: Export format ('csv', 'parquet', 'excel', 'json', 'summary_excel', 'summary_json', 'explanations', 'html')
                   - summary formats include only researcher name, CRS score, and h-index
                   - explanations format provides detailed CRS breakdowns for top researchers
                   - html format generates comprehensive interactive report
        """
        if not self.rankings:
            raise ValueError("No rankings available. Run analyze() first.")
        
        # Handle special formats
        if format.lower() == 'explanations':
            self._export_explanations(output_path)
            return
        elif format.lower() == 'html':
            self._export_html_report(output_path)
            return
        
        # Choose DataFrame format based on export type
        if format.lower() in ['summary_excel', 'summary_json']:
            df = self._create_summary_dataframe()
        else:
            df = self._rankings_to_dataframe()
        
        output_path = Path(output_path)
        
        if format.lower() == 'csv':
            df.to_csv(output_path, index=False)
        elif format.lower() == 'parquet':
            df.to_parquet(output_path, index=False)
        elif format.lower() in ['excel', 'summary_excel']:
            df.to_excel(output_path, index=False)
        elif format.lower() in ['json', 'summary_json']:
            df.to_json(output_path, orient='records', indent=2, force_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Results exported to {output_path} (format: {format})")
    
    def _rankings_to_dataframe(self) -> pd.DataFrame:
        """Convert rankings to pandas DataFrame."""
        data = []
        for ranking in self.rankings:
            data.append({
                'rank': ranking.rank,
                'author_display_name': ranking.author_display_name,
                'author_id': ranking.author_id,
                'orcid': ranking.orcid,
                'crs_final': ranking.crs_final,
                'crs_raw': ranking.crs_raw,
                'centrality_degree': ranking.centrality_deg,
                'centrality_pagerank': ranking.centrality_pagerank,
                'centrality_betweenness': ranking.centrality_betweenness,
                'h_index_global': ranking.h_index_global,
                'h_index_local': ranking.h_index_local,
                'leadership_rate': ranking.leadership_rate,
                'n_in_corpus_works': ranking.n_in_corpus_works
            })
        
        return pd.DataFrame(data)
    
    def _create_summary_dataframe(self) -> pd.DataFrame:
        """Create summary DataFrame with researcher name, CRS score, and h-index only.
        
        Returns:
            DataFrame with columns: rank, researcher_name, crs_score, h_index_global, h_index_local
        """
        data = []
        for ranking in self.rankings:
            data.append({
                'rank': ranking.rank,
                'researcher_name': ranking.author_display_name,
                'crs_score': round(ranking.crs_final, 4),
                'h_index_global': ranking.h_index_global or 0,
                'h_index_local': ranking.h_index_local or 0,
                'papers_in_corpus': ranking.n_in_corpus_works
            })
        
        return pd.DataFrame(data)
    
    def _export_explanations(self, output_path: str) -> None:
        """Export detailed explanations for top researchers to JSON file.
        
        Args:
            output_path: Path to output JSON file
        """
        explanations = self.get_top_explanations(top_n=10)
        
        output_path = Path(output_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(explanations, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Explanations exported to {output_path}")
    
    def get_top_researchers(self, n: int = 10) -> List[ResearcherRanking]:
        """Get top N researchers by CRS score."""
        return self.rankings[:n]
    
    def filter_by_country(self, country_codes: List[str]) -> List[ResearcherRanking]:
        """Filter rankings by country codes.
        
        Args:
            country_codes: List of country codes to filter by (e.g., ['US', 'JP', 'KR'])
            
        Returns:
            Filtered list of ResearcherRanking objects
        """
        if not self.rankings or not self.institutions:
            return []
        
        # Normalize country codes to uppercase
        normalized_countries = [code.upper() for code in country_codes]
        
        filtered_rankings = []
        for ranking in self.rankings:
            author_data = self.authors.get(ranking.author_id)
            if not author_data or not author_data.last_known_institution_ids:
                continue
            
            # Check if author has institutions in target countries
            for inst_id in author_data.last_known_institution_ids:
                institution = self.institutions.get(inst_id)
                if institution and institution.country_code and institution.country_code.upper() in normalized_countries:
                    filtered_rankings.append(ranking)
                    break
        
        return filtered_rankings
    
    def filter_by_institution(self, institution_names: List[str], fuzzy_match: bool = True) -> List[ResearcherRanking]:
        """Filter rankings by institution names.
        
        Args:
            institution_names: List of institution names to filter by
            fuzzy_match: If True, use partial string matching for name variations
            
        Returns:
            Filtered list of ResearcherRanking objects
        """
        if not self.rankings or not self.institutions:
            return []
        
        # Normalize institution names for comparison
        normalized_names = [self._normalize_institution_name(name) for name in institution_names]
        
        filtered_rankings = []
        for ranking in self.rankings:
            author_data = self.authors.get(ranking.author_id)
            if not author_data or not author_data.last_known_institution_ids:
                continue
            
            # Check if author has target institutions
            for inst_id in author_data.last_known_institution_ids:
                institution = self.institutions.get(inst_id)
                if not institution:
                    continue
                
                inst_name_normalized = self._normalize_institution_name(institution.display_name)
                
                if fuzzy_match:
                    # Use partial matching for name variations
                    for target_name in normalized_names:
                        if target_name in inst_name_normalized or inst_name_normalized in target_name:
                            filtered_rankings.append(ranking)
                            break
                    else:
                        continue
                    break
                else:
                    # Exact matching
                    if inst_name_normalized in normalized_names:
                        filtered_rankings.append(ranking)
                        break
        
        return filtered_rankings
    
    def _normalize_institution_name(self, name: str) -> str:
        """Normalize institution name for comparison.
        
        Handles common variations and abbreviations.
        """
        normalized = name.lower().strip()
        
        # Common abbreviation mappings
        abbreviation_map = {
            'university': 'univ',
            'institute': 'inst',
            'college': 'coll',
            'school': 'sch',
            'hospital': 'hosp',
            'medical': 'med',
            'technology': 'tech',
            'national': 'natl',
            'international': 'intl',
            'research': 'res',
            'center': 'ctr',
            'centre': 'ctr',
            'laboratory': 'lab',
        }
        
        # Apply abbreviation mappings
        for full_word, abbrev in abbreviation_map.items():
            normalized = normalized.replace(full_word, abbrev)
        
        # Remove common punctuation and extra spaces
        normalized = re.sub(r'[^\w\s]', ' ', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def get_country_distribution(self) -> Dict[str, int]:
        """Get distribution of researchers by country.
        
        Returns:
            Dict mapping country code to researcher count
        """
        country_counts = {}
        
        for ranking in self.rankings:
            author_data = self.authors.get(ranking.author_id)
            if not author_data or not author_data.last_known_institution_ids:
                continue
            
            # Count unique countries for this author
            author_countries = set()
            for inst_id in author_data.last_known_institution_ids:
                institution = self.institutions.get(inst_id)
                if institution and institution.country_code:
                    author_countries.add(institution.country_code)
            
            # Add to counts (author counted once per country)
            for country in author_countries:
                country_counts[country] = country_counts.get(country, 0) + 1
        
        return dict(sorted(country_counts.items(), key=lambda x: x[1], reverse=True))
    
    def get_institution_distribution(self, top_n: int = 20) -> Dict[str, int]:
        """Get distribution of researchers by institution.
        
        Args:
            top_n: Number of top institutions to return
            
        Returns:
            Dict mapping institution name to researcher count
        """
        institution_counts = {}
        
        for ranking in self.rankings:
            author_data = self.authors.get(ranking.author_id)
            if not author_data or not author_data.last_known_institution_ids:
                continue
            
            # Count unique institutions for this author
            author_institutions = set()
            for inst_id in author_data.last_known_institution_ids:
                institution = self.institutions.get(inst_id)
                if institution:
                    author_institutions.add(institution.display_name)
            
            # Add to counts (author counted once per institution)
            for inst_name in author_institutions:
                institution_counts[inst_name] = institution_counts.get(inst_name, 0) + 1
        
        # Return top N institutions
        sorted_counts = sorted(institution_counts.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_counts[:top_n])
    
    def get_researcher_details(self, author_id: str) -> Optional[Dict]:
        """Get detailed information for a specific researcher."""
        # Find the researcher in rankings
        researcher_ranking = None
        for ranking in self.rankings:
            if ranking.author_id == author_id:
                researcher_ranking = ranking
                break
        
        if not researcher_ranking:
            return None
        
        # Get additional details
        author_data = self.authors.get(author_id)
        centrality_data = {}
        for metric, values in self.centralities.items():
            centrality_data[metric] = values.get(author_id, 0.0)
        
        citation_data = self.citations.get(author_id, {})
        leadership_data = self.leadership.get(author_id, {})
        
        return {
            'ranking': researcher_ranking,
            'author_data': author_data,
            'centrality_metrics': centrality_data,
            'citation_metrics': citation_data,
            'leadership_metrics': leadership_data
        }
    
    def get_network_statistics(self) -> Dict:
        """Get network statistics."""
        if self.network is None:
            return {}
        
        return self.network_builder.get_network_statistics(self.network)
    
    def get_author_explanation(self, author_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed explanation for an author's ranking.
        
        Args:
            author_id: ID of the author to explain
            
        Returns:
            Dictionary with detailed explanation or None if author not found
        """
        if not self.rankings or not self.author_features:
            return None
        
        # Find the ranking for this author
        ranking_info = None
        for ranking in self.rankings:
            if ranking.author_id == author_id:
                ranking_info = ranking
                break
        
        if not ranking_info:
            return None
        
        features = self.author_features.get(author_id)
        if not features:
            return None
        
        return self.explainability.generate_author_explanation(
            author_id=author_id,
            graph=self.network,
            edge_data=self.edge_data,
            works=self.works,
            author_features=features,
            authors=self.authors,
            ranking_info=ranking_info
        )
    
    def get_top_explanations(self, top_n: int = 10) -> Dict[str, Dict[str, Any]]:
        """Get explanations for top N researchers.
        
        Args:
            top_n: Number of top researchers to explain
            
        Returns:
            Dictionary mapping author IDs to explanation data
        """
        if not self.rankings or not self.author_features:
            return {}
        
        return self.explainability.export_explanations_to_dict(
            rankings=self.rankings,
            graph=self.network,
            edge_data=self.edge_data,
            works=self.works,
            author_features=self.author_features,
            authors=self.authors,
            top_n=top_n
        )
    
    def _export_html_report(self, output_path: str, field_name: str = "Research Field", language: str = "en") -> None:
        """Export comprehensive HTML report with visualizations.
        
        Args:
            output_path: Path to output HTML file
            field_name: Name of the research field being analyzed
        """
        if not all([self.rankings, self.network, self.works, self.authors]):
            raise ValueError("Analysis not complete. Missing required data for HTML report.")
        
        # Get network statistics
        network_stats = self.get_network_statistics()
        
        # Generate HTML report
        self.html_generator.generate_report(
            rankings=self.rankings,
            graph=self.network,
            edge_data=self.edge_data,
            works=self.works,
            authors=self.authors,
            institutions=self.institutions,
            network_stats=network_stats,
            field_name=field_name,
            output_path=output_path,
            language=language
        )
        
        logger.info(f"HTML report exported to {output_path}")
    
    def generate_html_report(self, output_path: str, field_name: str = "Research Field", language: str = "en") -> str:
        """Generate comprehensive HTML report with visualizations.
        
        Args:
            output_path: Path to output HTML file
            field_name: Name of the research field being analyzed
            language: Language for report ('en' or 'ja')
            
        Returns:
            Path to generated HTML report
        """
        self._export_html_report(output_path, field_name, language)
        return output_path