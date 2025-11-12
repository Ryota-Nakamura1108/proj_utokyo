"""Core keyword extraction logic with full pipeline implementation."""

import asyncio
import logging
import time
from typing import Dict, Optional, Any
from datetime import datetime
import numpy as np

from .config import config
from .models import FilterConfig, ExtractionResult
from .openalex_fetcher import OpenAlexFetcher
from .paper_filter import PaperFilter
from .embedding_generator import EmbeddingGenerator
from .clustering_engine import ClusteringEngine
from .representative_extractor import RepresentativeExtractor
from .keyword_generator import KeywordGenerator
from .researcher_summarizer import ResearcherSummarizer

logger = logging.getLogger(__name__)


class KeywordExtractor:
    """Main class for extracting keywords from researcher data."""

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        openalex_email: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small",
        llm_model: str = "gpt-4o-mini",
    ):
        """Initialize the keyword extractor.

        Args:
            openai_api_key: OpenAI API key (if None, uses config)
            openalex_email: Email for OpenAlex API (if None, uses config)
            embedding_model: Embedding model to use
            llm_model: LLM model for keyword generation
        """
        self.api_key = openai_api_key or config.openai_api_key
        self.email = openalex_email or config.openalex_email
        self.embedding_model = embedding_model
        self.llm_model = llm_model

        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        logger.info(f"Initialized KeywordExtractor")
        logger.info(f"  Embedding model: {self.embedding_model}")
        logger.info(f"  LLM model: {self.llm_model}")

    async def extract_by_id(
        self,
        openalex_id: str,
        years_back: int = 10,
        max_keywords: int = 10,
        min_citations: int = 0,
        include_clusters: bool = False,
    ) -> Dict[str, Any]:
        """Extract keywords for a researcher by OpenAlex ID.

        Args:
            openalex_id: OpenAlex author ID
            years_back: Years back to analyze
            max_keywords: Maximum number of keywords to extract
            min_citations: Minimum citations for paper inclusion
            include_clusters: Whether to include cluster information

        Returns:
            Dictionary containing extracted keywords and metadata
        """
        logger.info(f"Extracting keywords for OpenAlex ID: {openalex_id}")
        start_time = time.time()

        try:
            # Fetch author and papers
            async with OpenAlexFetcher(self.email) as fetcher:
                author = await fetcher.get_author_by_id(openalex_id)

                if not author:
                    return self._create_error_result(
                        openalex_id, "unknown", "Author not found"
                    )

                papers = await fetcher.fetch_author_papers(
                    openalex_id, years_back=years_back
                )

            if not papers:
                return self._create_error_result(
                    openalex_id, author.display_name, "No papers found"
                )

            # Run extraction pipeline
            result = await self._run_extraction_pipeline(
                author_id=openalex_id,
                author_name=author.display_name,
                papers=papers,
                years_back=years_back,
                max_keywords=max_keywords,
                min_citations=min_citations,
                include_clusters=include_clusters,
            )

            result.processing_time = time.time() - start_time
            return result.to_dict()

        except Exception as e:
            logger.exception(f"Extraction failed: {e}")
            return self._create_error_result(
                openalex_id, "unknown", f"Error: {str(e)}"
            )

    async def extract_by_name(
        self,
        researcher_name: str,
        years_back: int = 10,
        max_keywords: int = 10,
        min_citations: int = 0,
        include_clusters: bool = False,
    ) -> Dict[str, Any]:
        """Extract keywords for a researcher by name.

        Args:
            researcher_name: Name of the researcher
            years_back: Years back to analyze
            max_keywords: Maximum number of keywords to extract
            min_citations: Minimum citations for paper inclusion
            include_clusters: Whether to include cluster information

        Returns:
            Dictionary containing extracted keywords and metadata
        """
        logger.info(f"Extracting keywords for researcher: {researcher_name}")
        start_time = time.time()

        try:
            # Search for author by name
            async with OpenAlexFetcher(self.email) as fetcher:
                author = await fetcher.search_author_by_name(researcher_name)

                if not author:
                    return self._create_error_result(
                        "unknown", researcher_name, "Author not found"
                    )

                papers = await fetcher.fetch_author_papers(
                    author.id, years_back=years_back
                )

            if not papers:
                return self._create_error_result(
                    author.id, author.display_name, "No papers found"
                )

            # Run extraction pipeline
            result = await self._run_extraction_pipeline(
                author_id=author.id,
                author_name=author.display_name,
                papers=papers,
                years_back=years_back,
                max_keywords=max_keywords,
                min_citations=min_citations,
                include_clusters=include_clusters,
            )

            result.processing_time = time.time() - start_time
            return result.to_dict()

        except Exception as e:
            logger.exception(f"Extraction failed: {e}")
            return self._create_error_result(
                "unknown", researcher_name, f"Error: {str(e)}"
            )

    async def _run_extraction_pipeline(
        self,
        author_id: str,
        author_name: str,
        papers: list,
        years_back: int,
        max_keywords: int,
        min_citations: int,
        include_clusters: bool,
    ) -> ExtractionResult:
        """Run the complete extraction pipeline.

        Args:
            author_id: OpenAlex author ID
            author_name: Author name
            papers: List of papers
            years_back: Years back
            max_keywords: Max keywords
            min_citations: Min citations
            include_clusters: Include cluster info

        Returns:
            ExtractionResult object
        """
        # Step 1: Filter papers
        filter_config = FilterConfig(
            years_back=years_back,
            author_positions=["first", "last"],
            min_citations=min_citations,
        )
        paper_filter = PaperFilter(filter_config)
        filtered_papers = paper_filter.filter_papers(papers)

        if not filtered_papers:
            logger.warning("No papers remaining after filtering")
            return self._create_empty_result(
                author_id, author_name, years_back
            )

        stats = paper_filter.get_statistics(filtered_papers)
        logger.info(f"Statistics: {stats}")

        # Step 2: Generate embeddings
        async with EmbeddingGenerator(
            self.api_key, model=self.embedding_model
        ) as emb_gen:
            embeddings = await emb_gen.generate_embeddings(filtered_papers)

        # Step 3: Cluster papers
        clustering_engine = ClusteringEngine(method="kmeans")
        clusters, labels = clustering_engine.cluster_papers(
            filtered_papers, embeddings
        )

        # Step 4: Extract representatives
        # Map embeddings to clusters
        embeddings_by_cluster = {}
        for i, label in enumerate(labels):
            if label not in embeddings_by_cluster:
                embeddings_by_cluster[label] = []
            embeddings_by_cluster[label].append(embeddings[i])

        rep_extractor = RepresentativeExtractor(
            strategy="weighted", max_representatives=3
        )
        clusters = rep_extractor.extract_representatives(
            clusters, embeddings_by_cluster
        )

        # Step 5: Generate keywords
        current_year = datetime.now().year
        start_year = current_year - years_back

        async with KeywordGenerator(
            self.api_key, model=self.llm_model, max_keywords=max_keywords
        ) as kw_gen:
            keywords = await kw_gen.generate_keywords(
                clusters,
                author_name,
                len(filtered_papers),
                start_year,
                current_year,
            )

        # Step 6: Generate researcher summary
        async with ResearcherSummarizer(
            self.api_key, model=self.llm_model
        ) as summarizer:
            summary = await summarizer.generate_summary(
                researcher_name=author_name,
                keywords=keywords,
                clusters=clusters,
                statistics=stats,
                analysis_period={"start_year": start_year, "end_year": current_year},
            )

        # Create result
        result = ExtractionResult(
            researcher_id=author_id,
            researcher_name=author_name,
            analysis_period={"start_year": start_year, "end_year": current_year},
            statistics=stats,
            keywords=keywords,
            summary=summary,
            clusters=clusters if include_clusters else None,
        )

        return result

    def _create_error_result(
        self, author_id: str, author_name: str, error_msg: str
    ) -> Dict[str, Any]:
        """Create an error result dictionary.

        Args:
            author_id: Author ID
            author_name: Author name
            error_msg: Error message

        Returns:
            Error result dictionary
        """
        return {
            "researcher_id": author_id,
            "researcher_name": author_name,
            "keywords": [],
            "statistics": {},
            "status": "error",
            "error": error_msg,
        }

    def _create_empty_result(
        self, author_id: str, author_name: str, years_back: int
    ) -> ExtractionResult:
        """Create an empty result when no papers found.

        Args:
            author_id: Author ID
            author_name: Author name
            years_back: Years back

        Returns:
            Empty ExtractionResult
        """
        current_year = datetime.now().year
        start_year = current_year - years_back

        return ExtractionResult(
            researcher_id=author_id,
            researcher_name=author_name,
            analysis_period={"start_year": start_year, "end_year": current_year},
            statistics={"total_papers": 0},
            keywords=[],
            status="no_papers",
        )


# Synchronous wrapper functions


def extract_keywords(
    identifier: str,
    years_back: int = 10,
    max_keywords: int = 10,
    min_citations: int = 0,
    include_clusters: bool = False,
    openai_api_key: Optional[str] = None,
    openalex_email: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract keywords for a researcher (synchronous wrapper).

    Args:
        identifier: Researcher name or OpenAlex ID
        years_back: Years back to analyze
        max_keywords: Maximum number of keywords to extract
        min_citations: Minimum citations filter
        include_clusters: Whether to include cluster information
        openai_api_key: OpenAI API key (optional)
        openalex_email: Email for OpenAlex API (optional)

    Returns:
        Dictionary containing extracted keywords and metadata
    """
    extractor = KeywordExtractor(
        openai_api_key=openai_api_key,
        openalex_email=openalex_email,
    )

    # Detect if identifier is an OpenAlex ID
    if identifier.startswith("A") or "openalex.org" in identifier:
        if "openalex.org" in identifier:
            identifier = identifier.split("/")[-1]
        return asyncio.run(
            extractor.extract_by_id(
                identifier,
                years_back=years_back,
                max_keywords=max_keywords,
                min_citations=min_citations,
                include_clusters=include_clusters,
            )
        )
    else:
        return asyncio.run(
            extractor.extract_by_name(
                identifier,
                years_back=years_back,
                max_keywords=max_keywords,
                min_citations=min_citations,
                include_clusters=include_clusters,
            )
        )


def extract_keywords_by_id(
    openalex_id: str,
    years_back: int = 10,
    max_keywords: int = 10,
    min_citations: int = 0,
    include_clusters: bool = False,
    openai_api_key: Optional[str] = None,
    openalex_email: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract keywords by OpenAlex ID (synchronous wrapper).

    Args:
        openalex_id: OpenAlex author ID
        years_back: Years back to analyze
        max_keywords: Maximum number of keywords to extract
        min_citations: Minimum citations filter
        include_clusters: Whether to include cluster information
        openai_api_key: OpenAI API key (optional)
        openalex_email: Email for OpenAlex API (optional)

    Returns:
        Dictionary containing extracted keywords and metadata
    """
    extractor = KeywordExtractor(
        openai_api_key=openai_api_key,
        openalex_email=openalex_email,
    )
    return asyncio.run(
        extractor.extract_by_id(
            openalex_id,
            years_back=years_back,
            max_keywords=max_keywords,
            min_citations=min_citations,
            include_clusters=include_clusters,
        )
    )


def extract_keywords_by_name(
    researcher_name: str,
    years_back: int = 10,
    max_keywords: int = 10,
    min_citations: int = 0,
    include_clusters: bool = False,
    openai_api_key: Optional[str] = None,
    openalex_email: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract keywords by name (synchronous wrapper).

    Args:
        researcher_name: Name of the researcher
        years_back: Years back to analyze
        max_keywords: Maximum number of keywords to extract
        min_citations: Minimum citations filter
        include_clusters: Whether to include cluster information
        openai_api_key: OpenAI API key (optional)
        openalex_email: Email for OpenAlex API (optional)

    Returns:
        Dictionary containing extracted keywords and metadata
    """
    extractor = KeywordExtractor(
        openai_api_key=openai_api_key,
        openalex_email=openalex_email,
    )
    return asyncio.run(
        extractor.extract_by_name(
            researcher_name,
            years_back=years_back,
            max_keywords=max_keywords,
            min_citations=min_citations,
            include_clusters=include_clusters,
        )
    )
