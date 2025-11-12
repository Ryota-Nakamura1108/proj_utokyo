"""Paper filtering logic."""

import logging
from typing import List
from datetime import datetime

from .models import Paper, FilterConfig

logger = logging.getLogger(__name__)


class PaperFilter:
    """Filters papers based on configured criteria."""

    def __init__(self, config: FilterConfig):
        """Initialize the filter.

        Args:
            config: FilterConfig with filtering parameters
        """
        self.config = config

    def filter_papers(self, papers: List[Paper]) -> List[Paper]:
        """Apply all filters to papers.

        Args:
            papers: List of papers to filter

        Returns:
            Filtered list of papers
        """
        logger.info(f"Filtering {len(papers)} papers...")

        filtered = papers

        # Filter by author position
        filtered = self._filter_by_author_position(filtered)

        # Filter by year
        filtered = self._filter_by_year(filtered)

        # Filter by citations
        filtered = self._filter_by_citations(filtered)

        # Limit to max papers
        filtered = self._limit_papers(filtered)

        logger.info(
            f"Filtering complete: {len(papers)} -> {len(filtered)} papers"
        )

        return filtered

    def _filter_by_author_position(self, papers: List[Paper]) -> List[Paper]:
        """Filter papers by author position.

        Args:
            papers: Papers to filter

        Returns:
            Filtered papers
        """
        if not self.config.author_positions:
            return papers

        filtered = [
            p
            for p in papers
            if p.author_position in self.config.author_positions
        ]

        logger.info(
            f"Author position filter ({self.config.author_positions}): "
            f"{len(papers)} -> {len(filtered)}"
        )

        return filtered

    def _filter_by_year(self, papers: List[Paper]) -> List[Paper]:
        """Filter papers by publication year.

        Args:
            papers: Papers to filter

        Returns:
            Filtered papers
        """
        current_year = datetime.now().year
        cutoff_year = current_year - self.config.years_back

        filtered = [
            p
            for p in papers
            if p.publication_year and p.publication_year >= cutoff_year
        ]

        logger.info(
            f"Year filter (>={cutoff_year}): {len(papers)} -> {len(filtered)}"
        )

        return filtered

    def _filter_by_citations(self, papers: List[Paper]) -> List[Paper]:
        """Filter papers by citation count.

        Args:
            papers: Papers to filter

        Returns:
            Filtered papers
        """
        if self.config.min_citations <= 0:
            return papers

        filtered = [
            p for p in papers if p.cited_by_count >= self.config.min_citations
        ]

        logger.info(
            f"Citation filter (>={self.config.min_citations}): "
            f"{len(papers)} -> {len(filtered)}"
        )

        return filtered

    def _limit_papers(self, papers: List[Paper]) -> List[Paper]:
        """Limit number of papers to max_papers.

        Papers are sorted by citation count before limiting.

        Args:
            papers: Papers to limit

        Returns:
            Limited papers
        """
        if len(papers) <= self.config.max_papers:
            return papers

        # Sort by citation count (descending) and limit
        sorted_papers = sorted(
            papers, key=lambda p: p.cited_by_count, reverse=True
        )
        limited = sorted_papers[: self.config.max_papers]

        logger.info(
            f"Paper limit ({self.config.max_papers}): {len(papers)} -> {len(limited)}"
        )

        return limited

    def get_statistics(self, papers: List[Paper]) -> dict:
        """Get statistics about filtered papers.

        Args:
            papers: Papers to analyze

        Returns:
            Dictionary with statistics
        """
        if not papers:
            return {
                "total_papers": 0,
                "papers_as_first_author": 0,
                "papers_as_last_author": 0,
                "total_citations": 0,
                "average_citations": 0,
                "year_range": None,
            }

        first_author_count = sum(
            1 for p in papers if p.author_position == "first"
        )
        last_author_count = sum(1 for p in papers if p.author_position == "last")
        total_citations = sum(p.cited_by_count for p in papers)

        years = [p.publication_year for p in papers if p.publication_year]
        year_range = None
        if years:
            year_range = {"start": min(years), "end": max(years)}

        return {
            "total_papers": len(papers),
            "papers_as_first_author": first_author_count,
            "papers_as_last_author": last_author_count,
            "total_citations": total_citations,
            "average_citations": total_citations / len(papers) if papers else 0,
            "year_range": year_range,
        }
