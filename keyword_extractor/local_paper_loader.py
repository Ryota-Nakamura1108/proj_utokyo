"""Local paper data loader for keyword extractor.

This module loads paper data from local JSON files instead of using OpenAlex API.
Designed to work with both local filesystem and Cloud Storage for Firebase hosting.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

from .models import Paper

logger = logging.getLogger(__name__)


class LocalPaperLoader:
    """Loads paper data from local JSON files."""

    def __init__(self, papers_dir: Optional[str] = None):
        """Initialize the local paper loader.

        Args:
            papers_dir: Directory containing paper JSON files.
                       If None, uses default location relative to this module.
        """
        if papers_dir:
            self.papers_dir = Path(papers_dir)
        else:
            # Default: ../papers directory relative to keyword_extractor
            module_dir = Path(__file__).parent
            self.papers_dir = module_dir.parent / "papers"

        logger.info(f"Initialized LocalPaperLoader with directory: {self.papers_dir}")

    def load_all_papers(self) -> List[Dict[str, Any]]:
        """Load all papers from JSON files in the papers directory.

        Returns:
            List of paper dictionaries (raw OpenAlex format)
        """
        if not self.papers_dir.exists():
            logger.error(f"Papers directory not found: {self.papers_dir}")
            return []

        all_papers = []
        json_files = sorted(self.papers_dir.glob("_*.json"))

        if not json_files:
            logger.warning(f"No JSON files found in {self.papers_dir}")
            return []

        logger.info(f"Found {len(json_files)} JSON files to load")

        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    papers = json.load(f)
                    if isinstance(papers, list):
                        all_papers.extend(papers)
                        logger.info(
                            f"Loaded {len(papers)} papers from {json_file.name}"
                        )
                    else:
                        logger.warning(
                            f"Unexpected data format in {json_file.name}: {type(papers)}"
                        )
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from {json_file.name}: {e}")
            except Exception as e:
                logger.error(f"Error loading {json_file.name}: {e}")

        logger.info(f"Total papers loaded: {len(all_papers)}")
        return all_papers

    def get_papers_by_author(
        self,
        author_id: str,
        years_back: int = 10,
        max_results: int = 200,
    ) -> List[Paper]:
        """Get papers for a specific author from local files.

        Args:
            author_id: OpenAlex author ID (e.g., "A5109349291" or full URL)
            years_back: How many years back to include papers
            max_results: Maximum number of papers to return

        Returns:
            List of Paper objects
        """
        # Clean up ID (remove URL if present)
        if "openalex.org" in author_id:
            author_id = author_id.split("/")[-1]

        logger.info(
            f"Filtering papers for author {author_id} "
            f"(years_back={years_back}, max_results={max_results})"
        )

        # Load all papers
        all_papers_raw = self.load_all_papers()

        # Filter by author and year
        from datetime import datetime

        current_year = datetime.now().year
        start_year = current_year - years_back

        filtered_papers = []
        for paper_data in all_papers_raw:
            # Check if author is in authorships
            if not self._is_author_in_paper(paper_data, author_id):
                continue

            # Check year filter
            pub_year = paper_data.get("publication_year")
            if pub_year and pub_year < start_year:
                continue

            # Parse paper
            paper = self._parse_paper(paper_data, author_id)
            if paper:
                filtered_papers.append(paper)

            # Stop if we've reached max results
            if len(filtered_papers) >= max_results:
                break

        logger.info(f"Found {len(filtered_papers)} papers for author {author_id}")
        return filtered_papers

    def _is_author_in_paper(self, paper_data: Dict[str, Any], author_id: str) -> bool:
        """Check if the given author is in the paper's authorships.

        Args:
            paper_data: Raw paper data
            author_id: OpenAlex author ID (without URL)

        Returns:
            True if author is in paper
        """
        authorships = paper_data.get("authorships", [])
        for authorship in authorships:
            author = authorship.get("author", {})
            auth_id = author.get("id", "")
            # Extract ID from URL if present
            if "/" in auth_id:
                auth_id = auth_id.split("/")[-1]
            if auth_id == author_id:
                return True
        return False

    def _parse_paper(
        self, work_data: Dict[str, Any], author_id: str
    ) -> Optional[Paper]:
        """Parse a work from raw data into Paper object.

        This method replicates the parsing logic from OpenAlexFetcher.

        Args:
            work_data: Raw work data
            author_id: ID of the author we're searching for

        Returns:
            Paper object or None if parsing fails
        """
        try:
            work_id = work_data.get("id", "")
            if "/" in work_id:
                work_id = work_id.split("/")[-1]

            # Find author position
            author_position = None
            is_corresponding = False

            for authorship in work_data.get("authorships", []):
                auth_id = authorship.get("author", {}).get("id", "")
                if "/" in auth_id:
                    auth_id = auth_id.split("/")[-1]
                if auth_id == author_id:
                    author_position = authorship.get("author_position")
                    is_corresponding = authorship.get("is_corresponding", False)
                    break

            # Extract topics
            topics = []
            for topic in work_data.get("topics", [])[:3]:  # Top 3 topics
                topics.append(topic.get("display_name", ""))

            # Extract journal name
            journal_name = None
            primary_location = work_data.get("primary_location", {})
            if primary_location:
                source = primary_location.get("source")
                if source:
                    journal_name = source.get("display_name")

            paper = Paper(
                id=work_id,
                title=work_data.get("display_name", ""),
                abstract=self._get_abstract(work_data),
                publication_year=work_data.get("publication_year"),
                cited_by_count=work_data.get("cited_by_count", 0),
                author_position=author_position,
                journal_name=journal_name,
                topics=topics,
                doi=work_data.get("doi"),
                is_corresponding=is_corresponding,
            )

            return paper

        except Exception as e:
            logger.warning(f"Failed to parse paper: {e}")
            return None

    def _get_abstract(self, work_data: Dict[str, Any]) -> Optional[str]:
        """Extract abstract from work data.

        Args:
            work_data: Raw work data

        Returns:
            Abstract text or None
        """
        # OpenAlex stores abstract in abstract_inverted_index
        inverted_index = work_data.get("abstract_inverted_index")
        if not inverted_index:
            return None

        try:
            # Reconstruct abstract from inverted index
            word_positions = []
            for word, positions in inverted_index.items():
                for pos in positions:
                    word_positions.append((pos, word))

            # Sort by position and join
            word_positions.sort(key=lambda x: x[0])
            abstract = " ".join([word for _, word in word_positions])
            return abstract

        except Exception as e:
            logger.warning(f"Failed to reconstruct abstract: {e}")
            return None

    def get_author_info_from_papers(
        self, author_id: str
    ) -> Optional[Dict[str, Any]]:
        """Extract author information from papers (since we don't have author API).

        Args:
            author_id: OpenAlex author ID

        Returns:
            Dictionary with author info or None
        """
        # Clean up ID
        if "openalex.org" in author_id:
            author_id = author_id.split("/")[-1]

        all_papers_raw = self.load_all_papers()

        # Find first paper with this author
        for paper_data in all_papers_raw:
            authorships = paper_data.get("authorships", [])
            for authorship in authorships:
                author = authorship.get("author", {})
                auth_id = author.get("id", "")
                if "/" in auth_id:
                    auth_id = auth_id.split("/")[-1]

                if auth_id == author_id:
                    return {
                        "id": author_id,
                        "display_name": author.get("display_name", "Unknown"),
                        "orcid": author.get("orcid"),
                    }

        logger.warning(f"No author info found for ID: {author_id}")
        return None
