"""OpenAlex API client for fetching author and paper data."""

import asyncio
import logging
from typing import List, Optional, Dict, Any
import httpx
from datetime import datetime

from .models import Paper, Author
from .config import config

logger = logging.getLogger(__name__)


class OpenAlexFetcher:
    """Fetches data from OpenAlex API."""

    BASE_URL = "https://api.openalex.org"

    def __init__(self, email: Optional[str] = None):
        """Initialize the fetcher.

        Args:
            email: Email for polite pool access
        """
        self.email = email or config.openalex_email
        self.client = httpx.AsyncClient(timeout=30.0)

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def search_author_by_name(self, name: str) -> Optional[Author]:
        """Search for an author by name.

        Args:
            name: Author name to search for

        Returns:
            Author object if found, None otherwise
        """
        logger.info(f"Searching for author: {name}")

        params = {"search": name, "mailto": self.email}

        try:
            response = await self.client.get(
                f"{self.BASE_URL}/authors", params=params
            )
            response.raise_for_status()
            data = response.json()

            if not data.get("results"):
                logger.warning(f"No author found for name: {name}")
                return None

            # Get the first (most relevant) result
            author_data = data["results"][0]

            author = Author(
                id=author_data["id"].split("/")[-1],  # Extract ID from URL
                display_name=author_data["display_name"],
                orcid=author_data.get("orcid"),
                h_index=author_data.get("summary_stats", {}).get("h_index"),
                works_count=author_data.get("works_count", 0),
                cited_by_count=author_data.get("cited_by_count", 0),
            )

            logger.info(f"Found author: {author.display_name} (ID: {author.id})")
            return author

        except httpx.HTTPError as e:
            logger.error(f"HTTP error while searching author: {e}")
            return None

    async def get_author_by_id(self, author_id: str) -> Optional[Author]:
        """Get author details by OpenAlex ID.

        Args:
            author_id: OpenAlex author ID

        Returns:
            Author object if found, None otherwise
        """
        # Clean up ID (remove URL if present)
        if "openalex.org" in author_id:
            author_id = author_id.split("/")[-1]

        logger.info(f"Fetching author with ID: {author_id}")

        params = {"mailto": self.email}

        try:
            response = await self.client.get(
                f"{self.BASE_URL}/authors/{author_id}", params=params
            )
            response.raise_for_status()
            author_data = response.json()

            author = Author(
                id=author_data["id"].split("/")[-1],
                display_name=author_data["display_name"],
                orcid=author_data.get("orcid"),
                h_index=author_data.get("summary_stats", {}).get("h_index"),
                works_count=author_data.get("works_count", 0),
                cited_by_count=author_data.get("cited_by_count", 0),
            )

            logger.info(f"Fetched author: {author.display_name}")
            return author

        except httpx.HTTPError as e:
            logger.error(f"HTTP error while fetching author: {e}")
            return None

    async def fetch_author_papers(
        self,
        author_id: str,
        years_back: int = 10,
        max_results: int = 200,
    ) -> List[Paper]:
        """Fetch all papers for a specific author.

        Args:
            author_id: OpenAlex author ID
            years_back: How many years back to fetch papers
            max_results: Maximum number of papers to fetch

        Returns:
            List of Paper objects
        """
        # Clean up ID
        if "openalex.org" in author_id:
            author_id = author_id.split("/")[-1]

        current_year = datetime.now().year
        start_year = current_year - years_back

        logger.info(
            f"Fetching papers for author {author_id} from {start_year} to {current_year}"
        )

        papers = []
        page = 1
        per_page = min(200, max_results)  # OpenAlex max is 200

        while len(papers) < max_results:
            params = {
                "filter": f"authorships.author.id:{author_id},publication_year:>{start_year - 1}",
                "per-page": per_page,
                "page": page,
                "mailto": self.email,
            }

            try:
                response = await self.client.get(
                    f"{self.BASE_URL}/works", params=params
                )
                response.raise_for_status()
                data = response.json()

                results = data.get("results", [])
                if not results:
                    break

                for work in results:
                    paper = self._parse_paper(work, author_id)
                    if paper:
                        papers.append(paper)

                logger.info(
                    f"Fetched page {page}: {len(results)} papers (total: {len(papers)})"
                )

                # Check if there are more pages
                if len(results) < per_page:
                    break

                page += 1

                # Avoid hitting rate limits
                await asyncio.sleep(0.1)

            except httpx.HTTPError as e:
                logger.error(f"HTTP error while fetching papers: {e}")
                break

        logger.info(f"Total papers fetched: {len(papers)}")
        return papers

    def _parse_paper(self, work_data: Dict[str, Any], author_id: str) -> Optional[Paper]:
        """Parse a work from OpenAlex API response into Paper object.

        Args:
            work_data: Raw work data from API
            author_id: ID of the author we're searching for

        Returns:
            Paper object or None if parsing fails
        """
        try:
            work_id = work_data.get("id", "").split("/")[-1]

            # Find author position
            author_position = None
            is_corresponding = False

            for authorship in work_data.get("authorships", []):
                auth_id = authorship.get("author", {}).get("id", "").split("/")[-1]
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
