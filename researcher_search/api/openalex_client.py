"""OpenAlex API client with rate limiting and batching."""

import time
import asyncio
import httpx
import logging
from typing import Dict, List, Optional, Union, Any
from dataclasses import asdict
import json

from ..core.models import WorkRaw, AuthorMaster, Authorship, SummaryStats, AuthorPosition, InstitutionMaster

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


class OpenAlexClient:
    """OpenAlex API client with rate limiting and batching."""
    
    BASE_URL = "https://api.openalex.org"
    RATE_LIMIT_PER_SEC = 10
    RATE_LIMIT_PER_DAY = 100000
    MAX_BATCH_SIZE = 100
    
    def __init__(self, email: str):
        """Initialize client with email for polite requests."""
        self.email = email
        self.session = httpx.AsyncClient(timeout=30.0)
        self.request_count = 0
        self.daily_request_count = 0
        self.last_request_time = 0.0
        
    async def __aenter__(self):
        """Async context manager entry."""
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.session.aclose()
        
    async def _rate_limit(self):
        """Apply rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        min_interval = 1.0 / self.RATE_LIMIT_PER_SEC
        
        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            await asyncio.sleep(sleep_time)
            
        self.last_request_time = time.time()
        self.request_count += 1
        self.daily_request_count += 1
        
        if self.daily_request_count >= self.RATE_LIMIT_PER_DAY:
            raise RuntimeError(f"Daily rate limit ({self.RATE_LIMIT_PER_DAY}) exceeded")
            
    async def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make rate-limited request to OpenAlex API."""
        await self._rate_limit()
        
        # Add email to params for polite requests
        params["mailto"] = self.email
        
        url = f"{self.BASE_URL}/{endpoint.lstrip('/')}"
        
        # logger.debug(f"Making request to {url} with params: {params}")
        
        for attempt in range(3):  # Retry up to 3 times
            try:
                response = await self.session.get(url, params=params)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:  # Rate limited
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limited, waiting {wait_time}s before retry {attempt + 1}")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise
            except Exception as e:
                if attempt == 2:  # Last attempt
                    raise
                await asyncio.sleep(2 ** attempt)
                
        raise RuntimeError("Failed to make request after 3 attempts")
    
    async def get_works_by_ids(self, work_ids: List[str]) -> List[WorkRaw]:
        """Get works by OpenAlex IDs, DOIs, or mixed identifiers in batches."""
        all_works = []
        
        # Process in batches of 100
        for i in range(0, len(work_ids), self.MAX_BATCH_SIZE):
            batch = work_ids[i:i + self.MAX_BATCH_SIZE]
            works_batch = await self._get_works_batch(batch)
            all_works.extend(works_batch)
            
        return all_works
    
    async def _get_works_batch(self, identifiers: List[str]) -> List[WorkRaw]:
        """Get a batch of works."""
        # Separate DOIs and OpenAlex IDs
        dois = []
        openalex_ids = []
        
        for identifier in identifiers:
            if identifier.startswith("W") or identifier.startswith("https://openalex.org/W"):
                # OpenAlex ID
                clean_id = identifier.replace("https://openalex.org/", "")
                openalex_ids.append(clean_id)
            elif "doi.org" in identifier or identifier.count("/") >= 1:
                # DOI
                clean_doi = identifier.replace("https://doi.org/", "").lower()
                dois.append(clean_doi)
            else:
                # Assume it's an OpenAlex ID
                openalex_ids.append(identifier)
        
        works = []
        
        # Fetch DOI batch
        if dois:
            doi_filter = "|".join(dois)
            works.extend(await self._fetch_works_with_filter(f"doi:{doi_filter}"))
            
        # Fetch OpenAlex ID batch  
        if openalex_ids:
            id_filter = "|".join(openalex_ids)
            works.extend(await self._fetch_works_with_filter(f"openalex:{id_filter}"))
            
        return works
    
    async def _fetch_works_with_filter(self, filter_str: str) -> List[WorkRaw]:
        """Fetch works with a given filter."""
        params = {
            "filter": filter_str,
            "select": "id,doi,display_name,publication_year,authorships,referenced_works,cited_by_count,topics,primary_topic,is_authors_truncated",
            "per-page": self.MAX_BATCH_SIZE
        }
        
        response = await self._make_request("/works", params)
        works = [self._parse_work(work_data) for work_data in response.get("results", [])]
        
        # Re-fetch works with truncated authors to get complete authorship data
        truncated_works = [work for work in works if work.is_authors_truncated]
        if truncated_works:
            logger.info(f"Re-fetching {len(truncated_works)} works with truncated authors")
            complete_works = await self._refetch_truncated_works(truncated_works)
            
            # Replace truncated works with complete ones
            work_lookup = {work.work_id: work for work in complete_works}
            for i, work in enumerate(works):
                if work.work_id in work_lookup:
                    works[i] = work_lookup[work.work_id]
        
        return works
    
    async def _refetch_truncated_works(self, truncated_works: List[WorkRaw]) -> List[WorkRaw]:
        """Re-fetch individual works to get complete author lists."""
        complete_works = []
        
        for work in truncated_works:
            try:
                # Use single work endpoint for complete data
                work_id = work.work_id.replace("https://openalex.org/", "")
                params = {
                    "select": "id,doi,display_name,publication_year,authorships,referenced_works,cited_by_count,topics,primary_topic"
                }
                
                response = await self._make_request(f"/works/{work_id}", params)
                complete_work = self._parse_work(response)
                complete_works.append(complete_work)
                
            except Exception as e:
                logger.warning(f"Failed to re-fetch work {work.work_id}: {e}")
                # Keep the original truncated work if re-fetch fails
                complete_works.append(work)
                
        return complete_works
    
    def _parse_work(self, work_data: Dict[str, Any]) -> WorkRaw:
        """Parse work data from OpenAlex API response."""
        from ..core.models import Topic
        
        authorships = []
        for auth_data in work_data.get("authorships", []):
            authorship = Authorship(
                author_id=auth_data.get("author", {}).get("id"),
                orcid=auth_data.get("author", {}).get("orcid"),
                raw_name=auth_data.get("raw_author_name", ""),
                author_position=AuthorPosition(auth_data.get("author_position", "middle")),
                is_corresponding=auth_data.get("is_corresponding", False),
                institution_ids=[inst.get("id", "") for inst in auth_data.get("institutions", [])]
            )
            authorships.append(authorship)
        
        # Parse topics
        topics = []
        for topic_data in work_data.get("topics", []):
            topic = Topic(
                topic_id=topic_data.get("id", ""),
                display_name=topic_data.get("display_name", ""),
                score=topic_data.get("score", 0.0)
            )
            topics.append(topic)
        
        # Parse primary topic
        primary_topic = None
        if "primary_topic" in work_data and work_data["primary_topic"]:
            primary_data = work_data["primary_topic"]
            primary_topic = Topic(
                topic_id=primary_data.get("id", ""),
                display_name=primary_data.get("display_name", ""),
                score=primary_data.get("score", 0.0)
            )
        
        # Detect author truncation
        is_authors_truncated = work_data.get("is_authors_truncated", False)
        # Also check if we have exactly 100 authors as an indicator
        if len(authorships) == 100:
            is_authors_truncated = True
        
        return WorkRaw(
            work_id=work_data.get("id", ""),
            doi=work_data.get("doi"),
            title=work_data.get("display_name", ""),
            year=work_data.get("publication_year"),
            cited_by_count=work_data.get("cited_by_count", 0),
            referenced_work_ids=work_data.get("referenced_works", []),
            authorships=authorships,
            topics=topics,
            primary_topic=primary_topic,
            is_authors_truncated=is_authors_truncated
        )
    
    async def get_authors_by_ids(self, author_ids: List[str]) -> Dict[str, AuthorMaster]:
        """Get authors by OpenAlex IDs in batches."""
        all_authors = {}
        
        # Process in batches of 100
        for i in range(0, len(author_ids), self.MAX_BATCH_SIZE):
            batch = author_ids[i:i + self.MAX_BATCH_SIZE]
            authors_batch = await self._get_authors_batch(batch)
            all_authors.update(authors_batch)
            
        return all_authors
    
    async def _get_authors_batch(self, author_ids: List[str]) -> Dict[str, AuthorMaster]:
        """Get a batch of authors."""
        # Clean author IDs
        clean_ids = []
        for author_id in author_ids:
            if author_id:
                clean_id = author_id.replace("https://openalex.org/", "")
                clean_ids.append(clean_id)
        
        if not clean_ids:
            return {}
            
        id_filter = "|".join(clean_ids)
        params = {
            "filter": f"openalex:{id_filter}",
            "select": "id,display_name,summary_stats,works_count,last_known_institutions",
            "per-page": self.MAX_BATCH_SIZE
        }
        
        response = await self._make_request("/authors", params)
        authors = {}
        
        for author_data in response.get("results", []):
            author = self._parse_author(author_data)
            authors[author.author_id] = author
            
        return authors
    
    def _parse_author(self, author_data: Dict[str, Any]) -> AuthorMaster:
        """Parse author data from OpenAlex API response."""
        summary_stats = None
        if "summary_stats" in author_data:
            stats_data = author_data["summary_stats"]
            summary_stats = SummaryStats(
                h_index=stats_data.get("h_index"),
                i10_index=stats_data.get("i10_index"), 
                two_yr_mean_citedness=stats_data.get("2yr_mean_citedness")
            )
        
        return AuthorMaster(
            author_id=author_data.get("id", ""),
            display_name=author_data.get("display_name", ""),
            orcid=author_data.get("orcid"),
            summary_stats=summary_stats,
            works_count=author_data.get("works_count", 0),
            last_known_institution_ids=[
                inst.get("id", "") for inst in author_data.get("last_known_institutions", [])
            ]
        )
    
    async def get_institutions_by_ids(self, institution_ids: List[str]) -> Dict[str, 'InstitutionMaster']:
        """Get institution data for multiple institution IDs.
        
        Args:
            institution_ids: List of OpenAlex institution IDs
            
        Returns:
            Dict mapping institution ID to InstitutionMaster
        """
        all_institutions = {}
        
        # Process in batches
        for i in range(0, len(institution_ids), self.MAX_BATCH_SIZE):
            batch = institution_ids[i:i + self.MAX_BATCH_SIZE]
            institutions_batch = await self._get_institutions_batch(batch)
            all_institutions.update(institutions_batch)
            
        return all_institutions
    
    async def _get_institutions_batch(self, institution_ids: List[str]) -> Dict[str, 'InstitutionMaster']:
        """Get a batch of institutions from OpenAlex API."""
        # Convert IDs to filter format
        filter_ids = "|".join([inst_id.replace("https://openalex.org/", "") for inst_id in institution_ids])
        
        params = {
            "filter": f"openalex:{filter_ids}",
            "select": "id,display_name,country_code,type,ror,homepage_url,works_count",
            "per-page": "100",
            "mailto": self.email
        }
        
        response = await self._make_request("/institutions", params)
        return self._parse_institutions_response(response)
    
    def _parse_institutions_response(self, response: Dict[str, Any]) -> Dict[str, 'InstitutionMaster']:
        """Parse institutions from OpenAlex API response."""
        from ..core.models import InstitutionMaster
        
        institutions = {}
        
        for inst_data in response.get("results", []):
            institution = self._parse_institution(inst_data)
            institutions[institution.institution_id] = institution
            
        return institutions
    
    def _parse_institution(self, inst_data: Dict[str, Any]) -> 'InstitutionMaster':
        """Parse institution data from OpenAlex API response."""
        from ..core.models import InstitutionMaster
        
        return InstitutionMaster(
            institution_id=inst_data.get("id", ""),
            display_name=inst_data.get("display_name", ""),
            country_code=inst_data.get("country_code"),
            type=inst_data.get("type"),
            ror=inst_data.get("ror"),
            homepage_url=inst_data.get("homepage_url"),
            works_count=inst_data.get("works_count", 0)
        )
    
    async def search_works_by_title(self, title: str, year: Optional[int] = None) -> List[WorkRaw]:
        """Search works by title with optional year filter."""
        params = {
            "search": title,
            "select": "id,doi,display_name,publication_year,authorships,referenced_works,cited_by_count,topics,primary_topic,is_authors_truncated",
            "per-page": 10  # Limit to top matches
        }
        
        if year:
            params["filter"] = f"publication_year:{year}"
        
        response = await self._make_request("/works", params)
        works = [self._parse_work(work_data) for work_data in response.get("results", [])]
        
        # Re-fetch works with truncated authors
        truncated_works = [work for work in works if work.is_authors_truncated]
        if truncated_works:
            logger.info(f"Re-fetching {len(truncated_works)} search results with truncated authors")
            complete_works = await self._refetch_truncated_works(truncated_works)
            
            # Replace truncated works with complete ones
            work_lookup = {work.work_id: work for work in complete_works}
            for i, work in enumerate(works):
                if work.work_id in work_lookup:
                    works[i] = work_lookup[work.work_id]
        
        return works