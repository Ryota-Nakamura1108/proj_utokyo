"""Input parsing and ID normalization for the Central Researcher system."""

import re
import logging
from typing import List, Tuple, Optional, Dict, Any
from urllib.parse import unquote

from .models import InputPaper, WorkRaw
from ..api.openalex_client import OpenAlexClient

logger = logging.getLogger(__name__)


class InputParser:
    """Handles parsing and normalization of input paper identifiers."""
    
    DOI_PATTERNS = [
        r'10\.\d{4,}[^\s]+',  # Standard DOI pattern
        r'doi:10\.\d{4,}[^\s]+',  # DOI with prefix
        r'https?://(?:dx\.)?doi\.org/10\.\d{4,}[^\s]+',  # DOI URLs
    ]
    
    OPENALEX_PATTERNS = [
        r'W\d{10}',  # OpenAlex Work ID
        r'https://openalex\.org/W\d{10}',  # OpenAlex Work URL
    ]
    
    PMID_PATTERNS = [
        r'pmid:(\d+)',  # PMID with prefix
        r'PMID:(\d+)',  # PMID with uppercase prefix
    ]
    
    def __init__(self, client: OpenAlexClient):
        """Initialize parser with OpenAlex client."""
        self.client = client
        
    async def normalize_and_resolve_papers(self, papers: List[InputPaper]) -> List[WorkRaw]:
        """Normalize paper identifiers and resolve to WorkRaw objects."""
        logger.info(f"Normalizing and resolving {len(papers)} input papers")
        
        # Group papers by identifier type for efficient batch processing
        doi_papers = []
        openalex_papers = []
        pmid_papers = []
        title_papers = []
        
        for paper in papers:
            normalized = self._normalize_single_paper(paper)
            
            if normalized.doi:
                doi_papers.append(normalized.doi)
            elif normalized.openalex_id:
                openalex_papers.append(normalized.openalex_id)
            elif normalized.pmid:
                # Convert PMID to search query format
                pmid_papers.append(f"pmid:{normalized.pmid}")
            elif normalized.title:
                title_papers.append(normalized)
        
        # Resolve identifiers to works
        resolved_works = []
        
        # Process DOIs and OpenAlex IDs together (they can use the same batch API)
        all_identifiers = doi_papers + openalex_papers + pmid_papers
        if all_identifiers:
            works = await self.client.get_works_by_ids(all_identifiers)
            resolved_works.extend(works)
        
        # Process title-only papers individually (requires search)
        for title_paper in title_papers:
            title_works = await self._resolve_title_paper(title_paper)
            resolved_works.extend(title_works)
        
        logger.info(f"Successfully resolved {len(resolved_works)} papers")
        return resolved_works
    
    def _normalize_single_paper(self, paper: InputPaper) -> InputPaper:
        """Normalize a single paper's identifiers."""
        # Create with dummy title to avoid validation, we'll clean this up later
        normalized = InputPaper(title="temp")
        
        # Normalize DOI
        if paper.doi:
            normalized.doi = self._normalize_doi(paper.doi)
        
        # Normalize OpenAlex ID
        if paper.openalex_id:
            normalized.openalex_id = self._normalize_openalex_id(paper.openalex_id)
        
        # Normalize PMID
        if paper.pmid:
            normalized.pmid = self._normalize_pmid(paper.pmid)
        
        # Keep title as-is but clean whitespace
        if paper.title:
            normalized.title = paper.title.strip()
        
        # If multiple identifiers provided, prioritize in order: OpenAlex ID > DOI > PMID > Title
        if normalized.openalex_id:
            normalized.doi = None
            normalized.pmid = None
            normalized.title = None
        elif normalized.doi:
            normalized.pmid = None
            normalized.title = None
        elif normalized.pmid:
            normalized.title = None
        else:
            # If no other identifier, keep title but remove dummy if it's still there
            if normalized.title == "temp" and paper.title:
                normalized.title = paper.title.strip()
            elif normalized.title == "temp":
                # No valid identifiers at all
                normalized.title = None
        
        return normalized
    
    def _normalize_doi(self, doi: str) -> str:
        """Normalize DOI to standard format."""
        # Remove URL prefixes and clean up
        doi = doi.strip()
        doi = re.sub(r'^(https?://(?:dx\.)?doi\.org/)', '', doi, flags=re.IGNORECASE)
        doi = re.sub(r'^doi:', '', doi, flags=re.IGNORECASE)
        
        # URL decode if needed
        doi = unquote(doi)
        
        # Convert to lowercase (DOIs are case-insensitive)
        doi = doi.lower()
        
        # Validate DOI format
        if not re.match(r'^10\.\d{4,}/', doi):
            logger.warning(f"Invalid DOI format: {doi}")
            return ""
        
        return doi
    
    def _normalize_openalex_id(self, openalex_id: str) -> str:
        """Normalize OpenAlex ID to standard format."""
        openalex_id = openalex_id.strip()
        
        # Remove URL prefix if present
        openalex_id = re.sub(r'^https://openalex\.org/', '', openalex_id)
        
        # Validate format
        if not re.match(r'^W\d{10}$', openalex_id):
            logger.warning(f"Invalid OpenAlex ID format: {openalex_id}")
            return ""
        
        return openalex_id
    
    def _normalize_pmid(self, pmid: str) -> str:
        """Normalize PMID to standard format."""
        pmid = pmid.strip()
        
        # Extract numeric part
        match = re.search(r'(\d+)', pmid)
        if not match:
            logger.warning(f"Invalid PMID format: {pmid}")
            return ""
        
        return match.group(1)
    
    async def _resolve_title_paper(self, paper: InputPaper) -> List[WorkRaw]:
        """Resolve paper by title search with disambiguation."""
        if not paper.title:
            return []
        
        logger.debug(f"Searching for paper by title: {paper.title[:100]}...")
        
        # Search for papers by title
        search_results = await self.client.search_works_by_title(paper.title)
        
        if not search_results:
            logger.warning(f"No results found for title: {paper.title}")
            return []
        
        # Try to find exact title match
        exact_matches = []
        for work in search_results:
            if self._titles_match(paper.title, work.title):
                exact_matches.append(work)
        
        if exact_matches:
            logger.debug(f"Found {len(exact_matches)} exact title matches")
            return exact_matches
        
        # If no exact match, return the top result with a warning
        logger.warning(f"No exact title match found. Using top search result for: {paper.title}")
        return [search_results[0]]
    
    def _titles_match(self, title1: str, title2: str, threshold: float = 0.9) -> bool:
        """Check if two titles match with fuzzy comparison."""
        # Simple fuzzy matching - normalize and compare
        def normalize_title(title: str) -> str:
            # Convert to lowercase, remove extra whitespace and punctuation
            title = re.sub(r'[^\w\s]', ' ', title.lower())
            title = ' '.join(title.split())  # Normalize whitespace
            return title
        
        norm_title1 = normalize_title(title1)
        norm_title2 = normalize_title(title2)
        
        # Exact match
        if norm_title1 == norm_title2:
            return True
        
        # Jaccard similarity for fuzzy matching
        words1 = set(norm_title1.split())
        words2 = set(norm_title2.split())
        
        if not words1 or not words2:
            return False
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        jaccard = intersection / union
        
        return jaccard >= threshold
    
    def detect_identifier_type(self, identifier: str) -> Tuple[str, str]:
        """Detect the type of identifier and return (type, normalized_value)."""
        identifier = identifier.strip()
        
        # Check for OpenAlex ID
        for pattern in self.OPENALEX_PATTERNS:
            if re.search(pattern, identifier, re.IGNORECASE):
                return "openalex_id", self._normalize_openalex_id(identifier)
        
        # Check for DOI
        for pattern in self.DOI_PATTERNS:
            if re.search(pattern, identifier, re.IGNORECASE):
                return "doi", self._normalize_doi(identifier)
        
        # Check for PMID
        for pattern in self.PMID_PATTERNS:
            match = re.search(pattern, identifier, re.IGNORECASE)
            if match:
                return "pmid", match.group(1)
        
        # Default to title
        return "title", identifier