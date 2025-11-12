"""LLM-based keyword generation."""

import logging
import json
from typing import List, Optional
from openai import AsyncOpenAI

from .models import Cluster, Keyword
from .config import config

logger = logging.getLogger(__name__)


class KeywordGenerator:
    """Generates keywords using LLM."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        max_keywords: int = 10,
    ):
        """Initialize the keyword generator.

        Args:
            api_key: OpenAI API key
            model: LLM model to use
            max_keywords: Maximum number of keywords to extract
        """
        self.api_key = api_key or config.openai_api_key
        self.model = model
        self.max_keywords = max_keywords

        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        self.client = AsyncOpenAI(api_key=self.api_key)
        logger.info(f"Initialized keyword generator with model: {self.model}")

    async def generate_keywords(
        self,
        clusters: List[Cluster],
        researcher_name: str,
        total_papers: int,
        start_year: int,
        end_year: int,
    ) -> List[Keyword]:
        """Generate keywords from clusters.

        Args:
            clusters: List of clusters with representative papers
            researcher_name: Name of the researcher
            total_papers: Total number of papers analyzed
            start_year: Start year of analysis
            end_year: End year of analysis

        Returns:
            List of Keyword objects
        """
        logger.info(f"Generating keywords from {len(clusters)} clusters...")

        # Build cluster summaries
        cluster_summaries = self._build_cluster_summaries(clusters)

        # Create prompt
        prompt = self._create_prompt(
            researcher_name,
            total_papers,
            start_year,
            end_year,
            cluster_summaries,
        )

        # Call LLM
        keywords = await self._extract_keywords_from_llm(prompt)

        logger.info(f"Generated {len(keywords)} keywords")
        return keywords

    def _build_cluster_summaries(self, clusters: List[Cluster]) -> str:
        """Build text summaries of clusters.

        Args:
            clusters: List of clusters

        Returns:
            Formatted cluster summaries
        """
        summaries = []

        for cluster in clusters:
            summary = f"\nCluster {cluster.cluster_id + 1} ({len(cluster.papers)} papers):\n"

            # Add representative papers
            if cluster.representative_papers:
                summary += "Representative papers:\n"
                for i, paper in enumerate(
                    cluster.representative_papers[:3], 1
                ):
                    summary += f"  {i}. {paper.title} ({paper.publication_year})\n"
                    if paper.abstract:
                        # First 150 chars of abstract
                        abstract_preview = paper.abstract[:150]
                        summary += f"     Abstract: {abstract_preview}...\n"
                    summary += f"     Citations: {paper.cited_by_count}\n"

            summaries.append(summary)

        return "\n".join(summaries)

    def _create_prompt(
        self,
        researcher_name: str,
        total_papers: int,
        start_year: int,
        end_year: int,
        cluster_summaries: str,
    ) -> str:
        """Create the LLM prompt for keyword extraction.

        Args:
            researcher_name: Researcher name
            total_papers: Total papers
            start_year: Start year
            end_year: End year
            cluster_summaries: Formatted cluster summaries

        Returns:
            Complete prompt string
        """
        prompt = f"""You are analyzing research papers from a specific researcher. Based on the following clusters of research papers, extract the most relevant keywords that represent this researcher's expertise.

Researcher: {researcher_name}
Total Papers: {total_papers}
Time Period: {start_year} - {end_year}

CLUSTERS:
{cluster_summaries}

INSTRUCTIONS:
1. Extract {self.max_keywords} keywords that best represent the researcher's main research areas
2. Include both broad field keywords and specific technical terms
3. Prioritize keywords that appear across multiple clusters
4. Consider the citation impact of papers when weighting importance
5. Format: Return a JSON array of keyword objects with relevance scores (0.0 to 1.0)

EXAMPLE OUTPUT:
[
  {{"keyword": "deep learning", "relevance_score": 0.95, "cluster_ids": [0, 1, 2]}},
  {{"keyword": "neural networks", "relevance_score": 0.89, "cluster_ids": [0, 1]}},
  {{"keyword": "computer vision", "relevance_score": 0.85, "cluster_ids": [1, 2]}}
]

Extract keywords (return only the JSON array, no additional text):
"""
        return prompt

    async def _extract_keywords_from_llm(self, prompt: str) -> List[Keyword]:
        """Call LLM to extract keywords.

        Args:
            prompt: The prompt to send

        Returns:
            List of Keyword objects
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at analyzing research papers and extracting relevant keywords. Always respond with valid JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,  # Lower temperature for more consistent output
                max_tokens=1000,
            )

            content = response.choices[0].message.content

            # Parse JSON response
            keywords_data = json.loads(content)

            # Convert to Keyword objects
            keywords = []
            for i, kw_data in enumerate(keywords_data[: self.max_keywords]):
                keyword = Keyword(
                    keyword=kw_data.get("keyword", ""),
                    relevance_score=kw_data.get("relevance_score", 0.0),
                    cluster_ids=kw_data.get("cluster_ids", []),
                )
                keywords.append(keyword)

            return keywords

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"Response content: {content}")
            return []

        except Exception as e:
            logger.error(f"LLM keyword extraction failed: {e}")
            return []

    async def close(self):
        """Close the OpenAI client."""
        await self.client.close()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
