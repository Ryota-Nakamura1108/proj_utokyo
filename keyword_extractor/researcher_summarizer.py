"""LLM-based researcher profile summarization."""

import logging
from typing import List, Optional, Dict, Any
from openai import AsyncOpenAI

from .models import Cluster, Keyword
from .config import config

logger = logging.getLogger(__name__)


class ResearcherSummarizer:
    """Generates comprehensive researcher summaries using LLM."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
    ):
        """Initialize the summarizer.

        Args:
            api_key: OpenAI API key
            model: LLM model to use
        """
        self.api_key = api_key or config.openai_api_key
        self.model = model

        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        self.client = AsyncOpenAI(api_key=self.api_key)
        logger.info(f"Initialized researcher summarizer with model: {self.model}")

    async def generate_summary(
        self,
        researcher_name: str,
        keywords: List[Keyword],
        clusters: List[Cluster],
        statistics: Dict[str, Any],
        analysis_period: Dict[str, int],
    ) -> str:
        """Generate a comprehensive researcher summary.

        Args:
            researcher_name: Name of the researcher
            keywords: Extracted keywords
            clusters: Paper clusters with representatives
            statistics: Research statistics
            analysis_period: Analysis time period

        Returns:
            English summary text
        """
        logger.info(f"Generating summary for {researcher_name}...")

        # Build the prompt
        prompt = self._create_summary_prompt(
            researcher_name,
            keywords,
            clusters,
            statistics,
            analysis_period,
        )

        # Call LLM
        summary = await self._generate_summary_from_llm(prompt)

        logger.info("Summary generation complete")
        return summary

    def _create_summary_prompt(
        self,
        researcher_name: str,
        keywords: List[Keyword],
        clusters: List[Cluster],
        statistics: Dict[str, Any],
        analysis_period: Dict[str, int],
    ) -> str:
        """Create the prompt for summary generation.

        Args:
            researcher_name: Researcher name
            keywords: Keywords
            clusters: Clusters
            statistics: Statistics
            analysis_period: Time period

        Returns:
            Complete prompt string
        """
        # Format keywords
        keywords_text = self._format_keywords(keywords)

        # Format clusters and representative papers
        clusters_text = self._format_clusters(clusters)

        # Format statistics
        stats_text = self._format_statistics(statistics, analysis_period)

        prompt = f"""You are an expert research analyst tasked with writing a comprehensive professional summary of a researcher's academic profile and contributions.

RESEARCHER INFORMATION:
{stats_text}

RESEARCH KEYWORDS:
{keywords_text}

RESEARCH CLUSTERS AND REPRESENTATIVE PAPERS:
{clusters_text}

TASK:
Write a comprehensive, professional summary (3-5 paragraphs) of {researcher_name}'s research profile in Japanese. The summary should:

1. **Opening paragraph**: Provide an overview of their primary research areas and expertise based on the keywords and paper clusters
2. **Main research themes**: Describe 2-3 major research themes or directions, referencing specific representative papers and their contributions
3. **Research impact**: Discuss the significance and impact of their work based on citation patterns and publication venues
4. **Research evolution** (if applicable): Note any shifts or expansion in research focus across different clusters
5. **Concluding assessment**: Summarize their overall standing and contribution to their field(s)

GUIDELINES:
- Write in a professional, objective academic tone
- Reference specific paper titles when highlighting key contributions
- Use the keywords naturally throughout the text
- Be concrete and specific rather than generic
- Maintain factual accuracy based only on the provided information
- Structure clearly with distinct paragraphs
- Target length: 250-350 words

Generate the researcher summary:
"""
        return prompt

    def _format_keywords(self, keywords: List[Keyword]) -> str:
        """Format keywords for the prompt.

        Args:
            keywords: List of keywords

        Returns:
            Formatted keywords text
        """
        if not keywords:
            return "No keywords available."

        lines = []
        for i, kw in enumerate(keywords[:10], 1):
            cluster_info = (
                f" (appears in clusters {', '.join(map(str, kw.cluster_ids))})"
                if kw.cluster_ids
                else ""
            )
            lines.append(
                f"  {i}. {kw.keyword} (relevance: {kw.relevance_score:.2f}){cluster_info}"
            )

        return "\n".join(lines)

    def _format_clusters(self, clusters: List[Cluster]) -> str:
        """Format cluster information for the prompt.

        Args:
            clusters: List of clusters

        Returns:
            Formatted clusters text
        """
        if not clusters:
            return "No cluster information available."

        lines = []
        for cluster in clusters:
            lines.append(f"\nCluster {cluster.cluster_id + 1} ({len(cluster.papers)} papers):")

            if cluster.theme:
                lines.append(f"  Theme: {cluster.theme}")

            if cluster.representative_papers:
                lines.append("  Representative Papers:")
                for i, paper in enumerate(
                    cluster.representative_papers[:3], 1
                ):
                    lines.append(f"    {i}. '{paper.title}' ({paper.publication_year})")

                    # Add abstract preview if available
                    if paper.abstract:
                        abstract_preview = paper.abstract[:150].replace("\n", " ")
                        lines.append(f"       Abstract: {abstract_preview}...")

                    lines.append(
                        f"       Journal: {paper.journal_name or 'N/A'}, "
                        f"Citations: {paper.cited_by_count}"
                    )

        return "\n".join(lines)

    def _format_statistics(
        self, statistics: Dict[str, Any], analysis_period: Dict[str, int]
    ) -> str:
        """Format statistics for the prompt.

        Args:
            statistics: Statistics dictionary
            analysis_period: Analysis period

        Returns:
            Formatted statistics text
        """
        lines = [
            f"Researcher: {statistics.get('researcher_name', 'Unknown')}",
            f"Analysis Period: {analysis_period.get('start_year', 'N/A')} - {analysis_period.get('end_year', 'N/A')}",
            f"Total Papers Analyzed: {statistics.get('total_papers', 0)}",
        ]

        if statistics.get("papers_as_first_author"):
            lines.append(
                f"Papers as First Author: {statistics['papers_as_first_author']}"
            )

        if statistics.get("papers_as_last_author"):
            lines.append(
                f"Papers as Last Author: {statistics['papers_as_last_author']}"
            )

        if statistics.get("total_citations"):
            lines.append(f"Total Citations: {statistics['total_citations']}")

        if statistics.get("average_citations"):
            lines.append(
                f"Average Citations per Paper: {statistics['average_citations']:.1f}"
            )

        year_range = statistics.get("year_range")
        if year_range:
            lines.append(
                f"Publication Years: {year_range['start']} - {year_range['end']}"
            )

        return "\n".join(lines)

    async def _generate_summary_from_llm(self, prompt: str) -> str:
        """Call LLM to generate summary.

        Args:
            prompt: The prompt to send

        Returns:
            Generated summary text
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert research analyst who writes clear, professional, and insightful summaries of researchers' academic profiles. Your summaries are factual, well-structured, and highlight key contributions.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,  # Balanced for coherent but varied text
                max_tokens=800,  # Enough for 250-350 word summary
            )

            summary = response.choices[0].message.content.strip()
            return summary

        except Exception as e:
            logger.error(f"LLM summary generation failed: {e}")
            return self._create_fallback_summary(prompt)

    def _create_fallback_summary(self, prompt: str) -> str:
        """Create a basic fallback summary if LLM fails.

        Args:
            prompt: Original prompt (for context extraction)

        Returns:
            Basic summary text
        """
        logger.warning("Using fallback summary generation")

        # Extract researcher name from prompt
        lines = prompt.split("\n")
        researcher_line = [l for l in lines if l.startswith("Researcher:")][0]
        researcher_name = researcher_line.split(":", 1)[1].strip()

        return f"""This is a summary of {researcher_name}'s research profile based on their recent publications. The analysis covers their primary research areas and key contributions. For a detailed breakdown, please refer to the keywords and cluster information in the full report.

Note: Detailed summary generation encountered an error. Please try again or contact support."""

    async def close(self):
        """Close the OpenAI client."""
        await self.client.close()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
