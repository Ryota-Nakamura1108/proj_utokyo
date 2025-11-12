"""OpenAI embedding generation."""

import logging
from typing import List, Optional
import numpy as np
from openai import OpenAI, AsyncOpenAI

from .models import Paper
from .config import config

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generates embeddings using OpenAI API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        batch_size: int = 20,
    ):
        """Initialize the embedding generator.

        Args:
            api_key: OpenAI API key
            model: Embedding model to use
            batch_size: Number of papers to process in one batch
        """
        self.api_key = api_key or config.openai_api_key
        self.model = model
        self.batch_size = batch_size

        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        self.client = AsyncOpenAI(api_key=self.api_key)
        logger.info(f"Initialized embedding generator with model: {self.model}")

    async def generate_embeddings(
        self, papers: List[Paper]
    ) -> List[np.ndarray]:
        """Generate embeddings for a list of papers.

        Args:
            papers: List of papers to embed

        Returns:
            List of embedding vectors (numpy arrays)
        """
        logger.info(f"Generating embeddings for {len(papers)} papers...")

        if not papers:
            return []

        embeddings = []

        # Process in batches
        for i in range(0, len(papers), self.batch_size):
            batch = papers[i : i + self.batch_size]
            batch_embeddings = await self._generate_batch_embeddings(batch)
            embeddings.extend(batch_embeddings)

            logger.info(
                f"Generated embeddings for batch {i // self.batch_size + 1}/{(len(papers) - 1) // self.batch_size + 1}"
            )

        logger.info(f"Successfully generated {len(embeddings)} embeddings")
        return embeddings

    async def _generate_batch_embeddings(
        self, papers: List[Paper]
    ) -> List[np.ndarray]:
        """Generate embeddings for a batch of papers.

        Args:
            papers: Batch of papers

        Returns:
            List of embedding vectors
        """
        # Prepare texts for embedding
        texts = []
        for paper in papers:
            text = paper.to_embedding_text()
            texts.append(text)

        try:
            # Call OpenAI API
            response = await self.client.embeddings.create(
                input=texts, model=self.model
            )

            # Extract embeddings
            embeddings = [
                np.array(data.embedding, dtype=np.float32)
                for data in response.data
            ]

            return embeddings

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")

            # Fallback: try with titles only
            logger.warning("Retrying with titles only...")
            return await self._generate_fallback_embeddings(papers)

    async def _generate_fallback_embeddings(
        self, papers: List[Paper]
    ) -> List[np.ndarray]:
        """Generate embeddings using only titles (fallback).

        Args:
            papers: Papers to embed

        Returns:
            List of embedding vectors
        """
        texts = [paper.title for paper in papers]

        try:
            response = await self.client.embeddings.create(
                input=texts, model=self.model
            )

            embeddings = [
                np.array(data.embedding, dtype=np.float32)
                for data in response.data
            ]

            logger.info("Successfully generated fallback embeddings")
            return embeddings

        except Exception as e:
            logger.error(f"Fallback embedding generation failed: {e}")
            # Return zero vectors as last resort
            return [np.zeros(1536, dtype=np.float32) for _ in papers]

    async def close(self):
        """Close the OpenAI client."""
        await self.client.close()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
