"""
Google Gemini API client implementation.
"""
import asyncio
import functools
import logging
import os
from typing import List

import google.generativeai as genai
from pydantic import SecretStr

from testteller.config import settings  # Ensure settings is imported
from testteller.constants import DEFAULT_GEMINI_GENERATION_MODEL, DEFAULT_GEMINI_EMBEDDING_MODEL
from testteller.utils.retry_helpers import api_retry_async, api_retry_sync

logger = logging.getLogger(__name__)


class GeminiClient:
    """Client for interacting with Google's Gemini API."""

    def __init__(self):
        """Initialize the Gemini client with API key from settings or environment."""
        self.api_key = self._get_api_key()
        genai.configure(api_key=self.api_key)

        # Get model names from settings
        try:
            if settings and settings.llm:
                self.generation_model = settings.llm.__dict__.get(
                    'generation_model', DEFAULT_GEMINI_GENERATION_MODEL)
                self.embedding_model = settings.llm.__dict__.get(
                    'embedding_model', DEFAULT_GEMINI_EMBEDDING_MODEL)
            else:
                self.generation_model = DEFAULT_GEMINI_GENERATION_MODEL
                self.embedding_model = DEFAULT_GEMINI_EMBEDDING_MODEL
        except Exception as e:
            logger.warning(
                "Could not get model names from settings: %s. Using defaults.", e)
            self.generation_model = DEFAULT_GEMINI_GENERATION_MODEL
            self.embedding_model = DEFAULT_GEMINI_EMBEDDING_MODEL

        self.model = genai.GenerativeModel(self.generation_model)
        logger.info("Initialized Gemini client with generation model '%s' and embedding model '%s'",
                    self.generation_model, self.embedding_model)

    def _get_api_key(self) -> str:
        """Get API key from settings or environment variables."""
        try:
            if settings and settings.api_keys:
                api_key = settings.api_keys.__dict__.get('google_api_key')
                if api_key and isinstance(api_key, SecretStr):
                    return api_key.get_secret_value()
        except Exception as e:
            logger.debug("Could not get API key from settings: %s", e)

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Google API key not found. Please set it in .env file as GOOGLE_API_KEY "
                "or provide it through settings configuration."
            )
        return api_key

    @api_retry_async
    async def get_embedding_async(self, text: str) -> List[float]:
        """
        Get embeddings for text asynchronously.

        Args:
            text: Text to get embeddings for

        Returns:
            List of embedding values
        """
        if not text or not text.strip():
            logger.warning(
                "Empty text provided for embedding, returning None.")
            return None
        try:
            loop = asyncio.get_running_loop()
            func_to_run = functools.partial(
                genai.embed_content,
                model=self.embedding_model,
                content=text,
                task_type="retrieval_document"
            )
            result = await loop.run_in_executor(None, func_to_run)
            return result['embedding']
        except Exception as e:
            logger.error(
                "Error generating embedding for text: '%s...': %s", text[:50], e, exc_info=True)
            return None

    @api_retry_sync
    def get_embedding_sync(self, text: str) -> List[float]:
        """
        Get embeddings for text synchronously.

        Args:
            text: Text to get embeddings for

        Returns:
            List of embedding values
        """
        if not text or not text.strip():
            logger.warning(
                "Empty text provided for sync embedding, returning None.")
            return None
        try:
            result = genai.embed_content(
                model=self.embedding_model,
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            logger.error(
                "Error generating sync embedding for text: '%s...': %s", text[:50], e, exc_info=True)
            return None

    async def get_embeddings_async(self, texts: list[str]) -> list[list[float] | None]:
        tasks = [self.get_embedding_async(text_chunk) for text_chunk in texts]
        embeddings = await asyncio.gather(*tasks, return_exceptions=True)

        processed_embeddings = []
        for i, emb_or_exc in enumerate(embeddings):
            if isinstance(emb_or_exc, Exception):
                logger.error(
                    "Failed to get embedding for text chunk %d after retries: %s", i, emb_or_exc)
                processed_embeddings.append(None)
            else:
                processed_embeddings.append(emb_or_exc)
        return processed_embeddings

    def get_embeddings_sync(self, texts: list[str]) -> list[list[float] | None]:
        embeddings = []
        for text_chunk in texts:
            emb = self.get_embedding_sync(text_chunk)
            embeddings.append(emb)
        return embeddings

    @api_retry_async
    async def generate_text_async(self, prompt: str) -> str:
        """
        Generate text using the Gemini model asynchronously.

        Args:
            prompt: The input prompt for text generation

        Returns:
            Generated text response
        """
        try:
            response = await self.model.generate_content_async(prompt)
            return response.text
        except Exception as e:
            logger.error("Error generating text with Gemini async: %s", e)
            raise

    @api_retry_sync
    def generate_text(self, prompt: str) -> str:
        """
        Generate text using the Gemini model.

        Args:
            prompt: The input prompt for text generation

        Returns:
            Generated text response
        """
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error("Error generating text with Gemini: %s", e)
            raise
