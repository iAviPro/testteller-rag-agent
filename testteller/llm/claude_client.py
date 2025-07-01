"""
Anthropic Claude API client implementation.
"""
import asyncio
import logging
import os
from typing import List

import anthropic
from pydantic import SecretStr

from testteller.config import settings
from testteller.constants import DEFAULT_CLAUDE_GENERATION_MODEL, DEFAULT_OPENAI_EMBEDDING_MODEL
from testteller.utils.retry_helpers import api_retry_async, api_retry_sync

logger = logging.getLogger(__name__)


class ClaudeClient:
    """Client for interacting with Anthropic's Claude API."""

    def __init__(self):
        """Initialize the Claude client with API key from settings or environment."""
        self.api_key = self._get_api_key()
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.async_client = anthropic.AsyncAnthropic(api_key=self.api_key)

        # Get model names from settings
        try:
            if settings and settings.llm:
                self.generation_model = settings.llm.__dict__.get(
                    'claude_generation_model', DEFAULT_CLAUDE_GENERATION_MODEL)
                # Claude doesn't provide embeddings, so we'll use OpenAI for embeddings
                self.embedding_model = settings.llm.__dict__.get(
                    'openai_embedding_model', DEFAULT_OPENAI_EMBEDDING_MODEL)
            else:
                self.generation_model = DEFAULT_CLAUDE_GENERATION_MODEL
                self.embedding_model = DEFAULT_OPENAI_EMBEDDING_MODEL
        except Exception as e:
            logger.warning(
                "Could not get model names from settings: %s. Using defaults.", e)
            self.generation_model = DEFAULT_CLAUDE_GENERATION_MODEL
            self.embedding_model = DEFAULT_OPENAI_EMBEDDING_MODEL

        # Initialize OpenAI client for embeddings (Claude doesn't provide embeddings)
        self._openai_client = None
        self._openai_async_client = None

        logger.info("Initialized Claude client with generation model '%s' and embedding model '%s' (via OpenAI)",
                    self.generation_model, self.embedding_model)

    def _get_api_key(self) -> str:
        """Get API key from settings or environment variables."""
        try:
            if settings and settings.api_keys:
                api_key = settings.api_keys.__dict__.get('claude_api_key')
                if api_key and isinstance(api_key, SecretStr):
                    return api_key.get_secret_value()
        except Exception as e:
            logger.debug("Could not get API key from settings: %s", e)

        api_key = os.getenv("CLAUDE_API_KEY")
        if not api_key:
            raise ValueError(
                "Claude API key not found. Please set it in .env file as CLAUDE_API_KEY "
                "or provide it through settings configuration."
            )
        return api_key

    def _get_openai_client(self):
        """Lazy initialization of OpenAI client for embeddings."""
        if self._openai_client is None:
            import openai
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError(
                    "OpenAI API key is required for embeddings when using Claude. "
                    "Please set OPENAI_API_KEY in your .env file."
                )
            self._openai_client = openai.OpenAI(api_key=openai_api_key)
        return self._openai_client

    def _get_openai_async_client(self):
        """Lazy initialization of OpenAI async client for embeddings."""
        if self._openai_async_client is None:
            import openai
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError(
                    "OpenAI API key is required for embeddings when using Claude. "
                    "Please set OPENAI_API_KEY in your .env file."
                )
            self._openai_async_client = openai.AsyncOpenAI(
                api_key=openai_api_key)
        return self._openai_async_client

    @api_retry_async
    async def get_embedding_async(self, text: str) -> List[float]:
        """
        Get embeddings for text asynchronously using OpenAI (Claude doesn't provide embeddings).

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
            openai_client = self._get_openai_async_client()
            response = await openai_client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(
                "Error generating embedding for text: '%s...': %s", text[:50], e, exc_info=True)
            return None

    @api_retry_sync
    def get_embedding_sync(self, text: str) -> List[float]:
        """
        Get embeddings for text synchronously using OpenAI (Claude doesn't provide embeddings).

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
            openai_client = self._get_openai_client()
            response = openai_client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
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
        Generate text using the Claude model asynchronously.

        Args:
            prompt: The input prompt for text generation

        Returns:
            Generated text response
        """
        try:
            response = await self.async_client.messages.create(
                model=self.generation_model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error("Error generating text with Claude async: %s", e)
            raise

    @api_retry_sync
    def generate_text(self, prompt: str) -> str:
        """
        Generate text using the Claude model.

        Args:
            prompt: The input prompt for text generation

        Returns:
            Generated text response
        """
        try:
            response = self.client.messages.create(
                model=self.generation_model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error("Error generating text with Claude: %s", e)
            raise
