"""
Llama client implementation using Ollama.
"""
import asyncio
import logging
import os
from typing import List

import httpx
from pydantic import SecretStr

from testteller.config import settings
from testteller.constants import DEFAULT_LLAMA_GENERATION_MODEL, DEFAULT_LLAMA_EMBEDDING_MODEL, DEFAULT_OLLAMA_BASE_URL
from testteller.utils.retry_helpers import api_retry_async, api_retry_sync

logger = logging.getLogger(__name__)


class LlamaClient:
    """Client for interacting with Llama models via Ollama."""

    def __init__(self):
        """Initialize the Llama client with Ollama configuration."""
        self.base_url = self._get_ollama_base_url()

        # Get model names from settings
        try:
            if settings and settings.llm:
                self.generation_model = settings.llm.__dict__.get(
                    'llama_generation_model', DEFAULT_LLAMA_GENERATION_MODEL)
                self.embedding_model = settings.llm.__dict__.get(
                    'llama_embedding_model', DEFAULT_LLAMA_EMBEDDING_MODEL)
            else:
                self.generation_model = DEFAULT_LLAMA_GENERATION_MODEL
                self.embedding_model = DEFAULT_LLAMA_EMBEDDING_MODEL
        except Exception as e:
            logger.warning(
                "Could not get model names from settings: %s. Using defaults.", e)
            self.generation_model = DEFAULT_LLAMA_GENERATION_MODEL
            self.embedding_model = DEFAULT_LLAMA_EMBEDDING_MODEL

        logger.info("Initialized Llama client with generation model '%s', embedding model '%s', and Ollama URL '%s'",
                    self.generation_model, self.embedding_model, self.base_url)

    def _get_ollama_base_url(self) -> str:
        """Get Ollama base URL from settings or environment variables."""
        try:
            if settings and settings.llm:
                base_url = settings.llm.__dict__.get('ollama_base_url')
                if base_url:
                    return base_url
        except Exception as e:
            logger.debug("Could not get Ollama base URL from settings: %s", e)

        base_url = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
        return base_url

    @api_retry_async
    async def get_embedding_async(self, text: str) -> List[float]:
        """
        Get embeddings for text asynchronously using Ollama.

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
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/embeddings",
                    json={
                        "model": self.embedding_model,
                        "prompt": text
                    }
                )
                response.raise_for_status()
                result = response.json()
                return result["embedding"]
        except Exception as e:
            logger.error(
                "Error generating embedding for text: '%s...': %s", text[:50], e, exc_info=True)
            return None

    @api_retry_sync
    def get_embedding_sync(self, text: str) -> List[float]:
        """
        Get embeddings for text synchronously using Ollama.

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
            with httpx.Client(timeout=60.0) as client:
                response = client.post(
                    f"{self.base_url}/api/embeddings",
                    json={
                        "model": self.embedding_model,
                        "prompt": text
                    }
                )
                response.raise_for_status()
                result = response.json()
                return result["embedding"]
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
        Generate text using the Llama model asynchronously via Ollama.

        Args:
            prompt: The input prompt for text generation

        Returns:
            Generated text response
        """
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.generation_model,
                        "prompt": prompt,
                        "stream": False
                    }
                )
                response.raise_for_status()
                result = response.json()
                return result["response"]
        except Exception as e:
            logger.error("Error generating text with Llama async: %s", e)
            raise

    @api_retry_sync
    def generate_text(self, prompt: str) -> str:
        """
        Generate text using the Llama model via Ollama.

        Args:
            prompt: The input prompt for text generation

        Returns:
            Generated text response
        """
        try:
            with httpx.Client(timeout=120.0) as client:
                response = client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.generation_model,
                        "prompt": prompt,
                        "stream": False
                    }
                )
                response.raise_for_status()
                result = response.json()
                return result["response"]
        except Exception as e:
            logger.error("Error generating text with Llama: %s", e)
            raise

    async def check_model_availability(self) -> bool:
        """
        Check if the configured models are available in Ollama.

        Returns:
            True if models are available, False otherwise
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                models = response.json()

                available_models = [model["name"]
                                    for model in models.get("models", [])]

                generation_available = any(
                    self.generation_model in model for model in available_models)
                embedding_available = any(
                    self.embedding_model in model for model in available_models)

                if not generation_available:
                    logger.warning(
                        "Generation model '%s' not found in Ollama", self.generation_model)
                if not embedding_available:
                    logger.warning(
                        "Embedding model '%s' not found in Ollama", self.embedding_model)

                return generation_available and embedding_available
        except Exception as e:
            logger.error("Error checking Ollama model availability: %s", e)
            return False
