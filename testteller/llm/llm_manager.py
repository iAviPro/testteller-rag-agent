"""
LLM Manager for unified access to different LLM providers.
"""
import logging
import os
from typing import List, Union, Optional

from testteller.config import settings
from testteller.constants import SUPPORTED_LLM_PROVIDERS, DEFAULT_LLM_PROVIDER
from testteller.llm.gemini_client import GeminiClient
from testteller.llm.openai_client import OpenAIClient
from testteller.llm.claude_client import ClaudeClient
from testteller.llm.llama_client import LlamaClient

logger = logging.getLogger(__name__)


class LLMManager:
    """Manager class that provides unified access to different LLM providers."""

    def __init__(self, provider: Optional[str] = None):
        """
        Initialize the LLM Manager.

        Args:
            provider: The LLM provider to use ('gemini', 'openai', 'claude', 'llama')
                     If None, will try to get from settings or environment
        """
        self.provider = self._get_provider(provider)
        self.client = self._initialize_client()

        logger.info("Initialized LLM Manager with provider: %s", self.provider)

    def _get_provider(self, provider: Optional[str] = None) -> str:
        """Get the LLM provider to use."""
        if provider:
            if provider not in SUPPORTED_LLM_PROVIDERS:
                raise ValueError(
                    f"Unsupported LLM provider: {provider}. Supported providers: {SUPPORTED_LLM_PROVIDERS}")
            return provider

        # Try to get from settings
        try:
            if settings and settings.llm:
                settings_provider = settings.llm.__dict__.get('provider')
                if settings_provider and settings_provider in SUPPORTED_LLM_PROVIDERS:
                    return settings_provider
        except Exception as e:
            logger.debug("Could not get LLM provider from settings: %s", e)

        # Try to get from environment
        env_provider = os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER).lower()
        if env_provider in SUPPORTED_LLM_PROVIDERS:
            return env_provider

        # Default fallback
        return DEFAULT_LLM_PROVIDER

    def _initialize_client(self) -> Union[GeminiClient, OpenAIClient, ClaudeClient, LlamaClient]:
        """Initialize the appropriate LLM client based on the provider."""
        try:
            if self.provider == "gemini":
                return GeminiClient()
            elif self.provider == "openai":
                return OpenAIClient()
            elif self.provider == "claude":
                return ClaudeClient()
            elif self.provider == "llama":
                return LlamaClient()
            else:
                raise ValueError(f"Unsupported LLM provider: {self.provider}")
        except Exception as e:
            # Check if it's an API key error and provide helpful guidance
            error_msg = str(e).lower()
            if "api key" in error_msg or "authentication" in error_msg:
                self._handle_api_key_error(e)
            raise

    def _handle_api_key_error(self, original_error: Exception):
        """Handle API key errors with helpful messages."""
        provider_key_map = {
            "gemini": "GOOGLE_API_KEY",
            "openai": "OPENAI_API_KEY",
            "claude": "CLAUDE_API_KEY",
            "llama": "No API key required (uses local Ollama)"
        }

        required_key = provider_key_map.get(self.provider, "API_KEY")

        if self.provider == "llama":
            error_message = (
                f"Failed to initialize {self.provider} client. "
                "Make sure Ollama is running locally at http://localhost:11434 "
                "and the required models are installed."
            )
        else:
            error_message = (
                f"Failed to initialize {self.provider} client due to missing or invalid API key. "
                f"Please set {required_key} in your .env file or run 'testteller configure' "
                "to set up your configuration."
            )

        logger.error(error_message)
        raise ValueError(error_message) from original_error

    async def get_embedding_async(self, text: str) -> List[float]:
        """Get embeddings for text asynchronously."""
        return await self.client.get_embedding_async(text)

    def get_embedding_sync(self, text: str) -> List[float]:
        """Get embeddings for text synchronously."""
        return self.client.get_embedding_sync(text)

    async def get_embeddings_async(self, texts: List[str]) -> List[List[float] | None]:
        """Get embeddings for multiple texts asynchronously."""
        return await self.client.get_embeddings_async(texts)

    def get_embeddings_sync(self, texts: List[str]) -> List[List[float] | None]:
        """Get embeddings for multiple texts synchronously."""
        return self.client.get_embeddings_sync(texts)

    async def generate_text_async(self, prompt: str) -> str:
        """Generate text asynchronously."""
        return await self.client.generate_text_async(prompt)

    def generate_text(self, prompt: str) -> str:
        """Generate text synchronously."""
        return self.client.generate_text(prompt)

    def get_provider_info(self) -> dict:
        """Get information about the current provider."""
        info = {
            "provider": self.provider,
            "generation_model": getattr(self.client, 'generation_model', 'Unknown'),
            "embedding_model": getattr(self.client, 'embedding_model', 'Unknown')
        }

        # Add provider-specific info
        if self.provider == "llama":
            info["ollama_url"] = getattr(self.client, 'base_url', 'Unknown')

        return info

    def get_current_provider(self) -> str:
        """Get the name of the currently active LLM provider."""
        return self.provider.lower()

    @classmethod
    def get_supported_providers(cls) -> List[str]:
        """Get list of supported LLM providers."""
        return SUPPORTED_LLM_PROVIDERS.copy()

    @classmethod
    def validate_provider_config(cls, provider: str) -> tuple[bool, str]:
        """
        Validate if a provider is properly configured.

        Args:
            provider: The provider to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if provider not in SUPPORTED_LLM_PROVIDERS:
            return False, f"Unsupported provider: {provider}"

        try:
            # Try to initialize the client to check configuration
            if provider == "gemini":
                GeminiClient()
            elif provider == "openai":
                OpenAIClient()
            elif provider == "claude":
                ClaudeClient()
            elif provider == "llama":
                LlamaClient()

            return True, "Configuration valid"
        except Exception as e:
            return False, str(e)
