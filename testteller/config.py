# config.py
"""
Configuration module for TestTeller RAG Agent.
Handles loading and validation of settings from environment variables and .env files.
"""

import os
from typing import Optional, List
from pydantic.v1 import BaseSettings, Field, validator

# Import version from the single source of truth
try:
    from ._version import __version__ as APP_VERSION
except ImportError:
    from .constants import FALLBACK_VERSION
    APP_VERSION = FALLBACK_VERSION  # Use fallback from constants

from .constants import (
    APP_NAME,
    DEFAULT_LOG_LEVEL, DEFAULT_LOG_FORMAT,
    DEFAULT_CHROMA_HOST, DEFAULT_CHROMA_PORT, DEFAULT_CHROMA_USE_REMOTE,
    DEFAULT_CHROMA_PERSIST_DIRECTORY, DEFAULT_COLLECTION_NAME,
    DEFAULT_LLM_PROVIDER, SUPPORTED_LLM_PROVIDERS,
    DEFAULT_GEMINI_EMBEDDING_MODEL, DEFAULT_GEMINI_GENERATION_MODEL,
    DEFAULT_OPENAI_EMBEDDING_MODEL, DEFAULT_OPENAI_GENERATION_MODEL,
    DEFAULT_CLAUDE_GENERATION_MODEL,
    DEFAULT_LLAMA_EMBEDDING_MODEL, DEFAULT_LLAMA_GENERATION_MODEL, DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CODE_EXTENSIONS, DEFAULT_TEMP_CLONE_DIR,
    DEFAULT_OUTPUT_FILE,
    DEFAULT_API_RETRY_ATTEMPTS, DEFAULT_API_RETRY_WAIT_SECONDS,
    ENV_GOOGLE_API_KEY, ENV_OPENAI_API_KEY, ENV_CLAUDE_API_KEY, ENV_GITHUB_TOKEN,
    ENV_LLM_PROVIDER, ENV_LOG_LEVEL,
    ENV_CHROMA_DB_HOST, ENV_CHROMA_DB_PORT, ENV_CHROMA_DB_USE_REMOTE,
    ENV_CHROMA_DB_PERSIST_DIRECTORY, ENV_DEFAULT_COLLECTION_NAME,
    ENV_GEMINI_EMBEDDING_MODEL, ENV_GEMINI_GENERATION_MODEL,
    ENV_OPENAI_EMBEDDING_MODEL, ENV_OPENAI_GENERATION_MODEL,
    ENV_CLAUDE_GENERATION_MODEL,
    ENV_LLAMA_EMBEDDING_MODEL, ENV_LLAMA_GENERATION_MODEL, ENV_OLLAMA_BASE_URL,
    ENV_CHUNK_SIZE, ENV_CHUNK_OVERLAP,
    ENV_CODE_EXTENSIONS, ENV_TEMP_CLONE_DIR_BASE,
    ENV_OUTPUT_FILE_PATH,
    ENV_API_RETRY_ATTEMPTS, ENV_API_RETRY_WAIT_SECONDS
)


def load_env():
    """Load environment variables from .env file."""
    env_path = os.path.join(os.getcwd(), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip().strip(
                        '"').strip("'")


# Load .env file
load_env()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class CommonSettings(BaseSettings):
    """Common application settings."""
    APP_NAME: str = APP_NAME
    APP_VERSION: str = APP_VERSION

    class Config:
        env_file = os.path.join(os.getcwd(), '.env')
        env_file_encoding = 'utf-8'
        extra = 'ignore'
        case_sensitive = False


class ApiKeysSettings(BaseSettings):
    """API key configurations."""
    class Config:
        extra = 'ignore'
        case_sensitive = False

    google_api_key: Optional[str] = Field(
        None,
        env=ENV_GOOGLE_API_KEY,
        description="Google Gemini API key"
    )

    openai_api_key: Optional[str] = Field(
        None,
        env=ENV_OPENAI_API_KEY,
        description="OpenAI API key"
    )

    claude_api_key: Optional[str] = Field(
        None,
        env=ENV_CLAUDE_API_KEY,
        description="Anthropic Claude API key"
    )

    github_token: Optional[str] = Field(
        None,
        env=ENV_GITHUB_TOKEN,
        description="GitHub Personal Access Token for private repos (optional)"
    )

    @validator("github_token")
    @classmethod
    def validate_github_token(cls, v: Optional[str]) -> Optional[str]:
        if v == "":
            raise ValueError(
                "GITHUB_TOKEN environment variable, if set, cannot be empty.")
        return v


class ChromaDBSettings(BaseSettings):
    """ChromaDB configurations."""
    class Config:
        extra = 'ignore'
        case_sensitive = False

    host: str = Field(
        default=DEFAULT_CHROMA_HOST,
        env=ENV_CHROMA_DB_HOST,
        description="ChromaDB host"
    )

    port: int = Field(
        default=DEFAULT_CHROMA_PORT,
        env=ENV_CHROMA_DB_PORT,
        description="ChromaDB port"
    )

    use_remote: bool = Field(
        default=DEFAULT_CHROMA_USE_REMOTE,
        env=ENV_CHROMA_DB_USE_REMOTE,
        description="Whether to use remote ChromaDB"
    )

    persist_directory: str = Field(
        default=DEFAULT_CHROMA_PERSIST_DIRECTORY,
        env=ENV_CHROMA_DB_PERSIST_DIRECTORY,
        description="Directory for ChromaDB persistence"
    )

    default_collection_name: str = Field(
        default=DEFAULT_COLLECTION_NAME,
        env=ENV_DEFAULT_COLLECTION_NAME,
        description="Default collection name for ChromaDB"
    )


class LLMSettings(BaseSettings):
    """LLM configurations."""
    class Config:
        extra = 'ignore'
        case_sensitive = False

    # Provider selection
    provider: str = Field(
        default=DEFAULT_LLM_PROVIDER,
        env=ENV_LLM_PROVIDER,
        description="LLM provider to use (gemini, openai, claude, llama)"
    )

    # Gemini settings
    gemini_embedding_model: str = Field(
        default=DEFAULT_GEMINI_EMBEDDING_MODEL,
        env=ENV_GEMINI_EMBEDDING_MODEL,
        description="Gemini model for embeddings"
    )

    gemini_generation_model: str = Field(
        default=DEFAULT_GEMINI_GENERATION_MODEL,
        env=ENV_GEMINI_GENERATION_MODEL,
        description="Gemini model for generation"
    )

    # OpenAI settings
    openai_embedding_model: str = Field(
        default=DEFAULT_OPENAI_EMBEDDING_MODEL,
        env=ENV_OPENAI_EMBEDDING_MODEL,
        description="OpenAI model for embeddings"
    )

    openai_generation_model: str = Field(
        default=DEFAULT_OPENAI_GENERATION_MODEL,
        env=ENV_OPENAI_GENERATION_MODEL,
        description="OpenAI model for generation"
    )

    # Claude settings
    claude_generation_model: str = Field(
        default=DEFAULT_CLAUDE_GENERATION_MODEL,
        env=ENV_CLAUDE_GENERATION_MODEL,
        description="Claude model for generation"
    )

    # Llama/Ollama settings
    llama_embedding_model: str = Field(
        default=DEFAULT_LLAMA_EMBEDDING_MODEL,
        env=ENV_LLAMA_EMBEDDING_MODEL,
        description="Llama model for embeddings (via Ollama)"
    )

    llama_generation_model: str = Field(
        default=DEFAULT_LLAMA_GENERATION_MODEL,
        env=ENV_LLAMA_GENERATION_MODEL,
        description="Llama model for generation (via Ollama)"
    )

    ollama_base_url: str = Field(
        default=DEFAULT_OLLAMA_BASE_URL,
        env=ENV_OLLAMA_BASE_URL,
        description="Ollama server base URL"
    )

    @validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        if v.lower() not in SUPPORTED_LLM_PROVIDERS:
            raise ValueError(
                f"Unsupported LLM provider: {v}. Supported providers: {SUPPORTED_LLM_PROVIDERS}")
        return v.lower()

    # Legacy fields for backward compatibility
    @property
    def embedding_model(self) -> str:
        """Get embedding model based on current provider."""
        if self.provider == "gemini":
            return self.gemini_embedding_model
        elif self.provider == "openai":
            return self.openai_embedding_model
        elif self.provider == "claude":
            return self.openai_embedding_model  # Claude uses OpenAI for embeddings
        elif self.provider == "llama":
            return self.llama_embedding_model
        return self.gemini_embedding_model

    @property
    def generation_model(self) -> str:
        """Get generation model based on current provider."""
        if self.provider == "gemini":
            return self.gemini_generation_model
        elif self.provider == "openai":
            return self.openai_generation_model
        elif self.provider == "claude":
            return self.claude_generation_model
        elif self.provider == "llama":
            return self.llama_generation_model
        return self.gemini_generation_model


class ProcessingSettings(BaseSettings):
    """Document and code processing configurations."""
    class Config:
        extra = 'ignore'
        case_sensitive = False

    chunk_size: int = Field(
        default=DEFAULT_CHUNK_SIZE,
        env=ENV_CHUNK_SIZE,
        description="Size of document chunks for processing"
    )

    chunk_overlap: int = Field(
        default=DEFAULT_CHUNK_OVERLAP,
        env=ENV_CHUNK_OVERLAP,
        description="Overlap between document chunks"
    )

    code_extensions: List[str] = Field(
        default=DEFAULT_CODE_EXTENSIONS,
        env=ENV_CODE_EXTENSIONS,
        description="List of file extensions to process"
    )

    temp_clone_dir: str = Field(
        default=DEFAULT_TEMP_CLONE_DIR,
        env=ENV_TEMP_CLONE_DIR_BASE,
        description="Base directory for temporary cloned repositories"
    )

    @validator("code_extensions", pre=True)
    @classmethod
    def parse_code_extensions(cls, v):
        """Parse comma-separated string of extensions into a list of properly formatted extensions."""
        if isinstance(v, str):
            # Split by comma and strip whitespace and dots
            extensions = [ext.strip().strip('.') for ext in v.split(',')]
            # Add dots back and filter out empty strings
            return [f".{ext}" for ext in extensions if ext]
        if isinstance(v, list):
            return [f".{ext.strip().strip('.')}" for ext in v if ext.strip()]
        return DEFAULT_CODE_EXTENSIONS


class OutputSettings(BaseSettings):
    """Output configurations."""
    class Config:
        extra = 'ignore'
        case_sensitive = False

    output_file_path: str = Field(
        default=DEFAULT_OUTPUT_FILE,
        env=ENV_OUTPUT_FILE_PATH,
        description="Path to save the generated output"
    )

    @validator("output_file_path")
    @classmethod
    def validate_output_file_path(cls, v: str) -> str:
        if not v.endswith('.md'):
            raise ValueError("Output file path must end with .md extension")
        return v


class LoggingSettings(BaseSettings):
    """Logging configurations."""
    class Config:
        case_sensitive = False
        extra = 'ignore'

    level: str = Field(
        default=DEFAULT_LOG_LEVEL,
        env=ENV_LOG_LEVEL,
        description="Logging level"
    )

    format: str = Field(
        default=DEFAULT_LOG_FORMAT,
        description="Logging format"
    )


class ApiRetrySettings(BaseSettings):
    """API retry configurations."""
    class Config:
        case_sensitive = False
        extra = 'ignore'

    api_retry_attempts: int = Field(
        default=DEFAULT_API_RETRY_ATTEMPTS,
        env=ENV_API_RETRY_ATTEMPTS,
        description="Number of retry attempts for API calls"
    )

    api_retry_wait_seconds: int = Field(
        default=DEFAULT_API_RETRY_WAIT_SECONDS,
        env=ENV_API_RETRY_WAIT_SECONDS,
        description="Wait time between retry attempts in seconds"
    )


class CodeLoaderSettings(BaseSettings):
    """Code loader configurations."""
    class Config:
        case_sensitive = False
        extra = 'ignore'

    code_extensions: List[str] = Field(
        default=DEFAULT_CODE_EXTENSIONS,
        env=ENV_CODE_EXTENSIONS,
        description="List of file extensions to process"
    )

    temp_clone_dir: str = Field(
        default=DEFAULT_TEMP_CLONE_DIR,
        env=ENV_TEMP_CLONE_DIR_BASE,
        description="Base directory for temporary cloned repositories"
    )

    @validator("code_extensions", pre=True)
    @classmethod
    def parse_code_extensions(cls, v):
        """Parse comma-separated string of extensions into a list of properly formatted extensions."""
        if isinstance(v, str):
            # Split by comma and strip whitespace and dots
            extensions = [ext.strip().strip('.') for ext in v.split(',')]
            # Add dots back and filter out empty strings
            return [f".{ext}" for ext in extensions if ext]
        if isinstance(v, list):
            return [f".{ext.strip().strip('.')}" for ext in v if ext.strip()]
        return DEFAULT_CODE_EXTENSIONS


class AppSettings:
    """Main application settings container."""

    def __init__(self):
        self.common = CommonSettings()
        self.api_keys = ApiKeysSettings()
        self.chromadb = ChromaDBSettings()
        self.llm = LLMSettings()
        self.processing = ProcessingSettings()
        self.output = OutputSettings()
        self.logging = LoggingSettings()
        self.api_retry = ApiRetrySettings()
        self.code_loader = CodeLoaderSettings()

    @classmethod
    def load_settings(cls) -> 'AppSettings':
        """Load all settings."""
        return cls()


# Initialize settings with graceful error handling
try:
    settings = AppSettings.load_settings()
except Exception as e:
    print(f"Error loading settings: {e}")
    settings = None
