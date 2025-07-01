"""
Constants and default values for the TestTeller RAG Agent.
This module centralizes all default values and configuration constants used throughout the application.
"""

# Application Information
APP_NAME = "TestTeller RAG Agent"
# APP_VERSION is now imported from __init__.py to maintain single source of truth
# Import will be handled by __init__.py
FALLBACK_VERSION = "0.1.3"  # Fallback version when _version.py import fails
APP_DESCRIPTION = "A versatile CLI-based RAG (Retrieval Augmented Generation) agent designed to generate software test cases."

# Default Environment Settings
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FORMAT = "json"

# ChromaDB Settings
DEFAULT_CHROMA_HOST = "localhost"
DEFAULT_CHROMA_PORT = 8000
DEFAULT_CHROMA_USE_REMOTE = False
DEFAULT_CHROMA_PERSIST_DIRECTORY = "./chroma_data"
DEFAULT_COLLECTION_NAME = "test_collection"

# LLM Settings
# Supported LLM providers
SUPPORTED_LLM_PROVIDERS = ["gemini", "openai", "claude", "llama"]
DEFAULT_LLM_PROVIDER = "gemini"

# Gemini Settings
DEFAULT_GEMINI_EMBEDDING_MODEL = "text-embedding-004"
DEFAULT_GEMINI_GENERATION_MODEL = "gemini-2.0-flash"

# OpenAI Settings
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_OPENAI_GENERATION_MODEL = "gpt-4o-mini"

# Claude Settings
DEFAULT_CLAUDE_GENERATION_MODEL = "claude-3-5-haiku-20241022"

# Llama Settings
DEFAULT_LLAMA_EMBEDDING_MODEL = "llama3.2:1b"
DEFAULT_LLAMA_GENERATION_MODEL = "llama3.2:3b"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"

# Document Processing Settings
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200

# Code Processing Settings
DEFAULT_CODE_EXTENSIONS = [
    ".py",   # Python
    ".js",   # JavaScript
    ".ts",   # TypeScript
    ".java",  # Java
    ".go",   # Go
    ".rs",   # Rust
    ".cpp",  # C++
    ".hpp",  # C++ Headers
    ".c",    # C
    ".h",    # C Headers
    ".cs",   # C#
    ".rb",   # Ruby
    ".php"   # PHP
]
DEFAULT_TEMP_CLONE_DIR = "./temp_cloned_repos"

# Output Settings
DEFAULT_OUTPUT_FILE = "testteller-testcases.md"

# API Retry Settings
DEFAULT_API_RETRY_ATTEMPTS = 3
DEFAULT_API_RETRY_WAIT_SECONDS = 2

# Docker Settings
DOCKER_HEALTHCHECK_INTERVAL = "30s"
DOCKER_HEALTHCHECK_TIMEOUT = "10s"
DOCKER_HEALTHCHECK_RETRIES = 3
DOCKER_HEALTHCHECK_START_PERIOD = "30s"
DOCKER_DEFAULT_CPU_LIMIT = "2"
DOCKER_DEFAULT_MEMORY_LIMIT = "4G"
DOCKER_DEFAULT_CPU_RESERVATION = "0.5"
DOCKER_DEFAULT_MEMORY_RESERVATION = "1G"

# Environment Variable Names
# API Keys
ENV_GOOGLE_API_KEY = "GOOGLE_API_KEY"
ENV_OPENAI_API_KEY = "OPENAI_API_KEY"
ENV_CLAUDE_API_KEY = "CLAUDE_API_KEY"
ENV_GITHUB_TOKEN = "GITHUB_TOKEN"

# LLM Configuration
ENV_LLM_PROVIDER = "LLM_PROVIDER"
ENV_LOG_LEVEL = "LOG_LEVEL"
ENV_CHROMA_DB_HOST = "CHROMA_DB_HOST"
ENV_CHROMA_DB_PORT = "CHROMA_DB_PORT"
ENV_CHROMA_DB_USE_REMOTE = "CHROMA_DB_USE_REMOTE"
ENV_CHROMA_DB_PERSIST_DIRECTORY = "CHROMA_DB_PERSIST_DIRECTORY"
ENV_DEFAULT_COLLECTION_NAME = "DEFAULT_COLLECTION_NAME"

# Gemini Model Environment Variables
ENV_GEMINI_EMBEDDING_MODEL = "GEMINI_EMBEDDING_MODEL"
ENV_GEMINI_GENERATION_MODEL = "GEMINI_GENERATION_MODEL"

# OpenAI Model Environment Variables
ENV_OPENAI_EMBEDDING_MODEL = "OPENAI_EMBEDDING_MODEL"
ENV_OPENAI_GENERATION_MODEL = "OPENAI_GENERATION_MODEL"

# Claude Model Environment Variables
ENV_CLAUDE_GENERATION_MODEL = "CLAUDE_GENERATION_MODEL"

# Llama/Ollama Model Environment Variables
ENV_LLAMA_EMBEDDING_MODEL = "LLAMA_EMBEDDING_MODEL"
ENV_LLAMA_GENERATION_MODEL = "LLAMA_GENERATION_MODEL"
ENV_OLLAMA_BASE_URL = "OLLAMA_BASE_URL"

# Other Environment Variables
ENV_CHUNK_SIZE = "CHUNK_SIZE"
ENV_CHUNK_OVERLAP = "CHUNK_OVERLAP"
ENV_CODE_EXTENSIONS = "CODE_EXTENSIONS"
ENV_TEMP_CLONE_DIR_BASE = "TEMP_CLONE_DIR_BASE"
ENV_OUTPUT_FILE_PATH = "OUTPUT_FILE_PATH"
ENV_API_RETRY_ATTEMPTS = "API_RETRY_ATTEMPTS"
ENV_API_RETRY_WAIT_SECONDS = "API_RETRY_WAIT_SECONDS"
