# Build stage
FROM ubuntu:22.04 AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# Set the working directory
WORKDIR /app

# Install Python and build dependencies
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    gcc \
    libpoppler-cpp-dev \
    pkg-config \
    sqlite3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && sqlite3 --version

# Create and activate virtual environment
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only requirements first for better caching
COPY requirements.txt .
COPY setup.py .
COPY README.md .
COPY MANIFEST.in .
COPY pyproject.toml .

# Install dependencies and package in development mode
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install -e .

# Final stage
FROM ubuntu:22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/opt/venv/bin:$PATH" \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    python3.11 \
    libpoppler-cpp-dev \
    tesseract-ocr \
    git \
    wget \
    curl \
    sqlite3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && sqlite3 --version

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create necessary directories with proper permissions
RUN mkdir -p /app/chroma_data \
    /app/temp_cloned_repos \
    && chmod -R 777 /app/chroma_data \
    && chmod -R 777 /app/temp_cloned_repos

# Create entrypoint script with multi-LLM support and improved error handling
RUN echo '#!/bin/bash\n\
    set -e\n\
    \n\
    # Function to check ChromaDB health\n\
    check_chromadb_health() {\n\
    for i in {1..30}; do\n\
    if curl -s -f http://chromadb:8000/api/v1/heartbeat > /dev/null; then\n\
    return 0\n\
    fi\n\
    echo "Waiting for ChromaDB to be ready... ($i/30)"\n\
    sleep 2\n\
    done\n\
    echo "ChromaDB is not ready after 60 seconds"\n\
    return 1\n\
    }\n\
    \n\
    # Function to validate environment based on LLM provider\n\
    validate_environment() {\n\
    local provider=${LLM_PROVIDER:-gemini}\n\
    echo "Validating environment for LLM provider: $provider"\n\
    \n\
    case "$provider" in\n\
    "gemini")\n\
    if [ -z "$GOOGLE_API_KEY" ]; then\n\
    echo "Error: GOOGLE_API_KEY is required for Gemini provider"\n\
    return 1\n\
    fi\n\
    ;;\n\
    "openai")\n\
    if [ -z "$OPENAI_API_KEY" ]; then\n\
    echo "Error: OPENAI_API_KEY is required for OpenAI provider"\n\
    return 1\n\
    fi\n\
    ;;\n\
    "claude")\n\
    if [ -z "$CLAUDE_API_KEY" ]; then\n\
    echo "Error: CLAUDE_API_KEY is required for Claude provider"\n\
    return 1\n\
    fi\n\
    if [ -z "$OPENAI_API_KEY" ]; then\n\
    echo "Error: OPENAI_API_KEY is also required for Claude provider (for embeddings)"\n\
    return 1\n\
    fi\n\
    ;;\n\
    "llama")\n\
    echo "Note: Llama provider uses local Ollama - no API key required"\n\
    echo "Make sure OLLAMA_BASE_URL is properly configured: ${OLLAMA_BASE_URL:-http://localhost:11434}"\n\
    ;;\n\
    *)\n\
    echo "Error: Unsupported LLM provider: $provider"\n\
    echo "Supported providers: gemini, openai, claude, llama"\n\
    return 1\n\
    ;;\n\
    esac\n\
    echo "Environment validation passed for $provider provider"\n\
    }\n\
    \n\
    # Function to display configuration info\n\
    display_config_info() {\n\
    echo "=== TestTeller Configuration ==="\n\
    echo "LLM Provider: ${LLM_PROVIDER:-gemini}"\n\
    echo "ChromaDB Host: ${CHROMA_DB_HOST:-chromadb}"\n\
    echo "Default Collection: ${DEFAULT_COLLECTION_NAME:-test_documents}"\n\
    echo "Log Level: ${LOG_LEVEL:-INFO}"\n\
    echo "================================"\n\
    }\n\
    \n\
    if [ "$1" = "serve" ]; then\n\
    display_config_info\n\
    echo "Container is running. Use one of the following commands:"\n\
    echo "  docker-compose exec app testteller --help"\n\
    echo "  docker-compose exec app testteller configure"\n\
    echo "  docker-compose exec app testteller <command> [options]"\n\
    echo ""\n\
    echo "Multi-LLM Support:"\n\
    echo "  Set LLM_PROVIDER environment variable to: gemini, openai, claude, or llama"\n\
    echo "  Configure appropriate API keys based on your chosen provider"\n\
    # Keep container running\n\
    tail -f /dev/null\n\
    else\n\
    # Validate environment\n\
    validate_environment\n\
    # Check ChromaDB health before running commands\n\
    check_chromadb_health\n\
    # Unset GITHUB_TOKEN if empty to avoid validation errors\n\
    if [ -z "$GITHUB_TOKEN" ]; then\n\
    unset GITHUB_TOKEN\n\
    fi\n\
    # Run the command\n\
    python -m testteller.main "$@"\n\
    fi' > /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

# Copy the application code
COPY . .

# Create a non-root user and switch to it
RUN useradd -m -u 1000 testteller && \
    chown -R testteller:testteller /app
USER testteller

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://chromadb:8000/api/v1/heartbeat || exit 1

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["serve"]