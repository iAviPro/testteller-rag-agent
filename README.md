# TestTeller RAG Agent

[![PyPI version](https://img.shields.io/pypi/v/testteller.svg)](https://pypi.org/project/testteller/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-supported-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


**TestTeller RAG Agent** is a versatile CLI-based RAG (Retrieval Augmented Generation) agent designed to generate software test cases. It supports multiple LLM providers including Google Gemini, OpenAI, Anthropic Claude, and local Llama models via Ollama. The agent uses ChromaDB as a vector store and can process various input sources, including PRD documentation, API contracts, technical design documents (HLD/LLD), and code from GitHub repositories or local folders.

## 🤖 Supported LLM Providers

TestTeller supports multiple LLM providers to give you flexibility in choosing the best model for your needs:

- **🟦 Google Gemini** (Default)
  - Models: `gemini-2.0-flash`, `text-embedding-004`
  - Fast and cost-effective
  - Requires: `GOOGLE_API_KEY`

- **🟢 OpenAI**
  - Models: `gpt-4o-mini`, `text-embedding-3-small`
  - High-quality responses
  - Requires: `OPENAI_API_KEY`

- **🟣 Anthropic Claude**
  - Models: `claude-3-5-haiku-20241022`
  - Excellent reasoning capabilities
  - Requires: `CLAUDE_API_KEY` + `OPENAI_API_KEY` (for embeddings)

- **🦙 Local Llama (via Ollama)**
  - Models: `llama3.2:3b`, `llama3.2:1b`
  - Privacy-focused, runs locally
  - Requires: Local Ollama installation

## 🚀 Quick Start

1. **Install the Package**
```bash
# Install from PyPI
pip install testteller

# Or clone and install locally
git clone https://github.com/iAviPro/testteller-rag-agent.git
cd testteller-rag-agent
pip install -e .
```

2. **Verify Installation**
```bash
# Check if testteller is correctly installed
testteller --version
```

3. **Configure the Agent**
```bash
# Run the interactive configuration wizard
testteller configure

# Or set environment variables manually
export LLM_PROVIDER="gemini"  # or "openai", "claude", "llama"
export GOOGLE_API_KEY="your_gemini_api_key"  # for Gemini
export OPENAI_API_KEY="your_openai_api_key"  # for OpenAI/Claude embeddings
export CLAUDE_API_KEY="your_claude_api_key"  # for Claude
export GITHUB_TOKEN="your_github_token"  # Optional, for private repos
```

4. **For Llama Users: Setup Ollama**
```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve

# Install required models
ollama pull llama3.2:3b
ollama pull llama3.2:1b
```

5. **Start ChromaDB**
```bash
# Using Docker (Optional)
docker run -d -p 8000:8000 chromadb/chroma:0.4.15
```

6. **Generate Test Cases**
```bash
# Ingest code or documentation
testteller ingest-code https://github.com/owner/repo.git --collection-name my_collection
testteller ingest-docs path/to/document.pdf --collection-name my_collection

# Generate tests
testteller generate "Create API integration tests for user authentication" --collection-name my_collection --output-file ./tests.md
```

## ✨ Features

### 🔄 Intelligent Test Generation

**TestTeller** generates a comprehensive suite of tests by analyzing your documentation and code. It covers multiple layers of your application with a focus on realism and technical depth.

- **Generates Multiple Test Types:**
  - **End-to-End (E2E) Tests:** Simulates complete user journeys, from UI interactions to backend processing, to validate entire workflows.
  - **Integration Tests:** Verifies the contracts and interactions between different components, services, and APIs, including event-driven architectures.
  - **Technical Tests:** Focuses on non-functional requirements, probing for weaknesses in performance, security, and resilience.
  - **Mocked System Tests:** Provides fast, isolated tests for individual components by mocking their dependencies.

- **Ensures Comprehensive Scenario Coverage:**
  - **Happy Paths:** Validates the primary, expected functionality.
  - **Negative & Edge Cases:** Explores system behavior with invalid inputs, at operational limits, and under stress.
  - **Failure & Recovery:** Tests resilience by simulating dependency failures and verifying recovery mechanisms.
  - **Security & Performance:** Assesses vulnerabilities and measures adherence to performance SLAs.

## 🧪 Test Case Types

### End-to-End (E2E) Tests
- Complete user journey coverage
- Focus on business workflows and user interactions
- Documentation includes:
  - User story or journey being tested
  - Prerequisites and test environment setup
  - Step-by-step test flow
  - Expected outcomes at each step
  - Test data requirements

### Integration Tests
- Component interaction verification
- API contract validation
- Documentation includes:
  - Components or services involved
  - Interface specifications
  - Data flow diagrams
  - Error handling scenarios
  - Dependencies and mocking requirements

### Technical Tests
- System limitations testing
- Edge case handling
- Documentation includes:
  - Technical constraints being tested
  - System boundaries and limits
  - Resource utilization scenarios
  - Error conditions and recovery
  - Performance thresholds

### Mocked System Tests
- Isolated component testing assuming the component is mocked
- Functional requirement verification
- Documentation includes:
  - Component specifications
  - Input/output requirements
  - State transitions
  - Configuration requirements
  - Environmental dependencies
  - Authentication and authorization
  - Data validation
  - Response times
  - Resource utilization
  - Load handling

### 📚 Document Processing
- **Multi-Format Support**
  - PDF documents (`.pdf`)
  - Word documents (`.docx`)
  - Excel spreadsheets (`.xlsx`)
  - Markdown files (`.md`)
  - Text files (`.txt`)
  - Source code files (multiple languages)

### 💻 Code Analysis
- **Repository Integration**
  - GitHub repository cloning (public and private)
  - Local codebase analysis
  - Multiple programming language support

### 🧠 Advanced RAG Pipeline
- **Multi-LLM Support**
  - Google Gemini 2.0 Flash for fast generation
  - OpenAI GPT-4o Mini for high-quality responses
  - Anthropic Claude for advanced reasoning
  - Local Llama models via Ollama for privacy
  - Optimized embeddings with provider-specific models
  - Context-aware prompt engineering
  - Streaming response support

### 📊 Output Management
- **Flexible Output Formats**
  - Markdown documentation
  - Structured test cases

## 📋 Prerequisites

*   Python 3.11 or higher (Required)
*   Docker and Docker Compose (for containerized deployment)
*   **At least one LLM provider API key:**
    - Google Gemini API key ([Get it here](https://aistudio.google.com/)) 
    - OpenAI API key ([Get it here](https://platform.openai.com/api-keys))
    - Anthropic Claude API key ([Get it here](https://console.anthropic.com/))
    - Or Ollama for local Llama models ([Install here](https://ollama.ai/))
*   (Optional) GitHub Personal Access Token for private repos

## 🛠️ Installation

### Option 1: Install from PyPI

```bash
# Install the package
pip install testteller

# Configure with interactive wizard
testteller configure

# Or set environment variables manually
export LLM_PROVIDER="gemini"  # Choose: gemini, openai, claude, llama
export GOOGLE_API_KEY="your_gemini_api_key"  # for Gemini
export OPENAI_API_KEY="your_openai_api_key"  # for OpenAI/Claude embeddings
export CLAUDE_API_KEY="your_claude_api_key"  # for Claude
export GITHUB_TOKEN="your_github_token"  # Optional, for private repos

# For Llama users: Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve
ollama pull llama3.2:3b
ollama pull llama3.2:1b

# Start ChromaDB (in a separate terminal) (Optional)
docker run -d -p 8000:8000 chromadb/chroma:0.4.15
```

### Option 2: Docker Installation

1. Clone the repository:
```bash
git clone https://github.com/iAviPro/testteller-rag-agent.git
cd testteller-rag-agent
```

2. Create environment file based on your LLM provider:
```bash
# Copy the Docker environment template
cp .env.docker.example .env

# Edit the .env file with your API keys and preferred LLM provider
# For Gemini (default):
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your_gemini_api_key

# For OpenAI:
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key

# For Claude:
LLM_PROVIDER=claude
CLAUDE_API_KEY=your_claude_api_key
OPENAI_API_KEY=your_openai_api_key  # Required for embeddings

# For Llama (local):
LLM_PROVIDER=llama
# No API key required - see Llama setup below
```

3. Start services:
```bash
# For cloud providers (Gemini, OpenAI, Claude)
docker-compose up -d

# For Llama provider (requires additional setup)
# 1. Uncomment ollama service in docker-compose.yml
# 2. Start services
docker-compose up -d
# 3. Install models
docker-compose exec ollama ollama pull llama3.2:3b
docker-compose exec ollama ollama pull llama3.2:1b
```

4. Verify installation:
```bash
# Check container status
docker-compose ps

# Test the installation
docker-compose exec app testteller --help
```

## 🎯 LLM Provider Selection Guide

Choose the best LLM provider for your use case:

### 🟦 Google Gemini (Recommended for most users)
- **Best for:** Cost-effective, fast responses, general-purpose testing
- **Pros:** Excellent price/performance ratio, fast generation
- **Cons:** May require more specific prompts for complex scenarios
- **Setup:** Only requires `GOOGLE_API_KEY`

### 🟢 OpenAI
- **Best for:** High-quality test cases, complex scenarios
- **Pros:** Superior reasoning, well-structured outputs
- **Cons:** Higher cost, rate limits
- **Setup:** Only requires `OPENAI_API_KEY`

### 🟣 Anthropic Claude
- **Best for:** Advanced reasoning, safety-critical applications
- **Pros:** Excellent at understanding context, safety-focused
- **Cons:** Requires both Claude and OpenAI keys (for embeddings)
- **Setup:** Requires `CLAUDE_API_KEY` + `OPENAI_API_KEY`

### 🦙 Local Llama (Privacy-focused)
- **Best for:** Privacy-sensitive environments, offline usage
- **Pros:** Complete data privacy, no API costs, offline capability
- **Cons:** Requires local compute resources, slower than cloud APIs
- **Setup:** Install Ollama and required models locally

## 🎯 Provider-Specific Optimizations

TestTeller automatically optimizes prompts for each LLM provider to maximize test case quality:

### 🤖 **Gemini (Default)**
- **Optimization:** Balanced approach with code integration
- **Strengths:** Integrates code examples, balances technical depth with clarity
- **Best For:** General-purpose test generation with good technical detail

### 🧠 **OpenAI**
- **Optimization:** Structured thinking with JSON emphasis
- **Strengths:** Systematic approach, consistent JSON formatting, detailed specs
- **Best For:** Structured test cases with precise technical specifications

### 🔥 **Claude**
- **Optimization:** Analytical approach with comprehensive reasoning
- **Strengths:** Context analysis, detailed reasoning, edge case emphasis
- **Best For:** Complex scenarios requiring deep analysis and edge case coverage

### 🦙 **Llama**
- **Optimization:** Simplified and direct instructions
- **Strengths:** Focused output, clear language, concise templates
- **Best For:** Straightforward test cases when local processing is preferred

These optimizations are applied automatically - no configuration required! The system detects your active provider and applies the appropriate prompt refinements for optimal results.

## 🐳 Docker Usage Examples

### Quick Start with Docker

```bash
# 1. Clone and setup
git clone https://github.com/iAviPro/testteller-rag-agent.git
cd testteller-rag-agent
cp .env.docker.example .env

# 2. Configure your LLM provider in .env
# Edit .env file with your API keys

# 3. Start services
docker-compose up -d

# 4. Run commands
docker-compose exec app testteller --help
docker-compose exec app testteller configure
```

### Provider-Specific Docker Commands

#### Using Gemini (Default)
```bash
# Set in .env: LLM_PROVIDER=gemini, GOOGLE_API_KEY=your_key
docker-compose up -d
docker-compose exec app testteller ingest-docs ./docs
docker-compose exec app testteller generate "Create API tests"
```

#### Using OpenAI
```bash
# Set in .env: LLM_PROVIDER=openai, OPENAI_API_KEY=your_key
docker-compose up -d
docker-compose exec app testteller ingest-code https://github.com/user/repo.git
docker-compose exec app testteller generate "Create integration tests"
```

#### Using Claude
```bash
# Set in .env: LLM_PROVIDER=claude, CLAUDE_API_KEY=your_key, OPENAI_API_KEY=your_key
docker-compose up -d
docker-compose exec app testteller status
```

#### Using Local Llama
```bash
# 1. Uncomment ollama service in docker-compose.yml
# 2. Set in .env: LLM_PROVIDER=llama
docker-compose up -d

# 3. Install models (one-time setup)
docker-compose exec ollama ollama pull llama3.2:3b
docker-compose exec ollama ollama pull llama3.2:1b

# 4. Use TestTeller
docker-compose exec app testteller generate "Create unit tests"
```

### Docker Management

```bash
# View logs
docker-compose logs app
docker-compose logs chromadb

# Stop services
docker-compose down

# Remove all data (caution!)
docker-compose down -v

# Update to latest
docker-compose pull
docker-compose up -d --build
```

### Docker Troubleshooting

**Container won't start:**
```bash
# Check logs for errors
docker-compose logs app
docker-compose logs chromadb

# Verify environment variables
docker-compose config
```

**API key errors:**
```bash
# Check if API keys are set correctly
docker-compose exec app env | grep -E "(GOOGLE_API_KEY|OPENAI_API_KEY|CLAUDE_API_KEY)"

# Test API connectivity
docker-compose exec app testteller configure
```

**ChromaDB connection issues:**
```bash
# Check ChromaDB health
curl http://localhost:8000/api/v1/heartbeat

# Restart ChromaDB
docker-compose restart chromadb
```

**Ollama issues (for Llama provider):**
```bash
# Check Ollama service
docker-compose exec ollama ollama list

# Pull missing models
docker-compose exec ollama ollama pull llama3.2:3b
```

## 📖 Available Commands

### Using pip installation (recommended)

### Configuration
```bash
# Run interactive configuration wizard
testteller configure

# Show all available commands
testteller --help

# Show help for specific command
testteller generate --help
```

### Ingest Documentation & Code
```bash
# Ingest a single document or directory
testteller ingest-docs path/to/document.pdf --collection-name my_collection

# Ingest a directory of documents
testteller ingest-docs path/to/docs/directory --collection-name my_collection

# Ingest code from GitHub repository or local folder
testteller ingest-code https://github.com/owner/repo.git --collection-name my_collection

# Ingest code with custom collection name
testteller ingest-code ./local/code/folder --collection-name my_collection
```

### Generate Test Cases
```bash
# Generate with default settings
testteller generate "Create API integration tests for user authentication" --collection-name my_collection

# Generate tests with custom output file
testteller generate "Create technical tests for login flow" --collection-name my_collection --output-file tests.md

# Generate tests with specific collection and number of retrieved docs
testteller generate "Create more than  end-to-end tests" --collection-name my_collection --num-retrieved 10 --output-file ./tests.md
```

### Manage Data
```bash
# Check collection status
testteller status --collection-name my_collection

# Clear collection data
testteller clear-data --collection-name my_collection --force
```

### Using Docker or Local Development

When using Docker or running from source, use the module format:

```bash
# Format for Docker:
docker-compose exec app python -m testteller.main [command]

# Format for local development:
python -m testteller.main [command]
```

First, ensure your environment variables are set in the `.env` file:
```bash
# Create .env file with required variables
cat > .env << EOL
GOOGLE_API_KEY=your_gemini_api_key
# Only set GITHUB_TOKEN if you need to access private repos
# GITHUB_TOKEN=your_github_token
LOG_LEVEL=INFO
LOG_FORMAT=json
DEFAULT_COLLECTION_NAME=my_test_collection
EOL
```

Note: Only set GITHUB_TOKEN if you actually need to access private repositories. If you don't need it, comment it out or remove it entirely. Setting it to an empty value will cause validation errors.

Then use the commands:
```bash
# Get help
docker-compose exec app python -m testteller.main --help

# Get command-specific help
docker-compose exec app python -m testteller.main generate --help

# Run configuration wizard
docker-compose exec app python -m testteller.main configure

# Ingest documentation
docker-compose exec app python -m testteller.main ingest-docs /path/to/doc.pdf --collection-name my_collection

# Generate tests
docker-compose exec app python -m testteller.main generate "Create API tests" --collection-name my_collection --output-file tests.md

# Check status
docker-compose exec app python -m testteller.main status --collection-name my_collection

# Clear data
docker-compose exec app python -m testteller.main clear-data --collection-name my_collection --force
```

Note: Make sure both the app and ChromaDB containers are healthy before running commands:
```bash
# Check container status
docker-compose ps

# Check container logs if needed
docker-compose logs -f
```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default | Required | Notes |
|----------|-------------|---------|----------|-------|
| `GOOGLE_API_KEY` | Google Gemini API key | - | Yes | Must be valid API key |
| `GITHUB_TOKEN` | GitHub Personal Access Token | - | No | Don't set if not using private repos |
| `LOG_LEVEL` | Logging level | INFO | No | DEBUG, INFO, WARNING, ERROR |
| `LOG_FORMAT` | Logging format | text | No | text or json |
| `GEMINI_GENERATION_MODEL` | Gemini model for generation | gemini-2.0-flash | No | |
| `GEMINI_EMBEDDING_MODEL` | Model for embeddings | text-embedding-004 | No | |
| `CHUNK_SIZE` | Document chunk size | 1000 | No | |
| `CHUNK_OVERLAP` | Chunk overlap size | 200 | No | |
| `API_RETRY_ATTEMPTS` | Number of API retry attempts | 3 | No | |
| `API_RETRY_WAIT_SECONDS` | Wait time between retries | 2 | No | Seconds |

## 🔧 Troubleshooting

### Common Issues

1. **Container Health Check Failures**
```bash
# Check container logs
docker-compose logs -f

# Restart services
docker-compose restart
```

2. **ChromaDB Connection Issues**
```bash
# Verify ChromaDB is running
curl http://localhost:8000/api/v1/heartbeat

# Check ChromaDB logs
docker-compose logs chromadb
```

3. **Permission Issues**
```bash
# Fix volume permissions
sudo chown -R 1000:1000 ./chroma_data
sudo chmod -R 777 ./temp_cloned_repos
```

## 📝 License

This project is licensed under the Apache-2.0 License - see the [LICENSE](LICENSE) file for details.
