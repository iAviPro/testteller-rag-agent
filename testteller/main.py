import asyncio
import logging
import os

import typer
from typing_extensions import Annotated

from .agent import TestTellerRagAgent
from .config import settings
from .utils.helpers import setup_logging


setup_logging()
logger = logging.getLogger(__name__)

app = typer.Typer(
    help="TestTeller: RAG Agent for AI Test Case Generation. Configure the agent via .env file.")

# Default .env template with descriptions
ENV_TEMPLATE = {
    "GOOGLE_API_KEY": {
        "value": "",
        "required": True,
        "description": "Your Google Gemini API key (required)"
    },
    "GITHUB_TOKEN": {
        "value": "",
        "required": False,
        "description": "GitHub Personal Access Token for private repos (optional)"
    },
    "LOG_LEVEL": {
        "value": "INFO",
        "required": False,
        "description": "Logging level (DEBUG, INFO, WARNING, ERROR)"
    },
    "CHROMA_DB_PATH": {
        "value": "./chroma_data_non_prod",
        "required": False,
        "description": "Path to ChromaDB persistent storage"
    },
    "DEFAULT_COLLECTION_NAME": {
        "value": "test_data_non_prod",
        "required": False,
        "description": "Default ChromaDB collection name"
    },
    "GEMINI_EMBEDDING_MODEL": {
        "value": "text-embedding-004",
        "required": False,
        "description": "Gemini model for embeddings"
    },
    "GEMINI_GENERATION_MODEL": {
        "value": "gemini-2.0-flash",
        "required": False,
        "description": "Gemini model for generation"
    },
    "OUTPUT_FILE_PATH": {
        "value": "testteller_output.md",
        "required": False,
        "description": "Default path to save generated test cases"
    }
}


def get_collection_name(provided_name: str | None = None) -> str:
    """
    Get the collection name to use, with the following priority:
    1. User-provided name
    2. Name from settings
    3. Default fallback name
    """
    if provided_name:
        return provided_name

    default_name = "test_documents_prod"

    try:
        if settings and settings.chroma_db:
            settings_dict = settings.chroma_db.__dict__
            if settings_dict.get('default_collection_name'):
                name = settings_dict['default_collection_name']
                logger.info(
                    "Using default collection name from settings: %s", name)
                return name
    except Exception as e:
        logger.warning("Failed to get collection name from settings: %s", e)

    logger.info("Using fallback default collection name: %s", default_name)
    return default_name


def check_settings():
    """Check if required settings are available and provide guidance if not."""
    if settings is None:
        env_path = os.path.join(os.getcwd(), '.env')
        print("\n⚠️  Configuration Error: Missing or invalid .env file")
        print("\nTo configure TestTeller, you have two options:")
        print("\n1. Run the configuration wizard:")
        print("   testteller configure")
        print("\n2. Manually create a .env file at:")
        print(f"   {env_path}")
        print("\nMinimum required configuration:")
        print('   GOOGLE_API_KEY="your-api-key-here"')
        print("\nFor more information about configuration, visit:")
        print("   https://github.com/yourusername/testteller#configuration")
        raise typer.Exit(code=1)
    return True


def _get_agent(collection_name: str) -> TestTellerRagAgent:
    check_settings()  # Ensure settings are available
    try:
        return TestTellerRagAgent(collection_name=collection_name)
    except Exception as e:
        logger.error(
            "Failed to initialize TestCaseAgent for collection '%s': %s", collection_name, e, exc_info=True)
        print(
            f"Error: Could not initialize agent. Check logs and GOOGLE_API_KEY. Details: {e}")
        raise typer.Exit(code=1)


async def ingest_docs_async(path: str, collection_name: str):
    agent = _get_agent(collection_name)
    await agent.ingest_documents_from_path(path)
    count = await agent.get_ingested_data_count()
    print(
        f"Successfully ingested documents. Collection '{collection_name}' now contains {count} items.")


async def ingest_code_async(source_path: str, collection_name: str, no_cleanup_github: bool):
    agent = _get_agent(collection_name)
    await agent.ingest_code_from_source(source_path, cleanup_github_after=not no_cleanup_github)
    count = await agent.get_ingested_data_count()
    print(
        f"Successfully ingested code from '{source_path}'. Collection '{collection_name}' now contains {count} items.")


async def generate_async(query: str, collection_name: str, num_retrieved: int, output_file: str | None):
    agent = _get_agent(collection_name)

    current_count = await agent.get_ingested_data_count()
    if current_count == 0:
        print(
            f"Warning: Collection '{collection_name}' is empty. Generation will rely on LLM's general knowledge.")
        if not typer.confirm("Proceed anyway?", default=True):
            print("Generation aborted.")
            return

    test_cases = await agent.generate_test_cases(query, n_retrieved_docs=num_retrieved)
    print("\n--- Generated Test Cases ---")
    print(test_cases)
    print("--- End of Test Cases ---\n")

    if output_file:
        if "Error:" in test_cases[:20]:
            logger.warning(
                "LLM generation resulted in an error, not saving to file: %s", test_cases)
            print(
                f"Warning: Test case generation seems to have failed. Not saving to {output_file}.")
        else:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(test_cases)
                print(f"Test cases saved to: {output_file}")
            except Exception as e:
                logger.error(
                    "Failed to save test cases to %s: %s", output_file, e, exc_info=True)
                print(
                    f"Error: Could not save test cases to {output_file}: {e}")


async def status_async(collection_name: str):
    """Check status of a collection asynchronously."""
    agent = _get_agent(collection_name)
    count = await agent.get_ingested_data_count()
    print(f"\nCollection '{collection_name}' contains {count} ingested items.")

    # Print ChromaDB connection info
    if agent.vector_store.use_remote:
        print(
            f"ChromaDB connection: Remote at {agent.vector_store.host}:{agent.vector_store.port}")
    else:
        print(f"ChromaDB persistent path: {agent.vector_store.db_path}")


async def clear_data_async(collection_name: str, force: bool):
    if not force:
        confirm = typer.confirm(
            f"Are you sure you want to clear all data from collection '{collection_name}' and remove related cloned repositories?")
        if not confirm:
            print("Operation cancelled.")
            raise typer.Exit()

    agent = _get_agent(collection_name)
    await agent.clear_ingested_data()
    print(f"Successfully cleared data from collection '{collection_name}'.")


@app.command()
def ingest_docs(
    path: Annotated[str, typer.Argument(help="Path to a document file or a directory.")],
    collection_name: Annotated[str, typer.Option(
        help="ChromaDB collection name.")] = None
):
    """Ingests documents from a file or directory into a collection."""
    # Get collection name from settings if not provided
    collection_name = get_collection_name(collection_name)

    logger.info("CLI: Ingesting documents from '%s' into collection '%s'",
                path, collection_name)

    if not os.path.exists(path):
        logger.error(
            "Document source path does not exist or is not accessible: %s", path)
        print(
            f"Error: Document source path '{path}' not found or not accessible.")
        raise typer.Exit(code=1)

    try:
        asyncio.run(ingest_docs_async(path, collection_name))
    except Exception as e:
        logger.error(
            "CLI: Unhandled error during document ingestion from '%s': %s", path, e, exc_info=True)
        print(f"An unexpected error occurred: {e}")
        raise typer.Exit(code=1)


@app.command()
def ingest_code(
    source_path: Annotated[str, typer.Argument(help="URL of the GitHub repository OR path to a local code folder.")],
    collection_name: Annotated[str, typer.Option(
        help="ChromaDB collection name.")] = None,
    no_cleanup_github: Annotated[bool, typer.Option(
        help="Do not delete cloned GitHub repo after ingestion (no effect for local folders).")] = False
):
    """Ingests code from a GitHub repository or local folder into a collection."""
    # Get collection name from settings if not provided
    collection_name = get_collection_name(collection_name)

    logger.info("CLI: Ingesting code from '%s' into collection '%s'",
                source_path, collection_name)

    # For local paths, check if they exist
    if not source_path.startswith(('http://', 'https://', 'git@')) and not os.path.exists(source_path):
        logger.error(
            "Local source path does not exist or is not accessible: %s", source_path)
        print(
            f"Error: Local source path '{source_path}' not found or not accessible.")
        raise typer.Exit(code=1)

    try:
        asyncio.run(ingest_code_async(
            source_path, collection_name, no_cleanup_github))
    except Exception as e:
        logger.error(
            "CLI: Unhandled error during code ingestion from '%s': %s", source_path, e, exc_info=True)
        print(f"An unexpected error occurred: {e}")
        raise typer.Exit(code=1)


@app.command()
def generate(
    query: Annotated[str, typer.Argument(help="Query for test case generation.")],
    collection_name: Annotated[str, typer.Option(
        help="ChromaDB collection name.")] = None,
    num_retrieved: Annotated[int, typer.Option(
        min=0, max=20, help="Number of docs for context.")] = 5,
    output_file: Annotated[str, typer.Option(
        help="Optional: Save test cases to this file. If not provided, uses OUTPUT_FILE_PATH from .env or defaults to testteller_output.md")] = None
):
    """Generates test cases based on query and knowledge base."""
    logger.info(
        "CLI: Generating test cases for query: '%s...', Collection: %s", query[:50], collection_name)

    # Get collection name from settings if not provided
    collection_name = get_collection_name(collection_name)

    # Determine output file path
    final_output_file = output_file
    if not final_output_file:
        try:
            if settings and settings.output:
                settings_dict = settings.output.__dict__
                if settings_dict.get('output_file_path'):
                    final_output_file = settings_dict['output_file_path']
                    logger.info(
                        "Using output file path from settings: %s", final_output_file)
        except Exception as e:
            logger.warning(
                "Failed to get output file path from settings: %s", e)

        if not final_output_file:
            final_output_file = "testteller_output.md"
            logger.info("Using default output file path: %s",
                        final_output_file)

    try:
        asyncio.run(generate_async(
            query, collection_name, num_retrieved, final_output_file))
    except Exception as e:
        logger.error(
            "CLI: Unhandled error during test case generation: %s", e, exc_info=True)
        print(f"An unexpected error occurred: {e}")
        raise typer.Exit(code=1)


@app.command()
def status(
    collection_name: Annotated[str, typer.Option(
        help="ChromaDB collection name.")] = None
):
    """Checks status of a collection."""
    # Get collection name from settings if not provided
    collection_name = get_collection_name(collection_name)

    logger.info("CLI: Checking status for collection: %s", collection_name)
    try:
        asyncio.run(status_async(collection_name))
    except Exception as e:
        logger.error(
            "CLI: Unhandled error during status check: %s", e, exc_info=True)
        print(f"An unexpected error occurred: {e}")
        raise typer.Exit(code=1)


@app.command()
def clear_data(
    collection_name: Annotated[str, typer.Option(
        help="ChromaDB collection to clear.")] = None,
    force: Annotated[bool, typer.Option(
        help="Force clear without confirmation.")] = False
):
    """Clears ingested data from a collection."""
    # Get collection name from settings if not provided
    collection_name = get_collection_name(collection_name)

    logger.info("CLI: Clearing data for collection: %s", collection_name)
    try:
        asyncio.run(clear_data_async(collection_name, force))
    except Exception as e:
        logger.error(
            "CLI: Unhandled error during data clearing: %s", e, exc_info=True)
        print(f"An unexpected error occurred: {e}")
        raise typer.Exit(code=1)


@app.command()
def configure():
    """Interactive configuration wizard to set up TestTeller."""
    env_path = os.path.join(os.getcwd(), '.env')
    env_example_path = os.path.join(os.getcwd(), '.env.example')

    # Check if .env already exists
    if os.path.exists(env_path):
        overwrite = typer.confirm(
            "⚠️  A .env file already exists. Do you want to reconfigure it?", default=False)
        if not overwrite:
            print("Configuration cancelled.")
            raise typer.Exit()

    print("\n🔧 TestTeller Configuration Wizard")
    print("==================================")

    env_values = {}

    # Collect values for each setting
    for key, config in ENV_TEMPLATE.items():
        description = config["description"]
        default = config["value"]
        required = config["required"]

        # Format prompt based on whether the field is required
        prompt = f"\n{description}"
        if required:
            prompt += " (required)"
        elif default:
            prompt += f" (default: {default})"

        # Get user input
        while True:
            value = typer.prompt(
                prompt, default=default if not required else None, show_default=True)
            if value or not required:
                break
            print("This field is required. Please provide a value.")

        if value:
            env_values[key] = value

    # Try to read additional non-critical configs from .env.example
    additional_configs = {}
    if os.path.exists(env_example_path):
        try:
            with open(env_example_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        # Only add if it's not already in env_values and not in ENV_TEMPLATE
                        if key not in env_values and key not in ENV_TEMPLATE:
                            additional_configs[key] = value

            if additional_configs:
                print("\nFound additional configurations in .env.example:")
                for key, value in additional_configs.items():
                    print(f"  - {key}")

                if typer.confirm("\nWould you like to include these additional configurations?", default=True):
                    env_values.update(additional_configs)
                    print("Additional configurations included.")
                else:
                    print("Additional configurations skipped.")
        except Exception as e:
            logger.warning("Could not read .env.example: %s", e)
            # Silently continue without additional configs

    # Write to .env file
    try:
        with open(env_path, 'w') as f:
            # Write critical configs first
            for key, config in ENV_TEMPLATE.items():
                if key in env_values:
                    f.write(f'{key}="{env_values[key]}"\n')

            # Write additional configs with a separator comment
            if any(key not in ENV_TEMPLATE for key in env_values):
                f.write('\n# Additional configurations\n')
                for key, value in env_values.items():
                    if key not in ENV_TEMPLATE:
                        f.write(f'{key}="{value}"\n')

        print("\n✅ Configuration complete!")
        print(f"Configuration saved to: {env_path}")
        print("\nYou can now use TestTeller commands. Try:")
        print("  testteller --help")

    except Exception as e:
        print(f"\n❌ Error saving configuration: {e}")
        raise typer.Exit(code=1)


def app_runner():
    """
    This function is the entry point for the CLI script defined in pyproject.toml.
    It ensures logging is set up and then runs the Typer application.
    """
    try:
        app()
    except Exception as e:
        logger.error("Unhandled error in CLI: %s", e, exc_info=True)
        print(f"\n❌ An unexpected error occurred: {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app_runner()
