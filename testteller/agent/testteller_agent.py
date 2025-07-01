"""
TestTellerAgent implementation for test case generation.
"""
import logging
import os
from typing import List, Dict, Any, Optional
from testteller.config import settings
from testteller.llm.llm_manager import LLMManager
from testteller.vector_store.chromadb_manager import ChromaDBManager
from testteller.data_ingestion.document_loader import DocumentLoader
from testteller.data_ingestion.code_loader import CodeLoader
from testteller.prompts import TEST_CASE_GENERATION_PROMPT_TEMPLATE, get_test_case_generation_prompt
import hashlib

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_COLLECTION_NAME = "test_documents_non_prod"


class TestTellerAgent:
    """Agent for generating test cases using RAG approach."""

    def __init__(
        self,
        collection_name: Optional[str] = None,
        llm_manager: Optional[LLMManager] = None
    ):
        """
        Initialize the TestTellerAgent.

        Args:
            collection_name: Name of the ChromaDB collection (optional)
            llm_manager: Instance of LLMManager (optional)
        """
        self.collection_name = collection_name or self._get_collection_name()
        self.llm_manager = llm_manager or LLMManager()
        self.vector_store = ChromaDBManager(
            llm_manager=self.llm_manager,
            collection_name=self.collection_name
        )
        self.document_loader = DocumentLoader()
        self.code_loader = CodeLoader()
        logger.info(
            "Initialized TestTellerAgent with collection '%s' and LLM provider '%s'",
            self.collection_name, self.llm_manager.provider)

    def _get_collection_name(self) -> str:
        """Get collection name from settings or use default."""
        try:
            if settings and settings.chromadb:
                return settings.chromadb.__dict__.get('default_collection_name', DEFAULT_COLLECTION_NAME)
        except Exception as e:
            logger.debug("Could not get collection name from settings: %s", e)
        return DEFAULT_COLLECTION_NAME

    async def ingest_documents_from_path(self, path: str) -> None:
        """Ingest documents from a file or directory."""
        try:
            if os.path.isfile(path):
                content = await self.document_loader.load_document(path)
                if content:
                    # Generate unique ID for the document
                    doc_id = hashlib.sha256(f"doc:{path}".encode()).hexdigest()
                    # Use sync call without await
                    self.vector_store.add_documents(
                        [content],
                        [{"source": path, "type": "document"}],
                        [doc_id]
                    )
                else:
                    logger.warning("No content loaded from document: %s", path)
                    return
            elif os.path.isdir(path):
                docs = await self.document_loader.load_from_directory(path)
                if docs:
                    contents, paths = zip(*docs)
                    # Generate unique IDs for each document
                    ids = [
                        hashlib.sha256(f"doc:{p}".encode()).hexdigest()
                        for p in paths
                    ]
                    # Use sync call without await
                    self.vector_store.add_documents(
                        list(contents),
                        [{"source": p, "type": "document"} for p in paths],
                        ids
                    )
                else:
                    logger.warning(
                        "No documents loaded from directory: %s", path)
                    return
            logger.info("Ingested documents from path: %s", path)
        except Exception as e:
            logger.error("Error ingesting documents: %s", e)
            raise

    async def ingest_code_from_source(self, source_path: str, cleanup_github_after: bool = True) -> None:
        """Ingest code from GitHub repository or local folder."""
        try:
            is_remote = "://" in source_path or source_path.startswith("git@")
            if is_remote:
                code_files = await self.code_loader.load_code_from_repo(source_path)
                if cleanup_github_after:
                    await self.code_loader.cleanup_repo(source_path)
            else:
                code_files = await self.code_loader.load_code_from_local_folder(source_path)

            if code_files:
                contents, paths = zip(*code_files)
                # Generate unique IDs based on source path and file path
                ids = [
                    hashlib.sha256(
                        f"{source_path}:{str(p)}".encode()).hexdigest()
                    for p in paths
                ]
                # Use sync call without await
                self.vector_store.add_documents(
                    list(contents),
                    [{"source": p, "type": "code"} for p in paths],
                    ids
                )
                logger.info("Ingested code from source: %s", source_path)
            else:
                logger.warning(
                    "No code files loaded from source: %s", source_path)
        except Exception as e:
            logger.error("Error ingesting code: %s", e)
            raise

    async def get_ingested_data_count(self) -> int:
        """Get count of ingested documents."""
        return self.vector_store.get_collection_count()

    async def clear_ingested_data(self) -> None:
        """Clear all ingested data."""
        try:
            self.vector_store.clear_collection()
            await self.code_loader.cleanup_all_repos()
            logger.info("Cleared all ingested data")
        except Exception as e:
            logger.error("Error clearing data: %s", e)
            raise

    async def generate_test_cases(
        self,
        code_context: str,
        n_retrieved_docs: int = 5
    ) -> str:
        """
        Generate test cases for given code context.

        Args:
            code_context: Code to generate tests for
            n_retrieved_docs: Number of similar documents to retrieve

        Returns:
            Generated test cases as string
        """
        try:
            # Query similar test cases
            results = self.vector_store.query_similar(
                query_text=code_context,
                n_results=n_retrieved_docs
            )
            similar_tests = results.get('documents', [[]])[0]

            # Get provider-optimized prompt
            current_provider = self.llm_manager.get_current_provider()
            prompt = get_test_case_generation_prompt(
                provider=current_provider,
                context="\n\n".join(similar_tests),
                query=code_context
            )

            # Generate test cases using LLM Manager
            response_text = await self.llm_manager.generate_text_async(prompt)
            logger.info("Generated test cases for code context using %s provider with optimized prompt",
                        self.llm_manager.provider)
            return response_text

        except Exception as e:
            logger.error("Error generating test cases: %s", e)
            raise

    def add_test_cases(
        self,
        test_cases: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> None:
        """
        Add test cases to the vector store.

        Args:
            test_cases: List of test case texts
            metadatas: Optional metadata for each test case
            ids: Optional IDs for each test case
        """
        try:
            self.vector_store.add_documents(
                documents=test_cases,
                metadatas=metadatas,
                ids=ids
            )
            logger.info("Added %d test cases to the vector store",
                        len(test_cases))
        except Exception as e:
            logger.error("Error adding test cases: %s", e)
            raise

    def clear_test_cases(self) -> None:
        """Clear all test cases from the vector store."""
        try:
            self.vector_store.clear_collection()
            logger.info("Cleared all test cases from the vector store")
        except Exception as e:
            logger.error("Error clearing test cases: %s", e)
            raise


# Create an alias for backward compatibility
TestTellerRagAgent = TestTellerAgent
