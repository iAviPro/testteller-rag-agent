"""
ChromaDB vector store manager implementation.
"""
import logging
import os
from typing import List, Dict, Any, Optional
import functools
import hashlib
import asyncio
import chromadb
from chromadb.api.types import (
    QueryResult,
    EmbeddingFunction,
    Documents,
    Metadatas,
    IDs,
    Where,
    WhereDocument
)
from testteller.config import settings
from testteller.constants import DEFAULT_COLLECTION_NAME, DEFAULT_CHROMA_HOST, DEFAULT_CHROMA_PORT, DEFAULT_CHROMA_PERSIST_DIRECTORY
from testteller.llm.llm_manager import LLMManager

logger = logging.getLogger(__name__)

# Default configuration values (imported from constants)
DEFAULT_PERSIST_DIRECTORY = DEFAULT_CHROMA_PERSIST_DIRECTORY
DEFAULT_HOST = DEFAULT_CHROMA_HOST
DEFAULT_PORT = DEFAULT_CHROMA_PORT


class ChromaDBManager:
    """Manager for ChromaDB vector store operations."""

    def __init__(
        self,
        llm_manager: LLMManager,
        collection_name: Optional[str] = None,
        persist_directory: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        use_remote: Optional[bool] = None
    ):
        """
        Initialize ChromaDB manager with configuration from settings or parameters.

        Args:
            llm_manager: Instance of LLMManager for embeddings
            collection_name: Name of the ChromaDB collection (optional)
            persist_directory: Directory for ChromaDB persistence (optional)
            host: Host for remote ChromaDB (optional)
            port: Port for remote ChromaDB (optional)
            use_remote: Whether to use remote ChromaDB (optional)
        """
        self.llm_manager = llm_manager

        # Get configuration from settings if available, otherwise use defaults/parameters
        try:
            if settings and settings.chromadb:
                self.collection_name = collection_name or settings.chromadb.__dict__.get(
                    'default_collection_name', DEFAULT_COLLECTION_NAME)
                self.persist_directory = persist_directory or settings.chromadb.__dict__.get(
                    'persist_directory', DEFAULT_PERSIST_DIRECTORY)
                self.host = host or settings.chromadb.__dict__.get(
                    'host', DEFAULT_HOST)
                self.port = port or settings.chromadb.__dict__.get(
                    'port', DEFAULT_PORT)
                self.use_remote = use_remote if use_remote is not None else settings.chromadb.__dict__.get(
                    'use_remote', False)
            else:
                logger.info(
                    "ChromaDB settings not found, using defaults and parameters")
                self.collection_name = collection_name or DEFAULT_COLLECTION_NAME
                self.persist_directory = persist_directory or DEFAULT_PERSIST_DIRECTORY
                self.host = host or DEFAULT_HOST
                self.port = port or DEFAULT_PORT
                self.use_remote = use_remote if use_remote is not None else False
        except Exception as e:
            logger.warning(
                "Error loading ChromaDB settings: %s. Using defaults.", e)
            self.collection_name = collection_name or DEFAULT_COLLECTION_NAME
            self.persist_directory = persist_directory or DEFAULT_PERSIST_DIRECTORY
            self.host = host or DEFAULT_HOST
            self.port = port or DEFAULT_PORT
            self.use_remote = use_remote if use_remote is not None else False

        # Store the actual db_path based on whether we're using remote or local
        self.db_path = None if self.use_remote else os.path.abspath(
            self.persist_directory)

        self.client = self._initialize_client()
        self.embedding_function = self._create_embedding_function()
        self.collection = self._get_or_create_collection()
        logger.info(
            "Initialized ChromaDB manager with collection '%s' using %s LLM %s",
            self.collection_name,
            self.llm_manager.provider,
            f"at {self.host}:{self.port}" if self.use_remote else f"in {self.persist_directory}"
        )

    def _initialize_client(self) -> chromadb.Client:
        """Initialize ChromaDB client based on configuration."""
        try:
            if self.use_remote:
                return chromadb.HttpClient(host=self.host, port=self.port)
            else:
                return chromadb.PersistentClient(path=self.persist_directory)
        except Exception as e:
            logger.error("Failed to initialize ChromaDB client: %s", e)
            raise

    def _get_or_create_collection(self) -> chromadb.Collection:
        """Get existing collection or create new one."""
        try:
            return self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            logger.error(
                "Failed to get or create collection '%s': %s", self.collection_name, e)
            raise

    def add_documents(
        self,
        documents: Documents,
        metadatas: Optional[Metadatas] = None,
        ids: Optional[IDs] = None
    ) -> None:
        """Add documents to the collection."""
        try:
            # Get embeddings for all documents
            embeddings = [
                self.llm_manager.get_embedding_sync(doc) for doc in documents
            ]

            # If no IDs provided, generate unique IDs
            if not ids:
                import uuid
                ids = [str(uuid.uuid4()) for _ in documents]

            # Get existing IDs to check for duplicates
            existing_docs = self.collection.get()
            existing_ids = set(
                existing_docs['ids']) if existing_docs and existing_docs['ids'] else set()

            # Track seen IDs to handle duplicates within the batch
            seen_ids = set()

            # Filter out duplicates while preserving order
            docs_to_add = []
            embeddings_to_add = []
            metadatas_to_add = []
            ids_to_add = []

            for i, doc_id in enumerate(ids):
                # Skip if ID is duplicate within batch or exists in collection
                if doc_id in seen_ids or doc_id in existing_ids:
                    logger.warning(
                        "Document with ID '%s' is duplicate %s, skipping",
                        doc_id,
                        "within batch" if doc_id in seen_ids else "in collection"
                    )
                    continue

                docs_to_add.append(documents[i])
                embeddings_to_add.append(embeddings[i])
                if metadatas:
                    metadatas_to_add.append(metadatas[i])
                ids_to_add.append(doc_id)
                seen_ids.add(doc_id)

            if docs_to_add:
                self.collection.add(
                    embeddings=embeddings_to_add,
                    documents=docs_to_add,
                    metadatas=metadatas_to_add if metadatas else None,
                    ids=ids_to_add
                )
                logger.info(
                    "Added %d new documents to collection '%s' (skipped %d duplicates)",
                    len(docs_to_add), self.collection_name, len(
                        documents) - len(docs_to_add)
                )
            else:
                logger.info(
                    "No new documents added to collection '%s' (all %d documents were duplicates)",
                    self.collection_name, len(documents)
                )

        except Exception as e:
            logger.error(
                "Error adding documents to collection '%s': %s", self.collection_name, e)
            raise

    def query_similar(
        self,
        query_text: str,
        n_results: int = 5,
        where: Optional[Where] = None,
        where_document: Optional[WhereDocument] = None
    ) -> QueryResult:
        """Query similar documents from the collection."""
        try:
            query_embedding = self.llm_manager.get_embedding_sync(query_text)
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
                where_document=where_document
            )
            logger.info(
                "Retrieved %d results for query from collection '%s'",
                len(results.get('documents', [[]])[0]),
                self.collection_name
            )
            return results
        except Exception as e:
            logger.error("Error querying collection '%s': %s",
                         self.collection_name, e)
            raise

    def clear_collection(self) -> None:
        """Clear all documents from the collection."""
        try:
            # Get all document IDs
            all_docs = self.collection.get()
            if all_docs and all_docs['ids']:
                # Delete all documents
                self.collection.delete(ids=all_docs['ids'])
                logger.info("Cleared %d documents from collection '%s'", len(
                    all_docs['ids']), self.collection_name)
            else:
                logger.info("Collection '%s' is already empty",
                            self.collection_name)
        except Exception as e:
            logger.error("Error clearing collection '%s': %s",
                         self.collection_name, e)
            raise

    def get_collection_count(self) -> int:
        """Get the number of documents in the collection."""
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(
                "Error getting count for collection '%s': %s", self.collection_name, e)
            raise

    def _create_embedding_function(self) -> EmbeddingFunction:
        """Create and return the LLM embedding function."""
        class LLMChromaEmbeddingFunction(EmbeddingFunction):
            def __init__(self, llm_client: LLMManager):
                self.llm_client = llm_client

            def __call__(self, input_texts: List[str]) -> List[List[float]]:
                raw_embeddings = self.llm_client.get_embedding_sync(
                    input_texts)
                valid_embeddings: List[List[float]] = []
                embedding_dim = 768  # For "models/embedding-001"

                for i, emb in enumerate(raw_embeddings):
                    if emb is None or len(emb) != embedding_dim:
                        logger.warning(
                            "Invalid embedding for text at index %d. Using zero vector.", i)
                        valid_embeddings.append([0.0] * embedding_dim)
                    else:
                        valid_embeddings.append(emb)

                return valid_embeddings

        return LLMChromaEmbeddingFunction(self.llm_manager)

    async def _run_collection_method(self, method_name: str, *pos_args, **kw_args) -> Any:
        """
        Helper to run a synchronous method of self.collection in a thread executor.
        It correctly uses functools.partial to bind all arguments.
        """
        loop = asyncio.get_running_loop()
        method_to_call = getattr(self.collection, method_name)

        # Create a partial function that has all arguments (positional and keyword) bound to it.
        # This partial function will then be called by run_in_executor without any additional arguments.
        func_with_bound_args = functools.partial(
            method_to_call, *pos_args, **kw_args)

        # DEBUG: Print what's being prepared for the executor
        # logger.debug(f"Executing in thread: {method_name} with pos_args={pos_args}, kw_args={kw_args}")
        # logger.debug(f"Partial function details: {func_with_bound_args.func}, {func_with_bound_args.args}, {func_with_bound_args.keywords}")

        return await loop.run_in_executor(None, func_with_bound_args)

    def generate_id_from_text_and_source(self, text: str, source: str) -> str:
        return hashlib.md5((text + source).encode('utf-8')).hexdigest()[:16]

    async def query_collection(self, query_text: str, n_results: int = 5) -> List[Dict[str, Any]]:
        if not query_text or not query_text.strip():
            logger.warning("Empty query text provided, returning empty list.")
            return []

        start_time = asyncio.get_event_loop().time()
        try:
            current_count = await self.get_collection_count_async()
            if current_count == 0:
                logger.warning(
                    "Querying empty collection '%s'. Returning no results.", self.collection_name)
                return []

            actual_n_results = min(n_results, current_count)
            if actual_n_results <= 0:
                # ensure n_results is positive if querying
                actual_n_results = 1 if current_count > 0 else 0

            if actual_n_results == 0:  # if collection is empty or n_results forced to 0
                logger.info(
                    "Query for '%.50s...' resulted in 0 n_results. Returning empty list.", query_text)
                return []

            results = await self._run_collection_method(
                'query',  # method_name
                # No positional arguments for collection.query, all are keyword.
                query_texts=[query_text],
                n_results=actual_n_results,
                include=['documents', 'metadatas', 'distances']
            )
            duration = asyncio.get_event_loop().time() - start_time

            formatted_results = []
            # ChromaDB query results structure: results['ids'] is list of lists, etc.
            # Check if the inner list is not None
            if results and results.get('ids') and results['ids'][0] is not None:
                for i in range(len(results['ids'][0])):
                    res = {
                        'id': results['ids'][0][i],
                        'document': results['documents'][0][i] if results.get('documents') and results['documents'][0] else None,
                        'metadata': results['metadatas'][0][i] if results.get('metadatas') and results['metadatas'][0] else None,
                        'distance': results['distances'][0][i] if results.get('distances') and results['distances'][0] else None,
                    }
                    formatted_results.append(res)

            logger.info(
                "Query '%.50s...' returned %d results in %.2fs.",
                query_text, len(formatted_results), duration)
            return formatted_results
        except Exception as e:
            logger.error(
                "Error querying ChromaDB collection: %s", e, exc_info=True)
            return []

    async def get_collection_count_async(self) -> int:
        try:
            # collection.count() takes no arguments
            return await self._run_collection_method('count')
        except Exception as e:
            logger.error("Error getting collection count: %s",
                         e, exc_info=True)
            return 0

    async def clear_collection_async(self):
        logger.warning(
            "Clearing collection '%s'. This will delete and recreate it.", self.collection_name)
        try:
            # These client operations are synchronous and not part of the _run_collection_method helper
            await asyncio.to_thread(self.client.delete_collection, name=self.collection_name)

            new_collection_instance = await asyncio.to_thread(
                self.client.get_or_create_collection,
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            self.collection = new_collection_instance

            new_count = await self.get_collection_count_async()
            logger.info(
                "Collection '%s' cleared and recreated. New count: %d", self.collection_name, new_count)
        except Exception as e:
            logger.error(
                "Error clearing collection '%s': %s", self.collection_name, e, exc_info=True)
