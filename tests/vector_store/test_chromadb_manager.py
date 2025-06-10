import pytest
import asyncio
from unittest import mock
import sys # For sys.modules mocking
import logging
import hashlib # For testing _generate_id_from_text_and_source
import functools # For checking functools.partial

# --- ChromaDB Mocking ---
chromadb_module_mock = mock.MagicMock(name="chromadb_module_mock")
mock_persistent_client_instance = mock.MagicMock(name="PersistentClient_instance_mock")
chromadb_module_mock.PersistentClient.return_value = mock_persistent_client_instance
mock_collection_instance = mock.MagicMock(name="Collection_instance_mock")
# Configure get_or_create_collection on the client instance to return the collection instance
mock_persistent_client_instance.get_or_create_collection.return_value = mock_collection_instance

# Apply the mock to sys.modules BEFORE ChromaDBManager is imported
original_sys_modules_chromadb = sys.modules.get('chromadb')
sys.modules['chromadb'] = chromadb_module_mock
# --- End ChromaDB Mocking ---

# --- Gemini Client Mocking ---
# Create a mock for the GeminiClient class that ChromaDBManager will receive
MockGeminiClientClass = mock.MagicMock(name="MockGeminiClient_Class") # This is the class
mock_gemini_client_instance = mock.MagicMock(spec_set=["get_embedding_async", "get_embedding_batch_async"], name="MockGeminiClient_Instance")
MockGeminiClientClass.return_value = mock_gemini_client_instance # Instantiating the class returns our instance
mock_gemini_client_instance.get_embedding_async = mock.AsyncMock(name="get_embedding_async_mock")
mock_gemini_client_instance.get_embedding_batch_async = mock.AsyncMock(name="get_embedding_batch_async_mock")
# --- End Gemini Client Mocking ---

# Now import ChromaDBManager and relevant settings/config
from vector_store.chromadb_manager import ChromaDBManager, ChromaEmbeddingFunction
from config import settings as global_settings_original, AppSettings, ChromaDbSettings
# Note: We don't need to import the real GeminiClient here for the tests themselves.

# --- Restore sys.modules after imports ---
if original_sys_modules_chromadb:
    sys.modules['chromadb'] = original_sys_modules_chromadb
else:
    if 'chromadb' in sys.modules: # Only delete if our mock is there
        del sys.modules['chromadb']
# --- End Restore sys.modules ---


# --- Fixtures ---
@pytest.fixture(scope="function")
def isolated_settings(monkeypatch):
    current_test_app_settings = AppSettings()
    monkeypatch.setattr("vector_store.chromadb_manager.settings", current_test_app_settings)
    monkeypatch.setattr("config.settings", current_test_app_settings, raising=False)
    yield current_test_app_settings

@pytest.fixture
def mock_env_vars_chroma(monkeypatch):
    monkeypatch.setenv("CHROMA_DB_PATH", "./test_chroma_data_fixt") # Unique path for fixture
    monkeypatch.setenv("DEFAULT_COLLECTION_NAME", "test_collection_fixt")

@pytest.fixture
def chromadb_manager_fixt(mock_env_vars_chroma, isolated_settings):
    chromadb_module_mock.PersistentClient.reset_mock()
    mock_persistent_client_instance.reset_mock() # Reset instance calls too
    mock_persistent_client_instance.get_or_create_collection.reset_mock()
    mock_collection_instance.reset_mock() # Reset collection instance calls

    mock_gemini_client_instance.get_embedding_async.reset_mock()
    mock_gemini_client_instance.get_embedding_batch_async.reset_mock()

    # Use the mocked Gemini Client *class* to create the instance passed to ChromaDBManager
    gemini_client_to_pass = MockGeminiClientClass()

    manager = ChromaDBManager(
        gemini_client=gemini_client_to_pass,
        collection_name="fixture_collection" # Specific name for clarity
    )
    return manager

# --- Tests for ChromaDBManager.__init__ ---
def test_chromadb_manager_init(chromadb_manager_fixt, isolated_settings):
    current_settings = isolated_settings

    chromadb_module_mock.PersistentClient.assert_called_once_with(
        path=current_settings.chroma_db.chroma_db_path
    )

    assert mock_persistent_client_instance.get_or_create_collection.call_count == 1
    call_args = mock_persistent_client_instance.get_or_create_collection.call_args
    assert call_args[1]['name'] == "fixture_collection"

    # Check the embedding_function type (it's an instance of ChromaEmbeddingFunction)
    ef_instance = call_args[1]['embedding_function']
    assert isinstance(ef_instance, ChromaEmbeddingFunction)
    # And that this instance wraps the Gemini client instance we provided
    assert ef_instance.gemini_client is mock_gemini_client_instance

    assert chromadb_manager_fixt.collection_name == "fixture_collection"
    assert chromadb_manager_fixt.client is mock_persistent_client_instance
    assert chromadb_manager_fixt.collection is mock_collection_instance
    assert chromadb_manager_fixt.gemini_client is mock_gemini_client_instance

# --- Test ChromaEmbeddingFunction.__call__ (used by _embedding_function) ---
def test_chroma_embedding_function_call_sync_manages_async(caplog):
    # Test ChromaEmbeddingFunction directly as it's a key part.
    # This function is synchronous but calls an async method.
    texts_to_embed = ["text1", "text2"]
    expected_embeddings = [[0.1, 0.2], [0.3, 0.4]]

    # Create a new mock Gemini client instance specifically for this test unit
    temp_mock_gemini_instance = mock.MagicMock(spec_set=["get_embedding_batch_async"])
    temp_mock_gemini_instance.get_embedding_batch_async = mock.AsyncMock(return_value=expected_embeddings)

    embedding_function_instance = ChromaEmbeddingFunction(temp_mock_gemini_instance)

    # Mock asyncio.run, which ChromaEmbeddingFunction uses internally.
    # The side_effect executes the coroutine for testing purposes.
    def run_coro_side_effect(coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    with mock.patch("asyncio.run", side_effect=run_coro_side_effect) as mock_asyncio_run:
        embeddings_result = embedding_function_instance(texts_to_embed) # This is a sync call

    assert embeddings_result == expected_embeddings
    temp_mock_gemini_instance.get_embedding_batch_async.assert_called_once_with(texts_to_embed)
    mock_asyncio_run.assert_called_once() # Verify asyncio.run was used

def test_chroma_embedding_function_call_empty_input(caplog):
    temp_mock_gemini_instance = mock.MagicMock(spec_set=["get_embedding_batch_async"])
    temp_mock_gemini_instance.get_embedding_batch_async = mock.AsyncMock()
    embedding_function_instance = ChromaEmbeddingFunction(temp_mock_gemini_instance)
    with mock.patch("asyncio.run") as mock_asyncio_run: # Should not be called
        result = embedding_function_instance([])
    assert result == []
    temp_mock_gemini_instance.get_embedding_batch_async.assert_not_called()
    mock_asyncio_run.assert_not_called()


# --- Test add_documents_async ---
@pytest.mark.asyncio
async def test_add_documents_async_success(chromadb_manager_fixt, caplog):
    caplog.set_level(logging.INFO)
    docs = ["doc1 text", "doc2 text"]
    metadatas = [{"source": "s1"}, {"source": "s2"}]
    ids = ["id1", "id2"]

    # Ensure the collection mock used by the fixture is configured for this call
    chromadb_manager_fixt.collection.add = mock.MagicMock()

    await chromadb_manager_fixt.add_documents_async(docs, metadatas, ids)

    chromadb_manager_fixt.collection.add.assert_called_once_with(
        documents=docs, metadatas=metadatas, ids=ids
    )
    assert f"Added {len(docs)} documents to collection '{chromadb_manager_fixt.collection_name}'" in caplog.text

@pytest.mark.asyncio
async def test_add_documents_async_empty_list(chromadb_manager_fixt, caplog):
    caplog.set_level(logging.INFO)
    await chromadb_manager_fixt.add_documents_async([], [], [])
    chromadb_manager_fixt.collection.add.assert_not_called()
    assert "Attempted to add an empty list of documents." in caplog.text

# --- Test query_collection_async ---
@pytest.mark.asyncio
async def test_query_collection_async_success(chromadb_manager_fixt):
    query_texts = ["query text for test"]
    n_results = 1
    mock_query_response = {
        'ids': [['id1']], 'documents': [['doc1 content']],
        'metadatas': [[{'source': 'test_source'}]], 'distances': [[0.01]]
    }
    chromadb_manager_fixt.collection.query = mock.MagicMock(return_value=mock_query_response)

    results = await chromadb_manager_fixt.query_collection_async(query_texts, n_results)

    chromadb_manager_fixt.collection.query.assert_called_once_with(
        query_texts=query_texts, n_results=n_results, include=['documents', 'metadatas', 'distances']
    )
    assert len(results) == 1
    assert results[0]['id'] == 'id1'
    assert results[0]['document'] == 'doc1 content'
    assert results[0]['metadata'] == {'source': 'test_source'}
    assert results[0]['distance'] == 0.01

@pytest.mark.asyncio
async def test_query_collection_async_no_results_found(chromadb_manager_fixt):
    chromadb_manager_fixt.collection.query = mock.MagicMock(return_value={
        'ids': [[]], 'documents': [[]], 'metadatas': [[]], 'distances': [[]]
    })
    results = await chromadb_manager_fixt.query_collection_async(["query"], 1)
    assert results == []

@pytest.mark.asyncio
async def test_query_collection_async_malformed_response(chromadb_manager_fixt, caplog):
    caplog.set_level(logging.WARNING)
    # Simulate results where lists have different lengths
    chromadb_manager_fixt.collection.query = mock.MagicMock(return_value={
        'ids': [['id1', 'id2']], # 2 ids
        'documents': [['doc1']], # 1 document
        'metadatas': [[{'s': 's1'}]], 'distances': [[0.1]]
    })
    results = await chromadb_manager_fixt.query_collection_async(["q"], 2)
    assert results == [] # Should return empty if data is inconsistent
    assert "Query results for query 0 had mismatched lengths or missing data." in caplog.text

    # Simulate results missing a key
    chromadb_manager_fixt.collection.query.return_value = {'ids': [['id1']]} # Missing documents, metadatas, distances
    results = await chromadb_manager_fixt.query_collection_async(["q"], 1)
    assert results == []
    assert "Query results for query 0 had mismatched lengths or missing data." in caplog.text


# --- Test get_collection_count_async ---
@pytest.mark.asyncio
async def test_get_collection_count_async(chromadb_manager_fixt):
    chromadb_manager_fixt.collection.count.return_value = 42
    count = await chromadb_manager_fixt.get_collection_count_async()
    assert count == 42
    chromadb_manager_fixt.collection.count.assert_called_once()

# --- Test clear_collection_async ---
@pytest.mark.asyncio
async def test_clear_collection_async(chromadb_manager_fixt, isolated_settings, caplog):
    caplog.set_level(logging.INFO)
    original_collection_name = chromadb_manager_fixt.collection_name

    # Mock for the client's delete_collection method
    chromadb_manager_fixt.client.delete_collection = mock.MagicMock()

    # Mock for the client's get_or_create_collection for when it's called after delete
    new_mock_collection_after_clear = mock.MagicMock(name="NewMockCollectionAfterClear")
    chromadb_manager_fixt.client.get_or_create_collection.return_value = new_mock_collection_after_clear

    await chromadb_manager_fixt.clear_collection_async()

    chromadb_manager_fixt.client.delete_collection.assert_called_once_with(original_collection_name)

    # get_or_create_collection was called once in __init__, and again after delete_collection
    assert chromadb_manager_fixt.client.get_or_create_collection.call_count == 2
    second_call_args = chromadb_manager_fixt.client.get_or_create_collection.call_args_list[1]
    assert second_call_args[1]['name'] == original_collection_name
    assert isinstance(second_call_args[1]['embedding_function'], ChromaEmbeddingFunction)

    assert chromadb_manager_fixt.collection is new_mock_collection_after_clear
    assert f"Collection '{original_collection_name}' cleared and recreated." in caplog.text

# --- Test _generate_id_from_text_and_source ---
def test_generate_id_from_text_and_source_consistency(chromadb_manager_fixt):
    text = "Hello, world!"
    source = "test_source.txt"
    expected_string_to_hash = f"{text}-{source}"
    expected_id = hashlib.sha256(expected_string_to_hash.encode('utf-8')).hexdigest()

    assert chromadb_manager_fixt.generate_id_from_text_and_source(text, source) == expected_id

def test_generate_id_from_text_and_source_uniqueness(chromadb_manager_fixt):
    id1 = chromadb_manager_fixt.generate_id_from_text_and_source("text1", "source1")
    id2 = chromadb_manager_fixt.generate_id_from_text_and_source("text2", "source1") # Different text
    id3 = chromadb_manager_fixt.generate_id_from_text_and_source("text1", "source2") # Different source
    id4 = chromadb_manager_fixt.generate_id_from_text_and_source("text1", "source1") # Same as id1

    assert id1 != id2
    assert id1 != id3
    assert id2 != id3
    assert id1 == id4

```
