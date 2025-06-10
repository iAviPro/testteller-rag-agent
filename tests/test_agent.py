import pytest
import asyncio
from unittest import mock
import logging
import os # For os.path mocks
from pathlib import Path # For Path mocks

# Import the class to be tested
from agent import TestTellerRagAgent

# Import classes that will be mocked
from llm.gemini_client import GeminiClient
from vector_store.chromadb_manager import ChromaDBManager
from data_ingestion.document_loader import DocumentLoader
from data_ingestion.code_loader import CodeLoader
from data_ingestion.text_splitter import TextSplitter
# Agent uses global settings from config.py
# We need to ensure this is controlled during tests.

@pytest.fixture(scope="function")
def isolated_agent_settings(monkeypatch):
    from config import AppSettings # Import here to get a fresh instance based on env
    current_test_app_settings = AppSettings()
    monkeypatch.setattr("agent.settings", current_test_app_settings)
    # If TestTellerRagAgent imports settings from config directly:
    monkeypatch.setattr("config.settings", current_test_app_settings, raising=False)
    yield current_test_app_settings

@pytest.fixture
def mock_gemini_client(mocker):
    mock_client = mocker.MagicMock(spec=GeminiClient)
    mock_client.generate_text_async = mocker.AsyncMock(return_value="mocked gemini text")
    mock_client.get_embedding_async = mocker.AsyncMock(return_value=[0.1, 0.2])
    return mock_client

@pytest.fixture
def mock_vector_store(mocker):
    mock_store = mocker.MagicMock(spec=ChromaDBManager)
    mock_store.add_documents_async = mocker.AsyncMock()
    mock_store.query_collection_async = mocker.AsyncMock(return_value=[])
    mock_store.get_collection_count_async = mocker.AsyncMock(return_value=0)
    mock_store.clear_collection_async = mocker.AsyncMock()
    mock_store.generate_id_from_text_and_source = mocker.MagicMock(return_value="mock_doc_id")
    return mock_store

@pytest.fixture
def mock_document_loader(mocker):
    mock_loader = mocker.MagicMock(spec=DocumentLoader)
    mock_loader.load_document = mocker.AsyncMock(return_value="mock file content")
    mock_loader.load_from_directory = mocker.AsyncMock(return_value=[("doc_path1", "doc_content1")])
    return mock_loader

@pytest.fixture
def mock_code_loader(mocker):
    mock_loader = mocker.MagicMock(spec=CodeLoader)
    mock_loader.load_code_from_repo = mocker.AsyncMock(return_value=[("repo/file.py", "repo code")])
    mock_loader.load_code_from_local_folder = mocker.AsyncMock(return_value=[("local/file.py", "local code")])
    mock_loader.cleanup_repo = mocker.AsyncMock()
    mock_loader.cleanup_all_repos = mocker.AsyncMock()
    return mock_loader

@pytest.fixture
def mock_text_splitter(mocker):
    mock_splitter = mocker.MagicMock(spec=TextSplitter)
    mock_splitter.split_text = mocker.MagicMock(return_value=["chunk1", "chunk2"]) # Sync method
    return mock_splitter

@pytest.fixture
def agent_fixt(
    isolated_agent_settings, # Ensure settings are patched before agent instantiation
    mocker,
    mock_gemini_client,
    mock_vector_store,
    mock_document_loader,
    mock_code_loader,
    mock_text_splitter
):
    # Patch constructors in the 'agent' module's namespace
    patched_gemini_constructor = mocker.patch("agent.GeminiClient", return_value=mock_gemini_client)
    patched_chromadb_constructor = mocker.patch("agent.ChromaDBManager", return_value=mock_vector_store)
    mocker.patch("agent.DocumentLoader", return_value=mock_document_loader)
    mocker.patch("agent.CodeLoader", return_value=mock_code_loader)
    mocker.patch("agent.TextSplitter", return_value=mock_text_splitter)

    agent = TestTellerRagAgent(collection_name="test_agent_collection")

    # Store patched constructors on the agent instance for easy access in tests, if needed,
    # or retrieve them via mocker. Mocker itself keeps track of patched objects.
    agent._patched_chromadb_constructor_for_test = patched_chromadb_constructor
    agent._patched_gemini_constructor_for_test = patched_gemini_constructor
    return agent

# --- Test __init__ ---
def test_agent_init(agent_fixt):
    # agent_fixt already creates an agent. This test verifies constructor calls.
    # Access the patched constructors through the agent if stored, or re-fetch via mocker

    # Check GeminiClient was constructed
    agent_fixt._patched_gemini_constructor_for_test.assert_called_once()

    # Check ChromaDBManager was constructed with correct args
    agent_fixt._patched_chromadb_constructor_for_test.assert_called_once()
    init_kwargs = agent_fixt._patched_chromadb_constructor_for_test.call_args[1]
    assert init_kwargs['collection_name'] == "test_agent_collection"
    # Ensure the gemini_client instance passed to ChromaDBManager is the one from our mock
    assert init_kwargs['gemini_client'] == agent_fixt.gemini_client

    assert agent_fixt.collection_name == "test_agent_collection"

# --- Test _ingest_content ---
@pytest.mark.asyncio
async def test_ingest_content_success(agent_fixt, caplog):
    caplog.set_level(logging.INFO)
    mock_contents_with_paths = [("file1.py", "code1"), ("file2.txt", "text1")]
    source_type = "test_src"

    def split_text_side_effect(text_content):
        return [f"chunk_{text_content}_1", f"chunk_{text_content}_2"]
    agent_fixt.text_splitter.split_text.side_effect = split_text_side_effect

    id_counter = 0
    def id_gen_side_effect(text, source):
        nonlocal id_counter; id_counter += 1; return f"id_{id_counter}"
    agent_fixt.vector_store.generate_id_from_text_and_source.side_effect = id_gen_side_effect

    await agent_fixt._ingest_content(mock_contents_with_paths, source_type)

    agent_fixt.text_splitter.split_text.assert_any_call("code1")
    agent_fixt.text_splitter.split_text.assert_any_call("text1")

    agent_fixt.vector_store.add_documents_async.assert_called_once()
    call_kwargs = agent_fixt.vector_store.add_documents_async.call_args[1]

    assert call_kwargs['documents'] == ["chunk_code1_1", "chunk_code1_2", "chunk_text1_1", "chunk_text1_2"]
    assert call_kwargs['ids'] == ["id_1", "id_2", "id_3", "id_4"]
    assert len(call_kwargs['metadatas']) == 4
    assert call_kwargs['metadatas'][0]['source'] == "file1.py"
    assert call_kwargs['metadatas'][2]['source'] == "file2.txt"
    assert "Content preparation for 4 chunks from 2 sources took" in caplog.text

@pytest.mark.asyncio
async def test_ingest_content_empty_input(agent_fixt, caplog):
    await agent_fixt._ingest_content([], "empty_src")
    agent_fixt.vector_store.add_documents_async.assert_not_called()
    assert "No content provided for ingestion from empty_src." in caplog.text

@pytest.mark.asyncio
async def test_ingest_content_no_chunks_produced(agent_fixt, caplog):
    agent_fixt.text_splitter.split_text.return_value = []
    await agent_fixt._ingest_content([("file.py", "content")], "no_chunks_src")
    agent_fixt.vector_store.add_documents_async.assert_not_called()
    assert "No valid chunks to ingest from no_chunks_src." in caplog.text

# --- Test ingest_documents_from_path ---
@pytest.mark.asyncio
async def test_ingest_documents_from_path_single_file(agent_fixt, mocker, caplog):
    path_str = "/fake/doc.txt"
    mocker.patch("os.path.isfile", return_value=True)
    mocker.patch("os.path.isdir", return_value=False)
    agent_fixt._ingest_content = mocker.AsyncMock() # Mock the actual ingestion logic

    await agent_fixt.ingest_documents_from_path(path_str)

    agent_fixt.document_loader.load_document.assert_called_once_with(path_str)
    agent_fixt._ingest_content.assert_called_once_with(
        [(path_str, "mock file content")], source_type="document_file"
    )

@pytest.mark.asyncio
async def test_ingest_documents_from_path_directory(agent_fixt, mocker, caplog):
    path_str = "/fake/docs_dir"
    mocker.patch("os.path.isfile", return_value=False)
    mocker.patch("os.path.isdir", return_value=True)
    agent_fixt._ingest_content = mocker.AsyncMock()

    await agent_fixt.ingest_documents_from_path(path_str)
    agent_fixt.document_loader.load_from_directory.assert_called_once_with(path_str)
    agent_fixt._ingest_content.assert_called_once_with(
        [("doc_path1", "doc_content1")], source_type="document_directory"
    )

@pytest.mark.asyncio
async def test_ingest_documents_from_path_non_existent(agent_fixt, mocker, caplog):
    path_str = "/non/existent"
    mocker.patch("os.path.isfile", return_value=False)
    mocker.patch("os.path.isdir", return_value=False)
    agent_fixt._ingest_content = mocker.AsyncMock()
    await agent_fixt.ingest_documents_from_path(path_str)
    agent_fixt._ingest_content.assert_not_called()
    assert f"Path does not exist or is not a file/directory: {path_str}" in caplog.text

# --- Test ingest_code_from_source ---
@pytest.mark.asyncio
async def test_ingest_code_from_source_repo_url(agent_fixt, mocker, caplog):
    repo_url = "https://github.com/user/repo.git" # This is a URL
    agent_fixt._ingest_content = mocker.AsyncMock()
    # Mock Path so that Path(repo_url).exists() is False for URL
    mocker.patch("agent.Path", lambda p: mocker.MagicMock(exists=lambda: False, __str__=lambda: p))

    await agent_fixt.ingest_code_from_source(repo_url, cleanup_github_after=True)
    agent_fixt.code_loader.load_code_from_repo.assert_called_once_with(repo_url)
    agent_fixt._ingest_content.assert_called_once_with(
        [("repo/file.py", "repo code")], source_type="github_code"
    )
    agent_fixt.code_loader.cleanup_repo.assert_called_once_with(repo_url)

@pytest.mark.asyncio
async def test_ingest_code_from_source_local_path(agent_fixt, mocker, caplog):
    local_path_str = "/local/codepath" # This is a local path
    # Mock Path for this specific path to exist and be a dir
    mock_path_obj = mocker.MagicMock(spec=Path)
    mock_path_obj.exists.return_value = True
    mock_path_obj.is_dir.return_value = True
    mocker.patch("agent.Path", lambda p: mock_path_obj if p == local_path_str else mocker.MagicMock(exists=lambda: False))

    agent_fixt._ingest_content = mocker.AsyncMock()
    await agent_fixt.ingest_code_from_source(local_path_str)
    agent_fixt.code_loader.load_code_from_local_folder.assert_called_once_with(local_path_str)
    agent_fixt._ingest_content.assert_called_once_with(
        [("local/file.py", "local code")], source_type="local_code_folder"
    )

@pytest.mark.asyncio
async def test_ingest_code_from_source_invalid_input(agent_fixt, mocker, caplog):
    invalid_source = "neither_url_nor_path"
    mocker.patch("agent.Path", lambda p: mocker.MagicMock(exists=lambda: False)) # Path does not exist
    agent_fixt._ingest_content = mocker.AsyncMock()
    await agent_fixt.ingest_code_from_source(invalid_source)
    agent_fixt._ingest_content.assert_not_called()
    assert f"Invalid source provided: {invalid_source}" in caplog.text

# --- Test get_ingested_data_count ---
@pytest.mark.asyncio
async def test_get_ingested_data_count(agent_fixt):
    agent_fixt.vector_store.get_collection_count_async.return_value = 42
    count = await agent_fixt.get_ingested_data_count()
    assert count == 42

# --- Test clear_ingested_data ---
@pytest.mark.asyncio
async def test_clear_ingested_data(agent_fixt, caplog):
    await agent_fixt.clear_ingested_data()
    agent_fixt.vector_store.clear_collection_async.assert_called_once()
    agent_fixt.code_loader.cleanup_all_repos.assert_called_once()
    assert "All ingested data cleared for collection" in caplog.text # Check log message

# --- Test generate_test_cases ---
@pytest.mark.asyncio
async def test_generate_test_cases_with_context(agent_fixt, caplog):
    query = "Feature X"
    agent_fixt.vector_store.get_collection_count_async.return_value = 5 # Data exists
    retrieved_docs = [{'document': "context doc1", 'metadata': {'source': 's1'}, 'distance': 0.1}]
    agent_fixt.vector_store.query_collection_async.return_value = retrieved_docs

    response = await agent_fixt.generate_test_cases(query, n_retrieved_docs=1)
    assert response == "mocked gemini text"
    agent_fixt.vector_store.query_collection_async.assert_called_once_with(query_text=query, n_results=1)
    prompt_arg = agent_fixt.gemini_client.generate_text_async.call_args[0][0]
    assert "context doc1" in prompt_arg
    assert "Query: Feature X" in prompt_arg

@pytest.mark.asyncio
async def test_generate_test_cases_no_context_docs_found(agent_fixt, caplog):
    query = "Feature Y"
    agent_fixt.vector_store.get_collection_count_async.return_value = 5
    agent_fixt.vector_store.query_collection_async.return_value = [] # No docs found

    await agent_fixt.generate_test_cases(query)
    prompt_arg = agent_fixt.gemini_client.generate_text_async.call_args[0][0]
    assert "No relevant context documents were found" in prompt_arg

@pytest.mark.asyncio
async def test_generate_test_cases_empty_kb(agent_fixt, caplog):
    query = "Feature Z"
    agent_fixt.vector_store.get_collection_count_async.return_value = 0 # KB is empty

    await agent_fixt.generate_test_cases(query)
    prompt_arg = agent_fixt.gemini_client.generate_text_async.call_args[0][0]
    assert "No specific context documents were found in the knowledge base" in prompt_arg
    agent_fixt.vector_store.query_collection_async.assert_not_called()

@pytest.mark.asyncio
async def test_generate_test_cases_n_retrieved_docs_zero(agent_fixt, caplog):
    query = "Feature A"
    agent_fixt.vector_store.get_collection_count_async.return_value = 5
    # query_collection_async mock default returns [] so this covers n_results=0 if it results in no docs

    await agent_fixt.generate_test_cases(query, n_retrieved_docs=0)
    agent_fixt.vector_store.query_collection_async.assert_called_once_with(query_text=query, n_results=0)
    prompt_arg = agent_fixt.gemini_client.generate_text_async.call_args[0][0]
    assert "No relevant context documents were found" in prompt_arg

```
