import pytest
from unittest import mock # To be used by mocker or directly
from pathlib import Path # For mocking Path objects

from typer.testing import CliRunner
from main import app # Assuming your Typer app instance is named 'app' in main.py
from config import settings as global_app_settings # To get default collection name

# We need to mock TestTellerRagAgent before it's used by the commands in main.py
# The agent is imported in main.py. So, we need to patch it in main's namespace.

@pytest.fixture(scope="module")
def mock_agent_class_for_main(mocker): # Renamed to avoid conflict if other test files use similar name
    """Mocks the TestTellerRagAgent class in the main module."""
    mock_agent_cls = mocker.patch("main.TestTellerRagAgent")
    mock_agent_instance = mocker.MagicMock(name="MockTestTellerRagAgentInstanceInMain")
    mock_agent_cls.return_value = mock_agent_instance

    # Define async methods on the MOCK INSTANCE
    mock_agent_instance.ingest_documents_from_path = mocker.AsyncMock(name="ingest_docs_mock")
    mock_agent_instance.ingest_code_from_source = mocker.AsyncMock(name="ingest_code_mock")
    mock_agent_instance.generate_test_cases = mocker.AsyncMock(name="generate_tests_mock", return_value="Generated test cases from mock agent.")
    mock_agent_instance.get_ingested_data_count = mocker.AsyncMock(name="get_count_mock", return_value=0)
    mock_agent_instance.clear_ingested_data = mocker.AsyncMock(name="clear_data_mock")
    return mock_agent_cls

@pytest.fixture
def runner():
    return CliRunner()

@pytest.fixture(autouse=True)
def reset_main_agent_instance_mocks(mock_agent_class_for_main): # Use the renamed fixture
    """Resets mocks on the agent INSTANCE before each test."""
    mock_instance = mock_agent_class_for_main.return_value
    methods_to_reset = [
        'ingest_documents_from_path', 'ingest_code_from_source',
        'generate_test_cases', 'get_ingested_data_count', 'clear_ingested_data'
    ]
    for method_name in methods_to_reset:
        getattr(mock_instance, method_name).reset_mock()
    mock_instance.get_ingested_data_count.return_value = 0
    mock_instance.generate_test_cases.return_value = "Generated test cases from mock agent."

# --- Tests for 'ingest-docs' command ---
def test_ingest_docs_success(runner, mock_agent_class_for_main, mocker):
    mock_instance = mock_agent_class_for_main.return_value
    test_path = "dummy_docs_path/"
    mocker.patch("os.path.exists", return_value=True)

    result = runner.invoke(app, ["ingest-docs", test_path, "--collection", "custom_docs"])

    assert result.exit_code == 0, result.stdout
    assert f"Starting ingestion for document(s) from path: {test_path}" in result.stdout
    assert "Document ingestion completed." in result.stdout
    mock_agent_class_for_main.assert_called_with(collection_name="custom_docs")
    mock_instance.ingest_documents_from_path.assert_called_once_with(test_path)

def test_ingest_docs_non_existent_path(runner, mock_agent_class_for_main, mocker):
    mock_instance = mock_agent_class_for_main.return_value
    test_path = "non_existent_path/"
    mocker.patch("os.path.exists", return_value=False)
    result = runner.invoke(app, ["ingest-docs", test_path])
    assert result.exit_code == 1 # Typer exits 1 for errors like this
    assert f"Error: Path '{test_path}' does not exist." in result.stdout
    mock_instance.ingest_documents_from_path.assert_not_called()

# --- Tests for 'ingest-code' command ---
def test_ingest_code_success_url(runner, mock_agent_class_for_main, mocker):
    mock_instance = mock_agent_class_for_main.return_value
    test_url = "https://github.com/example/repo.git"
    # For URL, os.path.exists and Path().exists might be false, which is fine for URLs
    mocker.patch("os.path.exists", return_value=False)
    mocker.patch("main.Path", lambda p: mocker.MagicMock(exists=lambda: False, __str__=lambda: p))


    result = runner.invoke(app, ["ingest-code", test_url, "--collection", "custom_code", "--no-cleanup"])
    assert result.exit_code == 0, result.stdout
    assert f"Starting ingestion for code from source: {test_url}" in result.stdout
    mock_agent_class_for_main.assert_called_with(collection_name="custom_code")
    mock_instance.ingest_code_from_source.assert_called_once_with(test_url, cleanup_github_after=False)

def test_ingest_code_success_local_path(runner, mock_agent_class_for_main, mocker):
    mock_instance = mock_agent_class_for_main.return_value
    test_path = "dummy_code_path/"
    # Mock Path(test_path).exists() to be True for local path
    mock_path_obj = mocker.MagicMock(spec=Path)
    mock_path_obj.exists.return_value = True
    mock_path_obj.__str__.return_value = test_path
    mocker.patch("main.Path", lambda p_arg: mock_path_obj if str(p_arg) == test_path else mocker.MagicMock(exists=lambda:False))

    result = runner.invoke(app, ["ingest-code", test_path, "--cleanup"])
    assert result.exit_code == 0, result.stdout
    # Default collection name is used here.
    # The agent constructor will be called with global_app_settings.default_collection_name
    default_col_name = global_app_settings.chroma_db.default_collection_name
    mock_agent_class_for_main.assert_called_with(collection_name=default_col_name)
    mock_instance.ingest_code_from_source.assert_called_once_with(test_path, cleanup_github_after=True)

def test_ingest_code_non_existent_local_path(runner, mock_agent_class_for_main, mocker):
    mock_instance = mock_agent_class_for_main.return_value
    test_path = "non_existent_code_path/"
    mocker.patch("main.Path", lambda p: mocker.MagicMock(exists=lambda: False, __str__=lambda:p)) # Path does not exist

    result = runner.invoke(app, ["ingest-code", test_path])
    assert result.exit_code == 1
    assert f"Error: Local path '{test_path}' does not exist and it's not a valid URL." in result.stdout
    mock_instance.ingest_code_from_source.assert_not_called()

# --- Tests for 'generate' command ---
def test_generate_success(runner, mock_agent_class_for_main):
    mock_instance = mock_agent_class_for_main.return_value
    mock_instance.get_ingested_data_count.return_value = 5
    query = "Test query for generation"
    result = runner.invoke(app, ["generate", query, "--n-docs", "3", "--collection", "gen_coll"])
    assert result.exit_code == 0, result.stdout
    assert "Generated test cases from mock agent." in result.stdout
    mock_agent_class_for_main.assert_called_with(collection_name="gen_coll")
    mock_instance.generate_test_cases.assert_called_once_with(query, n_retrieved_docs=3)

def test_generate_empty_collection_confirm_yes(runner, mock_agent_class_for_main, mocker):
    mock_instance = mock_agent_class_for_main.return_value
    mock_instance.get_ingested_data_count.return_value = 0
    mocker.patch("typer.confirm", return_value=True)
    result = runner.invoke(app, ["generate", "query"])
    assert result.exit_code == 0, result.stdout
    assert "Warning: The knowledge base is currently empty." in result.stdout
    assert "Generated test cases from mock agent." in result.stdout
    mock_instance.generate_test_cases.assert_called_once_with("query", n_retrieved_docs=5)

def test_generate_empty_collection_confirm_no(runner, mock_agent_class_for_main, mocker):
    mock_instance = mock_agent_class_for_main.return_value
    mock_instance.get_ingested_data_count.return_value = 0
    mocker.patch("typer.confirm", return_value=False)
    result = runner.invoke(app, ["generate", "query"])
    assert result.exit_code == 0, result.stdout # Typer confirm abort usually exits 0
    assert "Aborted!" in result.stdout # Typer's default abort message with !
    mock_instance.generate_test_cases.assert_not_called()

# --- Tests for 'status' command ---
def test_status_with_data(runner, mock_agent_class_for_main):
    mock_instance = mock_agent_class_for_main.return_value
    mock_instance.get_ingested_data_count.return_value = 120
    result = runner.invoke(app, ["status", "--collection", "status_coll"])
    assert result.exit_code == 0, result.stdout
    assert "Knowledge Base Status for collection 'status_coll':" in result.stdout
    assert "Number of ingested documents: 120" in result.stdout
    mock_agent_class_for_main.assert_called_with(collection_name="status_coll")
    mock_instance.get_ingested_data_count.assert_called_once()

def test_status_empty_kb(runner, mock_agent_class_for_main):
    mock_instance = mock_agent_class_for_main.return_value
    mock_instance.get_ingested_data_count.return_value = 0
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0, result.stdout
    assert "Number of ingested documents: 0" in result.stdout

# --- Tests for 'clear-data' command ---
def test_clear_data_force(runner, mock_agent_class_for_main, mocker):
    mock_instance = mock_agent_class_for_main.return_value
    confirm_mock = mocker.patch("typer.confirm")
    result = runner.invoke(app, ["clear-data", "--collection", "clear_coll", "--force"])
    assert result.exit_code == 0, result.stdout
    assert "Knowledge base 'clear_coll' cleared." in result.stdout
    mock_agent_class_for_main.assert_called_with(collection_name="clear_coll")
    mock_instance.clear_ingested_data.assert_called_once()
    confirm_mock.assert_not_called()

def test_clear_data_confirm_yes(runner, mock_agent_class_for_main, mocker):
    mock_instance = mock_agent_class_for_main.return_value
    mocker.patch("typer.confirm", return_value=True)
    result = runner.invoke(app, ["clear-data"])
    assert result.exit_code == 0, result.stdout
    assert "cleared." in result.stdout
    mocker.patch("typer.confirm").assert_called_once() # Check it was called
    mock_instance.clear_ingested_data.assert_called_once()

def test_clear_data_confirm_no(runner, mock_agent_class_for_main, mocker):
    mock_instance = mock_agent_class_for_main.return_value
    mocker.patch("typer.confirm", return_value=False)
    result = runner.invoke(app, ["clear-data"])
    assert result.exit_code == 0, result.stdout
    assert "Clear operation aborted." in result.stdout
    mocker.patch("typer.confirm").assert_called_once()
    mock_instance.clear_ingested_data.assert_not_called()

# --- Test missing arguments (Typer handling) ---
def test_missing_argument_ingest_docs(runner):
    result = runner.invoke(app, ["ingest-docs"])
    assert result.exit_code != 0 # Typer exits 2 for CLI errors like missing args
    assert "Missing argument 'PATH'." in result.stdout

def test_missing_argument_generate(runner):
    result = runner.invoke(app, ["generate"])
    assert result.exit_code != 0
    assert "Missing argument 'QUERY'." in result.stdout

```
