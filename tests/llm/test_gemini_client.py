import pytest
import asyncio
from unittest import mock
import sys # For sys.modules mocking
import logging

# --- Google GenerativeAI Mocking ---
# Mock google.generativeai before importing GeminiClient
google_module_mock = mock.MagicMock(name="google_module_mock") # Mock for 'google'
genai_mock = mock.MagicMock(name="genai_mock") # Mock for 'google.generativeai'

# Configure the 'google' module mock to have a 'generativeai' attribute
google_module_mock.generativeai = genai_mock

# Mock the specific functions/classes used by GeminiClient from google.generativeai
genai_mock.configure = mock.MagicMock(name="configure_mock")
genai_mock.GenerativeModel = mock.MagicMock(name="GenerativeModel_class_mock")
generative_model_instance_mock = mock.MagicMock(name="GenerativeModel_instance_mock")
genai_mock.GenerativeModel.return_value = generative_model_instance_mock # Model() returns our instance
# Mock methods on the instance, ensuring they are AsyncMocks for awaitable calls
generative_model_instance_mock.generate_content_async = mock.AsyncMock(name="generate_content_async_mock")
genai_mock.embed_content_async = mock.AsyncMock(name="embed_content_async_mock")


# Apply the mocks to sys.modules
# This setup ensures that when GeminiClient does `import google.generativeai as genai`,
# it gets our fully mocked `genai_mock`.
# Store original sys.modules state for 'google' and 'google.generativeai'
original_sys_modules_google = sys.modules.get('google')
original_sys_modules_google_genai = sys.modules.get('google.generativeai')

sys.modules['google'] = google_module_mock
sys.modules['google.generativeai'] = genai_mock
# --- End Google GenerativeAI Mocking ---

# Now import GeminiClient and relevant settings/config
from llm.gemini_client import GeminiClient
from config import settings as global_settings_config_module, AppSettings, ApiKeysSettings, GeminiModelSettings

# --- Restore sys.modules after imports ---
# This is crucial to prevent these top-level mocks from leaking into other test files.
# Restore 'google.generativeai'
if original_sys_modules_google_genai:
    sys.modules['google.generativeai'] = original_sys_modules_google_genai
else:
    if 'google.generativeai' in sys.modules:
        del sys.modules['google.generativeai']
# Restore 'google'
if original_sys_modules_google:
    sys.modules['google'] = original_sys_modules_google
else:
    if 'google' in sys.modules:
        del sys.modules['google']
# --- End Restore sys.modules ---


# --- Fixture to manage global settings for tests ---
@pytest.fixture(scope="function") # No autouse, explicitly request it
def isolated_settings(monkeypatch):
    """
    Ensures each test function gets a fresh AppSettings instance, loaded from an environment
    optionally modified by other fixtures. Patches this instance into llm.gemini_client.settings.
    """
    # Preserve the original settings object from the config module at the start of the test session
    # This is tricky because AppSettings() itself loads from env.
    # The goal is that llm.gemini_client.settings uses a version of settings
    # that reflects the env vars set for THAT specific test function.

    # Create a new AppSettings instance. It will load from the current state of os.environ.
    current_test_app_settings = AppSettings()

    # Monkeypatch the 'settings' object in the module where GeminiClient is defined
    monkeypatch.setattr("llm.gemini_client.settings", current_test_app_settings)

    # Also patch the 'settings' object in the config module itself, in case any other
    # part of the system indirectly uses it and expects it to be consistent.
    monkeypatch.setattr("config.settings", current_test_app_settings)

    yield current_test_app_settings


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mocks environment variables for GeminiClient settings for a typical success case."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test_api_key_from_env")
    monkeypatch.setenv("GEMINI_EMBEDDING_MODEL", "env-embedding-model")
    monkeypatch.setenv("GEMINI_GENERATION_MODEL", "env-generation-model")
    # The isolated_settings fixture, when called after this, will ensure AppSettings reloads with these.

@pytest.fixture
def gemini_client_fixt(mock_env_vars, isolated_settings):
    """Fixture to create a GeminiClient instance with mocked settings and genai."""
    # Reset mocks. These are global mocks, so they need resetting for each test.
    genai_mock.configure.reset_mock()
    genai_mock.GenerativeModel.reset_mock()
    generative_model_instance_mock.generate_content_async.reset_mock()
    genai_mock.embed_content_async.reset_mock()

    # `isolated_settings` has already run and patched `llm.gemini_client.settings`
    # with an `AppSettings` instance that reflects `mock_env_vars`.
    client = GeminiClient()
    return client

# --- Tests for GeminiClient.__init__ ---

def test_gemini_client_init_success(gemini_client_fixt, isolated_settings):
    current_settings = isolated_settings

    genai_mock.configure.assert_called_once_with(
        api_key=current_settings.api_keys.google_api_key.get_secret_value()
    )
    genai_mock.GenerativeModel.assert_called_once_with(
        current_settings.gemini_model.gemini_generation_model
    )
    assert gemini_client_fixt.generation_model is generative_model_instance_mock
    assert gemini_client_fixt.embedding_model_name == current_settings.gemini_model.gemini_embedding_model

def test_gemini_client_init_missing_api_key(monkeypatch, caplog): # No isolated_settings needed if AppSettings itself fails
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(Exception) as excinfo:
        AppSettings() # This will fail due to Pydantic validation in ApiKeysSettings
    assert "GOOGLE_API_KEY" in str(excinfo.value).upper()
    assert "field required" in str(excinfo.value).lower() or "must be set" in str(excinfo.value).lower()

# --- Tests for GeminiClient.generate_text_async ---

@pytest.mark.asyncio
async def test_generate_text_async_success(gemini_client_fixt):
    prompt = "Test prompt"
    expected_response_text = "Generated text response."
    mock_response = mock.MagicMock() # This mock is for the response object from the SDK
    mock_response.text = expected_response_text
    generative_model_instance_mock.generate_content_async.return_value = mock_response

    response = await gemini_client_fixt.generate_text_async(prompt)
    assert response == expected_response_text
    generative_model_instance_mock.generate_content_async.assert_called_once_with(prompt)

@pytest.mark.asyncio
async def test_generate_text_async_api_error(gemini_client_fixt, caplog):
    prompt = "Error prompt"
    caplog.set_level(logging.ERROR)
    generative_model_instance_mock.generate_content_async.side_effect = Exception("Gemini API Error")
    response = await gemini_client_fixt.generate_text_async(prompt)
    assert response == ""
    assert "Error generating text with Gemini: Gemini API Error" in caplog.text

@pytest.mark.asyncio
async def test_generate_text_async_empty_prompt(gemini_client_fixt, caplog):
    caplog.set_level(logging.WARNING)
    response = await gemini_client_fixt.generate_text_async("")
    assert response == ""
    assert "Empty prompt received. Returning empty response." in caplog.text
    generative_model_instance_mock.generate_content_async.assert_not_called()

# --- Tests for GeminiClient.get_embedding_async ---

@pytest.mark.asyncio
async def test_get_embedding_async_success(gemini_client_fixt, isolated_settings):
    text_to_embed = "Embed this text."
    expected_embedding = [0.1, 0.2, 0.3]
    current_settings = isolated_settings
    genai_mock.embed_content_async.return_value = {'embedding': expected_embedding}

    embedding = await gemini_client_fixt.get_embedding_async(text_to_embed)
    assert embedding == expected_embedding
    genai_mock.embed_content_async.assert_called_once_with(
        model=current_settings.gemini_model.gemini_embedding_model,
        content=text_to_embed,
        task_type="retrieval_document"
    )

@pytest.mark.asyncio
async def test_get_embedding_async_api_error(gemini_client_fixt, caplog):
    text_to_embed = "Error embedding this."
    caplog.set_level(logging.ERROR)
    genai_mock.embed_content_async.side_effect = Exception("Gemini Embedding Error")
    embedding = await gemini_client_fixt.get_embedding_async(text_to_embed)
    assert embedding == []
    assert f"Error getting embedding with Gemini for '{text_to_embed[:50]}...': Gemini Embedding Error" in caplog.text

@pytest.mark.asyncio
async def test_get_embedding_async_empty_text(gemini_client_fixt, caplog):
    caplog.set_level(logging.WARNING)
    embedding = await gemini_client_fixt.get_embedding_async("")
    assert embedding == []
    assert "Empty text received for embedding. Returning empty list." in caplog.text
    genai_mock.embed_content_async.assert_not_called()

# --- Placeholder for Batch Embedding Tests ---
# (If GeminiClient gets a get_embedding_batch_async method)

# --- Sanity check for mock states after a test using the client fixture ---
def test_mock_state_after_fixture_use(gemini_client_fixt, isolated_settings):
    # This test runs after gemini_client_fixt has initialized a client.
    # It verifies that the __init__ calls happened on the mocks.
    current_settings = isolated_settings
    genai_mock.configure.assert_called_once_with(
        api_key=current_settings.api_keys.google_api_key.get_secret_value()
    )
    genai_mock.GenerativeModel.assert_called_once_with(
        current_settings.gemini_model.gemini_generation_model
    )
    # These should not have been called yet if the test itself didn't call client methods
    generative_model_instance_mock.generate_content_async.assert_not_called()
    genai_mock.embed_content_async.assert_not_called()
```
