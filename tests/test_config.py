import pytest
from pydantic import ValidationError
import os
import pytest
from pydantic import ValidationError
from dotenv import load_dotenv

from config import AppSettings, ApiKeysSettings, LoggingSettings, CommonSettings, ChromaDbSettings, GeminiModelSettings, TextProcessingSettings, CodeLoaderSettings, ApiRetrySettings

# Load environment variables from .env file if it exists
load_dotenv()


def test_app_settings_defaults():
    """Test that AppSettings loads default values correctly."""
    # Unset environment variables to ensure defaults are used
    # Store original values to restore them later
    original_env = {}
    vars_to_unset = [
        "GOOGLE_API_KEY", "GITHUB_TOKEN", "CHROMA_DB_PATH",
        "DEFAULT_COLLECTION_NAME", "GEMINI_EMBEDDING_MODEL",
        "GEMINI_GENERATION_MODEL", "CHUNK_SIZE", "CHUNK_OVERLAP",
        "CODE_EXTENSIONS", "TEMP_CLONE_DIR_BASE", "API_RETRY_ATTEMPTS",
        "API_RETRY_WAIT_SECONDS", "LOG_LEVEL", "LOG_FORMAT"
    ]
    for var in vars_to_unset:
        if var in os.environ:
            original_env[var] = os.environ[var]
            del os.environ[var]

    # We need to set a dummy GOOGLE_API_KEY because it's required and has no default
    os.environ["GOOGLE_API_KEY"] = "dummy_key_for_default_testing"

    try:
        settings = AppSettings()

        # Test common settings defaults (although AppSettings doesn't directly use CommonSettings defaults)
        # common_defaults = CommonSettings()
        # assert settings.common.APP_NAME == common_defaults.APP_NAME
        # assert settings.common.APP_VERSION == common_defaults.APP_VERSION

        # Test ApiKeysSettings (GOOGLE_API_KEY is required, GITHUB_TOKEN is Optional)
        assert settings.api_keys.google_api_key.get_secret_value() == "dummy_key_for_default_testing"
        assert settings.api_keys.github_token is None

        # Test ChromaDbSettings defaults
        chroma_defaults = ChromaDbSettings()
        assert settings.chroma_db.chroma_db_path == chroma_defaults.chroma_db_path
        assert settings.chroma_db.default_collection_name == chroma_defaults.default_collection_name

        # Test GeminiModelSettings defaults
        gemini_defaults = GeminiModelSettings()
        assert settings.gemini_model.gemini_embedding_model == gemini_defaults.gemini_embedding_model
        assert settings.gemini_model.gemini_generation_model == gemini_defaults.gemini_generation_model

        # Test TextProcessingSettings defaults
        text_proc_defaults = TextProcessingSettings()
        assert settings.text_processing.chunk_size == text_proc_defaults.chunk_size
        assert settings.text_processing.chunk_overlap == text_proc_defaults.chunk_overlap

        # Test CodeLoaderSettings defaults
        code_loader_defaults = CodeLoaderSettings()
        assert settings.code_loader.code_extensions == code_loader_defaults.code_extensions
        assert settings.code_loader.temp_clone_dir_base == code_loader_defaults.temp_clone_dir_base

        # Test ApiRetrySettings defaults
        api_retry_defaults = ApiRetrySettings()
        assert settings.api_retry.api_retry_attempts == api_retry_defaults.api_retry_attempts
        assert settings.api_retry.api_retry_wait_seconds == api_retry_defaults.api_retry_wait_seconds

        # Test LoggingSettings defaults
        logging_defaults = LoggingSettings()
        assert settings.logging.log_level == logging_defaults.log_level.upper() # Validator makes it upper
        assert settings.logging.log_format == logging_defaults.log_format.lower() # Validator makes it lower

    finally:
        # Restore original environment variables
        for var, value in original_env.items():
            os.environ[var] = value
        # Clean up the dummy key
        if "GOOGLE_API_KEY" in os.environ and os.environ["GOOGLE_API_KEY"] == "dummy_key_for_default_testing":
            del os.environ["GOOGLE_API_KEY"]
        # Reload settings from AppSettings to avoid interference with other tests
        # This is important to ensure that subsequent tests that rely on os.environ get the correct values
        _ = AppSettings.__new__(AppSettings) # Re-instantiate to reload env vars

def test_app_settings_from_env(monkeypatch):
    """Test that AppSettings loads values from environment variables correctly."""
    # Define test values for environment variables
    test_env_vars = {
        "GOOGLE_API_KEY": "env_google_key",
        "GITHUB_TOKEN": "env_github_token",
        "CHROMA_DB_PATH": "/env/chroma/path",
        "DEFAULT_COLLECTION_NAME": "env_collection",
        "GEMINI_EMBEDDING_MODEL": "env/embedding-model",
        "GEMINI_GENERATION_MODEL": "env/generation-model",
        "CHUNK_SIZE": "2000",
        "CHUNK_OVERLAP": "300",
        "CODE_EXTENSIONS": ".py,.md",
        "TEMP_CLONE_DIR_BASE": "/env/temp/clone",
        "API_RETRY_ATTEMPTS": "5",
        "API_RETRY_WAIT_SECONDS": "10",
        "LOG_LEVEL": "DEBUG",
        "LOG_FORMAT": "text"
    }

    # Set environment variables using monkeypatch
    for key, value in test_env_vars.items():
        monkeypatch.setenv(key, value)

    settings = AppSettings()

    assert settings.api_keys.google_api_key.get_secret_value() == test_env_vars["GOOGLE_API_KEY"]
    assert settings.api_keys.github_token.get_secret_value() == test_env_vars["GITHUB_TOKEN"]
    assert settings.chroma_db.chroma_db_path == test_env_vars["CHROMA_DB_PATH"]
    assert settings.chroma_db.default_collection_name == test_env_vars["DEFAULT_COLLECTION_NAME"]
    assert settings.gemini_model.gemini_embedding_model == test_env_vars["GEMINI_EMBEDDING_MODEL"]
    assert settings.gemini_model.gemini_generation_model == test_env_vars["GEMINI_GENERATION_MODEL"]
    assert settings.text_processing.chunk_size == int(test_env_vars["CHUNK_SIZE"])
    assert settings.text_processing.chunk_overlap == int(test_env_vars["CHUNK_OVERLAP"])
    assert settings.code_loader.code_extensions == test_env_vars["CODE_EXTENSIONS"].split(',')
    assert settings.code_loader.temp_clone_dir_base == test_env_vars["TEMP_CLONE_DIR_BASE"]
    assert settings.api_retry.api_retry_attempts == int(test_env_vars["API_RETRY_ATTEMPTS"])
    assert settings.api_retry.api_retry_wait_seconds == int(test_env_vars["API_RETRY_WAIT_SECONDS"])
    assert settings.logging.log_level == test_env_vars["LOG_LEVEL"] # Already upper due to validator
    assert settings.logging.log_format == test_env_vars["LOG_FORMAT"] # Already lower due to validator

    # Clean up monkeypatched environment variables to avoid affecting other tests
    for key in test_env_vars.keys():
        monkeypatch.delenv(key, raising=False)

    # Reload settings from AppSettings to avoid interference with other tests
    _ = AppSettings.__new__(AppSettings) # Re-instantiate to reload env vars


def test_api_keys_settings_validation(monkeypatch):
    """Test validation logic for ApiKeysSettings."""
    # Test missing GOOGLE_API_KEY
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    # Ensure GITHUB_TOKEN is also not set or is valid if set, to isolate the error
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with pytest.raises(ValidationError) as excinfo:
        ApiKeysSettings()
    assert "GOOGLE_API_KEY environment variable must be set" in str(excinfo.value)

    # Test empty GOOGLE_API_KEY
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    with pytest.raises(ValidationError) as excinfo:
        ApiKeysSettings()
    assert "GOOGLE_API_KEY environment variable must be set and cannot be empty" in str(excinfo.value)

    # Test empty GITHUB_TOKEN (if set)
    monkeypatch.setenv("GOOGLE_API_KEY", "valid_google_key") # Needs to be valid for this test
    monkeypatch.setenv("GITHUB_TOKEN", "")
    with pytest.raises(ValidationError) as excinfo:
        ApiKeysSettings()
    assert "GITHUB_TOKEN environment variable, if set, cannot be empty" in str(excinfo.value)

    # Test valid keys
    monkeypatch.setenv("GOOGLE_API_KEY", "valid_google_key")
    monkeypatch.setenv("GITHUB_TOKEN", "valid_github_token")
    keys = ApiKeysSettings()
    assert keys.google_api_key.get_secret_value() == "valid_google_key"
    assert keys.github_token.get_secret_value() == "valid_github_token"

    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    # Reload settings from AppSettings to avoid interference with other tests
    _ = AppSettings.__new__(AppSettings)


def test_logging_settings_validation(monkeypatch):
    """Test validation logic for LoggingSettings."""
    # Test invalid log level
    monkeypatch.setenv("LOG_LEVEL", "INVALID_LEVEL")
    with pytest.raises(ValidationError) as excinfo:
        LoggingSettings()
    assert "Invalid log level 'INVALID_LEVEL'" in str(excinfo.value)

    # Test invalid log format
    monkeypatch.setenv("LOG_LEVEL", "INFO")  # Set valid log level
    monkeypatch.setenv("LOG_FORMAT", "INVALID_FORMAT")
    with pytest.raises(ValidationError) as excinfo:
        LoggingSettings()
    assert "Invalid log format 'INVALID_FORMAT'" in str(excinfo.value)

    # Test valid log settings (case-insensitivity)
    monkeypatch.setenv("LOG_LEVEL", "debug")
    monkeypatch.setenv("LOG_FORMAT", "TEXT")
    settings = LoggingSettings()
    assert settings.log_level == "DEBUG"
    assert settings.log_format == "text"

    monkeypatch.delenv("LOG_LEVEL", raising=False)
    monkeypatch.delenv("LOG_FORMAT", raising=False)
    # Reload settings from AppSettings to avoid interference with other tests
    _ = AppSettings.__new__(AppSettings)
