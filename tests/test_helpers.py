import pytest
import logging
import json
from unittest import mock
from io import StringIO

from utils.helpers import setup_logging, CustomJsonFormatter
from config import LoggingSettings, settings as global_settings


@pytest.fixture
def mock_logging_settings():
    """Fixture to mock LoggingSettings."""
    return LoggingSettings()

@pytest.fixture(autouse=True)
def reset_global_settings_logging(monkeypatch):
    """
    Fixture to reset the global logging settings before and after each test
    to ensure test isolation for logging configuration.
    """
    original_log_level = global_settings.logging.log_level
    original_log_format = global_settings.logging.log_format

    yield

    # Restore original logging settings by re-applying them
    # This forces pydantic to re-evaluate based on current env or defaults
    monkeypatch.setenv("LOG_LEVEL", original_log_level)
    monkeypatch.setenv("LOG_FORMAT", original_log_format)
    global_settings.logging = LoggingSettings() # Re-initialize logging settings
    setup_logging() # Re-apply logging based on restored settings


def test_setup_logging_json_format(mock_logging_settings, capsys, monkeypatch):
    """Test setup_logging with JSON format."""
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    monkeypatch.setenv("LOG_FORMAT", "json")

    # Re-initialize global settings with monkeypatched values
    global_settings.logging = LoggingSettings()

    setup_logging()

    logger = logging.getLogger("test_json_logger")
    test_message = "This is a JSON test message."
    logger.info(test_message)

    captured = capsys.readouterr()
    log_output = captured.err.strip() # setup_logging uses sys.stdout, but pytest captures it in err for loggers

    # Check if the initial setup log is present (it also goes to the logger)
    assert f"Logging initialized. Level: INFO, Format: json" in log_output

    # Parse the logged JSON message
    # We need to find the actual test message log entry, it might not be the first or last line
    json_log_entry = None
    for line in log_output.splitlines():
        try:
            log_entry = json.loads(line)
            if log_entry.get("message") == test_message:
                json_log_entry = log_entry
                break
        except json.JSONDecodeError:
            continue # Skip lines that are not valid JSON (like the initial text log)

    assert json_log_entry is not None, "JSON log entry not found or not valid JSON"
    assert json_log_entry["level"] == "INFO"
    assert json_log_entry["message"] == test_message
    assert "timestamp" in json_log_entry


def test_setup_logging_text_format(mock_logging_settings, capsys, monkeypatch):
    """Test setup_logging with text format."""
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("LOG_FORMAT", "text")
    global_settings.logging = LoggingSettings()
    setup_logging()

    logger = logging.getLogger("test_text_logger")
    test_message = "This is a text test message."
    logger.debug(test_message)

    captured = capsys.readouterr()
    log_output = captured.err.strip()

    assert f"Logging initialized. Level: DEBUG, Format: text" in log_output
    assert "DEBUG" in log_output
    assert test_message in log_output
    # Text format typically includes asctime, name, levelname, message
    assert logging.getLevelName(logger.level) == "DEBUG" # Check logger's effective level


def test_setup_logging_level_warning(mock_logging_settings, capsys, monkeypatch):
    """Test setup_logging with WARNING log level."""
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    monkeypatch.setenv("LOG_FORMAT", "text") # Format doesn't matter much for level testing
    global_settings.logging = LoggingSettings()
    setup_logging()

    logger = logging.getLogger("test_warning_logger")
    info_message = "This is an info message."
    warning_message = "This is a warning message."

    logger.info(info_message)
    logger.warning(warning_message)

    captured = capsys.readouterr()
    log_output = captured.err.strip()

    assert info_message not in log_output
    assert warning_message in log_output


def test_custom_json_formatter():
    """Test CustomJsonFormatter adds timestamp and uppercase level."""
    formatter = CustomJsonFormatter()
    record = logging.LogRecord(
        name='test', level=logging.INFO, pathname='test.py', lineno=10,
        msg='Test message', args=(), exc_info=None, func='test_func'
    )
    # Simulate a log record that doesn't have 'timestamp' or 'level' in the log_record dict yet
    log_record = {}
    formatter.add_fields(log_record, record, {})

    assert 'timestamp' in log_record
    assert isinstance(log_record['timestamp'], float)
    assert 'level' in log_record
    assert log_record['level'] == 'INFO'

    # Simulate a log record that already has 'level' (e.g. from a previous processor)
    log_record_with_level = {'level': 'debug'}
    formatter.add_fields(log_record_with_level, record, {})
    assert log_record_with_level['level'] == 'DEBUG' # Should be uppercased


def test_setup_logging_removes_existing_handlers(monkeypatch):
    """Test that setup_logging removes existing root handlers."""
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    monkeypatch.setenv("LOG_FORMAT", "text")
    global_settings.logging = LoggingSettings()

    # Add a dummy handler to the root logger
    dummy_handler = logging.StreamHandler(StringIO())
    dummy_handler.name = "DummyHandler"
    logging.root.addHandler(dummy_handler)

    assert any(h.name == "DummyHandler" for h in logging.root.handlers)

    setup_logging() # This should remove the DummyHandler

    assert not any(h.name == "DummyHandler" for h in logging.root.handlers)
    # It should have at least one handler (the one setup_logging configures)
    assert len(logging.root.handlers) >= 1
    # And none of them should be our dummy one unless setup_logging re-added one with the same name (unlikely)

    # Clean up by removing any handlers added by the test itself, if necessary,
    # though setup_logging should handle this.
    # For safety, explicitly remove the specific handler if it was not removed.
    if dummy_handler in logging.root.handlers:
         logging.root.removeHandler(dummy_handler)

# Ensure that the global logging state is clean for other test files
# This is a bit of a workaround for pytest's logger capture and module-level logging setup.
# The autouse fixture above (`reset_global_settings_logging`) should handle this better.
def teardown_module(module):
    # Restore to some default state if necessary, or rely on the autouse fixture
    pass
