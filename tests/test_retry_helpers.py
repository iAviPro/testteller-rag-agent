import pytest
import time
import logging
from unittest.mock import Mock, call, patch
import asyncio

from utils.retry_helpers import api_retry_sync, api_retry_async, TRANSIENT_EXCEPTIONS
from config import ApiRetrySettings, settings as global_settings


# --- Test Fixtures and Mocks ---

@pytest.fixture(autouse=True)
def mock_global_retry_settings(monkeypatch):
    """Mocks global retry settings for consistent tests."""
    monkeypatch.setattr(global_settings, "api_retry_attempts", 3)
    monkeypatch.setattr(global_settings, "api_retry_wait_seconds", 0.1) # Use short wait for tests


class NonTransientError(Exception):
    """A custom exception that is not in TRANSIENT_EXCEPTIONS."""
    pass

# --- Tests for api_retry_sync ---

def test_sync_retry_success_first_try(caplog):
    """Test that a function succeeds on the first try without retries."""
    mock_func = Mock(return_value="success")

    @api_retry_sync
    def decorated_func():
        return mock_func()

    caplog.set_level(logging.WARNING)
    result = decorated_func()

    assert result == "success"
    mock_func.assert_called_once()
    assert "Retrying decorated_func" not in caplog.text


def test_sync_retry_success_after_retries(caplog):
    """Test that a function succeeds after a few retries."""
    mock_func = Mock(side_effect=[TRANSIENT_EXCEPTIONS[0]("fail"), TRANSIENT_EXCEPTIONS[0]("fail"), "success"])

    @api_retry_sync
    def decorated_func():
        return mock_func()

    caplog.set_level(logging.WARNING)
    result = decorated_func()

    assert result == "success"
    assert mock_func.call_count == 3
    assert "Retrying decorated_func" in caplog.text
    assert f"attempt 1 of {global_settings.api_retry_attempts}" in caplog.text
    assert f"attempt 2 of {global_settings.api_retry_attempts}" in caplog.text


@patch('tenacity.nap.time.sleep', return_value=None) # Mock time.sleep to speed up test
def test_sync_retry_failure_all_attempts(mock_sleep, caplog):
    """Test that a function fails after all retry attempts for a transient error."""
    mock_func = Mock(side_effect=TRANSIENT_EXCEPTIONS[0]("persistent fail"))

    @api_retry_sync
    def decorated_func():
        return mock_func()

    caplog.set_level(logging.WARNING)
    with pytest.raises(TRANSIENT_EXCEPTIONS[0]) as excinfo:
        decorated_func()

    assert "persistent fail" in str(excinfo.value)
    assert mock_func.call_count == global_settings.api_retry_attempts
    assert f"attempt {global_settings.api_retry_attempts -1} of {global_settings.api_retry_attempts}" in caplog.text
    # Check that sleep was called the correct number of times (attempts - 1)
    assert mock_sleep.call_count == global_settings.api_retry_attempts - 1


def test_sync_retry_non_transient_error_no_retry(caplog):
    """Test that a function with a non-transient error does not retry."""
    mock_func = Mock(side_effect=NonTransientError("fatal error"))

    @api_retry_sync
    def decorated_func():
        return mock_func()

    caplog.set_level(logging.WARNING)
    with pytest.raises(NonTransientError) as excinfo:
        decorated_func()

    assert "fatal error" in str(excinfo.value)
    mock_func.assert_called_once() # Should only be called once
    assert "Retrying decorated_func" not in caplog.text


@patch('tenacity.nap.time.sleep', return_value=None) # Mock time.sleep
def test_sync_retry_respects_wait_times(mock_sleep, caplog, monkeypatch):
    """Test that retry waits for the specified time."""
    monkeypatch.setattr(global_settings, "api_retry_wait_seconds", 0.05) # Specific wait for this test

    # Must re-import or re-create the decorator if it captures settings at import time
    # For tenacity, it usually re-evaluates settings on each call or decorator application.
    # However, to be safe, let's redefine a local decorator or re-patch settings that tenacity uses.
    # The current implementation of api_retry_sync in utils.retry_helpers directly uses global_settings.
    # So, monkeypatching global_settings should be enough.

    mock_func = Mock(side_effect=[TRANSIENT_EXCEPTIONS[0]("fail"), "success"])

    @api_retry_sync # This will now use the monkeypatched api_retry_wait_seconds
    def decorated_func_for_wait_test():
        return mock_func()

    caplog.set_level(logging.WARNING)
    decorated_func_for_wait_test()

    assert mock_func.call_count == 2
    # Check that sleep was called with values that respect the exponential backoff
    # The first sleep should be multiplier * 2**0 = api_retry_wait_seconds * 1
    # tenacity's wait_exponential has a default multiplier of 1 if not specified for the `multiplier` arg of wait_exponential
    # but our `api_retry_sync` uses `settings.api_retry_wait_seconds` as the multiplier.
    # The first wait is `multiplier` (0.05s), then `multiplier * 2` (0.1s), etc., up to `max` (10s).
    # In our case, the first wait (multiplier * 2^0) should be global_settings.api_retry_wait_seconds

    # We expect one call to sleep because it succeeds on the second try
    assert mock_sleep.call_count == 1
    # The argument to sleep is the calculated wait time.
    # For the first retry, wait_exponential calculates wait as: multiplier * (2 ** (attempt - 1))
    # So for attempt 1 (first retry), it's multiplier * (2**0) = multiplier
    expected_first_wait = global_settings.api_retry_wait_seconds
    # Check that the sleep call was close to the expected wait time
    # tenacity adds some jitter by default, but with min=1, max=10, and a small multiplier, it should be close.
    # For more precise test, we might need to look into tenacity's internals or use a custom wait.
    # For now, let's check the log message which contains the wait time.
    assert f"Waiting {expected_first_wait:.2f}s before next attempt" in caplog.text


# --- Tests for api_retry_async (similar structure to sync) ---

@pytest.mark.asyncio
async def test_async_retry_success_first_try(caplog):
    mock_async_func = Mock(return_value=asyncio.Future())
    mock_async_func.return_value.set_result("async success")

    @api_retry_async
    async def decorated_async_func():
        return await mock_async_func()

    caplog.set_level(logging.WARNING)
    result = await decorated_async_func()

    assert result == "async success"
    mock_async_func.assert_called_once()
    assert "Retrying decorated_async_func" not in caplog.text


@pytest.mark.asyncio
async def test_async_retry_success_after_retries(caplog):
    mock_async_func = Mock()
    # Configure side effects for async: return awaitables (futures)
    f1 = asyncio.Future(); f1.set_exception(TRANSIENT_EXCEPTIONS[0]("async fail 1"))
    f2 = asyncio.Future(); f2.set_exception(TRANSIENT_EXCEPTIONS[0]("async fail 2"))
    f3 = asyncio.Future(); f3.set_result("async success after retries")
    mock_async_func.side_effect = [f1, f2, f3]


    @api_retry_async
    async def decorated_async_func():
        return await mock_async_func()

    caplog.set_level(logging.WARNING)
    result = await decorated_async_func()

    assert result == "async success after retries"
    assert mock_async_func.call_count == 3
    assert "Retrying decorated_async_func" in caplog.text
    assert f"attempt 1 of {global_settings.api_retry_attempts}" in caplog.text
    assert f"attempt 2 of {global_settings.api_retry_attempts}" in caplog.text


@pytest.mark.asyncio
@patch('asyncio.sleep', return_value=None) # Mock asyncio.sleep
async def test_async_retry_failure_all_attempts(mock_async_sleep, caplog):
    mock_async_func = Mock()
    f_fail = asyncio.Future(); f_fail.set_exception(TRANSIENT_EXCEPTIONS[0]("persistent async fail"))
    # Need to return a new future for each call if the function is called multiple times
    mock_async_func.side_effect = lambda: (fut := asyncio.Future(), fut.set_exception(TRANSIENT_EXCEPTIONS[0]("persistent async fail")), fut)[2]


    @api_retry_async
    async def decorated_async_func():
        return await mock_async_func()

    caplog.set_level(logging.WARNING)
    with pytest.raises(TRANSIENT_EXCEPTIONS[0]) as excinfo:
        await decorated_async_func()

    assert "persistent async fail" in str(excinfo.value)
    assert mock_async_func.call_count == global_settings.api_retry_attempts
    assert f"attempt {global_settings.api_retry_attempts -1} of {global_settings.api_retry_attempts}" in caplog.text
    assert mock_async_sleep.call_count == global_settings.api_retry_attempts - 1


@pytest.mark.asyncio
async def test_async_retry_non_transient_error_no_retry(caplog):
    mock_async_func = Mock()
    f_fatal = asyncio.Future(); f_fatal.set_exception(NonTransientError("fatal async error"))
    mock_async_func.return_value = f_fatal


    @api_retry_async
    async def decorated_async_func():
        return await mock_async_func()

    caplog.set_level(logging.WARNING)
    with pytest.raises(NonTransientError) as excinfo:
        await decorated_async_func()

    assert "fatal async error" in str(excinfo.value)
    mock_async_func.assert_called_once()
    assert "Retrying decorated_async_func" not in caplog.text
