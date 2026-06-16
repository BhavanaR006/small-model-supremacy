"""Unit tests for the API client with caching."""

import json
import sys
import time
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from src.data.api_client import APIClient, CachedResponse, GenerationParams


class TestGenerationParams:
    """Tests for GenerationParams dataclass."""

    def test_default_values(self):
        params = GenerationParams()
        assert params.temperature == 0.0
        assert params.max_tokens == 4096
        assert params.model == "claude-3-5-sonnet-20241022"

    def test_custom_values(self):
        params = GenerationParams(temperature=0.7, max_tokens=1024, model="gpt-4o")
        assert params.temperature == 0.7
        assert params.max_tokens == 1024
        assert params.model == "gpt-4o"


class TestAPIClientInit:
    """Tests for APIClient initialization."""

    def test_valid_claude_provider(self, tmp_path):
        client = APIClient(provider="claude", cache_dir=tmp_path / "cache")
        assert client.provider == "claude"
        assert (tmp_path / "cache").exists()

    def test_valid_openai_provider(self, tmp_path):
        client = APIClient(provider="openai", cache_dir=tmp_path / "cache")
        assert client.provider == "openai"

    def test_invalid_provider_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unsupported provider"):
            APIClient(provider="invalid", cache_dir=tmp_path / "cache")

    def test_creates_cache_directory(self, tmp_path):
        cache_dir = tmp_path / "nested" / "cache" / "dir"
        assert not cache_dir.exists()
        APIClient(provider="claude", cache_dir=cache_dir)
        assert cache_dir.exists()

    def test_custom_retries_and_timeout(self, tmp_path):
        client = APIClient(
            provider="claude", cache_dir=tmp_path, max_retries=3, timeout=30
        )
        assert client.max_retries == 3
        assert client.timeout == 30


class TestCacheKey:
    """Tests for deterministic cache key computation."""

    def test_same_input_same_key(self, tmp_path):
        client = APIClient(provider="claude", cache_dir=tmp_path)
        params = GenerationParams(temperature=0.0, max_tokens=100, model="test")
        key1 = client._compute_cache_key("hello", params)
        key2 = client._compute_cache_key("hello", params)
        assert key1 == key2

    def test_different_prompt_different_key(self, tmp_path):
        client = APIClient(provider="claude", cache_dir=tmp_path)
        params = GenerationParams()
        key1 = client._compute_cache_key("hello", params)
        key2 = client._compute_cache_key("world", params)
        assert key1 != key2

    def test_different_params_different_key(self, tmp_path):
        client = APIClient(provider="claude", cache_dir=tmp_path)
        params1 = GenerationParams(temperature=0.0)
        params2 = GenerationParams(temperature=0.5)
        key1 = client._compute_cache_key("hello", params1)
        key2 = client._compute_cache_key("hello", params2)
        assert key1 != key2

    def test_key_is_sha256_hex(self, tmp_path):
        client = APIClient(provider="claude", cache_dir=tmp_path)
        key = client._compute_cache_key("test", GenerationParams())
        assert len(key) == 64  # SHA-256 hex digest length
        assert all(c in "0123456789abcdef" for c in key)

    def test_different_model_different_key(self, tmp_path):
        client = APIClient(provider="claude", cache_dir=tmp_path)
        params1 = GenerationParams(model="model-a")
        params2 = GenerationParams(model="model-b")
        key1 = client._compute_cache_key("hello", params1)
        key2 = client._compute_cache_key("hello", params2)
        assert key1 != key2


class TestCaching:
    """Tests for cache save/load operations."""

    def test_save_and_load_cache(self, tmp_path):
        client = APIClient(provider="claude", cache_dir=tmp_path)
        params = GenerationParams(model="test-model")
        cache_key = client._compute_cache_key("test prompt", params)

        client._save_to_cache(cache_key, "test response", "test prompt", params)
        loaded = client._load_from_cache(cache_key)

        assert loaded == "test response"

    def test_load_missing_cache_returns_none(self, tmp_path):
        client = APIClient(provider="claude", cache_dir=tmp_path)
        result = client._load_from_cache("nonexistent_key")
        assert result is None

    def test_cache_file_contains_metadata(self, tmp_path):
        client = APIClient(provider="claude", cache_dir=tmp_path)
        params = GenerationParams(temperature=0.5, max_tokens=200, model="gpt-4o")
        cache_key = client._compute_cache_key("my prompt", params)

        client._save_to_cache(cache_key, "response text", "my prompt", params)

        cache_file = tmp_path / f"{cache_key}.json"
        data = json.loads(cache_file.read_text())

        assert data["response"] == "response text"
        assert data["prompt"] == "my prompt"
        assert data["model"] == "gpt-4o"
        assert data["params"]["temperature"] == 0.5
        assert data["params"]["max_tokens"] == 200
        assert "timestamp" in data
        assert data["cache_key"] == cache_key

    def test_corrupted_cache_returns_none(self, tmp_path):
        client = APIClient(provider="claude", cache_dir=tmp_path)
        cache_key = "test_corrupted_key"
        cache_file = tmp_path / f"{cache_key}.json"
        cache_file.write_text("not valid json {{{")

        result = client._load_from_cache(cache_key)
        assert result is None

    def test_cache_missing_response_key_returns_none(self, tmp_path):
        client = APIClient(provider="claude", cache_dir=tmp_path)
        cache_key = "test_missing_key"
        cache_file = tmp_path / f"{cache_key}.json"
        cache_file.write_text(json.dumps({"prompt": "hello"}))

        result = client._load_from_cache(cache_key)
        assert result is None


@pytest.fixture
def mock_anthropic_module():
    """Create a mock anthropic module for testing Claude API calls."""
    mock_module = ModuleType("anthropic")
    mock_module.Anthropic = MagicMock()
    mock_module.RateLimitError = type("RateLimitError", (Exception,), {})
    mock_module.APITimeoutError = type("APITimeoutError", (Exception,), {})
    mock_module.APIConnectionError = type("APIConnectionError", (Exception,), {})

    with patch.dict(sys.modules, {"anthropic": mock_module}):
        yield mock_module


@pytest.fixture
def mock_openai_module():
    """Create a mock openai module for testing OpenAI API calls."""
    mock_module = ModuleType("openai")
    mock_module.OpenAI = MagicMock()
    mock_module.RateLimitError = type("RateLimitError", (Exception,), {})
    mock_module.APITimeoutError = type("APITimeoutError", (Exception,), {})
    mock_module.APIConnectionError = type("APIConnectionError", (Exception,), {})

    with patch.dict(sys.modules, {"openai": mock_module}):
        yield mock_module


class TestGenerate:
    """Tests for the generate method with mocked API calls."""

    def test_returns_cached_response_without_api_call(self, tmp_path):
        client = APIClient(provider="claude", cache_dir=tmp_path)
        params = GenerationParams(model="test-model")
        cache_key = client._compute_cache_key("cached prompt", params)

        # Pre-populate cache
        client._save_to_cache(cache_key, "cached result", "cached prompt", params)

        # Should return cached result without calling API
        result = client.generate("cached prompt", params)
        assert result == "cached result"

    def test_calls_claude_api_and_caches(self, tmp_path, mock_anthropic_module):
        client = APIClient(provider="claude", cache_dir=tmp_path)
        params = GenerationParams(model="claude-3-5-sonnet-20241022")

        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="API response")]

        mock_client_instance = MagicMock()
        mock_client_instance.messages.create.return_value = mock_message
        mock_anthropic_module.Anthropic.return_value = mock_client_instance

        result = client.generate("new prompt", params)

        assert result == "API response"
        # Verify it was cached
        cache_key = client._compute_cache_key("new prompt", params)
        cached = client._load_from_cache(cache_key)
        assert cached == "API response"

    def test_calls_openai_api_and_caches(self, tmp_path, mock_openai_module):
        client = APIClient(provider="openai", cache_dir=tmp_path)
        params = GenerationParams(model="gpt-4o")

        mock_choice = MagicMock()
        mock_choice.message.content = "OpenAI response"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client_instance = MagicMock()
        mock_client_instance.chat.completions.create.return_value = mock_response
        mock_openai_module.OpenAI.return_value = mock_client_instance

        result = client.generate("new prompt", params)

        assert result == "OpenAI response"
        # Verify it was cached
        cache_key = client._compute_cache_key("new prompt", params)
        cached = client._load_from_cache(cache_key)
        assert cached == "OpenAI response"

    def test_uses_default_params_when_none(self, tmp_path, mock_anthropic_module):
        client = APIClient(provider="claude", cache_dir=tmp_path)

        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="response")]

        mock_client_instance = MagicMock()
        mock_client_instance.messages.create.return_value = mock_message
        mock_anthropic_module.Anthropic.return_value = mock_client_instance

        result = client.generate("prompt")

        assert result == "response"


class TestRetryLogic:
    """Tests for exponential backoff on rate limits."""

    def test_retries_on_rate_limit_with_backoff(self, tmp_path, mock_anthropic_module):
        client = APIClient(provider="claude", cache_dir=tmp_path, max_retries=3)
        params = GenerationParams()

        # Simulate rate limit errors first, then success
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="eventual response")]

        rate_limit_error = mock_anthropic_module.RateLimitError("rate limited")

        mock_client_instance = MagicMock()
        mock_client_instance.messages.create.side_effect = [
            rate_limit_error,
            rate_limit_error,
            mock_message,
        ]
        mock_anthropic_module.Anthropic.return_value = mock_client_instance

        with patch("time.sleep") as mock_sleep:
            result = client.generate("prompt", params)

        assert result == "eventual response"
        # Verify backoff timing: 2^0=1, 2^1=2
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(1)
        mock_sleep.assert_any_call(2)

    def test_raises_runtime_error_after_all_retries_exhausted(
        self, tmp_path, mock_anthropic_module
    ):
        client = APIClient(provider="claude", cache_dir=tmp_path, max_retries=2)
        params = GenerationParams()

        rate_limit_error = mock_anthropic_module.RateLimitError("rate limited")

        mock_client_instance = MagicMock()
        mock_client_instance.messages.create.side_effect = rate_limit_error
        mock_anthropic_module.Anthropic.return_value = mock_client_instance

        with patch("time.sleep"):
            with pytest.raises(RuntimeError, match="retries exhausted"):
                client.generate("prompt", params)

    def test_backoff_capped_at_60_seconds(self, tmp_path, mock_anthropic_module):
        client = APIClient(provider="claude", cache_dir=tmp_path, max_retries=5)
        params = GenerationParams()

        rate_limit_error = mock_anthropic_module.RateLimitError("rate limited")
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="response")]

        mock_client_instance = MagicMock()
        # Fail 4 times, succeed on 5th
        mock_client_instance.messages.create.side_effect = [
            rate_limit_error,
            rate_limit_error,
            rate_limit_error,
            rate_limit_error,
            mock_message,
        ]
        mock_anthropic_module.Anthropic.return_value = mock_client_instance

        with patch("time.sleep") as mock_sleep:
            result = client.generate("prompt", params)

        assert result == "response"
        # Backoff: 2^0=1, 2^1=2, 2^2=4, 2^3=8 (all within cap of 60)
        sleep_calls = [call.args[0] for call in mock_sleep.call_args_list]
        assert sleep_calls == [1, 2, 4, 8]
        assert all(s <= 60 for s in sleep_calls)


class TestErrorHandling:
    """Tests for network error, timeout, and HTTP error handling."""

    def test_timeout_with_cache_fallback(self, tmp_path, mock_anthropic_module):
        client = APIClient(provider="claude", cache_dir=tmp_path)
        params = GenerationParams()
        cache_key = client._compute_cache_key("prompt", params)

        # Pre-populate cache
        client._save_to_cache(cache_key, "cached fallback", "prompt", params)

        timeout_error = mock_anthropic_module.APITimeoutError("timed out")

        mock_client_instance = MagicMock()
        mock_client_instance.messages.create.side_effect = timeout_error
        mock_anthropic_module.Anthropic.return_value = mock_client_instance

        result = client.generate("prompt", params)
        assert result == "cached fallback"

    def test_timeout_without_cache_raises(self, tmp_path, mock_anthropic_module):
        client = APIClient(provider="claude", cache_dir=tmp_path)
        params = GenerationParams()

        timeout_error = mock_anthropic_module.APITimeoutError("timed out")

        mock_client_instance = MagicMock()
        mock_client_instance.messages.create.side_effect = timeout_error
        mock_anthropic_module.Anthropic.return_value = mock_client_instance

        with pytest.raises(TimeoutError, match="timed out"):
            client.generate("prompt", params)

    def test_connection_error_with_cache_fallback(self, tmp_path, mock_anthropic_module):
        client = APIClient(provider="claude", cache_dir=tmp_path)
        params = GenerationParams()
        cache_key = client._compute_cache_key("prompt", params)

        # Pre-populate cache
        client._save_to_cache(cache_key, "network fallback", "prompt", params)

        connection_error = mock_anthropic_module.APIConnectionError("no network")

        mock_client_instance = MagicMock()
        mock_client_instance.messages.create.side_effect = connection_error
        mock_anthropic_module.Anthropic.return_value = mock_client_instance

        result = client.generate("prompt", params)
        assert result == "network fallback"

    def test_connection_error_without_cache_raises(
        self, tmp_path, mock_anthropic_module
    ):
        client = APIClient(provider="claude", cache_dir=tmp_path)
        params = GenerationParams()

        connection_error = mock_anthropic_module.APIConnectionError("no network")

        mock_client_instance = MagicMock()
        mock_client_instance.messages.create.side_effect = connection_error
        mock_anthropic_module.Anthropic.return_value = mock_client_instance

        with pytest.raises(ConnectionError, match="no network"):
            client.generate("prompt", params)

    def test_generic_error_with_cache_fallback(self, tmp_path, mock_anthropic_module):
        client = APIClient(provider="claude", cache_dir=tmp_path)
        params = GenerationParams()
        cache_key = client._compute_cache_key("prompt", params)

        # Pre-populate cache
        client._save_to_cache(cache_key, "http fallback", "prompt", params)

        # Generic HTTP error (not rate limit, not timeout, not network)
        http_error = Exception("500 Internal Server Error")

        mock_client_instance = MagicMock()
        mock_client_instance.messages.create.side_effect = http_error
        mock_anthropic_module.Anthropic.return_value = mock_client_instance

        result = client.generate("prompt", params)
        assert result == "http fallback"

    def test_generic_error_without_cache_raises(self, tmp_path, mock_anthropic_module):
        client = APIClient(provider="claude", cache_dir=tmp_path)
        params = GenerationParams()

        http_error = Exception("500 Internal Server Error")

        mock_client_instance = MagicMock()
        mock_client_instance.messages.create.side_effect = http_error
        mock_anthropic_module.Anthropic.return_value = mock_client_instance

        with pytest.raises(Exception, match="500 Internal Server Error"):
            client.generate("prompt", params)

    def test_rate_limit_exhausted_with_cache_fallback(
        self, tmp_path, mock_anthropic_module
    ):
        client = APIClient(provider="claude", cache_dir=tmp_path, max_retries=2)
        params = GenerationParams()
        cache_key = client._compute_cache_key("prompt", params)

        # Pre-populate cache
        client._save_to_cache(cache_key, "rate limit fallback", "prompt", params)

        rate_limit_error = mock_anthropic_module.RateLimitError("rate limited")

        mock_client_instance = MagicMock()
        mock_client_instance.messages.create.side_effect = rate_limit_error
        mock_anthropic_module.Anthropic.return_value = mock_client_instance

        with patch("time.sleep"):
            result = client.generate("prompt", params)

        assert result == "rate limit fallback"

    def test_builtin_timeout_error_uses_cache(self, tmp_path, mock_anthropic_module):
        """Test that a Python built-in TimeoutError also triggers cache fallback."""
        client = APIClient(provider="claude", cache_dir=tmp_path)
        params = GenerationParams()
        cache_key = client._compute_cache_key("prompt", params)

        client._save_to_cache(cache_key, "timeout fallback", "prompt", params)

        mock_client_instance = MagicMock()
        mock_client_instance.messages.create.side_effect = TimeoutError("timeout")
        mock_anthropic_module.Anthropic.return_value = mock_client_instance

        result = client.generate("prompt", params)
        assert result == "timeout fallback"

    def test_builtin_connection_error_uses_cache(self, tmp_path, mock_anthropic_module):
        """Test that a Python built-in ConnectionError also triggers cache fallback."""
        client = APIClient(provider="claude", cache_dir=tmp_path)
        params = GenerationParams()
        cache_key = client._compute_cache_key("prompt", params)

        client._save_to_cache(cache_key, "conn fallback", "prompt", params)

        mock_client_instance = MagicMock()
        mock_client_instance.messages.create.side_effect = ConnectionError("reset")
        mock_anthropic_module.Anthropic.return_value = mock_client_instance

        result = client.generate("prompt", params)
        assert result == "conn fallback"
