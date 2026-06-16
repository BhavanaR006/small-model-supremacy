"""API client with caching for frontier model generation.

Provides the APIClient class that calls Claude or OpenAI APIs with:
- Prompt hash-based response caching
- Exponential backoff on rate limits (2^n seconds, max 60s, 5 retries)
- Cache fallback on network errors, timeouts (>60s), and HTTP errors

Usage:
    from src.data.api_client import APIClient, GenerationParams

    client = APIClient(provider="claude", cache_dir=Path("cache/api_responses"))
    result = client.generate("Extract fields from...", GenerationParams(temperature=0.0))
"""

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class GenerationParams:
    """Parameters for API generation requests."""

    temperature: float = 0.0
    max_tokens: int = 4096
    model: str = "claude-3-5-sonnet-20241022"


@dataclass
class CachedResponse:
    """A cached API response with metadata."""

    response: str
    prompt: str
    model: str
    params: dict
    timestamp: str
    cache_key: str


class APIClient:
    """Client for frontier model APIs with caching and retry logic.

    Supports "claude" and "openai" providers. Implements:
    - Deterministic prompt+params hash-based caching as JSON files
    - Exponential backoff (2^n seconds, max 60s) with 5 retries on rate limits
    - Cache fallback on network errors, timeouts (>60s), and HTTP errors

    Args:
        provider: API provider, one of "claude" or "openai".
        cache_dir: Directory for storing cached API responses.
        max_retries: Maximum retry attempts on rate limit errors.
        timeout: Request timeout in seconds.
    """

    SUPPORTED_PROVIDERS = ("claude", "openai")
    MAX_BACKOFF_SECONDS = 60
    DEFAULT_MAX_RETRIES = 5
    DEFAULT_TIMEOUT = 60

    def __init__(
        self,
        provider: str,
        cache_dir: Path,
        max_retries: int = DEFAULT_MAX_RETRIES,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        if provider not in self.SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unsupported provider '{provider}'. "
                f"Must be one of: {self.SUPPORTED_PROVIDERS}"
            )
        self.provider = provider
        self.cache_dir = Path(cache_dir)
        self.max_retries = max_retries
        self.timeout = timeout

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _compute_cache_key(self, prompt: str, params: GenerationParams) -> str:
        """Compute a deterministic hash of prompt + params for cache key.

        Uses SHA-256 of the canonical JSON representation of prompt and params
        to ensure identical requests map to the same cache entry.
        """
        key_data = {
            "prompt": prompt,
            "temperature": params.temperature,
            "max_tokens": params.max_tokens,
            "model": params.model,
        }
        canonical = json.dumps(key_data, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _cache_path(self, cache_key: str) -> Path:
        """Get the file path for a cache entry."""
        return self.cache_dir / f"{cache_key}.json"

    def _load_from_cache(self, cache_key: str) -> Optional[str]:
        """Load a cached response if available.

        Returns:
            The cached response text, or None if not cached.
        """
        cache_file = self._cache_path(cache_key)
        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text(encoding="utf-8"))
                logger.debug(
                    "Cache hit",
                    extra={"cache_key": cache_key[:12]},
                )
                return data["response"]
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(
                    "Corrupted cache entry, ignoring",
                    extra={"cache_key": cache_key[:12], "error": str(e)},
                )
        return None

    def _save_to_cache(
        self, cache_key: str, response: str, prompt: str, params: GenerationParams
    ) -> None:
        """Save a response to the cache with metadata."""
        cache_data = {
            "response": response,
            "prompt": prompt,
            "model": params.model,
            "params": {
                "temperature": params.temperature,
                "max_tokens": params.max_tokens,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cache_key": cache_key,
        }
        cache_file = self._cache_path(cache_key)
        cache_file.write_text(
            json.dumps(cache_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.debug(
            "Saved response to cache",
            extra={"cache_key": cache_key[:12]},
        )

    def _call_claude(self, prompt: str, params: GenerationParams) -> str:
        """Call the Anthropic Claude API.

        Uses lazy import so tests work without API keys or SDK installed.
        """
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "The 'anthropic' package is required for the Claude provider. "
                "Install it with: pip install anthropic"
            ) from e

        client = anthropic.Anthropic(timeout=self.timeout)
        message = client.messages.create(
            model=params.model,
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        # Extract text from the response
        return message.content[0].text

    def _call_openai(self, prompt: str, params: GenerationParams) -> str:
        """Call the OpenAI API.

        Uses lazy import so tests work without API keys or SDK installed.
        """
        try:
            import openai
        except ImportError as e:
            raise ImportError(
                "The 'openai' package is required for the OpenAI provider. "
                "Install it with: pip install openai"
            ) from e

        client = openai.OpenAI(timeout=self.timeout)
        response = client.chat.completions.create(
            model=params.model,
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if an exception is a rate limit error."""
        # Check Anthropic rate limit errors
        try:
            import anthropic

            if isinstance(error, anthropic.RateLimitError):
                return True
        except ImportError:
            pass

        # Check OpenAI rate limit errors
        try:
            import openai

            if isinstance(error, openai.RateLimitError):
                return True
        except ImportError:
            pass

        # Fallback: check status code if available
        status_code = getattr(error, "status_code", None)
        if status_code == 429:
            return True

        return False

    def _is_timeout_error(self, error: Exception) -> bool:
        """Check if an exception is a timeout error."""
        # Check common timeout exception types
        try:
            import anthropic

            if isinstance(error, anthropic.APITimeoutError):
                return True
        except ImportError:
            pass

        try:
            import openai

            if isinstance(error, openai.APITimeoutError):
                return True
        except ImportError:
            pass

        # Generic timeout detection
        if isinstance(error, (TimeoutError, ConnectionError)):
            return True

        error_str = str(error).lower()
        if "timeout" in error_str:
            return True

        return False

    def _is_network_error(self, error: Exception) -> bool:
        """Check if an exception is a network/connection error."""
        if isinstance(error, (ConnectionError, OSError)):
            return True

        # Check SDK-specific connection errors
        try:
            import anthropic

            if isinstance(error, anthropic.APIConnectionError):
                return True
        except ImportError:
            pass

        try:
            import openai

            if isinstance(error, openai.APIConnectionError):
                return True
        except ImportError:
            pass

        return False

    def generate(self, prompt: str, params: Optional[GenerationParams] = None) -> str:
        """Generate a response from the configured provider.

        Checks cache first, then calls the API with retry logic.
        On network errors, timeouts, or HTTP errors, falls back to cache
        if available, otherwise raises the error.

        Args:
            prompt: The prompt text to send to the model.
            params: Generation parameters. Defaults to GenerationParams().

        Returns:
            The generated text response.

        Raises:
            RuntimeError: If all retries are exhausted and no cache is available.
            ImportError: If the required SDK is not installed.
            TimeoutError: If the request times out and no cache fallback exists.
            ConnectionError: If a network error occurs and no cache fallback exists.
        """
        if params is None:
            params = GenerationParams()

        cache_key = self._compute_cache_key(prompt, params)

        # Check cache first
        cached = self._load_from_cache(cache_key)
        if cached is not None:
            return cached

        # Select the appropriate API call function
        if self.provider == "claude":
            api_call = self._call_claude
        else:
            api_call = self._call_openai

        # Attempt API call with exponential backoff on rate limits
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                response = api_call(prompt, params)
                # Cache the successful response
                self._save_to_cache(cache_key, response, prompt, params)
                return response

            except Exception as e:
                last_error = e

                # On rate limit: exponential backoff and retry
                if self._is_rate_limit_error(e):
                    backoff = min(2**attempt, self.MAX_BACKOFF_SECONDS)
                    logger.warning(
                        "Rate limited, backing off",
                        extra={
                            "attempt": attempt + 1,
                            "max_retries": self.max_retries,
                            "backoff_seconds": backoff,
                        },
                    )
                    time.sleep(backoff)
                    continue

                # On timeout or network error: try cache fallback
                if self._is_timeout_error(e) or self._is_network_error(e):
                    logger.warning(
                        "Network/timeout error, checking cache fallback",
                        extra={"error": str(e), "error_type": type(e).__name__},
                    )
                    cached = self._load_from_cache(cache_key)
                    if cached is not None:
                        logger.info("Using cached response as fallback")
                        return cached
                    # No cache available — raise immediately
                    if self._is_timeout_error(e):
                        raise TimeoutError(
                            f"Request timed out after {self.timeout}s and no "
                            f"cached response available for this prompt"
                        ) from e
                    else:
                        raise ConnectionError(
                            f"Network error and no cached response available: {e}"
                        ) from e

                # On other HTTP errors: try cache fallback, then raise
                logger.warning(
                    "API error, checking cache fallback",
                    extra={"error": str(e), "error_type": type(e).__name__},
                )
                cached = self._load_from_cache(cache_key)
                if cached is not None:
                    logger.info("Using cached response as fallback")
                    return cached
                raise

        # All retries exhausted (only reached for rate limit retries)
        logger.error(
            "All retries exhausted",
            extra={"max_retries": self.max_retries},
        )
        # Final cache fallback attempt
        cached = self._load_from_cache(cache_key)
        if cached is not None:
            logger.info("Using cached response as final fallback after retries exhausted")
            return cached

        raise RuntimeError(
            f"All {self.max_retries} retries exhausted due to rate limiting. "
            f"Last error: {last_error}"
        )
