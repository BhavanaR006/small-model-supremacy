"""Tokenizer utilities for token counting and length-based filtering.

Uses the Qwen tokenizer (Qwen/Qwen2.5-1.5B) for accurate token counting.
Falls back to a word-count approximation if the tokenizer cannot be loaded
(e.g., no network access, missing model files).

Usage:
    from src.data.tokenizer_utils import count_tokens, filter_by_token_length

    n = count_tokens("Some input text here")
    kept, discarded = filter_by_token_length(examples, min_tokens=50, max_tokens=2000)
"""

import warnings
from typing import Any, Protocol, Union, runtime_checkable

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Module-level singleton for the tokenizer (lazy-loaded on first use)
_tokenizer: Any = None
_tokenizer_loaded: bool = False
_using_fallback: bool = False

MODEL_NAME = "Qwen/Qwen2.5-1.5B"

# Approximation ratio: average tokens per word for English text.
# Qwen tokenizer typically produces ~1.3 tokens per whitespace-delimited word.
_TOKENS_PER_WORD_APPROX = 1.3


@runtime_checkable
class HasInputText(Protocol):
    """Protocol for objects with an input_text attribute."""

    input_text: str


def _load_tokenizer() -> Any:
    """Load the Qwen tokenizer, or return None if unavailable.

    Uses AutoTokenizer from the transformers library. On failure
    (network issues, missing files), logs a warning and returns None.
    """
    global _tokenizer, _tokenizer_loaded, _using_fallback

    if _tokenizer_loaded:
        return _tokenizer

    try:
        from transformers import AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        _tokenizer_loaded = True
        _using_fallback = False
        logger.info("Tokenizer loaded successfully", extra={"model": MODEL_NAME})
    except Exception as e:
        _tokenizer = None
        _tokenizer_loaded = True
        _using_fallback = True
        warnings.warn(
            f"Could not load tokenizer '{MODEL_NAME}': {e}. "
            f"Falling back to word-count approximation.",
            RuntimeWarning,
            stacklevel=2,
        )
        logger.warning(
            "Tokenizer load failed, using word-count fallback",
            extra={"model": MODEL_NAME, "error": str(e)},
        )

    return _tokenizer


def count_tokens(text: str) -> int:
    """Count the number of tokens in the given text.

    Uses the Qwen tokenizer if available, otherwise falls back to a
    word-count approximation (word_count * 1.3, rounded to nearest int).

    Args:
        text: The input text to tokenize.

    Returns:
        The token count as an integer.
    """
    tokenizer = _load_tokenizer()

    if tokenizer is not None:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)

    # Fallback: approximate tokens from word count
    word_count = len(text.split())
    return round(word_count * _TOKENS_PER_WORD_APPROX)


def _get_input_text(example: Union[dict, Any]) -> str:
    """Extract input_text from an example (dict or object with attribute).

    Args:
        example: Either a dict with an "input_text" key, or an object
            with an `input_text` attribute.

    Returns:
        The input text string.

    Raises:
        ValueError: If input_text cannot be extracted from the example.
    """
    if isinstance(example, dict):
        if "input_text" not in example:
            raise ValueError(
                f"Dict example missing 'input_text' key. Keys: {list(example.keys())}"
            )
        return example["input_text"]

    if hasattr(example, "input_text"):
        return example.input_text

    raise ValueError(
        f"Cannot extract input_text from example of type {type(example).__name__}. "
        f"Expected a dict with 'input_text' key or an object with 'input_text' attribute."
    )


def filter_by_token_length(
    examples: list,
    min_tokens: int = 50,
    max_tokens: int = 2000,
) -> tuple[list, list]:
    """Filter examples by token length of their input_text.

    Keeps examples whose input_text has a token count in [min_tokens, max_tokens].
    Discards examples outside this range.

    Args:
        examples: A list of examples. Each example must be either:
            - A dict with an "input_text" key
            - An object with an `input_text` attribute
        min_tokens: Minimum token count (inclusive). Default: 50.
        max_tokens: Maximum token count (inclusive). Default: 2000.

    Returns:
        A tuple of (kept, discarded) where:
            - kept: list of examples with token count in [min_tokens, max_tokens]
            - discarded: list of examples with token count outside the range
    """
    kept: list = []
    discarded: list = []

    for example in examples:
        try:
            text = _get_input_text(example)
        except ValueError as e:
            logger.warning(
                "Could not extract input_text, discarding example",
                extra={"error": str(e)},
            )
            discarded.append(example)
            continue

        token_count = count_tokens(text)

        if min_tokens <= token_count <= max_tokens:
            kept.append(example)
        else:
            discarded.append(example)

    logger.info(
        "Token length filtering complete",
        extra={
            "total": len(examples),
            "kept": len(kept),
            "discarded": len(discarded),
            "min_tokens": min_tokens,
            "max_tokens": max_tokens,
        },
    )

    return kept, discarded


def is_using_fallback() -> bool:
    """Check if the tokenizer is using the word-count fallback.

    Returns:
        True if the fallback is active (tokenizer failed to load),
        False if the real tokenizer is in use or hasn't been loaded yet.
    """
    return _using_fallback
