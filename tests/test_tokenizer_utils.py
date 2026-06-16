"""Unit tests for tokenizer utilities."""

from dataclasses import dataclass
from unittest.mock import patch

import pytest

from src.data.tokenizer_utils import (
    _TOKENS_PER_WORD_APPROX,
    _get_input_text,
    count_tokens,
    filter_by_token_length,
)


@dataclass
class MockExample:
    """Mock data example with input_text attribute."""

    input_text: str
    schema_id: str = "test_schema"


class TestCountTokens:
    """Tests for the count_tokens function."""

    def test_empty_string_returns_zero(self):
        """Empty string should have zero tokens."""
        result = count_tokens("")
        assert result == 0

    def test_single_word(self):
        """Single word should return at least 1 token."""
        result = count_tokens("hello")
        assert result >= 1

    def test_longer_text_has_more_tokens(self):
        """Longer text should produce more tokens than shorter text."""
        short = count_tokens("hello")
        long = count_tokens("hello world this is a longer text with more words")
        assert long > short

    def test_returns_integer(self):
        """Token count should always be an integer."""
        result = count_tokens("Some sample text for counting.")
        assert isinstance(result, int)

    def test_whitespace_only(self):
        """Whitespace-only text produces minimal tokens."""
        result = count_tokens("   ")
        # Depending on tokenizer, whitespace may produce 0 or a few tokens
        assert isinstance(result, int)
        assert result >= 0


class TestCountTokensFallback:
    """Tests for the fallback word-count approximation."""

    def test_fallback_approximation(self):
        """When tokenizer is unavailable, use word count * 1.3."""
        import src.data.tokenizer_utils as module

        # Save original state
        orig_tokenizer = module._tokenizer
        orig_loaded = module._tokenizer_loaded
        orig_fallback = module._using_fallback

        try:
            # Force fallback mode
            module._tokenizer = None
            module._tokenizer_loaded = True
            module._using_fallback = True

            text = "one two three four five"
            result = count_tokens(text)
            expected = round(5 * _TOKENS_PER_WORD_APPROX)
            assert result == expected
        finally:
            # Restore original state
            module._tokenizer = orig_tokenizer
            module._tokenizer_loaded = orig_loaded
            module._using_fallback = orig_fallback

    def test_fallback_empty_string(self):
        """Fallback with empty string returns 0."""
        import src.data.tokenizer_utils as module

        orig_tokenizer = module._tokenizer
        orig_loaded = module._tokenizer_loaded
        orig_fallback = module._using_fallback

        try:
            module._tokenizer = None
            module._tokenizer_loaded = True
            module._using_fallback = True

            result = count_tokens("")
            assert result == 0
        finally:
            module._tokenizer = orig_tokenizer
            module._tokenizer_loaded = orig_loaded
            module._using_fallback = orig_fallback


class TestGetInputText:
    """Tests for the _get_input_text helper."""

    def test_extracts_from_dict(self):
        """Should extract input_text from a dict."""
        example = {"input_text": "hello world", "schema_id": "test"}
        assert _get_input_text(example) == "hello world"

    def test_extracts_from_object(self):
        """Should extract input_text from an object attribute."""
        example = MockExample(input_text="object text")
        assert _get_input_text(example) == "object text"

    def test_raises_for_dict_missing_key(self):
        """Should raise ValueError for dict without input_text."""
        with pytest.raises(ValueError, match="missing 'input_text' key"):
            _get_input_text({"other_key": "value"})

    def test_raises_for_object_without_attribute(self):
        """Should raise ValueError for object without input_text attribute."""

        class NoInputText:
            pass

        with pytest.raises(ValueError, match="Cannot extract input_text"):
            _get_input_text(NoInputText())


class TestFilterByTokenLength:
    """Tests for the filter_by_token_length function."""

    def test_empty_list_returns_empty_tuples(self):
        """Empty input produces empty kept and discarded lists."""
        kept, discarded = filter_by_token_length([])
        assert kept == []
        assert discarded == []

    def test_keeps_examples_within_range(self):
        """Examples within token range should be kept."""
        # Create text that's likely within 50-2000 tokens
        text = " ".join(["word"] * 100)  # ~100 words -> ~130 tokens
        examples = [{"input_text": text}]

        kept, discarded = filter_by_token_length(examples, min_tokens=50, max_tokens=2000)
        assert len(kept) == 1
        assert len(discarded) == 0

    def test_discards_too_short(self):
        """Examples with too few tokens should be discarded."""
        text = "hi"  # Very few tokens
        examples = [{"input_text": text}]

        kept, discarded = filter_by_token_length(examples, min_tokens=50, max_tokens=2000)
        assert len(kept) == 0
        assert len(discarded) == 1

    def test_discards_too_long(self):
        """Examples with too many tokens should be discarded."""
        # Create very long text
        text = " ".join(["word"] * 5000)  # ~5000 words -> way over 2000 tokens
        examples = [{"input_text": text}]

        kept, discarded = filter_by_token_length(examples, min_tokens=50, max_tokens=2000)
        assert len(kept) == 0
        assert len(discarded) == 1

    def test_mixed_examples(self):
        """Mix of valid and invalid length examples."""
        short = {"input_text": "hi"}
        medium = {"input_text": " ".join(["word"] * 100)}
        long_text = {"input_text": " ".join(["word"] * 5000)}

        examples = [short, medium, long_text]
        kept, discarded = filter_by_token_length(examples, min_tokens=50, max_tokens=2000)

        assert len(kept) + len(discarded) == 3
        assert medium in kept
        assert short in discarded
        assert long_text in discarded

    def test_works_with_dataclass_objects(self):
        """Should work with objects having input_text attribute."""
        example = MockExample(input_text=" ".join(["word"] * 100))
        kept, discarded = filter_by_token_length(
            [example], min_tokens=50, max_tokens=2000
        )
        assert len(kept) == 1
        assert kept[0] is example

    def test_boundary_min_tokens(self):
        """Example at exactly min_tokens should be kept."""
        import src.data.tokenizer_utils as module

        orig_tokenizer = module._tokenizer
        orig_loaded = module._tokenizer_loaded
        orig_fallback = module._using_fallback

        try:
            # Force fallback for predictable counts
            module._tokenizer = None
            module._tokenizer_loaded = True
            module._using_fallback = True

            # 39 words * 1.3 = 50.7 -> rounds to 51, need exact 50
            # 38 words * 1.3 = 49.4 -> rounds to 49
            # 39 words * 1.3 = 50.7 -> rounds to 51
            # We need to find the right word count for exactly 50
            # round(n * 1.3) == 50 -> n ~ 38.46 -> n=38: round(49.4)=49, n=39: round(50.7)=51
            # Actually let's just test boundary behavior with min_tokens=49
            text = " ".join(["word"] * 38)  # 38 * 1.3 = 49.4 -> rounds to 49
            examples = [{"input_text": text}]
            kept, discarded = filter_by_token_length(
                examples, min_tokens=49, max_tokens=2000
            )
            assert len(kept) == 1
        finally:
            module._tokenizer = orig_tokenizer
            module._tokenizer_loaded = orig_loaded
            module._using_fallback = orig_fallback

    def test_boundary_max_tokens(self):
        """Example at exactly max_tokens should be kept."""
        import src.data.tokenizer_utils as module

        orig_tokenizer = module._tokenizer
        orig_loaded = module._tokenizer_loaded
        orig_fallback = module._using_fallback

        try:
            # Force fallback for predictable counts
            module._tokenizer = None
            module._tokenizer_loaded = True
            module._using_fallback = True

            # 77 words * 1.3 = 100.1 -> rounds to 100
            text = " ".join(["word"] * 77)
            examples = [{"input_text": text}]
            kept, discarded = filter_by_token_length(
                examples, min_tokens=1, max_tokens=100
            )
            assert len(kept) == 1
        finally:
            module._tokenizer = orig_tokenizer
            module._tokenizer_loaded = orig_loaded
            module._using_fallback = orig_fallback

    def test_invalid_example_discarded_gracefully(self):
        """Examples that can't provide input_text are discarded."""
        bad_example = {"no_input_text": "value"}
        good_example = {"input_text": " ".join(["word"] * 100)}

        kept, discarded = filter_by_token_length(
            [bad_example, good_example], min_tokens=50, max_tokens=2000
        )
        assert bad_example in discarded
        assert good_example in kept

    def test_returns_tuple(self):
        """Return value should be a tuple of two lists."""
        result = filter_by_token_length([])
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], list)
        assert isinstance(result[1], list)

    def test_custom_token_range(self):
        """Custom min/max token values work correctly."""
        import src.data.tokenizer_utils as module

        orig_tokenizer = module._tokenizer
        orig_loaded = module._tokenizer_loaded
        orig_fallback = module._using_fallback

        try:
            module._tokenizer = None
            module._tokenizer_loaded = True
            module._using_fallback = True

            # 10 words * 1.3 = 13 tokens
            text = " ".join(["word"] * 10)
            examples = [{"input_text": text}]

            kept, discarded = filter_by_token_length(
                examples, min_tokens=10, max_tokens=20
            )
            assert len(kept) == 1

            kept, discarded = filter_by_token_length(
                examples, min_tokens=14, max_tokens=20
            )
            assert len(discarded) == 1
        finally:
            module._tokenizer = orig_tokenizer
            module._tokenizer_loaded = orig_loaded
            module._using_fallback = orig_fallback
