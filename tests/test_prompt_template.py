"""Unit tests for prompt template formatting."""

import json

import pytest

from src.data.prompt_template import format_prompt, _validate_schema, _derive_task_description


@pytest.fixture
def simple_schema():
    """A minimal valid schema for testing."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Conference Talk",
        "description": "Structured extraction schema for conference talk information.",
        "type": "object",
        "properties": {
            "speaker_name": {"type": "string", "description": "Full name of the speaker"},
            "topic": {"type": "string", "description": "Topic of the talk"},
        },
        "required": ["speaker_name", "topic"],
    }


@pytest.fixture
def example_input():
    return "Dr. Jane Smith presented her research on CRISPR at BioConf 2024."


@pytest.fixture
def example_output():
    return {"speaker_name": "Dr. Jane Smith", "topic": "CRISPR"}


class TestFormatPrompt:
    """Tests for the format_prompt function."""

    def test_contains_schema_json(self, simple_schema, example_input, example_output):
        """Prompt must contain the full pretty-printed schema."""
        prompt = format_prompt(simple_schema, "some input", example_input, example_output)
        schema_json = json.dumps(simple_schema, indent=2)
        assert schema_json in prompt

    def test_contains_example_input(self, simple_schema, example_input, example_output):
        """Prompt must contain the example input text."""
        prompt = format_prompt(simple_schema, "some input", example_input, example_output)
        assert example_input in prompt

    def test_contains_example_output(self, simple_schema, example_input, example_output):
        """Prompt must contain the pretty-printed example output."""
        prompt = format_prompt(simple_schema, "some input", example_input, example_output)
        example_output_json = json.dumps(example_output, indent=2)
        assert example_output_json in prompt

    def test_contains_input_text(self, simple_schema, example_input, example_output):
        """Prompt must contain the actual input text for extraction."""
        input_text = "Professor Bob gave a keynote on AI safety."
        prompt = format_prompt(simple_schema, input_text, example_input, example_output)
        assert input_text in prompt

    def test_contains_natural_language_description(self, simple_schema, example_input, example_output):
        """Prompt must include a natural language description derived from schema."""
        prompt = format_prompt(simple_schema, "some input", example_input, example_output)
        assert "Conference Talk" in prompt
        assert "Structured extraction schema for conference talk information." in prompt

    def test_custom_task_description(self, simple_schema, example_input, example_output):
        """Custom task description overrides schema-derived description."""
        custom = "Extract speaker details from text."
        prompt = format_prompt(
            simple_schema, "some input", example_input, example_output,
            task_description=custom,
        )
        assert custom in prompt

    def test_ends_with_output_marker(self, simple_schema, example_input, example_output):
        """Prompt should end with 'Output:' ready for model completion."""
        prompt = format_prompt(simple_schema, "some input", example_input, example_output)
        assert prompt.endswith("Output:")

    def test_has_section_headers(self, simple_schema, example_input, example_output):
        """Prompt must contain Schema, Example, and Task section headers."""
        prompt = format_prompt(simple_schema, "some input", example_input, example_output)
        assert "## Schema" in prompt
        assert "## Example" in prompt
        assert "## Task" in prompt

    def test_raises_on_non_dict_schema(self, example_input, example_output):
        """Should raise ValueError if schema is not a dict."""
        with pytest.raises(ValueError, match="must be a dict"):
            format_prompt("not a dict", "input", example_input, example_output)

    def test_raises_on_missing_properties_key(self, example_input, example_output):
        """Should raise ValueError if schema lacks 'properties' key."""
        bad_schema = {"title": "Test", "type": "object"}
        with pytest.raises(ValueError, match="properties"):
            format_prompt(bad_schema, "input", example_input, example_output)


class TestValidateSchema:
    """Tests for schema validation helper."""

    def test_valid_schema_passes(self, simple_schema):
        """Valid schema with properties key should not raise."""
        _validate_schema(simple_schema)  # Should not raise

    def test_rejects_non_dict(self):
        with pytest.raises(ValueError, match="must be a dict"):
            _validate_schema([1, 2, 3])

    def test_rejects_missing_properties(self):
        with pytest.raises(ValueError, match="properties"):
            _validate_schema({"type": "object"})


class TestDeriveTaskDescription:
    """Tests for the description derivation helper."""

    def test_title_and_description(self):
        schema = {"title": "Product", "description": "A product listing.", "properties": {}}
        desc = _derive_task_description(schema)
        assert "Product" in desc
        assert "A product listing." in desc

    def test_title_only(self):
        schema = {"title": "Product", "properties": {}}
        desc = _derive_task_description(schema)
        assert "Product" in desc

    def test_description_only(self):
        schema = {"description": "A product listing.", "properties": {}}
        desc = _derive_task_description(schema)
        assert "A product listing." in desc

    def test_no_title_no_description(self):
        schema = {"properties": {}}
        desc = _derive_task_description(schema)
        assert "extract" in desc.lower()
