"""Prompt template formatting for structured data extraction.

Provides utilities to format prompts that include the full schema definition,
a natural language task description, and example input-output pairs for
fine-tuning and inference with extraction models.
"""

from __future__ import annotations

import json


def format_prompt(
    schema: dict,
    input_text: str,
    example_input: str,
    example_output: dict,
    task_description: str | None = None,
) -> str:
    """Format a complete prompt for structured data extraction.

    Constructs a prompt containing the schema definition, a natural language
    task description, an example input-output pair, and the actual input text
    to extract from.

    Args:
        schema: The full JSON Schema definition (must contain a "properties" key).
        input_text: The text to extract structured data from.
        example_input: An example input text demonstrating the task.
        example_output: The corresponding example output as a dict.
        task_description: Optional override for the natural language description.
            If not provided, it is derived from the schema's "title" and
            "description" fields.

    Returns:
        A formatted prompt string ready for model input.

    Raises:
        ValueError: If the schema is not a dict or lacks a "properties" key.
    """
    _validate_schema(schema)

    description = task_description or _derive_task_description(schema)
    schema_json = json.dumps(schema, indent=2)
    example_output_json = json.dumps(example_output, indent=2)

    prompt = (
        f"You are a structured data extraction assistant. "
        f"{description}\n"
        f"\n"
        f"## Schema\n"
        f"{schema_json}\n"
        f"\n"
        f"## Example\n"
        f"Input: {example_input}\n"
        f"Output: {example_output_json}\n"
        f"\n"
        f"## Task\n"
        f"Input: {input_text}\n"
        f"Output:"
    )

    return prompt


def _validate_schema(schema: dict) -> None:
    """Validate that the schema is a dict with at least a 'properties' key.

    Args:
        schema: The schema to validate.

    Raises:
        ValueError: If schema is not a dict or lacks "properties".
    """
    if not isinstance(schema, dict):
        raise ValueError(
            f"Schema must be a dict, got {type(schema).__name__}"
        )
    if "properties" not in schema:
        raise ValueError(
            "Schema must contain a 'properties' key defining the extraction fields"
        )


def _derive_task_description(schema: dict) -> str:
    """Derive a natural language task description from schema metadata.

    Uses the schema's "title" and "description" fields to produce a
    human-readable description of the extraction task.

    Args:
        schema: The JSON Schema dictionary.

    Returns:
        A natural language description string.
    """
    title = schema.get("title", "")
    description = schema.get("description", "")

    if title and description:
        return (
            f"Given the following schema and input text, extract the relevant "
            f"information as a JSON object. The target structure is: "
            f"{title} — {description}"
        )
    elif title:
        return (
            f"Given the following schema and input text, extract the relevant "
            f"information as a JSON object. The target structure is: {title}."
        )
    elif description:
        return (
            f"Given the following schema and input text, extract the relevant "
            f"information as a JSON object. {description}"
        )
    else:
        return (
            "Given the following schema and input text, extract the relevant "
            "information as a JSON object."
        )
