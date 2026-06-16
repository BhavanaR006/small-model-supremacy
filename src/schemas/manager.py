"""Schema Manager for loading, validating, and querying JSON Schema definitions.

Handles discovery of .schema.json files, validation of schemas against
JSON Schema draft 2020-12, and instance validation against loaded schemas.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import jsonschema
from jsonschema import Draft202012Validator, ValidationError

logger = logging.getLogger(__name__)

# JSON Schema draft 2020-12 meta-schema URI
DRAFT_2020_12_META_SCHEMA = "https://json-schema.org/draft/2020-12/schema"


@dataclass
class ValidationResult:
    """Result of a validation operation."""

    success: bool
    errors: list[str] = field(default_factory=list)


@dataclass
class SchemaDefinition:
    """A loaded and parsed schema definition."""

    domain: str
    complexity: Literal["simple", "medium", "complex"]
    schema: dict
    description: str
    field_count: int


class SchemaManager:
    """Manages discovery, loading, and validation of JSON Schema definitions.

    Schema files are expected in the format: {domain}_{complexity}.schema.json
    within the specified schemas directory.
    """

    def __init__(self, schemas_dir: Path) -> None:
        """Initialize SchemaManager with the directory containing schema files.

        Args:
            schemas_dir: Path to the directory containing .schema.json files.
        """
        self.schemas_dir = Path(schemas_dir)
        self._schemas: dict[str, SchemaDefinition] = {}

    def load_all(self) -> dict[str, SchemaDefinition]:
        """Discover and load all .schema.json files from the schemas directory.

        Returns:
            Dictionary mapping schema_id to SchemaDefinition.
        """
        self._schemas = {}

        if not self.schemas_dir.exists():
            logger.warning("Schemas directory does not exist: %s", self.schemas_dir)
            return self._schemas

        schema_files = sorted(self.schemas_dir.glob("*.schema.json"))

        for schema_file in schema_files:
            try:
                schema_def = self._load_schema_file(schema_file)
                schema_id = f"{schema_def.domain}_{schema_def.complexity}"
                self._schemas[schema_id] = schema_def
                logger.info("Loaded schema: %s", schema_id)
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.error("Failed to load schema file %s: %s", schema_file.name, e)

        return self._schemas

    def _load_schema_file(self, path: Path) -> SchemaDefinition:
        """Load and parse a single schema file.

        Args:
            path: Path to the .schema.json file.

        Returns:
            Parsed SchemaDefinition.

        Raises:
            ValueError: If the filename doesn't match expected format.
            json.JSONDecodeError: If the file isn't valid JSON.
        """
        with open(path, "r", encoding="utf-8") as f:
            schema = json.load(f)

        # Parse filename: {domain}_{complexity}.schema.json
        stem = path.name.removesuffix(".schema.json")
        parts = stem.rsplit("_", 1)

        if len(parts) != 2:
            raise ValueError(
                f"Schema filename '{path.name}' does not match expected format "
                f"'{{domain}}_{{complexity}}.schema.json'"
            )

        domain, complexity = parts

        if complexity not in ("simple", "medium", "complex"):
            raise ValueError(
                f"Invalid complexity '{complexity}' in filename '{path.name}'. "
                f"Expected one of: simple, medium, complex"
            )

        description = schema.get("description", "")
        field_count = self._count_fields(schema)

        return SchemaDefinition(
            domain=domain,
            complexity=complexity,
            schema=schema,
            description=description,
            field_count=field_count,
        )

    def _count_fields(self, schema: dict) -> int:
        """Count the number of top-level properties in a schema.

        Args:
            schema: The JSON Schema dictionary.

        Returns:
            Number of top-level properties defined.
        """
        properties = schema.get("properties", {})
        return len(properties)

    def validate_schema(self, schema: dict) -> ValidationResult:
        """Verify that a schema is valid JSON Schema draft 2020-12.

        Args:
            schema: The JSON Schema dictionary to validate.

        Returns:
            ValidationResult with success/failure and error details.
        """
        errors: list[str] = []

        try:
            # Check that the schema declares draft 2020-12
            declared_schema = schema.get("$schema", "")
            if declared_schema and declared_schema != DRAFT_2020_12_META_SCHEMA:
                errors.append(
                    f"Schema declares '$schema': '{declared_schema}', "
                    f"expected '{DRAFT_2020_12_META_SCHEMA}'"
                )

            # Validate the schema against the draft 2020-12 meta-schema
            Draft202012Validator.check_schema(schema)

        except jsonschema.SchemaError as e:
            error_path = " -> ".join(str(p) for p in e.absolute_path) if e.absolute_path else "root"
            errors.append(f"Invalid schema at '{error_path}': {e.message}")

        return ValidationResult(success=len(errors) == 0, errors=errors)

    def validate_instance(self, instance: dict, schema_id: str) -> ValidationResult:
        """Validate an extracted data instance against a loaded schema.

        Args:
            instance: The data dictionary to validate.
            schema_id: The identifier of the schema to validate against
                       (format: {domain}_{complexity}).

        Returns:
            ValidationResult with success/failure and error details.
        """
        if schema_id not in self._schemas:
            return ValidationResult(
                success=False,
                errors=[f"Schema '{schema_id}' not found. Available: {list(self._schemas.keys())}"],
            )

        schema_def = self._schemas[schema_id]
        return self._validate_against_schema(instance, schema_def.schema)

    def _validate_against_schema(self, instance: dict, schema: dict) -> ValidationResult:
        """Validate an instance against a raw schema dictionary.

        Args:
            instance: The data dictionary to validate.
            schema: The JSON Schema dictionary.

        Returns:
            ValidationResult with success/failure and error details.
        """
        errors: list[str] = []
        validator = Draft202012Validator(schema)

        for error in validator.iter_errors(instance):
            error_path = " -> ".join(str(p) for p in error.absolute_path) if error.absolute_path else "root"
            errors.append(f"Validation error at '{error_path}': {error.message}")

        return ValidationResult(success=len(errors) == 0, errors=errors)

    def get_by_complexity(self, complexity: str) -> list[SchemaDefinition]:
        """Filter loaded schemas by complexity level.

        Args:
            complexity: One of "simple", "medium", "complex".

        Returns:
            List of SchemaDefinition objects matching the given complexity.
        """
        return [
            schema_def
            for schema_def in self._schemas.values()
            if schema_def.complexity == complexity
        ]

    def get_schema(self, schema_id: str) -> SchemaDefinition | None:
        """Get a specific schema by its ID.

        Args:
            schema_id: The schema identifier (format: {domain}_{complexity}).

        Returns:
            The SchemaDefinition if found, None otherwise.
        """
        return self._schemas.get(schema_id)

    @property
    def schemas(self) -> dict[str, SchemaDefinition]:
        """Access the loaded schemas dictionary."""
        return self._schemas
