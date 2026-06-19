"""Tests for the generate_data CLI entry point."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from generate_data import main, _save_generation_log, _save_stats
from src.data.curator import DataExample, DatasetSplits, DatasetStats, SourceMetadata


@pytest.fixture
def sample_config_file(tmp_path):
    """Create a minimal valid config.yaml for testing."""
    config = {
        "model": {"name": "Qwen/Qwen2.5-1.5B", "max_seq_length": 2048},
        "data": {
            "schemas_dir": str(tmp_path / "schemas"),
            "output_dir": str(tmp_path / "data"),
            "seed": 42,
            "min_tokens": 50,
            "max_tokens": 2000,
            "adversarial_ratio": 0.15,
            "min_examples_per_schema": 10,
            "min_test_per_schema": 5,
            "generation_model": "claude-3-5-sonnet-20241022",
            "cache_dir": str(tmp_path / "cache"),
        },
    }
    import yaml

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config))
    return config_path


@pytest.fixture
def sample_schema_dir(tmp_path):
    """Create a schemas directory with a valid schema file."""
    schemas_dir = tmp_path / "schemas"
    schemas_dir.mkdir()
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "conference_talk_simple",
        "title": "Conference Talk",
        "description": "A simple conference talk schema",
        "type": "object",
        "properties": {
            "speaker_name": {"type": "string"},
            "topic": {"type": "string"},
        },
        "required": ["speaker_name", "topic"],
    }
    (schemas_dir / "conference_talk_simple.schema.json").write_text(
        json.dumps(schema)
    )
    return schemas_dir


@pytest.fixture
def sample_examples():
    """Create sample DataExample objects for testing."""
    examples = []
    for i in range(10):
        examples.append(
            DataExample(
                input_text=f"Dr. Smith presented on topic {i} at the conference in city {i}. " * 5,
                expected_output={"speaker_name": f"Dr. Smith {i}", "topic": f"Topic {i}"},
                schema_id="conference_talk_simple",
                difficulty_level="simple",
                source_metadata=SourceMetadata(
                    generation_model="claude-3-5-sonnet-20241022",
                    generation_timestamp="2024-12-01T10:00:00Z",
                    prompt_id=f"gen_conference_talk_simple_{i:04d}",
                ),
            )
        )
    return examples


class TestSaveGenerationLog:
    """Tests for the _save_generation_log helper."""

    def test_saves_log_file(self, tmp_path, sample_examples):
        _save_generation_log(sample_examples, tmp_path)
        log_path = tmp_path / "generation_log.jsonl"
        assert log_path.exists()

        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 10

        # Verify structure of first entry
        entry = json.loads(lines[0])
        assert "schema_id" in entry
        assert "difficulty_level" in entry
        assert "source_metadata" in entry
        assert "generation_model" in entry["source_metadata"]

    def test_creates_parent_dirs(self, tmp_path, sample_examples):
        output_dir = tmp_path / "nested" / "dir"
        _save_generation_log(sample_examples, output_dir)
        assert (output_dir / "generation_log.jsonl").exists()


class TestSaveStats:
    """Tests for the _save_stats helper."""

    def test_saves_stats_json(self, tmp_path):
        stats = {"total": 100, "splits": {"train": 80, "val": 10, "test": 10}}
        _save_stats(stats, tmp_path)

        stats_path = tmp_path / "stats.json"
        assert stats_path.exists()

        loaded = json.loads(stats_path.read_text())
        assert loaded == stats

    def test_creates_parent_dirs(self, tmp_path):
        output_dir = tmp_path / "deep" / "nested"
        _save_stats({"key": "value"}, output_dir)
        assert (output_dir / "stats.json").exists()


class TestMainCLI:
    """Tests for the main CLI command."""

    def test_help_flag(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "config" in result.output.lower()

    def test_missing_config_file(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(main, ["--config", str(tmp_path / "nonexistent.yaml")])
        assert result.exit_code != 0
        assert "Error" in result.output or "error" in result.output.lower()

    @patch("generate_data.DatasetCurator")
    @patch("generate_data.APIClient")
    @patch("generate_data.SchemaManager")
    def test_full_pipeline(
        self,
        mock_schema_manager_cls,
        mock_api_client_cls,
        mock_curator_cls,
        tmp_path,
        sample_examples,
    ):
        """Test the full pipeline end-to-end with mocked components."""
        import yaml
        from src.schemas.manager import SchemaDefinition, ValidationResult

        # Set up schemas directory
        schemas_dir = tmp_path / "schemas"
        schemas_dir.mkdir()
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "Test",
            "description": "Test schema",
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        (schemas_dir / "conference_talk_simple.schema.json").write_text(
            json.dumps(schema)
        )

        # Write config
        config = {
            "model": {"name": "Qwen/Qwen2.5-1.5B"},
            "data": {
                "schemas_dir": str(schemas_dir),
                "output_dir": str(tmp_path / "data"),
                "seed": 42,
                "min_tokens": 50,
                "max_tokens": 2000,
                "adversarial_ratio": 0.15,
                "min_examples_per_schema": 10,
                "min_test_per_schema": 5,
                "generation_model": "claude-3-5-sonnet-20241022",
                "cache_dir": str(tmp_path / "cache"),
            },
        }
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump(config))

        # Mock SchemaManager
        mock_sm = MagicMock()
        mock_sm.load_all.return_value = {
            "conference_talk_simple": SchemaDefinition(
                domain="conference_talk",
                complexity="simple",
                schema=schema,
                description="Test schema",
                field_count=1,
            )
        }
        mock_schema_manager_cls.return_value = mock_sm

        # Mock APIClient
        mock_api = MagicMock()
        mock_api_client_cls.return_value = mock_api

        # Mock DatasetCurator
        mock_curator = MagicMock()
        mock_curator.generate_examples.return_value = sample_examples
        mock_curator.validate_example.return_value = ValidationResult(
            success=True, errors=[]
        )
        mock_curator.filter_by_token_length.return_value = (
            sample_examples,
            [],
        )
        splits = DatasetSplits(
            train=sample_examples[:8],
            val=sample_examples[8:9],
            test=sample_examples[9:],
        )
        mock_curator.split_dataset.return_value = splits
        mock_curator.compute_statistics.return_value = DatasetStats(
            token_length_distributions={
                "conference_talk_simple": {
                    "min": 50, "max": 200, "mean": 120.5, "median": 115.0, "p95": 190
                }
            },
            field_coverage_rates={
                "conference_talk_simple": {"speaker_name": 100.0, "topic": 95.0}
            },
            difficulty_distribution={
                "simple": {"count": 10, "percentage": 100.0}
            },
        )
        mock_curator_cls.return_value = mock_curator

        # Run CLI
        runner = CliRunner()
        result = runner.invoke(main, ["--config", str(config_path)])

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert "Dataset Generation Complete" in result.output

        # Verify stats.json was saved
        stats_path = tmp_path / "data" / "stats.json"
        assert stats_path.exists()

        stats = json.loads(stats_path.read_text())
        assert "token_length_distributions" in stats
        assert "field_coverage_rates" in stats
        assert "difficulty_distribution" in stats
        assert "splits" in stats
        assert stats["splits"]["train"] == 8
        assert stats["splits"]["val"] == 1
        assert stats["splits"]["test"] == 1

        # Verify generation_log.jsonl was saved
        log_path = tmp_path / "data" / "generation_log.jsonl"
        assert log_path.exists()

    def test_no_schemas_exits_with_error(self, tmp_path):
        """Test that the script exits if no schemas are found."""
        import yaml

        schemas_dir = tmp_path / "empty_schemas"
        schemas_dir.mkdir()

        config = {
            "model": {"name": "Qwen/Qwen2.5-1.5B"},
            "data": {
                "schemas_dir": str(schemas_dir),
                "output_dir": str(tmp_path / "data"),
                "seed": 42,
                "min_tokens": 50,
                "max_tokens": 2000,
                "adversarial_ratio": 0.15,
                "min_examples_per_schema": 10,
                "min_test_per_schema": 5,
                "generation_model": "claude-3-5-sonnet-20241022",
                "cache_dir": str(tmp_path / "cache"),
            },
        }
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump(config))

        runner = CliRunner()
        result = runner.invoke(main, ["--config", str(config_path)])
        assert result.exit_code != 0
        assert "No schemas found" in result.output
