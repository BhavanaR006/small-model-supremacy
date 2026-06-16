"""Data generation CLI entry point for the Small Model Supremacy project.

Generates synthetic training data for all schemas, validates, filters by token
length, splits into train/val/test sets, saves to JSONL files, and outputs
dataset statistics.

Usage:
    python generate_data.py --config config.yaml
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import click

from src.config.schema import ProjectConfig
from src.data.api_client import APIClient
from src.data.curator import DataExample, DatasetCurator, DatasetSplits
from src.schemas.manager import SchemaManager
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _save_generation_log(
    all_examples: list[DataExample], output_dir: Path
) -> None:
    """Save generation log with prompts, model, and parameters for each example.

    Args:
        all_examples: All generated examples (before filtering/splitting).
        output_dir: Directory to save generation_log.jsonl.
    """
    log_path = output_dir / "generation_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w", encoding="utf-8") as f:
        for example in all_examples:
            record = {
                "schema_id": example.schema_id,
                "difficulty_level": example.difficulty_level,
                "source_metadata": {
                    "generation_model": example.source_metadata.generation_model,
                    "generation_timestamp": example.source_metadata.generation_timestamp,
                    "prompt_id": example.source_metadata.prompt_id,
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(
        "Saved generation log",
        extra={"path": str(log_path), "count": len(all_examples)},
    )


def _save_stats(stats_dict: dict, output_dir: Path) -> None:
    """Save dataset statistics to data/stats.json.

    Args:
        stats_dict: Statistics dictionary to save.
        output_dir: Directory to save stats.json.
    """
    stats_path = output_dir / "stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_dict, f, indent=2, ensure_ascii=False)

    logger.info("Saved dataset statistics", extra={"path": str(stats_path)})


@click.command()
@click.option("--config", default="config.yaml", help="Path to config file")
def main(config: str) -> None:
    """Generate training data for structured extraction fine-tuning.

    Loads configuration, generates synthetic examples for all schemas using a
    frontier model API, validates against schemas, filters by token length,
    splits into train/val/test sets, and saves everything to JSONL files.
    """
    # 1. Load ProjectConfig from the config file
    config_path = Path(config)
    click.echo(f"Loading configuration from: {config_path}")

    try:
        project_config = ProjectConfig.from_yaml(config_path)
    except (FileNotFoundError, ValueError) as e:
        click.echo(f"Error loading config: {e}", err=True)
        sys.exit(1)

    # Validate configuration
    errors = project_config.validate_all()
    if errors:
        click.echo("Configuration validation errors:", err=True)
        for error in errors:
            click.echo(f"  - {error}", err=True)
        sys.exit(1)

    data_config = project_config.data
    output_dir = Path(data_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Create SchemaManager and load all schemas
    click.echo(f"Loading schemas from: {data_config.schemas_dir}")
    schema_manager = SchemaManager(Path(data_config.schemas_dir))
    schemas = schema_manager.load_all()

    if not schemas:
        click.echo("Error: No schemas found. Cannot generate data.", err=True)
        sys.exit(1)

    click.echo(f"Loaded {len(schemas)} schema(s): {list(schemas.keys())}")

    # 3. Create APIClient and DatasetCurator
    api_client = APIClient(
        provider="claude",
        cache_dir=Path(data_config.cache_dir),
    )
    curator = DatasetCurator(
        config=data_config,
        schema_manager=schema_manager,
        api_client=api_client,
    )

    # 4. For each schema, generate examples (min_examples_per_schema)
    all_examples: list[DataExample] = []
    generation_summary: dict[str, dict] = {}

    for schema_id in schemas:
        count = data_config.min_examples_per_schema
        click.echo(f"Generating {count} examples for schema: {schema_id}")

        examples = curator.generate_examples(schema_id, count)
        click.echo(f"  Generated {len(examples)} raw examples")

        generation_summary[schema_id] = {
            "requested": count,
            "generated": len(examples),
        }
        all_examples.extend(examples)

    click.echo(f"\nTotal raw examples generated: {len(all_examples)}")

    # Save generation log alongside the dataset
    _save_generation_log(all_examples, output_dir)

    # 5. Validate all examples against schemas, discard failures
    click.echo("Validating examples against schemas...")
    valid_examples: list[DataExample] = []
    validation_failures = 0

    for example in all_examples:
        result = curator.validate_example(example)
        if result.success:
            valid_examples.append(example)
        else:
            validation_failures += 1

    click.echo(
        f"  Validation: {len(valid_examples)} passed, "
        f"{validation_failures} discarded"
    )

    # 6. Filter by token length
    click.echo(
        f"Filtering by token length [{data_config.min_tokens}, "
        f"{data_config.max_tokens}]..."
    )
    kept_examples, discarded_examples = curator.filter_by_token_length(
        valid_examples,
        min_tokens=data_config.min_tokens,
        max_tokens=data_config.max_tokens,
    )

    click.echo(
        f"  Token filter: {len(kept_examples)} kept, "
        f"{len(discarded_examples)} discarded"
    )

    if not kept_examples:
        click.echo(
            "Error: No examples remaining after validation and filtering.",
            err=True,
        )
        sys.exit(1)

    # 7. Split into train/val/test
    click.echo(f"Splitting dataset (seed={data_config.seed})...")
    splits = curator.split_dataset(kept_examples, seed=data_config.seed)

    click.echo(
        f"  Split: train={len(splits.train)}, "
        f"val={len(splits.val)}, test={len(splits.test)}"
    )

    # 8. Save to data/train.jsonl, data/val.jsonl, data/test.jsonl
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    test_path = output_dir / "test.jsonl"

    curator.save_jsonl(splits.train, train_path)
    curator.save_jsonl(splits.val, val_path)
    curator.save_jsonl(splits.test, test_path)

    click.echo(f"  Saved: {train_path}, {val_path}, {test_path}")

    # 9. Compute and save stats to data/stats.json
    click.echo("Computing dataset statistics...")
    stats = curator.compute_statistics(splits)

    stats_dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "seed": data_config.seed,
            "min_tokens": data_config.min_tokens,
            "max_tokens": data_config.max_tokens,
            "min_examples_per_schema": data_config.min_examples_per_schema,
            "generation_model": data_config.generation_model,
            "adversarial_ratio": data_config.adversarial_ratio,
        },
        "generation_summary": generation_summary,
        "validation": {
            "total_generated": len(all_examples),
            "passed_validation": len(valid_examples),
            "passed_token_filter": len(kept_examples),
            "validation_failures": validation_failures,
            "token_filter_discarded": len(discarded_examples),
        },
        "splits": {
            "train": len(splits.train),
            "val": len(splits.val),
            "test": len(splits.test),
        },
        "token_length_distributions": stats.token_length_distributions,
        "field_coverage_rates": stats.field_coverage_rates,
        "difficulty_distribution": stats.difficulty_distribution,
    }

    _save_stats(stats_dict, output_dir)

    # 10. Print summary report
    click.echo("\n" + "=" * 60)
    click.echo("Dataset Generation Complete")
    click.echo("=" * 60)
    click.echo(f"  Total generated:       {len(all_examples)}")
    click.echo(f"  Passed validation:     {len(valid_examples)}")
    click.echo(f"  Passed token filter:   {len(kept_examples)}")
    click.echo(f"  Train set size:        {len(splits.train)}")
    click.echo(f"  Validation set size:   {len(splits.val)}")
    click.echo(f"  Test set size:         {len(splits.test)}")
    click.echo(f"  Statistics saved to:   {output_dir / 'stats.json'}")
    click.echo(f"  Generation log:        {output_dir / 'generation_log.jsonl'}")
    click.echo("=" * 60)


if __name__ == "__main__":
    main()
