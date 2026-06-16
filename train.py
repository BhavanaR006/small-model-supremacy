"""Training entry point for the Small Model Supremacy pipeline.

Loads configuration, instantiates the QLoRA trainer, loads datasets,
and runs training with early stopping, checkpointing, and W&B logging.

Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --resume checkpoints/checkpoint-step-500
"""

import sys
import time
from pathlib import Path

import click

from src.config.schema import ProjectConfig
from src.data.curator import load_jsonl
from src.training.trainer import Trainer
from src.utils.logging import get_logger

logger = get_logger(__name__)


@click.command()
@click.option("--config", default="config.yaml", help="Path to config file")
@click.option("--resume", default=None, help="Path to checkpoint to resume from")
def main(config: str, resume: str | None) -> None:
    """Train a Qwen2.5 model with QLoRA for structured data extraction.

    Loads the project configuration, validates it, loads training and
    validation datasets from JSONL, and runs the full training loop with
    early stopping, checkpointing, and Weights & Biases logging.
    """
    click.echo("=" * 60)
    click.echo("Small Model Supremacy — Training Pipeline")
    click.echo("=" * 60)

    # 1. Load ProjectConfig from config file
    config_path = Path(config)
    click.echo(f"\n[1/7] Loading configuration from: {config_path}")

    try:
        project_config = ProjectConfig.from_yaml(config_path)
    except FileNotFoundError as e:
        click.echo(f"ERROR: {e}", err=True)
        sys.exit(1)
    except ValueError as e:
        click.echo(f"ERROR: Invalid configuration: {e}", err=True)
        sys.exit(1)

    # 2. Validate config (exit with error if invalid)
    click.echo("[2/7] Validating configuration...")
    errors = project_config.validate_all()
    if errors:
        click.echo("ERROR: Configuration validation failed:", err=True)
        for error in errors:
            click.echo(f"  - {error}", err=True)
        sys.exit(1)

    click.echo(f"  Model: {project_config.model.name}")
    click.echo(f"  LoRA rank: {project_config.training.lora_rank}")
    click.echo(f"  LoRA alpha: {project_config.training.lora_alpha}")
    click.echo(f"  Learning rate: {project_config.training.learning_rate}")
    click.echo(f"  Batch size: {project_config.training.batch_size}")
    click.echo(f"  Epochs: {project_config.training.num_epochs}")
    click.echo(f"  Max memory: {project_config.training.max_memory_gb} GB")

    # 3. Load train/val datasets from JSONL files
    train_path = Path(project_config.data.output_dir) / "train.jsonl"
    val_path = Path(project_config.data.output_dir) / "val.jsonl"

    click.echo(f"\n[3/7] Loading datasets...")
    click.echo(f"  Train: {train_path}")
    click.echo(f"  Val:   {val_path}")

    if not train_path.exists():
        click.echo(f"ERROR: Training dataset not found: {train_path}", err=True)
        click.echo("  Run 'python generate_data.py' first to create the dataset.", err=True)
        sys.exit(1)

    if not val_path.exists():
        click.echo(f"ERROR: Validation dataset not found: {val_path}", err=True)
        click.echo("  Run 'python generate_data.py' first to create the dataset.", err=True)
        sys.exit(1)

    train_dataset = load_jsonl(train_path)
    val_dataset = load_jsonl(val_path)

    if not train_dataset:
        click.echo("ERROR: Training dataset is empty.", err=True)
        sys.exit(1)

    if not val_dataset:
        click.echo("ERROR: Validation dataset is empty.", err=True)
        sys.exit(1)

    click.echo(f"  Loaded {len(train_dataset)} training examples")
    click.echo(f"  Loaded {len(val_dataset)} validation examples")

    # 4. Create Trainer with config settings
    click.echo("\n[4/7] Initializing trainer...")
    trainer = Trainer(project_config)

    # 5. Setup model and QLoRA
    click.echo("[5/7] Loading model and applying QLoRA...")
    trainer.setup_model()
    trainer.setup_qlora()

    # 6. If --resume is provided, resume from checkpoint
    if resume is not None:
        resume_path = Path(resume)
        click.echo(f"\n[6/7] Resuming from checkpoint: {resume_path}")
        if not resume_path.exists():
            click.echo(f"ERROR: Checkpoint not found: {resume_path}", err=True)
            sys.exit(1)
        trainer.resume_from_checkpoint(resume_path)
    else:
        click.echo("\n[6/7] Starting fresh training (no checkpoint to resume)")

    # 7. Run training
    click.echo("\n[7/7] Starting training...")
    click.echo("-" * 60)

    start_time = time.time()
    result = trainer.train(train_dataset, val_dataset)
    elapsed = time.time() - start_time

    # 8. Print final results
    click.echo("\n" + "=" * 60)
    click.echo("Training Complete!")
    click.echo("=" * 60)
    click.echo(f"\n  Final train loss:     {result.final_train_loss:.6f}")
    click.echo(f"  Final val loss:       {result.final_val_loss:.6f}")
    click.echo(f"  Best val loss:        {result.best_val_loss:.6f}")
    click.echo(f"  Total steps:          {result.total_steps}")
    click.echo(f"  Total time:           {elapsed:.1f}s ({elapsed/60:.1f}min)")
    click.echo(f"  Early stopped:        {result.early_stopped}")

    if result.best_checkpoint_path:
        click.echo(f"  Best checkpoint:      {result.best_checkpoint_path}")
    else:
        click.echo("  Best checkpoint:      (none saved)")

    if result.early_stopped and result.stopped_at_step is not None:
        click.echo(f"  Stopped at step:      {result.stopped_at_step}")

    click.echo("\nDone.")


if __name__ == "__main__":
    main()
