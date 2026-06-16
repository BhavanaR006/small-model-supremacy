"""Training entry point for the Small Model Supremacy pipeline.

Loads configuration, instantiates the QLoRA trainer, loads datasets,
tokenizes them into DataLoaders, and runs training with early stopping,
checkpointing, and W&B logging.

Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --resume checkpoints/checkpoint-step-500
"""

import json
import sys
import time
from pathlib import Path

import click

from src.config.schema import ProjectConfig
from src.data.curator import load_jsonl
from src.data.prompt_template import format_prompt
from src.schemas.manager import SchemaManager
from src.training.trainer import Trainer
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _build_prompts(examples, schema_manager):
    """Convert DataExample objects into formatted prompts with target outputs."""
    prompts = []
    targets = []

    for ex in examples:
        schema_def = schema_manager.get_schema(ex.schema_id)
        if schema_def is None:
            continue

        # Use a simple example for the one-shot (first field values)
        example_input = "Example input text for demonstration."
        example_output = {k: f"example_{k}" for k in list(schema_def.schema.get("properties", {}).keys())[:3]}

        prompt = format_prompt(
            schema=schema_def.schema,
            input_text=ex.input_text,
            example_input=example_input,
            example_output=example_output,
        )
        target = json.dumps(ex.expected_output, ensure_ascii=False)
        prompts.append(prompt)
        targets.append(target)

    return prompts, targets


def _create_dataloader(prompts, targets, tokenizer, batch_size, max_seq_length):
    """Tokenize prompt+target pairs and create a PyTorch DataLoader."""
    import torch
    from torch.utils.data import DataLoader, Dataset

    class ExtractionDataset(Dataset):
        def __init__(self, input_ids_list, attention_mask_list, labels_list):
            self.input_ids = input_ids_list
            self.attention_mask = attention_mask_list
            self.labels = labels_list

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            return {
                "input_ids": self.input_ids[idx],
                "attention_mask": self.attention_mask[idx],
                "labels": self.labels[idx],
            }

    all_input_ids = []
    all_attention_masks = []
    all_labels = []

    for prompt, target in zip(prompts, targets):
        # Concatenate prompt + target for causal LM training
        full_text = prompt + target + tokenizer.eos_token

        encoded = tokenizer(
            full_text,
            max_length=max_seq_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # Create labels: mask the prompt tokens with -100, only predict target
        prompt_encoded = tokenizer(
            prompt,
            max_length=max_seq_length,
            truncation=True,
            return_tensors="pt",
        )
        prompt_length = prompt_encoded["input_ids"].shape[1]

        labels = input_ids.clone()
        labels[:prompt_length] = -100  # Don't compute loss on prompt
        # Also mask padding
        labels[attention_mask == 0] = -100

        all_input_ids.append(input_ids)
        all_attention_masks.append(attention_mask)
        all_labels.append(labels)

    dataset = ExtractionDataset(all_input_ids, all_attention_masks, all_labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


@click.command()
@click.option("--config", default="config.yaml", help="Path to config file")
@click.option("--resume", default=None, help="Path to checkpoint to resume from")
def main(config: str, resume: str | None) -> None:
    """Train a Qwen2.5 model with QLoRA for structured data extraction."""
    click.echo("=" * 60)
    click.echo("Small Model Supremacy — Training Pipeline")
    click.echo("=" * 60)

    # 1. Load ProjectConfig
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

    # 2. Validate config
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

    # 3. Load datasets
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

    train_examples = load_jsonl(train_path)
    val_examples = load_jsonl(val_path)

    if not train_examples:
        click.echo("ERROR: Training dataset is empty.", err=True)
        sys.exit(1)
    if not val_examples:
        click.echo("ERROR: Validation dataset is empty.", err=True)
        sys.exit(1)

    click.echo(f"  Loaded {len(train_examples)} training examples")
    click.echo(f"  Loaded {len(val_examples)} validation examples")

    # 4. Initialize trainer
    click.echo("\n[4/7] Initializing trainer...")
    trainer = Trainer(
        config=project_config.training,
        model_config=project_config.model,
        checkpoint_dir=Path("checkpoints/"),
        wandb_project=project_config.infrastructure.wandb_project,
        wandb_enabled=False,  # Disable W&B for free-tier usage
    )

    # 5. Setup model and QLoRA
    click.echo("[5/7] Loading model and applying QLoRA...")
    trainer.setup_model()
    trainer.setup_qlora()

    # 6. Tokenize data and create DataLoaders
    click.echo("\n[6/7] Tokenizing data and creating DataLoaders...")

    # Load schemas for prompt formatting
    schema_manager = SchemaManager(Path(project_config.data.schemas_dir))
    schema_manager.load_all()

    train_prompts, train_targets = _build_prompts(train_examples, schema_manager)
    val_prompts, val_targets = _build_prompts(val_examples, schema_manager)

    click.echo(f"  Tokenizing {len(train_prompts)} training prompts...")
    train_dataloader = _create_dataloader(
        train_prompts, train_targets, trainer.tokenizer,
        batch_size=project_config.training.batch_size,
        max_seq_length=project_config.model.max_seq_length,
    )

    click.echo(f"  Tokenizing {len(val_prompts)} validation prompts...")
    val_dataloader = _create_dataloader(
        val_prompts, val_targets, trainer.tokenizer,
        batch_size=project_config.training.batch_size,
        max_seq_length=project_config.model.max_seq_length,
    )

    click.echo(f"  Train batches: {len(train_dataloader)}")
    click.echo(f"  Val batches: {len(val_dataloader)}")

    # 7. Resume or start fresh
    if resume is not None:
        resume_path = Path(resume)
        click.echo(f"\n  Resuming from checkpoint: {resume_path}")
        if not resume_path.exists():
            click.echo(f"ERROR: Checkpoint not found: {resume_path}", err=True)
            sys.exit(1)
        trainer.resume_from_checkpoint(resume_path)
    else:
        click.echo("\n  Starting fresh training (no checkpoint to resume)")

    # 8. Run training
    click.echo("\n[7/7] Starting training...")
    click.echo("-" * 60)

    start_time = time.time()
    result = trainer.train(train_dataloader, val_dataloader)
    elapsed = time.time() - start_time

    # Print final results
    click.echo("\n" + "=" * 60)
    click.echo("Training Complete!")
    click.echo("=" * 60)
    click.echo(f"\n  Final train loss:     {result.final_train_loss:.6f}")
    click.echo(f"  Final val loss:       {result.final_val_loss:.6f}")
    click.echo(f"  Total steps:          {result.total_steps}")
    click.echo(f"  Total time:           {elapsed:.1f}s ({elapsed/60:.1f}min)")
    click.echo(f"  Early stopped:        {result.early_stopped}")

    if result.best_checkpoint_path:
        click.echo(f"  Best checkpoint:      {result.best_checkpoint_path}")
    else:
        click.echo("  Best checkpoint:      (none saved)")

    click.echo("\nDone.")


if __name__ == "__main__":
    main()
