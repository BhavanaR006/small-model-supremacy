"""Evaluation entry point for Small Model Supremacy.

Loads the fine-tuned model and evaluates it against baselines (GPT-4o,
Claude 3.5 Sonnet, base model). Generates results CSV and comparison charts
in the results/ directory.

Usage:
    python evaluate.py --model-path checkpoints/best --config config.yaml
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click

from src.config.schema import ProjectConfig
from src.evaluation.evaluator import EvalExample, Evaluator
from src.evaluation.metrics import MetricsCalculator
from src.evaluation.visualization import generate_charts
from src.parsing.output_parser import OutputParser
from src.schemas.manager import SchemaManager

logger = logging.getLogger(__name__)


def _load_test_set(path: Path) -> list[EvalExample]:
    """Load evaluation test set from a JSONL file.

    Each line must be a JSON object with keys: input_text, expected_output,
    schema_id, difficulty_level.

    Args:
        path: Path to the JSONL test set file.

    Returns:
        List of EvalExample instances.

    Raises:
        FileNotFoundError: If the test set file does not exist.
        ValueError: If the file is empty or contains no valid examples.
    """
    if not path.exists():
        raise FileNotFoundError(f"Test set file not found: {path}")

    examples: list[EvalExample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                example = EvalExample(
                    input_text=data["input_text"],
                    expected_output=data["expected_output"],
                    schema_id=data["schema_id"],
                    difficulty_level=data.get("difficulty_level", "simple"),
                )
                examples.append(example)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(
                    "Skipping invalid line %d in test set: %s", line_num, e
                )

    if not examples:
        raise ValueError(f"No valid examples found in test set: {path}")

    return examples


def _load_fine_tuned_model(model_path: str):
    """Load a fine-tuned model from the given path.

    Attempts to load via HuggingFace transformers with PEFT adapter.
    Falls back to a simple wrapper if transformers/PEFT are not available
    or model loading fails.

    Args:
        model_path: Path to the fine-tuned model checkpoint directory.

    Returns:
        A model object with a generate(prompt: str) -> str method.
    """
    model_path_obj = Path(model_path)

    if not model_path_obj.exists():
        logger.error("Model path does not exist: %s", model_path)
        raise FileNotFoundError(f"Model path not found: {model_path}")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        logger.info("Loading fine-tuned model from %s", model_path)

        # Check for adapter config to determine if this is a PEFT model
        adapter_config = model_path_obj / "adapter_config.json"
        if adapter_config.exists():
            with open(adapter_config, "r") as f:
                config_data = json.load(f)
            base_model_name = config_data.get("base_model_name_or_path", "")

            tokenizer = AutoTokenizer.from_pretrained(
                base_model_name, trust_remote_code=True
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                device_map="auto",
                trust_remote_code=True,
            )
            model = PeftModel.from_pretrained(base_model, model_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path, device_map="auto", trust_remote_code=True
            )

        class HFModelWrapper:
            """Wrapper around HuggingFace model providing generate(prompt) interface."""

            def __init__(self, model, tokenizer):
                self._model = model
                self._tokenizer = tokenizer

            def generate(self, prompt: str) -> str:
                inputs = self._tokenizer(prompt, return_tensors="pt").to(
                    self._model.device
                )
                outputs = self._model.generate(
                    **inputs, max_new_tokens=2048, temperature=0.0, do_sample=False
                )
                # Decode only the generated tokens (skip input)
                generated = outputs[0][inputs["input_ids"].shape[1] :]
                return self._tokenizer.decode(generated, skip_special_tokens=True)

        return HFModelWrapper(model, tokenizer)

    except (ImportError, OSError, Exception) as e:
        logger.error("Failed to load model from %s: %s", model_path, e)
        raise RuntimeError(
            f"Could not load fine-tuned model from '{model_path}': {e}"
        ) from e


@click.command()
@click.option(
    "--config",
    "config_path",
    default="config.yaml",
    help="Path to config file",
    type=click.Path(exists=False),
)
@click.option(
    "--model-path",
    required=True,
    help="Path to fine-tuned model checkpoint",
    type=click.Path(),
)
@click.option(
    "--test-set",
    "test_set_path",
    default=None,
    help="Override test set path (default: from config)",
    type=click.Path(),
)
def main(config_path: str, model_path: str, test_set_path: str | None) -> None:
    """Run evaluation on the fine-tuned model and baselines.

    Loads the config, test set, and fine-tuned model. Runs evaluation
    against all configured baselines and generates results CSV and
    comparison charts in the results/ directory.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    # 1. Load ProjectConfig from config file
    logger.info("Loading configuration from %s", config_path)
    try:
        project_config = ProjectConfig.from_yaml(Path(config_path))
    except (FileNotFoundError, ValueError) as e:
        click.echo(f"Error loading config: {e}", err=True)
        sys.exit(1)

    # Validate config
    errors = project_config.validate_all()
    if errors:
        click.echo("Configuration validation errors:", err=True)
        for error in errors:
            click.echo(f"  - {error}", err=True)
        sys.exit(1)

    # 2. Load test set from JSONL
    test_path = Path(test_set_path) if test_set_path else Path(project_config.evaluation.test_set_path)
    logger.info("Loading test set from %s", test_path)
    try:
        test_set = _load_test_set(test_path)
    except (FileNotFoundError, ValueError) as e:
        click.echo(f"Error loading test set: {e}", err=True)
        sys.exit(1)

    logger.info("Loaded %d test examples", len(test_set))

    # 3. Load fine-tuned model
    logger.info("Loading fine-tuned model from %s", model_path)
    try:
        model = _load_fine_tuned_model(model_path)
    except (FileNotFoundError, RuntimeError) as e:
        click.echo(f"Error loading model: {e}", err=True)
        sys.exit(1)

    # 4. Create Evaluator with all components
    schemas_dir = Path(project_config.data.schemas_dir)
    schema_manager = SchemaManager(schemas_dir)
    schema_manager.load_all()

    output_parser = OutputParser()
    metrics_calculator = MetricsCalculator(schema_manager)

    evaluator = Evaluator(
        config=project_config.evaluation,
        schema_manager=schema_manager,
        output_parser=output_parser,
        metrics_calculator=metrics_calculator,
    )

    # 5. Run full evaluation (fine-tuned + baselines)
    logger.info(
        "Running evaluation against baselines: %s",
        project_config.evaluation.baselines,
    )
    results = evaluator.run_full_evaluation(
        model=model,
        test_set=test_set,
        baselines=project_config.evaluation.baselines,
    )

    # 6. Save results CSV to results/
    output_dir = Path(project_config.evaluation.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = evaluator.save_results(results, output_dir)
    logger.info("Results CSV saved to %s", csv_path)

    # 7. Generate comparison charts to results/
    # Pass "overall" metrics per model for chart generation
    chart_data = {
        model_name: metrics["overall"]
        for model_name, metrics in results.items()
        if "overall" in metrics
    }
    generate_charts(chart_data, output_dir)
    logger.info("Charts saved to %s", output_dir)

    # 8. Print summary
    click.echo(f"\nEvaluation complete. Results saved to: {output_dir}/")
    click.echo(f"  - CSV: {csv_path}")
    click.echo(f"  - Charts: {output_dir / 'metrics_comparison.png'}")
    click.echo(f"  - Charts: {output_dir / 'latency_comparison.png'}")


if __name__ == "__main__":
    main()
