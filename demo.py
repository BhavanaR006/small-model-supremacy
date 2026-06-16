"""Interactive demo CLI for structured data extraction.

Loads a fine-tuned model checkpoint, accepts free-text input, and outputs
structured JSON extraction using the trained model and output parser.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from src.parsing.output_parser import OutputParser
from src.schemas.manager import SchemaManager


def _load_model_and_tokenizer(model_path: str):
    """Lazily load the fine-tuned model and tokenizer.

    Uses lazy imports for torch/transformers to avoid import-time failures
    when these heavy dependencies are not installed or CUDA is unavailable.

    Args:
        model_path: Path to the fine-tuned model checkpoint directory.

    Returns:
        Tuple of (model, tokenizer).

    Raises:
        click.ClickException: If model loading fails.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise click.ClickException(
            f"Required dependencies not installed: {e}. "
            "Install with: pip install torch transformers"
        )

    model_dir = Path(model_path)
    if not model_dir.exists():
        raise click.ClickException(f"Model path not found: {model_path}")

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            click.echo("Warning: CUDA not available, using CPU (inference will be slow).", err=True)

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )
        if device == "cpu":
            model = model.to(device)

        model.eval()
        return model, tokenizer
    except Exception as e:
        raise click.ClickException(f"Failed to load model from '{model_path}': {e}")


def _load_schemas(schemas_dir: Path) -> SchemaManager:
    """Load all schemas from the schemas directory.

    Args:
        schemas_dir: Path to the schemas directory.

    Returns:
        Initialized SchemaManager with loaded schemas.

    Raises:
        click.ClickException: If no schemas are found.
    """
    manager = SchemaManager(schemas_dir)
    schemas = manager.load_all()
    if not schemas:
        raise click.ClickException(
            f"No schemas found in '{schemas_dir}'. "
            "Ensure .schema.json files exist in the schemas/ directory."
        )
    return manager


def _get_example_for_schema(schema_def) -> tuple[str, dict]:
    """Generate a simple example input-output pair for a schema.

    Args:
        schema_def: The SchemaDefinition to generate an example for.

    Returns:
        Tuple of (example_input, example_output).
    """
    properties = schema_def.schema.get("properties", {})
    example_output = {}
    for field_name, field_def in properties.items():
        field_type = field_def.get("type", "string")
        if field_type == "string":
            example_output[field_name] = f"example_{field_name}"
        elif field_type == "number" or field_type == "integer":
            example_output[field_name] = 0
        elif field_type == "boolean":
            example_output[field_name] = True
        elif field_type == "array":
            example_output[field_name] = []
        elif field_type == "object":
            example_output[field_name] = {}
        else:
            example_output[field_name] = f"example_{field_name}"

    example_input = "This is an example input text for demonstration purposes."
    return example_input, example_output


def _run_inference(model, tokenizer, prompt: str) -> str:
    """Run model inference on a formatted prompt.

    Args:
        model: The loaded model.
        tokenizer: The loaded tokenizer.
        prompt: The formatted prompt string.

    Returns:
        The generated text output (excluding the prompt).

    Raises:
        click.ClickException: If inference fails.
    """
    try:
        import torch
    except ImportError as e:
        raise click.ClickException(f"torch not available: {e}")

    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the generated tokens (after the prompt)
        generated_ids = outputs[0][input_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return generated_text.strip()
    except Exception as e:
        raise click.ClickException(f"Inference failed: {e}")


def _extract_and_display(
    model,
    tokenizer,
    parser: OutputParser,
    schema_manager: SchemaManager,
    schema_id: str,
    input_text: str,
) -> None:
    """Run extraction on input text and display results.

    Args:
        model: The loaded model.
        tokenizer: The loaded tokenizer.
        parser: The OutputParser instance.
        schema_manager: The SchemaManager with loaded schemas.
        schema_id: The schema ID to use for extraction.
        input_text: The text to extract from.
    """
    from src.data.prompt_template import format_prompt

    schema_def = schema_manager.get_schema(schema_id)
    if schema_def is None:
        click.echo(f"Error: Schema '{schema_id}' not found.", err=True)
        click.echo(f"Available schemas: {list(schema_manager.schemas.keys())}", err=True)
        return

    example_input, example_output = _get_example_for_schema(schema_def)

    prompt = format_prompt(
        schema=schema_def.schema,
        input_text=input_text,
        example_input=example_input,
        example_output=example_output,
    )

    raw_output = _run_inference(model, tokenizer, prompt)

    result = parser.parse(raw_output)

    if result.success:
        click.echo(json.dumps(result.parsed_output, indent=2))
        if result.repair_applied:
            click.echo(f"\n(Note: JSON repair applied — {result.repair_type})", err=True)
    else:
        click.echo("Error: Failed to extract structured data from model output.", err=True)
        click.echo(f"Raw output: {raw_output[:500]}", err=True)


def _select_schema_interactive(schema_manager: SchemaManager) -> str | None:
    """Prompt user to select a schema interactively.

    Args:
        schema_manager: The SchemaManager with loaded schemas.

    Returns:
        Selected schema ID, or None if user cancels.
    """
    schemas = list(schema_manager.schemas.keys())
    click.echo("\nAvailable schemas:")
    for i, schema_id in enumerate(schemas, 1):
        schema_def = schema_manager.get_schema(schema_id)
        desc = schema_def.description[:60] if schema_def else ""
        click.echo(f"  {i}. {schema_id} — {desc}")

    choice = click.prompt("\nSelect schema (number or ID)", default="1")

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(schemas):
            return schemas[idx]
    except ValueError:
        pass

    # Try as schema ID directly
    if choice in schemas:
        return choice

    click.echo(f"Invalid selection: {choice}", err=True)
    return None


@click.command()
@click.option("--model-path", required=True, help="Path to fine-tuned model checkpoint")
@click.option("--schema", default=None, help="Schema ID to use (e.g., 'conference_talk_simple')")
@click.option("--input", "input_text", default=None, help="Input text to extract from")
@click.option("--interactive", is_flag=True, help="Run in interactive mode")
@click.option(
    "--schemas-dir",
    default="schemas",
    help="Path to schemas directory (default: schemas/)",
    type=click.Path(),
)
def main(model_path: str, schema: str | None, input_text: str | None, interactive: bool, schemas_dir: str) -> None:
    """Structured data extraction demo using a fine-tuned model.

    Loads a fine-tuned Qwen model and extracts structured JSON from
    unstructured text input based on a selected schema.

    Examples:

        # Single extraction from command line
        python demo.py --model-path ./checkpoints/best --schema conference_talk_simple --input "Dr. Smith spoke about AI at NeurIPS in Vancouver"

        # Interactive mode
        python demo.py --model-path ./checkpoints/best --interactive

        # Read from stdin
        echo "Some text" | python demo.py --model-path ./checkpoints/best --schema conference_talk_simple
    """
    # Load schemas
    schema_manager = _load_schemas(Path(schemas_dir))

    # Load model
    click.echo("Loading model...", err=True)
    model, tokenizer = _load_model_and_tokenizer(model_path)
    click.echo("Model loaded successfully.", err=True)

    parser = OutputParser()

    # Determine schema to use
    selected_schema = schema
    if selected_schema is None and not interactive:
        # Default to first available schema for non-interactive single-shot
        available = list(schema_manager.schemas.keys())
        selected_schema = available[0] if available else None

    if input_text is not None:
        # Single extraction from --input argument
        if selected_schema is None:
            raise click.ClickException("No schema selected. Use --schema to specify one.")
        _extract_and_display(model, tokenizer, parser, schema_manager, selected_schema, input_text)

    elif interactive:
        # Interactive loop
        click.echo("\n--- Interactive Extraction Mode ---", err=True)
        click.echo("Type 'quit' or 'exit' to stop.", err=True)
        click.echo("Type 'schema' to change the active schema.\n", err=True)

        if selected_schema is None:
            selected_schema = _select_schema_interactive(schema_manager)
            if selected_schema is None:
                raise click.ClickException("No schema selected.")

        click.echo(f"Active schema: {selected_schema}\n", err=True)

        while True:
            try:
                text = click.prompt("Input", prompt_suffix="> ")
            except (EOFError, KeyboardInterrupt):
                click.echo("\nExiting.", err=True)
                break

            text = text.strip()
            if not text:
                continue
            if text.lower() in ("quit", "exit"):
                click.echo("Exiting.", err=True)
                break
            if text.lower() == "schema":
                new_schema = _select_schema_interactive(schema_manager)
                if new_schema:
                    selected_schema = new_schema
                    click.echo(f"Active schema: {selected_schema}\n", err=True)
                continue

            _extract_and_display(model, tokenizer, parser, schema_manager, selected_schema, text)
            click.echo("")  # Blank line between extractions

    else:
        # Read from stdin
        if selected_schema is None:
            raise click.ClickException("No schema selected. Use --schema to specify one.")

        if sys.stdin.isatty():
            click.echo("Reading from stdin (press Ctrl+D when done):", err=True)

        text = sys.stdin.read().strip()
        if not text:
            raise click.ClickException("No input text provided via stdin.")

        _extract_and_display(model, tokenizer, parser, schema_manager, selected_schema, text)


if __name__ == "__main__":
    main()
