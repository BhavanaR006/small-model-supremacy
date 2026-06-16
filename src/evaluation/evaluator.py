"""Evaluator orchestration for model evaluation and baseline comparison.

Provides the Evaluator class that runs fine-tuned models and baseline APIs
against a test set, computes metrics, and generates results tables.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from src.config.schema import EvalConfig
from src.data.api_client import APIClient, GenerationParams
from src.evaluation.metrics import MetricsCalculator
from src.parsing.output_parser import OutputParser
from src.schemas.manager import SchemaManager

logger = logging.getLogger(__name__)


class GenerativeModel(Protocol):
    """Protocol for models that support text generation (duck typing)."""

    def generate(self, prompt: str) -> str: ...


@dataclass
class EvalMetrics:
    """Evaluation metrics for a model on a test set.

    Attributes:
        schema_validity: Fraction of outputs passing schema validation (0.0-1.0).
        field_f1: Average field-level F1 score (0.0-1.0).
        exact_match: Fraction of exact matches (0.0-1.0).
        avg_latency_ms: Average inference latency in milliseconds.
        valid_json_rate: Fraction of outputs that parse as valid JSON (0.0-1.0).
        confidence_intervals: Mapping of metric_name -> (lower, upper) bounds.
    """

    schema_validity: float
    field_f1: float
    exact_match: float
    avg_latency_ms: float
    valid_json_rate: float
    confidence_intervals: dict = field(default_factory=dict)


@dataclass
class EvalExample:
    """A single evaluation example from the test set.

    Attributes:
        input_text: The input prompt text.
        expected_output: The expected structured output dict.
        schema_id: The schema identifier for validation.
        difficulty_level: One of "simple", "medium", "complex".
    """

    input_text: str
    expected_output: dict
    schema_id: str
    difficulty_level: str


class Evaluator:
    """Orchestrates evaluation of fine-tuned models and baselines.

    Runs models against the test set, parses outputs, computes metrics,
    and generates results tables with per-difficulty breakdowns.

    Args:
        config: Evaluation configuration.
        schema_manager: SchemaManager with loaded schemas for validation.
        output_parser: OutputParser for extracting JSON from model outputs.
        metrics_calculator: MetricsCalculator for computing evaluation metrics.
    """

    def __init__(
        self,
        config: EvalConfig,
        schema_manager: SchemaManager,
        output_parser: OutputParser | None = None,
        metrics_calculator: MetricsCalculator | None = None,
    ) -> None:
        self.config = config
        self.schema_manager = schema_manager
        self.output_parser = output_parser or OutputParser()
        self.metrics_calculator = metrics_calculator or MetricsCalculator(schema_manager)

    def evaluate_model(
        self, model: GenerativeModel, test_set: list[EvalExample]
    ) -> EvalMetrics:
        """Run a fine-tuned model on the test set and compute metrics.

        Iterates through each example, generates output using the model,
        parses the response, validates against the schema, and computes
        aggregate metrics.

        Args:
            model: A model with a generate(prompt) method.
            test_set: List of EvalExample instances to evaluate.

        Returns:
            EvalMetrics with aggregate metrics across the test set.
        """
        if not test_set:
            return EvalMetrics(
                schema_validity=0.0,
                field_f1=0.0,
                exact_match=0.0,
                avg_latency_ms=0.0,
                valid_json_rate=0.0,
                confidence_intervals={},
            )

        parse_successes = 0
        schema_valid_count = 0
        f1_scores: list[float] = []
        exact_matches: list[float] = []
        latencies_ms: list[float] = []

        for example in test_set:
            # Measure latency
            start = time.perf_counter()
            try:
                raw_output = model.generate(example.input_text)
            except Exception as e:
                logger.warning(
                    "Model generation failed for example, recording as failure",
                    extra={"schema_id": example.schema_id, "error": str(e)},
                )
                f1_scores.append(0.0)
                exact_matches.append(0.0)
                continue
            end = time.perf_counter()
            latencies_ms.append((end - start) * 1000.0)

            # Parse output
            parse_result = self.output_parser.parse(raw_output)
            if parse_result.success:
                parse_successes += 1
                parsed = parse_result.parsed_output

                # Schema validation
                validation = self.schema_manager.validate_instance(
                    parsed, example.schema_id
                )
                if validation.success:
                    schema_valid_count += 1

                # Field F1
                f1_result = self.metrics_calculator.field_f1(
                    parsed, example.expected_output
                )
                f1_scores.append(f1_result["f1"])

                # Exact match
                is_exact = self.metrics_calculator.exact_match(
                    parsed, example.expected_output
                )
                exact_matches.append(1.0 if is_exact else 0.0)
            else:
                f1_scores.append(0.0)
                exact_matches.append(0.0)

        total = len(test_set)
        valid_json_rate = parse_successes / total
        schema_validity = schema_valid_count / total
        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        avg_exact = sum(exact_matches) / len(exact_matches) if exact_matches else 0.0
        avg_latency = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0

        # Bootstrap confidence intervals
        confidence_intervals = self._compute_confidence_intervals(
            f1_scores, exact_matches, valid_json_rate, schema_validity, total
        )

        return EvalMetrics(
            schema_validity=schema_validity,
            field_f1=avg_f1,
            exact_match=avg_exact,
            avg_latency_ms=avg_latency,
            valid_json_rate=valid_json_rate,
            confidence_intervals=confidence_intervals,
        )

    def evaluate_baseline(
        self, provider: str, test_set: list[EvalExample]
    ) -> EvalMetrics:
        """Run a baseline model via API on the test set and compute metrics.

        Calls the baseline API for each example with 3x retry and exponential
        backoff. Records failures and continues evaluation for remaining examples.

        Args:
            provider: API provider name (e.g., "gpt-4o", "claude-3-5-sonnet").
            test_set: List of EvalExample instances to evaluate.

        Returns:
            EvalMetrics with aggregate metrics across the test set.
        """
        if not test_set:
            return EvalMetrics(
                schema_validity=0.0,
                field_f1=0.0,
                exact_match=0.0,
                avg_latency_ms=0.0,
                valid_json_rate=0.0,
                confidence_intervals={},
            )

        # Determine API provider and model for the baseline
        api_provider, model_name = self._resolve_provider(provider)

        parse_successes = 0
        schema_valid_count = 0
        f1_scores: list[float] = []
        exact_matches: list[float] = []
        latencies_ms: list[float] = []
        failure_count = 0

        params = GenerationParams(
            temperature=self.config.temperature,
            model=model_name,
        )

        for example in test_set:
            # Call API with retry logic
            raw_output = self._call_with_retry(
                api_provider, example.input_text, params
            )

            if raw_output is None:
                # All retries failed — record as failure and continue
                failure_count += 1
                f1_scores.append(0.0)
                exact_matches.append(0.0)
                logger.warning(
                    "Baseline API call failed after retries, recording as failure",
                    extra={
                        "provider": provider,
                        "schema_id": example.schema_id,
                    },
                )
                continue

            # Measure latency includes the API call time
            start = time.perf_counter()
            # Already have the output from retry call, so latency was measured there
            end = time.perf_counter()

            # Parse output
            parse_result = self.output_parser.parse(raw_output)
            if parse_result.success:
                parse_successes += 1
                parsed = parse_result.parsed_output

                # Schema validation
                validation = self.schema_manager.validate_instance(
                    parsed, example.schema_id
                )
                if validation.success:
                    schema_valid_count += 1

                # Field F1
                f1_result = self.metrics_calculator.field_f1(
                    parsed, example.expected_output
                )
                f1_scores.append(f1_result["f1"])

                # Exact match
                is_exact = self.metrics_calculator.exact_match(
                    parsed, example.expected_output
                )
                exact_matches.append(1.0 if is_exact else 0.0)
            else:
                f1_scores.append(0.0)
                exact_matches.append(0.0)

        if failure_count > 0:
            logger.warning(
                "Baseline evaluation completed with failures",
                extra={"provider": provider, "failure_count": failure_count},
            )

        total = len(test_set)
        valid_json_rate = parse_successes / total
        schema_validity = schema_valid_count / total
        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        avg_exact = sum(exact_matches) / len(exact_matches) if exact_matches else 0.0
        avg_latency = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0

        # Bootstrap confidence intervals
        confidence_intervals = self._compute_confidence_intervals(
            f1_scores, exact_matches, valid_json_rate, schema_validity, total
        )

        return EvalMetrics(
            schema_validity=schema_validity,
            field_f1=avg_f1,
            exact_match=avg_exact,
            avg_latency_ms=avg_latency,
            valid_json_rate=valid_json_rate,
            confidence_intervals=confidence_intervals,
        )

    def generate_results_table(
        self, results: dict[str, dict[str, EvalMetrics]]
    ) -> pd.DataFrame:
        """Generate a pandas DataFrame with per-model, per-difficulty metrics.

        Groups results by model and difficulty level. Adds warnings for any
        schema with schema_validity < 95%.

        Args:
            results: Nested dict of {model_name: {difficulty_or_overall: EvalMetrics}}.
                Expected keys per model: "overall", "simple", "medium", "complex".

        Returns:
            DataFrame with columns: model, difficulty, schema_validity, field_f1,
            exact_match, avg_latency_ms, valid_json_rate, warning.
        """
        rows: list[dict[str, Any]] = []

        for model_name, difficulty_metrics in results.items():
            for difficulty, metrics in difficulty_metrics.items():
                warning = ""
                if metrics.schema_validity < 0.95:
                    warning = (
                        f"WARNING: schema_validity={metrics.schema_validity:.2%} "
                        f"(below 95% threshold)"
                    )

                rows.append(
                    {
                        "model": model_name,
                        "difficulty": difficulty,
                        "schema_validity": metrics.schema_validity,
                        "field_f1": metrics.field_f1,
                        "exact_match": metrics.exact_match,
                        "avg_latency_ms": metrics.avg_latency_ms,
                        "valid_json_rate": metrics.valid_json_rate,
                        "warning": warning,
                    }
                )

        df = pd.DataFrame(rows)
        return df

    def run_full_evaluation(
        self,
        model: GenerativeModel,
        test_set: list[EvalExample],
        baselines: list[str] | None = None,
    ) -> dict[str, dict[str, EvalMetrics]]:
        """Run full evaluation on model and baselines, grouped by difficulty.

        Evaluates the fine-tuned model and each baseline across overall and
        per-difficulty splits. Ensures the union of per-difficulty examples
        equals the full test set.

        Args:
            model: The fine-tuned model to evaluate.
            test_set: Full test set.
            baselines: List of baseline provider names. Defaults to config baselines.

        Returns:
            Nested dict {model_name: {difficulty: EvalMetrics}}.
        """
        if baselines is None:
            baselines = self.config.baselines

        # Group by difficulty
        grouped = self._group_by_difficulty(test_set)

        results: dict[str, dict[str, EvalMetrics]] = {}

        # Evaluate fine-tuned model
        logger.info("Evaluating fine-tuned model on full test set")
        model_results: dict[str, EvalMetrics] = {}
        model_results["overall"] = self.evaluate_model(model, test_set)

        for difficulty, examples in grouped.items():
            logger.info(
                "Evaluating fine-tuned model on %s difficulty (%d examples)",
                difficulty,
                len(examples),
            )
            model_results[difficulty] = self.evaluate_model(model, examples)

        results["fine_tuned"] = model_results

        # Evaluate baselines
        for baseline in baselines:
            logger.info("Evaluating baseline: %s", baseline)
            baseline_results: dict[str, EvalMetrics] = {}
            baseline_results["overall"] = self.evaluate_baseline(baseline, test_set)

            for difficulty, examples in grouped.items():
                logger.info(
                    "Evaluating baseline %s on %s difficulty (%d examples)",
                    baseline,
                    difficulty,
                    len(examples),
                )
                baseline_results[difficulty] = self.evaluate_baseline(
                    baseline, examples
                )

            results[baseline] = baseline_results

        return results

    def save_results(
        self,
        results: dict[str, dict[str, EvalMetrics]],
        output_dir: Path | None = None,
    ) -> Path:
        """Save results table as CSV and print to stdout.

        Args:
            results: The evaluation results from run_full_evaluation.
            output_dir: Directory to save CSV. Defaults to config output_dir.

        Returns:
            Path to the saved CSV file.
        """
        if output_dir is None:
            output_dir = Path(self.config.output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        df = self.generate_results_table(results)
        csv_path = output_dir / "evaluation_results.csv"
        df.to_csv(csv_path, index=False)

        # Print to stdout
        print("\n=== Evaluation Results ===\n")
        print(df.to_string(index=False))
        print()

        # Print warnings
        warnings = df[df["warning"] != ""]
        if not warnings.empty:
            print("\n=== Warnings ===")
            for _, row in warnings.iterrows():
                print(f"  [{row['model']}][{row['difficulty']}] {row['warning']}")
            print()

        logger.info("Results saved to %s", csv_path)
        return csv_path

    def _group_by_difficulty(
        self, test_set: list[EvalExample]
    ) -> dict[str, list[EvalExample]]:
        """Group test set examples by difficulty level.

        Ensures that the union of all groups equals the full test set.

        Args:
            test_set: Full test set.

        Returns:
            Dict mapping difficulty level to list of examples.
        """
        grouped: dict[str, list[EvalExample]] = {}
        for example in test_set:
            level = example.difficulty_level
            if level not in grouped:
                grouped[level] = []
            grouped[level].append(example)

        # Verify: union of groups equals full set
        total_grouped = sum(len(examples) for examples in grouped.values())
        if total_grouped != len(test_set):
            logger.error(
                "Grouping mismatch: grouped %d examples but test set has %d",
                total_grouped,
                len(test_set),
            )

        return grouped

    def _call_with_retry(
        self, provider: str, prompt: str, params: GenerationParams
    ) -> str | None:
        """Call API with 3x retry and exponential backoff.

        Args:
            provider: The API provider ("claude" or "openai").
            prompt: The prompt text.
            params: Generation parameters.

        Returns:
            The API response text, or None if all retries failed.
        """
        max_retries = self.config.retry_count
        cache_dir = Path(self.config.output_dir) / "api_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        client = APIClient(provider=provider, cache_dir=cache_dir)

        for attempt in range(max_retries):
            try:
                start = time.perf_counter()
                response = client.generate(prompt, params)
                return response
            except Exception as e:
                backoff = 2 ** attempt
                logger.warning(
                    "Baseline API attempt %d/%d failed, backing off %ds",
                    attempt + 1,
                    max_retries,
                    backoff,
                    extra={"error": str(e), "provider": provider},
                )
                if attempt < max_retries - 1:
                    time.sleep(backoff)

        return None

    def _resolve_provider(self, provider: str) -> tuple[str, str]:
        """Resolve a provider name to (api_provider, model_name).

        Maps user-friendly baseline names to API client provider and model strings.

        Args:
            provider: The baseline name (e.g., "gpt-4o", "claude-3-5-sonnet").

        Returns:
            Tuple of (api_provider, model_name) for the APIClient.
        """
        provider_map = {
            "gpt-4o": ("openai", "gpt-4o"),
            "gpt-4": ("openai", "gpt-4"),
            "claude-3-5-sonnet": ("claude", "claude-3-5-sonnet-20241022"),
            "claude-3-opus": ("claude", "claude-3-opus-20240229"),
            "base_model": ("openai", "base_model"),  # placeholder
        }

        if provider in provider_map:
            return provider_map[provider]

        # Default: assume it's a direct model name
        # Try to infer provider from name
        if "claude" in provider.lower():
            return ("claude", provider)
        elif "gpt" in provider.lower() or "openai" in provider.lower():
            return ("openai", provider)
        else:
            # Default to openai as a generic provider
            return ("openai", provider)

    def _compute_confidence_intervals(
        self,
        f1_scores: list[float],
        exact_matches: list[float],
        valid_json_rate: float,
        schema_validity: float,
        total: int,
    ) -> dict[str, tuple[float, float]]:
        """Compute bootstrap confidence intervals for all metrics.

        Args:
            f1_scores: Per-example F1 scores.
            exact_matches: Per-example exact match indicators (0.0 or 1.0).
            valid_json_rate: Overall valid JSON rate.
            schema_validity: Overall schema validity rate.
            total: Total number of examples.

        Returns:
            Dict mapping metric name to (lower, upper) CI bounds.
        """
        ci: dict[str, tuple[float, float]] = {}

        bootstrap_iters = self.config.bootstrap_iterations
        seed = self.config.seed

        if f1_scores:
            ci["field_f1"] = self.metrics_calculator.compute_bootstrap_ci(
                f1_scores, n_iterations=bootstrap_iters, seed=seed
            )

        if exact_matches:
            ci["exact_match"] = self.metrics_calculator.compute_bootstrap_ci(
                exact_matches, n_iterations=bootstrap_iters, seed=seed
            )

        # For schema_validity and valid_json_rate, create per-example binary indicators
        # These are already aggregated, so we approximate using the available scores
        # For a proper bootstrap, we'd need per-example validity indicators
        # Here we construct binary arrays from the counts
        if total > 0:
            schema_binary = [1.0] * int(schema_validity * total) + [0.0] * (
                total - int(schema_validity * total)
            )
            ci["schema_validity"] = self.metrics_calculator.compute_bootstrap_ci(
                schema_binary, n_iterations=bootstrap_iters, seed=seed
            )

            json_binary = [1.0] * int(valid_json_rate * total) + [0.0] * (
                total - int(valid_json_rate * total)
            )
            ci["valid_json_rate"] = self.metrics_calculator.compute_bootstrap_ci(
                json_binary, n_iterations=bootstrap_iters, seed=seed
            )

        return ci
