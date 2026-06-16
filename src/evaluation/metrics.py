"""Metrics calculator for structured data extraction evaluation.

Provides schema validity, field-level F1, exact match, and bootstrap
confidence interval computation for the evaluation harness.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.schemas.manager import SchemaManager

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Computes evaluation metrics for structured data extraction models.

    Metrics include schema validity rate, field-level F1, exact match,
    and bootstrap confidence intervals.
    """

    def __init__(self, schema_manager: SchemaManager) -> None:
        """Initialize MetricsCalculator with a SchemaManager for validation.

        Args:
            schema_manager: A SchemaManager instance with loaded schemas.
        """
        self.schema_manager = schema_manager

    def schema_validity(self, outputs: list[dict], schema_id: str) -> float:
        """Compute the fraction of outputs passing schema validation.

        Args:
            outputs: List of parsed output dictionaries to validate.
            schema_id: The schema identifier to validate against.

        Returns:
            Fraction of outputs that pass schema validation, in [0.0, 1.0].
            Returns 0.0 if outputs is empty.
        """
        if not outputs:
            return 0.0

        valid_count = 0
        for output in outputs:
            result = self.schema_manager.validate_instance(output, schema_id)
            if result.success:
                valid_count += 1

        return valid_count / len(outputs)

    def field_f1(self, predicted: dict, expected: dict) -> dict[str, float]:
        """Compute per-field precision, recall, and F1 score.

        Works on both flat and nested dicts by flattening nested keys
        with dot notation for comparison. A field is "correct" if both
        the key exists and the value matches after type normalization
        (str(v).strip().lower()).

        Args:
            predicted: The predicted output dictionary.
            expected: The expected/ground-truth output dictionary.

        Returns:
            Dictionary with keys "precision", "recall", "f1", each in [0.0, 1.0].
        """
        predicted_flat = self._flatten_dict(predicted)
        expected_flat = self._flatten_dict(expected)

        predicted_fields = set(predicted_flat.keys())
        expected_fields = set(expected_flat.keys())

        if not predicted_fields and not expected_fields:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

        # Count correct fields: key exists in both and normalized value matches
        correct = 0
        for key in predicted_fields & expected_fields:
            if self._normalize_value(predicted_flat[key]) == self._normalize_value(expected_flat[key]):
                correct += 1

        precision = correct / len(predicted_fields) if predicted_fields else 0.0
        recall = correct / len(expected_fields) if expected_fields else 0.0

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        return {"precision": precision, "recall": recall, "f1": f1}

    def exact_match(self, predicted: dict, expected: dict) -> bool:
        """Check strict equality after normalization.

        Normalizes both dicts by sorting keys, lowercasing strings,
        and stripping whitespace before comparison.

        Args:
            predicted: The predicted output dictionary.
            expected: The expected/ground-truth output dictionary.

        Returns:
            True if normalized dicts are equal, False otherwise.
        """
        normalized_predicted = self._normalize_dict(predicted)
        normalized_expected = self._normalize_dict(expected)
        return normalized_predicted == normalized_expected

    def compute_bootstrap_ci(
        self,
        metric_values: list[float],
        n_iterations: int = 1000,
        confidence: float = 0.95,
        seed: int = 42,
    ) -> tuple[float, float]:
        """Compute bootstrap confidence intervals for a metric.

        Uses bootstrap resampling to estimate confidence intervals
        for the mean of the provided metric values.

        Args:
            metric_values: List of metric values (e.g., per-example F1 scores).
            n_iterations: Number of bootstrap iterations.
            confidence: Confidence level (default 0.95 for 95% CI).
            seed: Random seed for reproducibility.

        Returns:
            Tuple of (lower_bound, upper_bound) at the specified confidence level.
            Both bounds are in [0.0, 1.0].
        """
        if not metric_values:
            return (0.0, 0.0)

        values = np.array(metric_values, dtype=np.float64)
        rng = np.random.default_rng(seed)

        bootstrap_means = np.empty(n_iterations, dtype=np.float64)
        n = len(values)

        for i in range(n_iterations):
            sample = rng.choice(values, size=n, replace=True)
            bootstrap_means[i] = np.mean(sample)

        alpha = 1.0 - confidence
        lower = float(np.percentile(bootstrap_means, 100 * alpha / 2))
        upper = float(np.percentile(bootstrap_means, 100 * (1 - alpha / 2)))

        # Clamp to [0.0, 1.0]
        lower = max(0.0, min(1.0, lower))
        upper = max(0.0, min(1.0, upper))

        # Ensure lower <= upper (should always be true, but defensive)
        if lower > upper:
            lower, upper = upper, lower

        return (lower, upper)

    def _flatten_dict(self, d: dict, parent_key: str = "", sep: str = ".") -> dict[str, Any]:
        """Flatten a nested dictionary using dot notation for keys.

        Args:
            d: The dictionary to flatten.
            parent_key: Prefix for nested keys.
            sep: Separator between key levels.

        Returns:
            Flat dictionary with dot-notation keys.
        """
        items: list[tuple[str, Any]] = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def _normalize_value(self, value: Any) -> str:
        """Normalize a value for comparison.

        Converts to string, strips whitespace, and lowercases.

        Args:
            value: Any value to normalize.

        Returns:
            Normalized string representation.
        """
        return str(value).strip().lower()

    def _normalize_dict(self, d: dict) -> dict:
        """Normalize a dictionary for exact match comparison.

        Recursively sorts keys, lowercases string values, and strips whitespace.

        Args:
            d: The dictionary to normalize.

        Returns:
            Normalized dictionary with sorted keys and normalized values.
        """
        normalized = {}
        for key in sorted(d.keys()):
            value = d[key]
            if isinstance(value, dict):
                normalized[key] = self._normalize_dict(value)
            elif isinstance(value, list):
                normalized[key] = self._normalize_list(value)
            elif isinstance(value, str):
                normalized[key] = value.strip().lower()
            else:
                normalized[key] = value
        return normalized

    def _normalize_list(self, lst: list) -> list:
        """Normalize a list for exact match comparison.

        Args:
            lst: The list to normalize.

        Returns:
            Normalized list with normalized elements.
        """
        normalized = []
        for item in lst:
            if isinstance(item, dict):
                normalized.append(self._normalize_dict(item))
            elif isinstance(item, str):
                normalized.append(item.strip().lower())
            else:
                normalized.append(item)
        return normalized
