"""Visualization module for evaluation results.

Generates comparison charts for model metrics and latency,
saved as PNG files suitable for GitHub README rendering.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import matplotlib
matplotlib.use("Agg")  # Headless rendering — no display needed

import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

# Chart configuration
_DPI = 150
_FIGSIZE_METRICS = (10, 6)
_FIGSIZE_LATENCY = (8, 5)


class EvalMetricsLike(Protocol):
    """Protocol for objects with evaluation metric attributes."""

    schema_validity: float
    field_f1: float
    exact_match: float
    avg_latency_ms: float


def generate_charts(results: dict[str, Any], output_dir: Path) -> None:
    """Generate comparison charts from evaluation results.

    Produces two charts:
    - metrics_comparison.png: grouped bar chart comparing schema_validity,
      field_f1, and exact_match across models.
    - latency_comparison.png: bar chart comparing avg_latency_ms across models.

    Args:
        results: Dictionary mapping model names to objects with EvalMetrics-like
                 attributes (schema_validity, field_f1, exact_match, avg_latency_ms).
        output_dir: Directory where PNG files will be saved. Created if it does not exist.
    """
    if not results:
        logger.warning("No results provided for chart generation — skipping.")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_style("whitegrid")
    sns.set_palette("muted")

    _generate_metrics_comparison(results, output_dir)
    _generate_latency_comparison(results, output_dir)

    logger.info("Charts saved to %s", output_dir)


def _generate_metrics_comparison(results: dict[str, Any], output_dir: Path) -> None:
    """Generate grouped bar chart comparing models across quality metrics.

    Args:
        results: Dictionary mapping model names to EvalMetrics-like objects.
        output_dir: Directory to save the chart.
    """
    model_names = list(results.keys())
    metrics = ["schema_validity", "field_f1", "exact_match"]
    metric_labels = ["Schema Validity", "Field F1", "Exact Match"]

    # Extract metric values per model
    data: dict[str, list[float]] = {label: [] for label in metric_labels}
    for model_name in model_names:
        metrics_obj = results[model_name]
        data["Schema Validity"].append(_get_metric(metrics_obj, "schema_validity"))
        data["Field F1"].append(_get_metric(metrics_obj, "field_f1"))
        data["Exact Match"].append(_get_metric(metrics_obj, "exact_match"))

    fig, ax = plt.subplots(figsize=_FIGSIZE_METRICS)

    x = range(len(model_names))
    n_metrics = len(metric_labels)
    bar_width = 0.25
    offsets = [i * bar_width for i in range(n_metrics)]

    colors = sns.color_palette("muted", n_colors=n_metrics)

    for i, (label, color) in enumerate(zip(metric_labels, colors)):
        positions = [pos + offsets[i] for pos in x]
        ax.bar(positions, data[label], bar_width, label=label, color=color, edgecolor="white", linewidth=0.5)

    # Center tick labels under grouped bars
    center_offset = bar_width * (n_metrics - 1) / 2
    ax.set_xticks([pos + center_offset for pos in x])
    ax.set_xticklabels(model_names, rotation=0, ha="center")

    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Model Metrics Comparison")
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="upper right")

    plt.tight_layout()
    filepath = output_dir / "metrics_comparison.png"
    fig.savefig(filepath, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved metrics comparison chart: %s", filepath)


def _generate_latency_comparison(results: dict[str, Any], output_dir: Path) -> None:
    """Generate bar chart comparing average latency across models.

    Args:
        results: Dictionary mapping model names to EvalMetrics-like objects.
        output_dir: Directory to save the chart.
    """
    model_names = list(results.keys())
    latencies = [_get_metric(results[name], "avg_latency_ms") for name in model_names]

    fig, ax = plt.subplots(figsize=_FIGSIZE_LATENCY)

    colors = sns.color_palette("muted", n_colors=len(model_names))
    bars = ax.bar(model_names, latencies, color=colors, edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Model")
    ax.set_ylabel("Average Latency (ms)")
    ax.set_title("Inference Latency Comparison")

    # Add value labels on top of bars
    for bar, latency in zip(bars, latencies):
        ax.annotate(
            f"{latency:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Add a bit of headroom for labels
    if latencies:
        ax.set_ylim(0, max(latencies) * 1.15)

    plt.tight_layout()
    filepath = output_dir / "latency_comparison.png"
    fig.savefig(filepath, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved latency comparison chart: %s", filepath)


def _get_metric(obj: Any, attr: str) -> float:
    """Safely extract a metric value from an object or dict.

    Supports both attribute access (dataclass/object) and dict-style access.

    Args:
        obj: The metrics object or dictionary.
        attr: The attribute/key name to extract.

    Returns:
        The metric value as a float, or 0.0 if not found.
    """
    if isinstance(obj, dict):
        return float(obj.get(attr, 0.0))
    return float(getattr(obj, attr, 0.0))
