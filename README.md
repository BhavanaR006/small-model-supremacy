# Small Model Supremacy

> Beating frontier models with a 3B parameter model on structured data extraction.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Weights & Biases](https://img.shields.io/badge/Weights_%26_Biases-FFBE00?logo=weightsandbiases&logoColor=black)](https://wandb.ai)

---

## Overview

**Small Model Supremacy** demonstrates that a carefully fine-tuned 3B parameter model can outperform GPT-4o and Claude 3.5 Sonnet on structured data extraction from unstructured text.

The core insight: frontier models frequently hallucinate fields or produce malformed JSON on complex schemas. A small model trained on high-quality extraction pairs achieves near-perfect schema compliance — faster, cheaper, and fully under your control.

This project provides a complete, reproducible ML pipeline:

- 🎯 **Synthetic data generation** using frontier model APIs with validation and caching
- ⚡ **QLoRA fine-tuning** that fits on a single RTX 4090 (24GB VRAM)
- 📊 **Rigorous evaluation** with bootstrap confidence intervals and statistical significance testing
- 🔧 **Robust output parsing** with JSON repair heuristics for production reliability

---

## Key Results

| Model | Schema Validity | Field F1 | Exact Match | Avg Latency |
|-------|:-:|:-:|:-:|:-:|
| **Qwen2.5-3B (ours)** | **98.2%** | **0.96** | **89.4%** | **45ms** |
| GPT-4o | 91.7% | 0.91 | 78.2% | 820ms |
| Claude 3.5 Sonnet | 93.1% | 0.92 | 81.5% | 950ms |
| Qwen2.5-3B (base) | 34.8% | 0.42 | 12.1% | 42ms |

*Results on held-out test set (500+ examples across 3 schemas). 95% confidence intervals computed via bootstrap resampling (n=1000). Full results in `results/`.*

---

## Quick Start

### Prerequisites

- Python 3.11+
- CUDA-capable GPU with 24GB+ VRAM (RTX 4090, A10G, or equivalent)
- API keys for data generation (Anthropic or OpenAI)

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/small-model-supremacy.git
cd small-model-supremacy

# Create environment and install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Configure API keys
export ANTHROPIC_API_KEY="your-key-here"
# or
export OPENAI_API_KEY="your-key-here"
```

### Generate Training Data

```bash
python generate_data.py --config config.yaml
```

Generates 5,000+ validated training pairs across all schemas using frontier model APIs. Responses are cached for reproducibility.

### Train

```bash
python train.py --config config.yaml
```

Fine-tunes Qwen2.5-3B with QLoRA. Training logs stream to Weights & Biases. Checkpoints saved to `checkpoints/`.

### Evaluate

```bash
python evaluate.py --config config.yaml
```

Runs the evaluation harness against all baselines. Produces comparison tables and charts in `results/`.

### Interactive Demo

```bash
python demo.py --text "Dr. Jane Smith presented her research on CRISPR gene editing at NeurIPS 2024..."
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Data Pipeline                             │
│  Schema Definitions → Dataset Curator → Validated JSONL Dataset  │
│        ↑                    ↑                                    │
│   schemas/*.json    Frontier Model APIs (cached)                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Training Pipeline                           │
│  Dataset → Tokenizer → QLoRA Trainer → Checkpoints → Model      │
│                              ↑                                   │
│                     config.yaml + W&B logging                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Evaluation Harness                           │
│  Model + Baselines → Output Parser → Schema Validator → Metrics │
│                                                    → Charts/CSV  │
└─────────────────────────────────────────────────────────────────┘
```

**Training approach:** QLoRA with 4-bit NF4 quantization, LoRA rank 64, alpha 128. Effective batch size of 16 via gradient accumulation. Early stopping monitors validation loss with patience of 3 intervals.

---

## Project Structure

```
small-model-supremacy/
├── config.yaml              # All hyperparameters and paths
├── pyproject.toml           # Dependencies (pinned versions)
├── Dockerfile               # Reproducible training environment
├── train.py                 # Training entry point
├── evaluate.py              # Evaluation entry point
├── generate_data.py         # Data generation entry point
├── demo.py                  # Interactive inference demo
├── schemas/                 # JSON Schema definitions
│   ├── conference_talk_simple.schema.json
│   ├── product_listing_medium.schema.json
│   └── scientific_paper_complex.schema.json
├── src/
│   ├── config/              # Configuration loading and validation
│   │   └── schema.py
│   ├── schemas/             # Schema management and validation
│   │   └── manager.py
│   ├── data/                # Data generation pipeline
│   │   ├── api_client.py    # Frontier model API with caching
│   │   ├── curator.py       # Dataset curation and validation
│   │   ├── prompt_template.py
│   │   └── tokenizer_utils.py
│   ├── training/            # Fine-tuning pipeline
│   │   ├── trainer.py       # QLoRA training loop
│   │   ├── callbacks.py     # Early stopping, checkpointing
│   │   └── memory.py        # GPU memory management
│   ├── evaluation/          # Benchmarking harness
│   │   ├── evaluator.py     # Model evaluation orchestration
│   │   ├── metrics.py       # F1, exact match, bootstrap CIs
│   │   └── visualization.py # Chart generation
│   ├── parsing/             # Output extraction and repair
│   │   └── output_parser.py
│   └── utils/
│       └── logging.py       # Structured logging setup
├── tests/                   # Test suite (pytest + hypothesis)
│   ├── properties/          # Property-based tests
│   ├── test_api_client.py
│   ├── test_callbacks.py
│   └── ...
├── data/                    # Generated datasets (JSONL)
├── cache/                   # API response cache
├── checkpoints/             # Model checkpoints (top-3 by val_loss)
├── results/                 # Evaluation outputs (CSV + PNG charts)
└── notebooks/               # Analysis notebooks
```

---

## Configuration

All settings live in `config.yaml`. No hardcoded values in source code.

```yaml
model:
  name: "Qwen/Qwen2.5-3B"       # Base model from HuggingFace
  max_seq_length: 2048            # Maximum input sequence length

training:
  lora_rank: 64                   # LoRA rank (higher = more capacity)
  lora_alpha: 128                 # LoRA scaling factor
  learning_rate: 2.0e-4           # Peak learning rate
  batch_size: 4                   # Per-device batch size
  gradient_accumulation_steps: 4  # Effective batch = 16
  num_epochs: 3                   # Maximum training epochs
  max_memory_gb: 24.0             # GPU memory budget

data:
  seed: 42                        # Reproducibility seed
  adversarial_ratio: 0.15         # 15% adversarial examples
  min_tokens: 50                  # Minimum input length
  max_tokens: 2000                # Maximum input length

evaluation:
  baselines: ["gpt-4o", "claude-3-5-sonnet", "base_model"]
  bootstrap_iterations: 1000      # For confidence intervals
  temperature: 0.0                # Deterministic inference
```

See `config.yaml` for the complete configuration reference.

---

## CLI Commands

### `generate_data.py` — Dataset Generation

```bash
# Generate full dataset
python generate_data.py --config config.yaml

# Generate for a specific schema
python generate_data.py --config config.yaml --schema conference_talk_simple

# Dry run (validate schemas only)
python generate_data.py --config config.yaml --dry-run
```

### `train.py` — Model Training

```bash
# Start training
python train.py --config config.yaml

# Resume from checkpoint
python train.py --config config.yaml --resume checkpoints/checkpoint-1500

# Train with a different base model
python train.py --config config.yaml --model Qwen/Qwen2.5-1.5B
```

### `evaluate.py` — Evaluation Harness

```bash
# Full evaluation against all baselines
python evaluate.py --config config.yaml

# Evaluate specific model checkpoint
python evaluate.py --config config.yaml --checkpoint checkpoints/best

# Skip API baselines (offline mode)
python evaluate.py --config config.yaml --local-only
```

### `demo.py` — Interactive Demo

```bash
# Single extraction
python demo.py --text "Your unstructured text here..."

# Specify schema
python demo.py --schema product_listing_medium --text "..."

# Read from stdin
echo "Some text to extract from" | python demo.py --schema conference_talk_simple
```

---

## Evaluation Methodology

The evaluation harness ensures fair, rigorous comparison:

1. **Identical prompts** — All models receive the same prompt template including the full schema definition, task description, and one-shot example
2. **Deterministic inference** — Temperature 0, fixed seed across all runs
3. **Statistical rigor** — 95% confidence intervals via bootstrap resampling (1,000 iterations)
4. **Multi-dimensional metrics:**
   - **Schema Validity** — Does the output conform to the JSON Schema?
   - **Field F1** — Precision/recall at the individual field level
   - **Exact Match** — Percentage of perfectly correct extractions
   - **Latency** — End-to-end inference time (local GPU time vs. API round-trip)
5. **Stratified reporting** — Results broken down by schema complexity (simple/medium/complex)
6. **Robust parsing** — JSON repair heuristics applied uniformly to all models before scoring

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Base Model | Qwen2.5-1.5B / Qwen2.5-3B |
| Fine-tuning | QLoRA via PEFT + bitsandbytes |
| Training Framework | HuggingFace Transformers 4.46 |
| Experiment Tracking | Weights & Biases |
| Data Validation | jsonschema (draft 2020-12) |
| Configuration | YAML + Pydantic v2 |
| Testing | pytest + Hypothesis (property-based) |
| Visualization | Matplotlib + Seaborn |
| Containerization | Docker (CUDA 12.1 base) |
| Language | Python 3.11+ |

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
