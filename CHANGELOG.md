# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-06-16

### Added

- Project scaffolding with complete directory structure
- Configuration system with YAML loading and Pydantic validation (`src/config/schema.py`)
- Schema management for JSON Schema definitions (`src/schemas/manager.py`)
- Dataset curation pipeline with frontier model API integration (`src/data/curator.py`)
- API client with response caching and exponential backoff (`src/data/api_client.py`)
- Prompt template system for consistent model inputs (`src/data/prompt_template.py`)
- Tokenizer utilities for input length validation (`src/data/tokenizer_utils.py`)
- QLoRA training pipeline with 4-bit NF4 quantization (`src/training/trainer.py`)
- Training callbacks: early stopping, checkpoint management (`src/training/callbacks.py`)
- GPU memory monitoring and enforcement (`src/training/memory.py`)
- Output parser with JSON repair heuristics (`src/parsing/output_parser.py`)
- Evaluation harness with multi-model benchmarking (`src/evaluation/evaluator.py`)
- Metrics computation with bootstrap confidence intervals (`src/evaluation/metrics.py`)
- Visualization module for comparison charts (`src/evaluation/visualization.py`)
- Structured logging setup (`src/utils/logging.py`)
- CLI entry points: `train.py`, `evaluate.py`, `generate_data.py`, `demo.py`
- Three extraction schemas: conference talk (simple), product listing (medium), scientific paper (complex)
- Dockerfile for reproducible training environment (CUDA 12.1)
- `pyproject.toml` with pinned dependencies
- Property-based test infrastructure with Hypothesis
- Unit tests for core components

### Technical Decisions

- **Base model:** Qwen2.5-1.5B selected for strong instruction-following at small scale
- **Fine-tuning:** QLoRA (rank 64, alpha 128) to fit within 24GB VRAM budget
- **Data generation:** Claude 3.5 Sonnet as the primary generation model with caching for reproducibility
- **Evaluation:** Bootstrap resampling (n=1000) for 95% confidence intervals
