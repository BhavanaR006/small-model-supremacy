# Implementation Plan: Small Model Supremacy

## Overview

End-to-end implementation of a QLoRA fine-tuning pipeline for Qwen2.5 models targeting structured data extraction. The plan progresses from project scaffolding and configuration through data generation, training, parsing, evaluation, and finally CLI entry points.

## Tasks

- [x] 1. Set up project structure and dependencies
  - [x] 1.1 Create project scaffolding and pyproject.toml
    - Create directory structure: `src/`, `tests/`, `schemas/`, `data/`, `cache/`, `checkpoints/`, `results/`, `notebooks/`
    - Add all `__init__.py` files for packages (`src/`, `src/config/`, `src/schemas/`, `src/data/`, `src/training/`, `src/parsing/`, `src/evaluation/`, `src/utils/`, `tests/`, `tests/properties/`)
    - Create `pyproject.toml` with pinned dependencies: transformers, peft, bitsandbytes, datasets, jsonschema, pydantic, wandb, hypothesis, pytest, pytest-cov, pytest-timeout, tiktoken, pandas, matplotlib, seaborn, click, pyyaml
    - Create `Dockerfile` based on `nvidia/cuda:12.1.0-devel-ubuntu22.04` with Python 3.11, pip install from pyproject.toml
    - _Requirements: 5.1, 5.2, 5.3, 6.1_

  - [x] 1.2 Create JSON Schema definition files
    - Create `schemas/conference_talk_simple.schema.json` (4 fields: speaker_name, topic, conference, location)
    - Create `schemas/product_listing_medium.schema.json` (8-10 fields with nested objects)
    - Create `schemas/scientific_paper_complex.schema.json` (15+ fields with arrays and nested objects)
    - All schemas must use JSON Schema draft 2020-12 with `$schema`, `$id`, `title`, `description`, `properties`, `required`, `additionalProperties: false`
    - _Requirements: 1.1, 1.2, 1.3_

  - [x] 1.3 Implement structured logging utility
    - Create `src/utils/logging.py` with structured JSON logging
    - Configure log levels (DEBUG, INFO, WARNING, ERROR) via environment variable
    - Include context fields: timestamp, module, schema_id, step number where relevant
    - _Requirements: 7.5_

- [x] 2. Implement Configuration and Schema Management
  - [x] 2.1 Implement Pydantic configuration models
    - Create `src/config/schema.py` with Pydantic models: `ModelConfig`, `TrainConfig`, `DataConfig`, `EvalConfig`, `InfraConfig`, `ProjectConfig`
    - Implement `ProjectConfig.from_yaml(path)` to load and validate `config.yaml`
    - Implement `validate_all()` returning list of validation errors
    - Reject missing required fields, wrong types, out-of-range values with descriptive error messages
    - Create default `config.yaml` matching the design document specification
    - _Requirements: 5.4, 5.5, 5.6, 5.7, 5.8_

  - [ ]* 2.2 Write property test for configuration validation (Property 11)
    - **Property 11: Configuration Validation**
    - Generate arbitrary config dicts with missing fields, wrong types, out-of-range numbers
    - Assert validator rejects invalid configs with descriptive error identifying the problematic field
    - Assert valid configs pass validation without errors
    - **Validates: Requirements 5.8**

  - [x] 2.3 Implement Schema Manager
    - Create `src/schemas/manager.py` with `SchemaDefinition` dataclass and `SchemaManager` class
    - Implement `load_all()` to discover and load all `.schema.json` files from schemas directory
    - Implement `validate_schema(schema)` to verify JSON Schema draft 2020-12 compliance
    - Implement `validate_instance(instance, schema_id)` to validate extracted data against schema
    - Implement `get_by_complexity(complexity)` to filter schemas
    - Return `ValidationResult` with success/failure and error details
    - _Requirements: 1.2, 1.4, 1.5_

  - [ ]* 2.4 Write property test for schema document validation (Property 1)
    - **Property 1: Schema Document Validation**
    - Generate arbitrary JSON documents (valid and invalid JSON Schema 2020-12)
    - Assert validator accepts valid schemas and rejects invalid ones with error messages identifying the invalid portion
    - **Validates: Requirements 1.2**

  - [ ]* 2.5 Write property test for schema validation on parsed output (Property 15)
    - **Property 15: Schema Validation on Parsed Output**
    - Generate arbitrary JSON objects and schema pairs
    - Assert validation accepts conforming outputs and rejects non-conforming ones
    - **Validates: Requirements 8.4**

  - [ ]* 2.6 Write unit tests for config and schema manager
    - Test `ProjectConfig.from_yaml` with valid config
    - Test config validation with missing fields, wrong types, boundary values
    - Test `SchemaManager.load_all` with valid/invalid schema files
    - Test `validate_instance` with conforming and non-conforming instances
    - _Requirements: 1.2, 5.8_

- [x] 3. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Implement Dataset Curation Pipeline
  - [x] 4.1 Implement API client with caching
    - Create `src/data/api_client.py` with `APIClient` class
    - Implement prompt hash-based caching in `cache/api_responses/` directory
    - Implement `generate(prompt, params)` calling Claude/GPT-4 APIs
    - Implement exponential backoff (2^n seconds, max 60s) with 5 retries on rate limits
    - Handle network errors, timeouts (>60s), HTTP errors — use cache fallback when available
    - _Requirements: 2.1, 2.3, 7.2, 7.3_

  - [x] 4.2 Implement tokenizer utilities
    - Create `src/data/tokenizer_utils.py` with token counting using the Qwen tokenizer
    - Implement `count_tokens(text)` returning token count
    - Implement `filter_by_token_length(examples, min_tokens=50, max_tokens=2000)` returning filtered list
    - _Requirements: 2.8_

  - [x] 4.3 Implement Dataset Curator
    - Create `src/data/curator.py` with `DataExample`, `SourceMetadata`, `DatasetSplits`, `DatasetStats` dataclasses
    - Implement `generate_examples(schema_id, count)` using API client with prompt templates
    - Implement `validate_example(example)` checking output against schema
    - Implement `filter_by_token_length(examples, min_tokens, max_tokens)`
    - Implement `split_dataset(examples, seed=42)` with deterministic train/val/test split
    - Ensure adversarial examples (ambiguous_input, missing_fields, extraneous_text, contradictory_info) comprise at least 15% of training set
    - Implement `compute_statistics(dataset)` returning token length distribution, field coverage, difficulty distribution
    - Implement `save_jsonl(examples, path)` and corresponding load function
    - Discard invalid examples (log schema_id, error, output) — never halt on single bad example
    - _Requirements: 2.1, 2.2, 2.4, 2.5, 2.6, 2.7, 2.8_

  - [ ]* 4.4 Write property test for train/test non-overlap (Property 2)
    - **Property 2: Train/Test Set Non-Overlap**
    - Generate arbitrary lists of DataExample objects
    - After splitting, assert no input_text (whitespace-normalized) appears in both train and test sets
    - **Validates: Requirements 2.2**

  - [ ]* 4.5 Write property test for invalid example filtering (Property 3)
    - **Property 3: Invalid Example Filtering**
    - Generate DataExample objects with outputs that do not conform to their schema
    - Assert filtering discards them and logs schema_id, validation error, and generated output
    - **Validates: Requirements 2.4**

  - [ ]* 4.6 Write property test for adversarial ratio invariant (Property 4)
    - **Property 4: Adversarial Ratio Invariant**
    - Generate training sets with varying adversarial proportions
    - Assert adversarial examples comprise at least 15% of total
    - **Validates: Requirements 2.6**

  - [ ]* 4.7 Write property test for data serialization round-trip (Property 5)
    - **Property 5: Data Serialization Round-Trip**
    - Generate arbitrary valid DataExample objects
    - Serialize to JSONL and deserialize back
    - Assert all required fields (input_text, expected_output, schema_id, difficulty_level, source_metadata) preserved
    - **Validates: Requirements 2.7**

  - [ ]* 4.8 Write property test for token length filtering (Property 6)
    - **Property 6: Token Length Filtering**
    - Generate input examples with various token counts
    - After filtering, assert all remaining have token count in [50, 2000] and all discarded are outside that range
    - **Validates: Requirements 2.8**

  - [ ]* 4.9 Write property test for seed determinism (Property 12)
    - **Property 12: Seed Determinism**
    - For arbitrary seed values, run splitting/shuffling/sampling twice
    - Assert byte-identical results
    - **Validates: Requirements 7.1**

  - [ ]* 4.10 Write property test for dataset statistics consistency (Property 13)
    - **Property 13: Dataset Statistics Consistency**
    - Generate arbitrary datasets and compute statistics
    - Assert: min <= median <= max for token lengths, field coverage in [0%, 100%], difficulty counts sum to total, percentages sum to 100%
    - **Validates: Requirements 7.6**

- [x] 5. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Implement Output Parser
  - [x] 6.1 Implement JSON extraction and repair logic
    - Create `src/parsing/output_parser.py` with `OutputParser` class and `ParseResult` dataclass
    - Implement `parse(raw_output)` as main entry point returning `ParseResult`
    - Implement `_extract_json(text)` to find JSON objects embedded in arbitrary text (code fences, surrounding prose)
    - Implement `_repair_json(text)` with heuristics: `_complete_brackets`, `_remove_trailing_commas`, `_fix_quotes`
    - Reject whitespace-only strings immediately as parse failure without attempting repair
    - Track whether repair was applied and repair type in `ParseResult`
    - _Requirements: 8.1, 8.2, 8.3, 8.7_

  - [ ]* 6.2 Write property test for JSON extraction and repair round-trip (Property 14)
    - **Property 14: JSON Extraction and Repair Round-Trip**
    - Generate valid JSON objects embedded in arbitrary surrounding text; assert correct extraction
    - Generate valid JSON with applied corruptions (missing brackets, trailing commas, single quotes); assert repair restores parseability
    - **Validates: Requirements 8.1, 8.2**

  - [ ]* 6.3 Write property test for whitespace-only rejection (Property 18)
    - **Property 18: Whitespace-Only Rejection**
    - Generate strings composed entirely of whitespace (spaces, tabs, newlines, empty string)
    - Assert parser records as failure without attempting repair
    - **Validates: Requirements 8.7**

  - [ ]* 6.4 Write unit tests for output parser
    - Test known JSON repair cases: missing closing brackets, trailing commas, single quotes
    - Test extraction from code fences, surrounding prose
    - Test empty/whitespace input handling
    - Test deeply nested JSON extraction
    - _Requirements: 8.1, 8.2, 8.7_

- [x] 7. Implement Evaluation Harness
  - [x] 7.1 Implement metrics calculator
    - Create `src/evaluation/metrics.py` with `MetricsCalculator` class
    - Implement `schema_validity(outputs, schema_id)` — fraction of outputs passing schema validation
    - Implement `field_f1(predicted, expected)` — per-field precision/recall/F1 score
    - Implement `exact_match(predicted, expected)` — strict equality check
    - Implement `compute_bootstrap_ci(metric_values, n_iterations=1000)` — bootstrap confidence intervals
    - Ensure all metrics in [0.0, 1.0], exact_match=true implies field_f1=1.0, CI bounds satisfy lower <= mean <= upper
    - _Requirements: 4.1, 4.2, 4.8_

  - [ ]* 7.2 Write property test for metric mathematical invariants (Property 9)
    - **Property 9: Metric Mathematical Invariants**
    - Generate arbitrary predicted/expected output pairs
    - Assert: Field_F1 in [0,1], Schema_Validity in [0,1], Exact_Match in [0,1], exact_match=true implies field_f1=1.0, bootstrap CI lower <= mean <= upper with bounds in [0,1]
    - **Validates: Requirements 4.1, 4.8**

  - [ ]* 7.3 Write property test for parse rate invariant (Property 16)
    - **Property 16: Parse Rate Invariant**
    - Generate sets of model outputs with varying parse/schema validity rates
    - Assert schema_validity <= valid_json_rate (schema validity requires valid JSON as precondition)
    - **Validates: Requirements 8.5**

  - [x] 7.4 Implement evaluator orchestration
    - Create `src/evaluation/evaluator.py` with `Evaluator` class and `EvalMetrics` dataclass
    - Implement `evaluate_model(model, test_set)` — run fine-tuned model, parse outputs, compute metrics
    - Implement `evaluate_baseline(provider, test_set)` — run baseline via API, compute metrics
    - Implement `generate_results_table(results)` — produce pandas DataFrame with per-model, per-difficulty metrics
    - Group results by difficulty level; ensure union of per-difficulty examples equals full test set
    - Handle baseline API failures with 3x retry + exponential backoff; record failures and continue
    - _Requirements: 4.1, 4.3, 4.4, 4.5, 4.6, 4.7_

  - [ ]* 7.5 Write property test for results grouped by difficulty (Property 10)
    - **Property 10: Results Grouped by Difficulty**
    - Generate test sets with examples across multiple difficulty levels
    - Assert results contain separate rows per difficulty, union of per-difficulty examples equals full set
    - **Validates: Requirements 4.3, 4.4**

  - [x] 7.6 Implement visualization module
    - Create `src/evaluation/visualization.py` with chart generation
    - Implement `generate_charts(results, output_dir)` producing `metrics_comparison.png` and `latency_comparison.png`
    - Use matplotlib/seaborn for bar charts comparing models across metrics with labeled axes, legend, title
    - _Requirements: 4.9_

  - [ ]* 7.7 Write unit tests for metrics and evaluator
    - Test field_f1 with known input/output pairs
    - Test exact_match edge cases (extra whitespace, key ordering)
    - Test bootstrap CI with known distributions
    - Test results table generation with multiple models and difficulty levels
    - _Requirements: 4.1, 4.3_

- [x] 8. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 9. Implement Training Pipeline
  - [x] 9.1 Implement GPU memory monitoring
    - Create `src/training/memory.py` with memory monitoring utilities
    - Implement `check_memory()` returning current GPU memory usage
    - Implement `get_peak_memory()` for post-training reporting
    - Terminate run if memory exceeds `max_memory_gb` (24GB default), report peak usage
    - Handle CUDA unavailable — fall back to CPU with warning to stderr
    - _Requirements: 3.1, 3.8, 6.2_

  - [x] 9.2 Implement training callbacks
    - Create `src/training/callbacks.py` with early stopping and checkpointing logic
    - Implement `EarlyStoppingCallback` — trigger if 3 consecutive eval intervals show < 0.001 improvement
    - Implement `CheckpointCallback` — save checkpoint every N steps, retain top 3 by lowest val_loss, delete others
    - Save checkpoint metadata: step, val_loss, train_loss, timestamp, config_hash
    - Handle NaN/Inf in loss — halt training, save last good checkpoint, report step number
    - _Requirements: 3.4, 3.5, 3.7_

  - [ ]* 9.3 Write property test for checkpoint retention top-3 (Property 7)
    - **Property 7: Checkpoint Retention Top-3**
    - Generate arbitrary sequences of checkpoint saves with validation losses
    - Assert exactly top 3 by lowest val_loss retained, all others deleted
    - **Validates: Requirements 3.4**

  - [ ]* 9.4 Write property test for early stopping trigger (Property 8)
    - **Property 8: Early Stopping Trigger**
    - Generate arbitrary sequences of validation loss values (every 100 steps)
    - Assert early stopping triggers iff 3 consecutive intervals show < 0.001 improvement
    - **Validates: Requirements 3.7**

  - [x] 9.5 Implement QLoRA trainer
    - Create `src/training/trainer.py` with `Trainer` class
    - Implement `setup_model()` — load Qwen2.5 base model with 4-bit NF4 quantization via bitsandbytes
    - Implement `setup_qlora()` — apply LoRA (r=64, alpha=128) via PEFT to attention layers
    - Implement `train(train_dataset, val_dataset)` — training loop with gradient accumulation, eval at intervals
    - Implement `resume_from_checkpoint(checkpoint_path)` for training resumption
    - Integrate memory monitoring, early stopping callback, checkpoint callback, W&B logging
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8_

  - [ ]* 9.6 Write unit tests for training callbacks
    - Test early stopping with known loss sequences (should trigger / should not trigger)
    - Test checkpoint retention with 5+ checkpoints, verify only top 3 retained
    - Test NaN detection halts training
    - _Requirements: 3.4, 3.7_

- [x] 10. Implement Prompt Template and Integration
  - [x] 10.1 Implement prompt formatting
    - Add prompt template formatting to the data curator or a dedicated utility
    - Template must include: full schema definition, natural language task description, at least one example input-output pair
    - Implement `format_prompt(schema, input_text, example)` returning completed prompt string
    - _Requirements: 8.6_

  - [ ]* 10.2 Write property test for prompt template completeness (Property 17)
    - **Property 17: Prompt Template Completeness**
    - Generate arbitrary schema definitions and input texts
    - Assert formatted prompt contains full schema, natural language description, and at least one example pair
    - **Validates: Requirements 8.6**

- [x] 11. Implement CLI Entry Points and Wiring
  - [x] 11.1 Implement data generation entry point
    - Create `generate_data.py` CLI script
    - Load config, instantiate SchemaManager + APIClient + DatasetCurator
    - Generate examples for all schemas, validate, filter, split, save to JSONL
    - Output dataset statistics report to `data/stats.json`
    - _Requirements: 2.1, 2.5, 7.6_

  - [x] 11.2 Implement training entry point
    - Create `train.py` CLI script
    - Load config, instantiate Trainer with QLoRA settings
    - Load training/validation datasets from JSONL
    - Run training with early stopping, checkpointing, W&B logging
    - Support `--resume` flag for checkpoint resumption
    - _Requirements: 3.1, 3.6, 5.4_

  - [x] 11.3 Implement evaluation entry point
    - Create `evaluate.py` CLI script
    - Load config, instantiate Evaluator + SchemaManager + OutputParser + MetricsCalculator
    - Run evaluation on fine-tuned model and baselines (GPT-4o, Claude 3.5 Sonnet, base model)
    - Generate results table (CSV) and comparison charts
    - Output to `results/` directory
    - _Requirements: 4.1, 4.5, 4.6, 4.9_

  - [x] 11.4 Implement interactive demo CLI
    - Create `demo.py` with Click-based CLI
    - Accept input text and schema selection
    - Load fine-tuned model, run inference, parse output, display extracted data
    - _Requirements: 6.3_

- [x] 12. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate the 18 universal correctness properties from the design document using Hypothesis
- Unit tests validate specific examples and edge cases with pytest
- All PBT test files follow the tag format: `# Feature: small-model-supremacy, Property {N}: {title}`
- Implementation language: Python
- Minimum 100 iterations per property test

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2", "1.3"] },
    { "id": 1, "tasks": ["2.1", "2.3"] },
    { "id": 2, "tasks": ["2.2", "2.4", "2.5", "2.6"] },
    { "id": 3, "tasks": ["4.1", "4.2"] },
    { "id": 4, "tasks": ["4.3"] },
    { "id": 5, "tasks": ["4.4", "4.5", "4.6", "4.7", "4.8", "4.9", "4.10"] },
    { "id": 6, "tasks": ["6.1", "7.1"] },
    { "id": 7, "tasks": ["6.2", "6.3", "6.4", "7.2", "7.3"] },
    { "id": 8, "tasks": ["7.4", "7.6"] },
    { "id": 9, "tasks": ["7.5", "7.7"] },
    { "id": 10, "tasks": ["9.1", "9.2"] },
    { "id": 11, "tasks": ["9.3", "9.4", "9.5"] },
    { "id": 12, "tasks": ["9.6"] },
    { "id": 13, "tasks": ["10.1"] },
    { "id": 14, "tasks": ["10.2"] },
    { "id": 15, "tasks": ["11.1", "11.2", "11.3", "11.4"] }
  ]
}
```
