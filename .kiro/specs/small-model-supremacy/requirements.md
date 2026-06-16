# Requirements Document

## Introduction

**Small Model Supremacy** is a personal ML project that fine-tunes a small Qwen model (1.7B–3B parameters) to outperform state-of-the-art large models on a specific narrow task. The project demonstrates that targeted fine-tuning on curated data can beat general-purpose frontier models, producing a resume-worthy GitHub portfolio piece with reproducible results, clear documentation, and a published model on Hugging Face.

The selected target task is **structured data extraction from unstructured text** — specifically, extracting structured JSON objects from natural language descriptions of real-world entities (e.g., extracting product attributes, event details, or scientific paper metadata from free-form text). This task is chosen because:
- Large models frequently hallucinate fields or produce malformed JSON on complex schemas
- A small model trained on high-quality extraction pairs can achieve near-perfect schema compliance
- The task has clear, automatable evaluation metrics (exact match, field-level F1, schema validity)
- It is practically useful and demonstrates production-relevant skills

## Glossary

- **Fine_Tuned_Model**: The Qwen 1.7B–3B parameter model after task-specific training
- **Base_Model**: The pre-trained Qwen model before fine-tuning (Qwen2.5-3B or Qwen2.5-1.5B)
- **Training_Pipeline**: The end-to-end system for data loading, model training, checkpointing, and evaluation
- **Evaluation_Harness**: The automated system that runs the Fine_Tuned_Model and baseline models against the test set and computes metrics
- **Dataset_Curator**: The component responsible for generating, filtering, and validating training data
- **Extraction_Schema**: A JSON Schema definition describing the target structured output format
- **Schema_Validity**: Whether a model output is parseable JSON that conforms to the Extraction_Schema
- **Field_F1**: The harmonic mean of precision and recall computed at the individual field level across all test examples
- **Exact_Match**: The percentage of test examples where model output exactly matches the reference output after normalization

## Requirements

### Requirement 1: Task Definition and Schema Design

**User Story:** As a developer, I want a clearly defined extraction task with formal schemas, so that evaluation is unambiguous and reproducible.

#### Acceptance Criteria

1. THE Training_Pipeline SHALL support at least 3 distinct Extraction_Schemas of varying complexity (simple: 3–5 fields, medium: 6–10 fields, complex: 11+ fields with nested objects)
2. WHEN an Extraction_Schema is defined, THE Dataset_Curator SHALL validate that the schema is a valid JSON Schema draft 2020-12 document and reject any schema that fails validation with an error message identifying the invalid portion
3. THE Extraction_Schema definitions SHALL include field types, required fields, enum constraints, and description annotations for each field
4. WHEN a new domain is added, THE Training_Pipeline SHALL require an accompanying Extraction_Schema and at least 10 seed examples before allowing training on that domain
5. THE Training_Pipeline SHALL store all Extraction_Schemas in a dedicated `schemas/` directory, each as a standalone JSON file named `{domain}_{complexity}.schema.json`

### Requirement 2: Dataset Curation

**User Story:** As a developer, I want a high-quality curated dataset for fine-tuning, so that the model learns accurate extraction patterns.

#### Acceptance Criteria

1. THE Dataset_Curator SHALL produce a training set of at least 5,000 input-output pairs across all schemas
2. THE Dataset_Curator SHALL produce a held-out test set of at least 500 examples where no input text appears in both the training set and test set (source overlap is defined as identical input_text content after whitespace normalization)
3. WHEN generating synthetic training data, THE Dataset_Curator SHALL use a frontier model (Claude or GPT-4) to produce candidate pairs and then validate each pair against the Extraction_Schema
4. IF a generated example fails schema validation, THEN THE Dataset_Curator SHALL discard that example and log the failure reason including the schema_id, the validation error message, and the generated output
5. THE Dataset_Curator SHALL ensure each schema has at least 1,000 training examples and 100 test examples
6. WHEN curating data, THE Dataset_Curator SHALL include adversarial examples (classified as one of: ambiguous_input, missing_fields, extraneous_text, contradictory_info) comprising at least 15% of the training set
7. THE Dataset_Curator SHALL store all data in JSONL format with fields: input_text, expected_output, schema_id, difficulty_level (one of: simple, medium, complex), and source_metadata (object containing: generation_model, generation_timestamp, prompt_id)
8. THE Dataset_Curator SHALL enforce input_text length between 50 and 2,000 tokens (measured by the Qwen tokenizer) and discard examples outside this range

### Requirement 3: Fine-Tuning Methodology

**User Story:** As a developer, I want an efficient fine-tuning approach that works within consumer GPU constraints, so that I can train the model affordably.

#### Acceptance Criteria

1. THE Training_Pipeline SHALL use QLoRA (4-bit quantization with Low-Rank Adaptation) as the default fine-tuning method with default LoRA rank of 64 and LoRA alpha of 128
2. THE Training_Pipeline SHALL not exceed 24GB of peak GPU memory usage during training (compatible with a single RTX 4090 or A10G)
3. WHEN training begins, THE Training_Pipeline SHALL log hyperparameters including: learning rate, batch size, LoRA rank, LoRA alpha, number of epochs, warmup steps, and quantization configuration
4. THE Training_Pipeline SHALL save checkpoints at configurable intervals (default: every 500 steps) and retain the top 3 checkpoints ranked by lowest validation loss, deleting older checkpoints that fall outside the top 3
5. WHILE training is in progress, THE Training_Pipeline SHALL compute and log validation loss on a held-out validation split every 100 steps
6. THE Training_Pipeline SHALL support training on the Qwen2.5-1.5B and Qwen2.5-3B base models
7. IF validation loss does not improve by at least 0.001 for 3 consecutive evaluation intervals (every 100 steps), THEN THE Training_Pipeline SHALL apply early stopping and save the checkpoint with the lowest validation loss observed during the run
8. THE Training_Pipeline SHALL use the Hugging Face Transformers library with PEFT (Parameter-Efficient Fine-Tuning) and bitsandbytes for quantization
9. WHEN training completes normally (all epochs finished without early stopping), THE Training_Pipeline SHALL save the final model and log total training time, final training loss, and final validation loss
10. IF peak GPU memory usage exceeds 24GB during training, THEN THE Training_Pipeline SHALL terminate the run and report an error message indicating the memory limit was exceeded along with the observed peak usage

### Requirement 4: Evaluation and Benchmarking

**User Story:** As a developer, I want rigorous evaluation comparing my fine-tuned model against frontier models, so that I can demonstrate the model beats larger models on this task.

#### Acceptance Criteria

1. THE Evaluation_Harness SHALL compute the following metrics for each model: Schema_Validity rate (percentage), Field_F1 (0.0–1.0), Exact_Match (percentage), and average inference latency in milliseconds measured from prompt submission to complete response receipt
2. THE Evaluation_Harness SHALL evaluate the Fine_Tuned_Model against at least 3 baseline models: GPT-4o, Claude 3.5 Sonnet, and the unmodified Base_Model
3. WHEN evaluation is complete, THE Evaluation_Harness SHALL produce a results table comparing all models across all metrics and schemas, saved as both a CSV file and printed to standard output
4. THE Evaluation_Harness SHALL report metrics separately for each difficulty level (simple, medium, complex schemas)
5. IF the Fine_Tuned_Model achieves a Schema_Validity rate below 95% on any schema, THEN THE Evaluation_Harness SHALL include a warning entry in the results output identifying the schema name and its achieved Schema_Validity rate
6. THE Evaluation_Harness SHALL use identical prompts across all models to ensure fair comparison
7. WHEN running evaluation, THE Evaluation_Harness SHALL use temperature 0 and a fixed random seed for reproducibility
8. THE Evaluation_Harness SHALL compute 95% confidence intervals for all metrics using bootstrap resampling with 1,000 iterations
9. THE Evaluation_Harness SHALL report inference latency separately for local models (measuring GPU computation time) and API-based models (measuring round-trip request time including network overhead)
10. IF a baseline model API call fails during evaluation, THEN THE Evaluation_Harness SHALL retry the request up to 3 times with exponential backoff, and if all retries fail, record the example as a failure and continue evaluation of remaining examples

### Requirement 5: Infrastructure and Tooling

**User Story:** As a developer, I want a reproducible training environment with clear setup instructions, so that others can verify my results.

#### Acceptance Criteria

1. THE Training_Pipeline SHALL be runnable via a single command after environment setup (e.g., `python train.py --config config.yaml`)
2. THE Training_Pipeline SHALL use a YAML configuration file for all hyperparameters and paths, with no hardcoded values in source code
3. THE Training_Pipeline SHALL log all experiments to Weights & Biases (wandb) for tracking and visualization
4. WHEN the project is cloned, THE Training_Pipeline SHALL provide a pyproject.toml with pinned dependency versions (exact versions, not ranges)
5. THE Training_Pipeline SHALL include a Dockerfile that reproduces the training environment exactly, using a pinned CUDA base image
6. IF a CUDA-capable GPU is not available, THEN THE Training_Pipeline SHALL fall back to CPU mode with a warning logged to stderr that training will be impractically slow
7. THE Training_Pipeline SHALL support resuming training from any saved checkpoint by specifying the checkpoint path in the configuration file
8. THE Training_Pipeline SHALL validate the configuration file at startup and exit with a descriptive error message if any required field is missing or has an invalid value

### Requirement 6: Project Deliverables and Documentation

**User Story:** As a developer, I want polished deliverables suitable for a portfolio, so that the project demonstrates professional ML engineering skills.

#### Acceptance Criteria

1. THE Fine_Tuned_Model SHALL be published to Hugging Face Hub with a model card documenting: training data summary, hyperparameters, evaluation results, and usage instructions
2. THE Training_Pipeline SHALL include a README.md with: project overview, setup instructions, training commands, evaluation commands, and results summary with visualizations
3. WHEN evaluation is complete, THE Evaluation_Harness SHALL generate comparison charts (bar charts for metrics, latency comparison) saved as PNG files with labeled axes, legend, and title
4. THE Training_Pipeline SHALL include a CLI-based interactive demo script (`demo.py`) that accepts free-text input via stdin or command-line argument and outputs structured JSON extraction to stdout using the Fine_Tuned_Model
5. THE Training_Pipeline SHALL include a Jupyter notebook walking through the end-to-end process: data exploration, training monitoring, and results analysis; the notebook SHALL be executable from top to bottom without error given the project dependencies are installed
6. THE Training_Pipeline SHALL maintain a CHANGELOG.md documenting all experiments, their configurations, and outcomes
7. WHEN the model is published, THE Fine_Tuned_Model SHALL include example inference code compatible with the Hugging Face transformers library

### Requirement 7: Data Pipeline Reproducibility

**User Story:** As a developer, I want the data generation pipeline to be fully reproducible, so that others can recreate or extend the dataset.

#### Acceptance Criteria

1. THE Dataset_Curator SHALL use a configurable random seed (default: 42) for all data splitting, shuffling, and sampling operations, and SHALL record the seed value in the dataset metadata
2. WHEN synthetic data is generated, THE Dataset_Curator SHALL save the exact prompts, model identifier, and generation parameters (temperature, max tokens) used to generate each example in a JSONL file alongside the dataset
3. THE Dataset_Curator SHALL include a script that regenerates the full dataset from seed examples and saved prompts; WHEN cached API responses are available, the script SHALL produce a byte-identical dataset to the original
4. IF the source API (Claude/GPT-4) is unavailable due to network error, HTTP error response, or request timeout exceeding 60 seconds, THEN THE Dataset_Curator SHALL use cached responses from prior runs
5. IF cached responses are not available for a required generation request, THEN THE Dataset_Curator SHALL skip that example, log a warning including the prompt identifier, and report the total number of skipped examples upon completion
6. WHEN dataset generation is complete, THE Dataset_Curator SHALL compute and log dataset statistics to a JSON report file including: token length distributions (min, max, mean, median, p95 per schema), field coverage rates (percentage of examples containing each field per schema), and difficulty distribution (count and percentage per difficulty level)

### Requirement 8: Model Output Parsing and Validation

**User Story:** As a developer, I want robust output parsing so that the model's raw text output is reliably converted to structured data.

#### Acceptance Criteria

1. WHEN the Fine_Tuned_Model produces output, THE Training_Pipeline SHALL extract the first JSON object or array from the output text and attempt to parse it as JSON, ignoring any surrounding non-JSON text
2. IF the model output is not valid JSON, THEN THE Training_Pipeline SHALL apply the following repair heuristics in order — bracket/brace completion, trailing comma removal, and single-to-double quote replacement — and re-attempt parsing after each repair
3. IF the model output remains unparseable after all repair heuristics are applied, THEN THE Training_Pipeline SHALL record the output as a parse failure, increment the parse failure count for the corresponding schema, and proceed to the next example without halting
4. WHEN parsed output is available, THE Training_Pipeline SHALL validate the output against the corresponding Extraction_Schema using JSON Schema validation
5. THE Training_Pipeline SHALL track and report a valid_json_rate metric (percentage of outputs that parse as valid JSON, with or without repair) separately from the Schema_Validity metric (percentage of outputs that also pass Extraction_Schema validation)
6. THE Training_Pipeline SHALL format model inputs using a consistent prompt template that includes the full Extraction_Schema definition, a natural language description of the extraction task, and at least one example input-output pair
7. IF the Fine_Tuned_Model produces empty or whitespace-only output, THEN THE Training_Pipeline SHALL record the output as a parse failure without attempting repair
