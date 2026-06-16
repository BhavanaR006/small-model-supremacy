"""Pydantic configuration models for the Small Model Supremacy project.

Provides typed, validated configuration via Pydantic v2 BaseModel classes.
Supports loading from YAML and comprehensive validation with descriptive errors.
"""

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class ModelConfig(BaseModel):
    """Configuration for the base model."""

    name: str = Field(
        description="HuggingFace model identifier (e.g., 'Qwen/Qwen2.5-3B')"
    )
    max_seq_length: int = Field(
        default=2048,
        gt=0,
        le=32768,
        description="Maximum sequence length for tokenization",
    )

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("model name must not be empty or whitespace-only")
        return v


class TrainConfig(BaseModel):
    """Configuration for QLoRA training."""

    lora_rank: int = Field(
        default=64, gt=0, le=1024, description="LoRA rank (r parameter)"
    )
    lora_alpha: int = Field(
        default=128, gt=0, le=2048, description="LoRA alpha scaling factor"
    )
    learning_rate: float = Field(
        default=2e-4, gt=0.0, lt=1.0, description="Learning rate"
    )
    batch_size: int = Field(default=4, gt=0, le=256, description="Training batch size")
    gradient_accumulation_steps: int = Field(
        default=4, gt=0, le=128, description="Gradient accumulation steps"
    )
    num_epochs: int = Field(
        default=3, gt=0, le=100, description="Number of training epochs"
    )
    warmup_steps: int = Field(
        default=100, ge=0, le=10000, description="Number of warmup steps"
    )
    checkpoint_interval: int = Field(
        default=500, gt=0, description="Save checkpoint every N steps"
    )
    eval_interval: int = Field(
        default=100, gt=0, description="Evaluate every N steps"
    )
    early_stopping_patience: int = Field(
        default=3,
        gt=0,
        le=100,
        description="Number of consecutive eval intervals without improvement before stopping",
    )
    early_stopping_threshold: float = Field(
        default=0.001,
        gt=0.0,
        lt=1.0,
        description="Minimum improvement to reset early stopping counter",
    )
    max_memory_gb: float = Field(
        default=24.0, gt=0.0, le=1024.0, description="Maximum GPU memory in GB"
    )
    resume_from: Optional[str] = Field(
        default=None, description="Path to checkpoint to resume training from"
    )


class DataConfig(BaseModel):
    """Configuration for dataset curation."""

    schemas_dir: str = Field(
        default="schemas/", description="Directory containing JSON Schema files"
    )
    output_dir: str = Field(
        default="data/", description="Directory for generated dataset output"
    )
    seed: int = Field(
        default=42, ge=0, description="Random seed for reproducibility"
    )
    min_tokens: int = Field(
        default=50, gt=0, description="Minimum token count per example"
    )
    max_tokens: int = Field(
        default=2000, gt=0, description="Maximum token count per example"
    )
    adversarial_ratio: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Minimum fraction of adversarial examples in training set",
    )
    min_examples_per_schema: int = Field(
        default=1000, gt=0, description="Minimum training examples per schema"
    )
    min_test_per_schema: int = Field(
        default=100, gt=0, description="Minimum test examples per schema"
    )
    generation_model: str = Field(
        default="claude-3-5-sonnet-20241022",
        description="Model used for synthetic data generation",
    )
    cache_dir: str = Field(
        default="cache/api_responses/",
        description="Directory for caching API responses",
    )

    @model_validator(mode="after")
    def min_less_than_max_tokens(self) -> "DataConfig":
        if self.min_tokens >= self.max_tokens:
            raise ValueError(
                f"min_tokens ({self.min_tokens}) must be less than max_tokens ({self.max_tokens})"
            )
        return self


class EvalConfig(BaseModel):
    """Configuration for evaluation harness."""

    test_set_path: str = Field(
        default="data/test.jsonl", description="Path to the test dataset"
    )
    baselines: list[str] = Field(
        default=["gpt-4o", "claude-3-5-sonnet", "base_model"],
        description="List of baseline models to compare against",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for evaluation",
    )
    seed: int = Field(
        default=42, ge=0, description="Random seed for evaluation reproducibility"
    )
    bootstrap_iterations: int = Field(
        default=1000,
        gt=0,
        le=100000,
        description="Number of bootstrap iterations for confidence intervals",
    )
    output_dir: str = Field(
        default="results/", description="Directory for evaluation results"
    )
    retry_count: int = Field(
        default=3,
        gt=0,
        le=10,
        description="Number of retries for failed API calls during evaluation",
    )

    @field_validator("baselines")
    @classmethod
    def baselines_not_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("baselines list must contain at least one model")
        return v


class InfraConfig(BaseModel):
    """Configuration for infrastructure and tooling."""

    wandb_project: str = Field(
        default="small-model-supremacy",
        description="Weights & Biases project name",
    )
    wandb_entity: Optional[str] = Field(
        default=None, description="Weights & Biases entity (team or username)"
    )
    device: str = Field(
        default="auto",
        description="Device for training: 'auto', 'cuda', 'cpu'",
    )
    docker_image: str = Field(
        default="nvidia/cuda:12.1.0-devel-ubuntu22.04",
        description="Base Docker image for reproducible environment",
    )

    @field_validator("device")
    @classmethod
    def valid_device(cls, v: str) -> str:
        allowed = {"auto", "cuda", "cpu"}
        if v not in allowed:
            raise ValueError(
                f"device must be one of {sorted(allowed)}, got '{v}'"
            )
        return v

    @field_validator("wandb_project")
    @classmethod
    def wandb_project_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("wandb_project must not be empty or whitespace-only")
        return v


class ProjectConfig(BaseModel):
    """Top-level project configuration combining all sub-configs."""

    model: ModelConfig
    training: TrainConfig = Field(default_factory=TrainConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    evaluation: EvalConfig = Field(default_factory=EvalConfig)
    infrastructure: InfraConfig = Field(default_factory=InfraConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "ProjectConfig":
        """Load and validate configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            A validated ProjectConfig instance.

        Raises:
            FileNotFoundError: If the config file does not exist.
            ValueError: If the YAML is invalid or config validation fails.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        if not isinstance(raw, dict):
            raise ValueError(
                f"Configuration file must contain a YAML mapping, got {type(raw).__name__}"
            )

        return cls.model_validate(raw)

    def validate_all(self) -> list[str]:
        """Perform comprehensive validation and return a list of error messages.

        Returns an empty list if the configuration is fully valid.
        Each error message identifies the problematic field and the issue.
        """
        errors: list[str] = []

        # Cross-field validations beyond what Pydantic handles
        if self.training.eval_interval > self.training.checkpoint_interval:
            errors.append(
                f"training.eval_interval ({self.training.eval_interval}) should not exceed "
                f"training.checkpoint_interval ({self.training.checkpoint_interval})"
            )

        if self.data.min_tokens >= self.data.max_tokens:
            errors.append(
                f"data.min_tokens ({self.data.min_tokens}) must be less than "
                f"data.max_tokens ({self.data.max_tokens})"
            )

        if self.training.resume_from is not None:
            resume_path = Path(self.training.resume_from)
            if not resume_path.exists():
                errors.append(
                    f"training.resume_from path does not exist: {self.training.resume_from}"
                )

        return errors
