"""QLoRA Trainer for fine-tuning Qwen2.5 models on structured data extraction.

Implements 4-bit NF4 quantization via bitsandbytes and LoRA via PEFT,
with integrated memory monitoring, early stopping, checkpointing, and W&B logging.

All heavy imports (torch, transformers, peft, bitsandbytes) are deferred to
method calls to avoid import-time failures in test environments without GPU.

Usage:
    from src.config.schema import TrainConfig, ModelConfig
    from src.training.trainer import Trainer, TrainResult

    config = TrainConfig()
    model_config = ModelConfig(name="Qwen/Qwen2.5-1.5B")
    trainer = Trainer(config=config, model_config=model_config)
    trainer.setup_model()
    trainer.setup_qlora()
    result = trainer.train(train_dataloader, val_dataloader)
"""

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from src.config.schema import ModelConfig, TrainConfig
from src.training.callbacks import (
    CheckpointCallback,
    EarlyStoppingCallback,
    TrainingHaltError,
)
from src.training.memory import MemoryMonitor, get_device
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TrainResult:
    """Result of a completed training run."""

    final_train_loss: float
    final_val_loss: float
    total_steps: int
    total_time_seconds: float
    best_checkpoint_path: Path
    early_stopped: bool


class Trainer:
    """QLoRA trainer for Qwen2.5 fine-tuning.

    Orchestrates model loading with 4-bit quantization, LoRA adapter application,
    training loop with gradient accumulation, evaluation, checkpointing, early
    stopping, and experiment logging to Weights & Biases.

    Args:
        config: Training hyperparameters (TrainConfig).
        model_config: Model configuration (ModelConfig).
        checkpoint_dir: Directory for saving checkpoints.
        wandb_project: W&B project name.
        wandb_enabled: Whether to log to W&B.
    """

    LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

    def __init__(
        self,
        config: TrainConfig,
        model_config: ModelConfig,
        checkpoint_dir: Path = Path("checkpoints/"),
        wandb_project: str = "small-model-supremacy",
        wandb_enabled: bool = True,
    ) -> None:
        self.config = config
        self.model_config = model_config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.wandb_project = wandb_project
        self.wandb_enabled = wandb_enabled

        self.model: Any = None
        self.tokenizer: Any = None
        self.optimizer: Any = None
        self.scheduler: Any = None
        self.device: str = "cpu"

        self.memory_monitor = MemoryMonitor()
        self.early_stopping = EarlyStoppingCallback(
            patience=config.early_stopping_patience,
            threshold=config.early_stopping_threshold,
        )
        self.checkpointer = CheckpointCallback(
            checkpoint_dir=self.checkpoint_dir,
            max_checkpoints=3,
        )

        self._config_hash = self._compute_config_hash()

    def _compute_config_hash(self) -> str:
        """Compute a hash of the training configuration for provenance."""
        config_dict = {
            "model_name": self.model_config.name,
            "max_seq_length": self.model_config.max_seq_length,
            "lora_rank": self.config.lora_rank,
            "lora_alpha": self.config.lora_alpha,
            "learning_rate": self.config.learning_rate,
            "batch_size": self.config.batch_size,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "num_epochs": self.config.num_epochs,
            "warmup_steps": self.config.warmup_steps,
        }
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def setup_model(self) -> None:
        """Load the base Qwen2.5 model with 4-bit NF4 quantization.

        Uses bitsandbytes BitsAndBytesConfig for memory-efficient loading.
        Falls back to CPU without quantization if CUDA is unavailable.
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.device = get_device()

        logger.info(
            "Loading base model with 4-bit NF4 quantization",
            extra={
                "model_name": self.model_config.name,
                "device": self.device,
                "max_seq_length": self.model_config.max_seq_length,
            },
        )

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        if self.device == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_config.name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_config.name,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.name,
            trust_remote_code=True,
            padding_side="right",
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

        self.model.config.use_cache = False
        self.model.gradient_checkpointing_enable()

        self.memory_monitor.check_and_enforce(max_memory_gb=self.config.max_memory_gb)

        logger.info(
            "Base model loaded successfully",
            extra={"model_name": self.model_config.name},
        )

    def setup_qlora(self) -> None:
        """Apply LoRA adapters to the model's attention layers via PEFT.

        Targets q_proj, k_proj, v_proj, o_proj with configured rank and alpha.

        Raises:
            RuntimeError: If setup_model() has not been called first.
        """
        if self.model is None:
            raise RuntimeError(
                "Model not loaded. Call setup_model() before setup_qlora()."
            )

        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

        logger.info(
            "Applying LoRA adapters",
            extra={
                "lora_rank": self.config.lora_rank,
                "lora_alpha": self.config.lora_alpha,
                "target_modules": self.LORA_TARGET_MODULES,
            },
        )

        self.model = prepare_model_for_kbit_training(self.model)

        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.LORA_TARGET_MODULES,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        self.model = get_peft_model(self.model, lora_config)

        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_pct = (
            100.0 * trainable_params / total_params if total_params > 0 else 0.0
        )

        logger.info(
            "LoRA adapters applied",
            extra={
                "trainable_params": trainable_params,
                "total_params": total_params,
                "trainable_percentage": round(trainable_pct, 4),
            },
        )

        self.memory_monitor.check_and_enforce(max_memory_gb=self.config.max_memory_gb)

    def _setup_optimizer_and_scheduler(self, total_steps: int) -> None:
        """Set up AdamW optimizer and cosine LR scheduler."""
        import torch
        from transformers import get_cosine_schedule_with_warmup

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01,
        )

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps,
        )

    def _log_hyperparameters(self) -> dict[str, Any]:
        """Log and return training hyperparameters dict."""
        hyperparams = {
            "model_name": self.model_config.name,
            "max_seq_length": self.model_config.max_seq_length,
            "lora_rank": self.config.lora_rank,
            "lora_alpha": self.config.lora_alpha,
            "learning_rate": self.config.learning_rate,
            "batch_size": self.config.batch_size,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "num_epochs": self.config.num_epochs,
            "warmup_steps": self.config.warmup_steps,
            "eval_interval": self.config.eval_interval,
            "checkpoint_interval": self.config.checkpoint_interval,
            "early_stopping_patience": self.config.early_stopping_patience,
            "early_stopping_threshold": self.config.early_stopping_threshold,
            "max_memory_gb": self.config.max_memory_gb,
            "quantization": "4-bit NF4",
            "lora_target_modules": self.LORA_TARGET_MODULES,
            "config_hash": self._config_hash,
        }
        logger.info("Training hyperparameters", extra=hyperparams)
        return hyperparams

    def _init_wandb(self, hyperparams: dict[str, Any]) -> None:
        """Initialize W&B logging if enabled."""
        if not self.wandb_enabled:
            return
        try:
            import wandb

            wandb.init(
                project=self.wandb_project,
                config=hyperparams,
                name=f"qlora-{self.model_config.name.split('/')[-1]}-{self._config_hash[:8]}",
            )
            logger.info("W&B logging initialized", extra={"project": self.wandb_project})
        except Exception as e:
            logger.warning(
                "Failed to initialize W&B, continuing without logging",
                extra={"error": str(e)},
            )
            self.wandb_enabled = False

    def _evaluate(self, val_dataloader: Any) -> float:
        """Run evaluation on the validation dataloader.

        Returns:
            Average validation loss.
        """
        import torch

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_dataloader:
                batch = {
                    k: v.to(self.device) if hasattr(v, "to") else v
                    for k, v in batch.items()
                }
                outputs = self.model(**batch)
                total_loss += outputs.loss.item()
                num_batches += 1

        self.model.train()
        return total_loss / max(num_batches, 1)

    def _log_to_wandb(self, metrics: dict[str, Any], step: int) -> None:
        """Log metrics to W&B if enabled."""
        if not self.wandb_enabled:
            return
        try:
            import wandb

            wandb.log(metrics, step=step)
        except Exception as e:
            logger.warning(
                "Failed to log to W&B",
                extra={"error": str(e), "step": step},
            )

    def train(self, train_dataloader: Any, val_dataloader: Any) -> TrainResult:
        """Execute the training loop with gradient accumulation and periodic eval.

        Args:
            train_dataloader: Iterable of tokenized batches (dicts with
                input_ids, attention_mask, labels).
            val_dataloader: Iterable of tokenized validation batches.

        Returns:
            TrainResult with final metrics and best checkpoint path.

        Raises:
            RuntimeError: If model not set up.
            MemoryError: If GPU memory exceeds limit.
            TrainingHaltError: If NaN/Inf loss detected.
        """
        import torch

        if self.model is None:
            raise RuntimeError(
                "Model not initialized. Call setup_model() and setup_qlora() first."
            )

        hyperparams = self._log_hyperparameters()
        self._init_wandb(hyperparams)

        self.checkpointer.load_existing()

        num_batches_per_epoch = len(train_dataloader)
        total_steps = (
            num_batches_per_epoch * self.config.num_epochs
        ) // self.config.gradient_accumulation_steps

        self._setup_optimizer_and_scheduler(total_steps)

        logger.info(
            "Starting training",
            extra={
                "total_steps": total_steps,
                "num_epochs": self.config.num_epochs,
                "batches_per_epoch": num_batches_per_epoch,
            },
        )

        start_time = time.time()
        global_step = 0
        accumulated_loss = 0.0
        accumulation_count = 0
        final_train_loss = 0.0
        final_val_loss = float("inf")
        early_stopped = False

        self.model.train()

        try:
            for epoch in range(self.config.num_epochs):
                logger.info("Starting epoch", extra={"epoch": epoch + 1})

                for batch_idx, batch in enumerate(train_dataloader):
                    batch = {
                        k: v.to(self.device) if hasattr(v, "to") else v
                        for k, v in batch.items()
                    }

                    outputs = self.model(**batch)
                    loss = outputs.loss / self.config.gradient_accumulation_steps
                    loss.backward()

                    accumulated_loss += outputs.loss.item()
                    accumulation_count += 1

                    if accumulation_count >= self.config.gradient_accumulation_steps:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_norm=1.0
                        )

                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()

                        step_loss = accumulated_loss / accumulation_count
                        final_train_loss = step_loss
                        accumulated_loss = 0.0
                        accumulation_count = 0
                        global_step += 1

                        self._log_to_wandb(
                            {
                                "train/loss": step_loss,
                                "train/learning_rate": self.scheduler.get_last_lr()[0],
                                "train/epoch": epoch + 1,
                            },
                            step=global_step,
                        )

                        if global_step % self.config.eval_interval == 0:
                            val_loss = self._evaluate(val_dataloader)
                            final_val_loss = val_loss

                            logger.info(
                                "Evaluation",
                                extra={
                                    "step": global_step,
                                    "train_loss": step_loss,
                                    "val_loss": val_loss,
                                },
                            )

                            self._log_to_wandb(
                                {
                                    "eval/val_loss": val_loss,
                                    "eval/train_loss": step_loss,
                                },
                                step=global_step,
                            )

                            if global_step % self.config.checkpoint_interval == 0:
                                self.checkpointer.save(
                                    step=global_step,
                                    val_loss=val_loss,
                                    train_loss=step_loss,
                                    model=self.model,
                                    config_hash=self._config_hash,
                                )

                            should_stop = self.early_stopping.check(val_loss)
                            if should_stop:
                                self.checkpointer.save(
                                    step=global_step,
                                    val_loss=val_loss,
                                    train_loss=step_loss,
                                    model=self.model,
                                    config_hash=self._config_hash,
                                )
                                early_stopped = True
                                logger.warning(
                                    "Early stopping triggered",
                                    extra={
                                        "step": global_step,
                                        "val_loss": val_loss,
                                    },
                                )
                                break

                        self.memory_monitor.check_and_enforce(
                            max_memory_gb=self.config.max_memory_gb
                        )

                if early_stopped:
                    break

        except TrainingHaltError as e:
            logger.error(
                "Training halted due to invalid loss",
                extra={"step": e.step, "loss_value": e.loss_value},
            )
            raise

        finally:
            if self.wandb_enabled:
                try:
                    import wandb

                    wandb.finish()
                except Exception:
                    pass

        total_time = time.time() - start_time

        best_checkpoint = self.checkpointer.get_best()
        if best_checkpoint is None:
            best_checkpoint = self.checkpointer.save(
                step=global_step,
                val_loss=final_val_loss,
                train_loss=final_train_loss,
                model=self.model,
                config_hash=self._config_hash,
            )

        logger.info(
            "Training complete",
            extra={
                "total_steps": global_step,
                "total_time_seconds": round(total_time, 2),
                "final_train_loss": final_train_loss,
                "final_val_loss": final_val_loss,
                "early_stopped": early_stopped,
                "best_checkpoint": str(best_checkpoint),
                "peak_memory_gb": self.memory_monitor.get_peak(),
            },
        )

        return TrainResult(
            final_train_loss=final_train_loss,
            final_val_loss=final_val_loss,
            total_steps=global_step,
            total_time_seconds=total_time,
            best_checkpoint_path=best_checkpoint,
            early_stopped=early_stopped,
        )

    def resume_from_checkpoint(self, checkpoint_path: Path) -> None:
        """Resume training state from a saved checkpoint.

        Loads adapter weights from the checkpoint directory.

        Args:
            checkpoint_path: Path to the checkpoint directory.

        Raises:
            FileNotFoundError: If the checkpoint path does not exist.
            RuntimeError: If model has not been loaded.
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint path does not exist: {checkpoint_path}"
            )

        if self.model is None:
            raise RuntimeError(
                "Model not loaded. Call setup_model() before resuming."
            )

        from peft import PeftModel

        logger.info(
            "Resuming from checkpoint",
            extra={"checkpoint_path": str(checkpoint_path)},
        )

        if isinstance(self.model, PeftModel):
            self.model.load_adapter(str(checkpoint_path), adapter_name="default")
        else:
            self.model = PeftModel.from_pretrained(
                self.model,
                str(checkpoint_path),
                is_trainable=True,
            )

        metadata_path = checkpoint_path / "metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            logger.info(
                "Checkpoint metadata loaded",
                extra={
                    "resumed_from_step": metadata.get("step"),
                    "resumed_val_loss": metadata.get("val_loss"),
                },
            )

        self.checkpointer.load_existing()
        logger.info("Successfully resumed from checkpoint")
