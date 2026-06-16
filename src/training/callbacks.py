"""Training callbacks for early stopping and checkpoint management.

Provides composable callback classes for the training loop:
- EarlyStoppingCallback: Halts training when validation loss plateaus
- CheckpointCallback: Saves model checkpoints and retains only the top-3 by val_loss

Both callbacks handle NaN/Inf loss values by halting training and preserving
the last known-good checkpoint.

Usage:
    from src.training.callbacks import EarlyStoppingCallback, CheckpointCallback

    early_stopping = EarlyStoppingCallback(patience=3, threshold=0.001)
    checkpointer = CheckpointCallback(checkpoint_dir=Path("checkpoints/"))

    if early_stopping.check(val_loss):
        # Training should stop
        ...

    checkpoint_path = checkpointer.save(step, val_loss, train_loss, model, config_hash)
"""

import hashlib
import json
import math
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol

from src.utils.logging import get_logger

logger = get_logger(__name__)


class Saveable(Protocol):
    """Protocol for objects that can save their state to a directory."""

    def save_pretrained(self, path: str) -> None: ...


@dataclass
class CheckpointMetadata:
    """Metadata stored alongside each checkpoint.

    Attributes:
        step: Training step at which the checkpoint was saved.
        val_loss: Validation loss at the checkpoint step.
        train_loss: Training loss at the checkpoint step.
        timestamp: ISO 8601 timestamp of when the checkpoint was saved.
        config_hash: Hash of the training configuration for provenance.
    """

    step: int
    val_loss: float
    train_loss: float
    timestamp: str
    config_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize metadata to a dictionary."""
        return {
            "step": self.step,
            "val_loss": self.val_loss,
            "train_loss": self.train_loss,
            "timestamp": self.timestamp,
            "config_hash": self.config_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CheckpointMetadata":
        """Deserialize metadata from a dictionary."""
        return cls(
            step=data["step"],
            val_loss=data["val_loss"],
            train_loss=data["train_loss"],
            timestamp=data["timestamp"],
            config_hash=data["config_hash"],
        )


@dataclass
class TrainingHaltError(Exception):
    """Raised when training must halt due to NaN/Inf loss.

    Attributes:
        step: The step at which the invalid loss was detected.
        loss_value: The invalid loss value (NaN or Inf).
        last_good_checkpoint: Path to the last good checkpoint, if any.
    """

    step: int
    loss_value: float
    last_good_checkpoint: Optional[Path] = None

    def __str__(self) -> str:
        return (
            f"Training halted at step {self.step}: "
            f"loss value is {self.loss_value}. "
            f"Last good checkpoint: {self.last_good_checkpoint}"
        )


class EarlyStoppingCallback:
    """Monitors validation loss and triggers early stopping on plateau.

    Early stopping is triggered when validation loss does not improve by at
    least `threshold` for `patience` consecutive evaluation intervals.

    Improvement is measured relative to the best observed validation loss.
    A new evaluation is considered an improvement if:
        best_loss - val_loss >= threshold

    Args:
        patience: Number of consecutive intervals without improvement
            before triggering early stopping. Defaults to 3.
        threshold: Minimum improvement in validation loss required to
            reset the patience counter. Defaults to 0.001.
    """

    def __init__(self, patience: int = 3, threshold: float = 0.001) -> None:
        self.patience = patience
        self.threshold = threshold
        self.best_loss: float = float("inf")
        self.no_improvement_count: int = 0

    def check(self, val_loss: float) -> bool:
        """Check whether training should stop based on validation loss.

        Handles NaN/Inf values by raising TrainingHaltError, since these
        indicate a catastrophic training failure.

        Args:
            val_loss: Current validation loss value.

        Returns:
            True if training should stop (patience exhausted), False otherwise.

        Raises:
            TrainingHaltError: If val_loss is NaN or Inf.
        """
        if math.isnan(val_loss) or math.isinf(val_loss):
            raise TrainingHaltError(
                step=-1,  # Caller should set the actual step
                loss_value=val_loss,
            )

        # Check if there's sufficient improvement
        improvement = self.best_loss - val_loss
        if improvement >= self.threshold:
            # Meaningful improvement — reset counter
            self.best_loss = val_loss
            self.no_improvement_count = 0
            logger.info(
                "Validation loss improved",
                extra={
                    "val_loss": val_loss,
                    "best_loss": self.best_loss,
                    "improvement": improvement,
                },
            )
        else:
            # No meaningful improvement
            self.no_improvement_count += 1
            logger.info(
                "No sufficient improvement in validation loss",
                extra={
                    "val_loss": val_loss,
                    "best_loss": self.best_loss,
                    "improvement": improvement,
                    "no_improvement_count": self.no_improvement_count,
                    "patience": self.patience,
                },
            )

        if self.no_improvement_count >= self.patience:
            logger.warning(
                "Early stopping triggered",
                extra={
                    "patience": self.patience,
                    "threshold": self.threshold,
                    "best_loss": self.best_loss,
                    "no_improvement_count": self.no_improvement_count,
                },
            )
            return True

        return False

    def reset(self) -> None:
        """Reset the early stopping state."""
        self.best_loss = float("inf")
        self.no_improvement_count = 0


class CheckpointCallback:
    """Manages model checkpoint saving with top-K retention by validation loss.

    Saves checkpoints at specified intervals and retains only the top
    `max_checkpoints` checkpoints ranked by lowest validation loss.
    Older checkpoints outside the top-K are automatically deleted.

    Each checkpoint is stored as a directory containing model weights
    and a `metadata.json` file with training state information.

    Args:
        checkpoint_dir: Directory where checkpoints are saved.
        max_checkpoints: Maximum number of checkpoints to retain.
            Defaults to 3.
    """

    def __init__(self, checkpoint_dir: Path, max_checkpoints: int = 3) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.checkpoints: list[CheckpointMetadata] = []  # sorted by val_loss ascending

        # Ensure checkpoint directory exists
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        step: int,
        val_loss: float,
        train_loss: float,
        model: Any,
        config_hash: str,
    ) -> Path:
        """Save a model checkpoint and prune checkpoints exceeding max_checkpoints.

        Handles NaN/Inf in val_loss by raising TrainingHaltError and saving
        the last known-good checkpoint information.

        Args:
            step: Current training step.
            val_loss: Validation loss at this step.
            train_loss: Training loss at this step.
            model: Model object with a `save_pretrained(path)` method.
            config_hash: Hash identifying the training configuration.

        Returns:
            Path to the saved checkpoint directory.

        Raises:
            TrainingHaltError: If val_loss is NaN or Inf.
        """
        # Check for NaN/Inf — halt training
        if math.isnan(val_loss) or math.isinf(val_loss):
            last_good = self.get_best()
            raise TrainingHaltError(
                step=step,
                loss_value=val_loss,
                last_good_checkpoint=last_good,
            )

        # Create checkpoint directory
        checkpoint_name = f"checkpoint-step-{step}"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Save model
        model.save_pretrained(str(checkpoint_path))

        # Create and save metadata
        metadata = CheckpointMetadata(
            step=step,
            val_loss=val_loss,
            train_loss=train_loss,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            config_hash=config_hash,
        )

        metadata_path = checkpoint_path / "metadata.json"
        metadata_path.write_text(json.dumps(metadata.to_dict(), indent=2))

        # Add to tracked checkpoints and sort by val_loss (ascending)
        self.checkpoints.append(metadata)
        self.checkpoints.sort(key=lambda m: m.val_loss)

        logger.info(
            "Checkpoint saved",
            extra={
                "step": step,
                "val_loss": val_loss,
                "train_loss": train_loss,
                "checkpoint_path": str(checkpoint_path),
                "total_checkpoints": len(self.checkpoints),
            },
        )

        # Prune checkpoints exceeding max_checkpoints
        self._prune()

        return checkpoint_path

    def _prune(self) -> None:
        """Remove checkpoints that fall outside the top-K by lowest val_loss."""
        while len(self.checkpoints) > self.max_checkpoints:
            # Remove the worst checkpoint (highest val_loss, last in sorted list)
            worst = self.checkpoints.pop()
            checkpoint_name = f"checkpoint-step-{worst.step}"
            checkpoint_path = self.checkpoint_dir / checkpoint_name

            if checkpoint_path.exists():
                shutil.rmtree(checkpoint_path)
                logger.info(
                    "Pruned checkpoint",
                    extra={
                        "step": worst.step,
                        "val_loss": worst.val_loss,
                        "checkpoint_path": str(checkpoint_path),
                    },
                )

    def get_best(self) -> Optional[Path]:
        """Return the path to the best checkpoint (lowest val_loss).

        Returns:
            Path to the best checkpoint directory, or None if no
            checkpoints have been saved.
        """
        if not self.checkpoints:
            return None

        best = self.checkpoints[0]
        checkpoint_name = f"checkpoint-step-{best.step}"
        return self.checkpoint_dir / checkpoint_name

    def get_all(self) -> list[CheckpointMetadata]:
        """Return all retained checkpoint metadata, sorted by val_loss ascending.

        Returns:
            List of CheckpointMetadata for all retained checkpoints.
        """
        return list(self.checkpoints)

    def load_existing(self) -> None:
        """Scan checkpoint_dir for existing checkpoints and load their metadata.

        Useful when resuming training — reconstructs the checkpoint tracking
        state from disk.
        """
        self.checkpoints.clear()

        if not self.checkpoint_dir.exists():
            return

        for entry in self.checkpoint_dir.iterdir():
            if entry.is_dir() and entry.name.startswith("checkpoint-step-"):
                metadata_path = entry / "metadata.json"
                if metadata_path.exists():
                    data = json.loads(metadata_path.read_text())
                    metadata = CheckpointMetadata.from_dict(data)
                    self.checkpoints.append(metadata)

        self.checkpoints.sort(key=lambda m: m.val_loss)
        logger.info(
            "Loaded existing checkpoints",
            extra={"count": len(self.checkpoints)},
        )
