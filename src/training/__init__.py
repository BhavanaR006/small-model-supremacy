"""Training pipeline for QLoRA fine-tuning."""

from src.training.callbacks import (
    CheckpointCallback,
    CheckpointMetadata,
    EarlyStoppingCallback,
    TrainingHaltError,
)
from src.training.memory import (
    MemoryMonitor,
    MemoryStats,
    check_memory,
    get_device,
    get_peak_memory,
)

__all__ = [
    "CheckpointCallback",
    "CheckpointMetadata",
    "EarlyStoppingCallback",
    "MemoryMonitor",
    "MemoryStats",
    "TrainingHaltError",
    "check_memory",
    "get_device",
    "get_peak_memory",
]

# Trainer and TrainResult require ML dependencies (transformers, peft, torch)
# Import them lazily to avoid import errors when only using callbacks/memory
try:
    from src.training.trainer import Trainer, TrainResult

    __all__ += ["Trainer", "TrainResult"]
except ImportError:
    pass
