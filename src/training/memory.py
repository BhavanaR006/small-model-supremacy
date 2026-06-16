"""GPU memory monitoring utilities for the training pipeline.

Provides memory tracking, peak usage reporting, and enforcement of memory limits
to ensure training stays within consumer GPU constraints (24GB default).

Falls back gracefully to CPU with a warning when CUDA is not available.

Usage:
    from src.training.memory import MemoryMonitor, get_device, check_memory

    device = get_device()
    monitor = MemoryMonitor()
    monitor.check_and_enforce(max_memory_gb=24.0)
"""

import sys
from dataclasses import dataclass, field
from typing import Optional

from src.utils.logging import get_logger

logger = get_logger(__name__)


def _is_cuda_available() -> bool:
    """Check if CUDA is available via PyTorch."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def get_device() -> str:
    """Return the appropriate device string for training.

    Returns "cuda" if a CUDA-capable GPU is available, otherwise "cpu"
    with a warning logged to stderr indicating training will be slow.

    Returns:
        Device string: "cuda" or "cpu".
    """
    if _is_cuda_available():
        return "cuda"

    print(
        "WARNING: CUDA is not available. Falling back to CPU. "
        "Training will be impractically slow.",
        file=sys.stderr,
    )
    logger.warning(
        "CUDA is not available. Falling back to CPU. "
        "Training will be impractically slow."
    )
    return "cpu"


@dataclass
class MemoryStats:
    """Current GPU memory usage statistics in GB."""

    allocated_gb: float
    reserved_gb: float
    peak_gb: float


def check_memory() -> dict:
    """Return current GPU memory usage in GB.

    Returns a dict with keys:
    - allocated_gb: Currently allocated memory
    - reserved_gb: Total memory reserved by the allocator
    - peak_gb: Peak allocated memory since last reset

    If CUDA is not available, returns dummy values (0.0) for all fields.

    Returns:
        Dict with allocated_gb, reserved_gb, and peak_gb.
    """
    if not _is_cuda_available():
        return {
            "allocated_gb": 0.0,
            "reserved_gb": 0.0,
            "peak_gb": 0.0,
        }

    import torch

    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    peak = torch.cuda.max_memory_allocated() / (1024**3)

    return {
        "allocated_gb": round(allocated, 4),
        "reserved_gb": round(reserved, 4),
        "peak_gb": round(peak, 4),
    }


def get_peak_memory() -> float:
    """Return the maximum allocated GPU memory observed during training in GB.

    This reports the peak memory allocated since the last call to
    `torch.cuda.reset_peak_memory_stats()` or since training started.

    If CUDA is not available, returns 0.0.

    Returns:
        Peak allocated memory in GB.
    """
    if not _is_cuda_available():
        return 0.0

    import torch

    return round(torch.cuda.max_memory_allocated() / (1024**3), 4)


class MemoryMonitor:
    """Tracks GPU memory usage over time and enforces memory limits.

    Attributes:
        cuda_available: Whether CUDA is available on the system.
        history: List of MemoryStats snapshots taken during training.
        peak_observed_gb: The highest allocated memory observed across all checks.
    """

    def __init__(self) -> None:
        self.cuda_available: bool = _is_cuda_available()
        self.history: list[MemoryStats] = []
        self.peak_observed_gb: float = 0.0

        if not self.cuda_available:
            print(
                "WARNING: CUDA is not available. MemoryMonitor will report "
                "dummy values (0.0). Training will run on CPU.",
                file=sys.stderr,
            )
            logger.warning(
                "CUDA is not available. MemoryMonitor will report dummy values. "
                "Training will run on CPU."
            )

    def check(self) -> MemoryStats:
        """Take a memory usage snapshot.

        Records current GPU memory statistics and updates peak observed value.
        If CUDA is unavailable, returns zeroed stats.

        Returns:
            MemoryStats with current allocated, reserved, and peak memory in GB.
        """
        if not self.cuda_available:
            stats = MemoryStats(
                allocated_gb=0.0,
                reserved_gb=0.0,
                peak_gb=0.0,
            )
            self.history.append(stats)
            return stats

        import torch

        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        peak = torch.cuda.max_memory_allocated() / (1024**3)

        stats = MemoryStats(
            allocated_gb=round(allocated, 4),
            reserved_gb=round(reserved, 4),
            peak_gb=round(peak, 4),
        )

        self.history.append(stats)

        # Update peak observed
        if stats.allocated_gb > self.peak_observed_gb:
            self.peak_observed_gb = stats.allocated_gb

        return stats

    def check_and_enforce(self, max_memory_gb: float = 24.0) -> MemoryStats:
        """Check memory usage and enforce the memory limit.

        Takes a memory snapshot and raises MemoryError if the currently
        allocated memory exceeds the specified limit.

        Args:
            max_memory_gb: Maximum allowed GPU memory in GB. Defaults to 24.0.

        Returns:
            MemoryStats with current usage.

        Raises:
            MemoryError: If allocated GPU memory exceeds max_memory_gb,
                with a message reporting peak usage and the limit.
        """
        stats = self.check()

        if not self.cuda_available:
            return stats

        if stats.allocated_gb > max_memory_gb:
            error_msg = (
                f"GPU memory limit exceeded: {stats.allocated_gb:.2f} GB allocated, "
                f"limit is {max_memory_gb:.2f} GB. "
                f"Peak usage observed: {self.peak_observed_gb:.2f} GB. "
                f"Terminating training run."
            )
            logger.error(error_msg)
            raise MemoryError(error_msg)

        return stats

    def get_peak(self) -> float:
        """Return the highest allocated memory observed across all checks in GB.

        Returns:
            Peak observed allocated memory in GB, or 0.0 if no checks have been
            performed or CUDA is unavailable.
        """
        return self.peak_observed_gb

    def reset(self) -> None:
        """Reset monitoring state and CUDA peak memory statistics.

        Clears history, resets peak observed value, and resets the CUDA
        peak memory tracking.
        """
        self.history.clear()
        self.peak_observed_gb = 0.0

        if self.cuda_available:
            import torch

            torch.cuda.reset_peak_memory_stats()

    def summary(self) -> dict:
        """Return a summary of memory usage during the monitoring session.

        Returns:
            Dict with peak_allocated_gb, num_snapshots, and cuda_available.
        """
        return {
            "peak_allocated_gb": self.peak_observed_gb,
            "num_snapshots": len(self.history),
            "cuda_available": self.cuda_available,
        }
