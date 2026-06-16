"""Unit tests for training callbacks (early stopping and checkpointing).

Tests cover:
- EarlyStoppingCallback: trigger/no-trigger with known loss sequences, NaN handling
- CheckpointCallback: save, retention (top-3), pruning, NaN handling, metadata
"""

import json
import math
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.training.callbacks import (
    CheckpointCallback,
    CheckpointMetadata,
    EarlyStoppingCallback,
    TrainingHaltError,
)


class TestEarlyStoppingCallback:
    """Tests for EarlyStoppingCallback."""

    def test_no_trigger_on_consistent_improvement(self):
        """Should not trigger when loss improves every interval."""
        cb = EarlyStoppingCallback(patience=3, threshold=0.001)
        losses = [1.0, 0.9, 0.8, 0.7, 0.6]
        for loss in losses:
            assert cb.check(loss) is False

    def test_triggers_after_patience_exhausted(self):
        """Should trigger after 3 consecutive intervals without improvement."""
        cb = EarlyStoppingCallback(patience=3, threshold=0.001)
        # First call sets best_loss
        assert cb.check(1.0) is False
        # Three intervals with no improvement (loss stays the same)
        assert cb.check(1.0) is False  # no_improvement_count = 1
        assert cb.check(1.0) is False  # no_improvement_count = 2
        assert cb.check(1.0) is True   # no_improvement_count = 3 → trigger

    def test_no_trigger_with_marginal_improvement_below_threshold(self):
        """Improvement below threshold should count as no-improvement."""
        cb = EarlyStoppingCallback(patience=3, threshold=0.001)
        assert cb.check(1.0) is False
        # Each step improves by 0.0001 relative to best — always below threshold
        assert cb.check(0.9999) is False  # no_improvement = 1
        assert cb.check(0.9998) is False  # no_improvement = 2
        assert cb.check(0.9997) is True   # no_improvement = 3 → trigger

    def test_resets_counter_on_sufficient_improvement(self):
        """Counter should reset when improvement exceeds threshold."""
        cb = EarlyStoppingCallback(patience=3, threshold=0.001)
        assert cb.check(1.0) is False
        assert cb.check(1.0) is False  # no_improvement = 1
        assert cb.check(1.0) is False  # no_improvement = 2
        # Big improvement resets counter
        assert cb.check(0.9) is False  # counter reset to 0
        assert cb.check(0.9) is False  # no_improvement = 1
        assert cb.check(0.9) is False  # no_improvement = 2
        assert cb.check(0.9) is True   # no_improvement = 3 → trigger

    def test_raises_on_nan_loss(self):
        """Should raise TrainingHaltError on NaN loss."""
        cb = EarlyStoppingCallback(patience=3, threshold=0.001)
        with pytest.raises(TrainingHaltError) as exc_info:
            cb.check(float("nan"))
        assert math.isnan(exc_info.value.loss_value)

    def test_raises_on_inf_loss(self):
        """Should raise TrainingHaltError on Inf loss."""
        cb = EarlyStoppingCallback(patience=3, threshold=0.001)
        with pytest.raises(TrainingHaltError) as exc_info:
            cb.check(float("inf"))
        assert math.isinf(exc_info.value.loss_value)

    def test_raises_on_negative_inf_loss(self):
        """Should raise TrainingHaltError on -Inf loss."""
        cb = EarlyStoppingCallback(patience=3, threshold=0.001)
        with pytest.raises(TrainingHaltError) as exc_info:
            cb.check(float("-inf"))
        assert math.isinf(exc_info.value.loss_value)

    def test_custom_patience(self):
        """Should respect custom patience value."""
        cb = EarlyStoppingCallback(patience=5, threshold=0.001)
        assert cb.check(1.0) is False
        for _ in range(4):
            assert cb.check(1.0) is False
        assert cb.check(1.0) is True  # 5th no-improvement → trigger

    def test_custom_threshold(self):
        """Should respect custom threshold value."""
        cb = EarlyStoppingCallback(patience=3, threshold=0.01)
        assert cb.check(1.0) is False
        # Each step improves by 0.001 relative to best — below 0.01 threshold
        assert cb.check(0.999) is False  # no_improvement = 1
        assert cb.check(0.998) is False  # no_improvement = 2
        assert cb.check(0.997) is True   # no_improvement = 3 → trigger

    def test_reset_clears_state(self):
        """Should clear all state on reset."""
        cb = EarlyStoppingCallback(patience=3, threshold=0.001)
        cb.check(1.0)
        cb.check(1.0)  # no_improvement = 1
        cb.reset()
        assert cb.best_loss == float("inf")
        assert cb.no_improvement_count == 0

    def test_exact_threshold_improvement_counts(self):
        """Improvement exactly equal to threshold should count as improvement."""
        cb = EarlyStoppingCallback(patience=3, threshold=0.001)
        assert cb.check(1.0) is False
        # Exactly 0.001 improvement — should reset counter
        assert cb.check(0.999) is False
        assert cb.no_improvement_count == 0


class TestCheckpointCallback:
    """Tests for CheckpointCallback."""

    def test_save_creates_checkpoint_directory(self, tmp_path):
        """Should create a checkpoint directory with model files."""
        model = MagicMock()
        cb = CheckpointCallback(checkpoint_dir=tmp_path, max_checkpoints=3)

        path = cb.save(step=100, val_loss=0.5, train_loss=0.6, model=model, config_hash="abc123")

        assert path.exists()
        assert path.name == "checkpoint-step-100"
        model.save_pretrained.assert_called_once_with(str(path))

    def test_save_writes_metadata_json(self, tmp_path):
        """Should write metadata.json with all required fields."""
        model = MagicMock()
        cb = CheckpointCallback(checkpoint_dir=tmp_path, max_checkpoints=3)

        path = cb.save(step=200, val_loss=0.4, train_loss=0.5, model=model, config_hash="hash456")

        metadata_path = path / "metadata.json"
        assert metadata_path.exists()
        metadata = json.loads(metadata_path.read_text())
        assert metadata["step"] == 200
        assert metadata["val_loss"] == 0.4
        assert metadata["train_loss"] == 0.5
        assert metadata["config_hash"] == "hash456"
        assert "timestamp" in metadata

    def test_retains_only_top_3_checkpoints(self, tmp_path):
        """Should keep only the 3 checkpoints with lowest val_loss."""
        model = MagicMock()
        cb = CheckpointCallback(checkpoint_dir=tmp_path, max_checkpoints=3)

        # Save 5 checkpoints with varying val_loss
        cb.save(step=100, val_loss=0.5, train_loss=0.6, model=model, config_hash="h1")
        cb.save(step=200, val_loss=0.3, train_loss=0.4, model=model, config_hash="h1")
        cb.save(step=300, val_loss=0.7, train_loss=0.8, model=model, config_hash="h1")
        cb.save(step=400, val_loss=0.2, train_loss=0.3, model=model, config_hash="h1")
        cb.save(step=500, val_loss=0.4, train_loss=0.5, model=model, config_hash="h1")

        # Should retain steps 200 (0.3), 400 (0.2), 500 (0.4) — top 3 lowest
        assert len(cb.checkpoints) == 3
        retained_steps = {m.step for m in cb.checkpoints}
        assert retained_steps == {200, 400, 500}

        # Pruned directories should not exist
        assert not (tmp_path / "checkpoint-step-100").exists()
        assert not (tmp_path / "checkpoint-step-300").exists()

        # Retained directories should exist
        assert (tmp_path / "checkpoint-step-200").exists()
        assert (tmp_path / "checkpoint-step-400").exists()
        assert (tmp_path / "checkpoint-step-500").exists()

    def test_get_best_returns_lowest_val_loss_checkpoint(self, tmp_path):
        """Should return path to checkpoint with lowest val_loss."""
        model = MagicMock()
        cb = CheckpointCallback(checkpoint_dir=tmp_path, max_checkpoints=3)

        cb.save(step=100, val_loss=0.5, train_loss=0.6, model=model, config_hash="h1")
        cb.save(step=200, val_loss=0.2, train_loss=0.3, model=model, config_hash="h1")
        cb.save(step=300, val_loss=0.4, train_loss=0.5, model=model, config_hash="h1")

        best = cb.get_best()
        assert best == tmp_path / "checkpoint-step-200"

    def test_get_best_returns_none_when_empty(self, tmp_path):
        """Should return None when no checkpoints exist."""
        cb = CheckpointCallback(checkpoint_dir=tmp_path, max_checkpoints=3)
        assert cb.get_best() is None

    def test_raises_on_nan_val_loss(self, tmp_path):
        """Should raise TrainingHaltError on NaN val_loss."""
        model = MagicMock()
        cb = CheckpointCallback(checkpoint_dir=tmp_path, max_checkpoints=3)

        # Save a good checkpoint first
        cb.save(step=100, val_loss=0.5, train_loss=0.6, model=model, config_hash="h1")

        with pytest.raises(TrainingHaltError) as exc_info:
            cb.save(step=200, val_loss=float("nan"), train_loss=0.6, model=model, config_hash="h1")

        assert exc_info.value.step == 200
        assert exc_info.value.last_good_checkpoint == tmp_path / "checkpoint-step-100"

    def test_raises_on_inf_val_loss(self, tmp_path):
        """Should raise TrainingHaltError on Inf val_loss."""
        model = MagicMock()
        cb = CheckpointCallback(checkpoint_dir=tmp_path, max_checkpoints=3)

        with pytest.raises(TrainingHaltError) as exc_info:
            cb.save(step=100, val_loss=float("inf"), train_loss=0.6, model=model, config_hash="h1")

        assert exc_info.value.step == 100
        assert exc_info.value.last_good_checkpoint is None  # No prior checkpoint

    def test_load_existing_reconstructs_state(self, tmp_path):
        """Should load existing checkpoints from disk on resume."""
        model = MagicMock()
        cb1 = CheckpointCallback(checkpoint_dir=tmp_path, max_checkpoints=3)

        # Save some checkpoints
        cb1.save(step=100, val_loss=0.5, train_loss=0.6, model=model, config_hash="h1")
        cb1.save(step=200, val_loss=0.3, train_loss=0.4, model=model, config_hash="h1")

        # Create a new callback and load existing
        cb2 = CheckpointCallback(checkpoint_dir=tmp_path, max_checkpoints=3)
        cb2.load_existing()

        assert len(cb2.checkpoints) == 2
        assert cb2.checkpoints[0].val_loss == 0.3  # sorted ascending
        assert cb2.checkpoints[1].val_loss == 0.5

    def test_get_all_returns_sorted_list(self, tmp_path):
        """Should return all checkpoints sorted by val_loss ascending."""
        model = MagicMock()
        cb = CheckpointCallback(checkpoint_dir=tmp_path, max_checkpoints=3)

        cb.save(step=100, val_loss=0.5, train_loss=0.6, model=model, config_hash="h1")
        cb.save(step=200, val_loss=0.2, train_loss=0.3, model=model, config_hash="h1")
        cb.save(step=300, val_loss=0.4, train_loss=0.5, model=model, config_hash="h1")

        all_ckpts = cb.get_all()
        assert len(all_ckpts) == 3
        assert all_ckpts[0].val_loss == 0.2
        assert all_ckpts[1].val_loss == 0.4
        assert all_ckpts[2].val_loss == 0.5

    def test_creates_checkpoint_dir_if_not_exists(self, tmp_path):
        """Should create the checkpoint directory if it doesn't exist."""
        new_dir = tmp_path / "new_checkpoints" / "nested"
        cb = CheckpointCallback(checkpoint_dir=new_dir, max_checkpoints=3)
        assert new_dir.exists()


class TestCheckpointMetadata:
    """Tests for CheckpointMetadata dataclass."""

    def test_to_dict_roundtrip(self):
        """Should serialize and deserialize correctly."""
        meta = CheckpointMetadata(
            step=100,
            val_loss=0.5,
            train_loss=0.6,
            timestamp="2024-01-01T00:00:00+0000",
            config_hash="abc123",
        )
        data = meta.to_dict()
        restored = CheckpointMetadata.from_dict(data)
        assert restored.step == meta.step
        assert restored.val_loss == meta.val_loss
        assert restored.train_loss == meta.train_loss
        assert restored.timestamp == meta.timestamp
        assert restored.config_hash == meta.config_hash


class TestTrainingHaltError:
    """Tests for TrainingHaltError."""

    def test_str_representation(self):
        """Should produce a descriptive string."""
        err = TrainingHaltError(
            step=500,
            loss_value=float("nan"),
            last_good_checkpoint=Path("/checkpoints/step-400"),
        )
        msg = str(err)
        assert "500" in msg
        assert "nan" in msg
        assert "step-400" in msg

    def test_is_exception(self):
        """Should be a proper exception subclass."""
        err = TrainingHaltError(step=100, loss_value=float("inf"))
        assert isinstance(err, Exception)
