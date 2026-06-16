"""Unit tests for GPU memory monitoring utilities.

Tests cover both the CUDA-available and CUDA-unavailable (CPU fallback) paths.
"""

import sys
from io import StringIO
from unittest.mock import patch, MagicMock

import pytest

from src.training.memory import (
    MemoryMonitor,
    MemoryStats,
    check_memory,
    get_device,
    get_peak_memory,
    _is_cuda_available,
)


class TestGetDevice:
    """Tests for the get_device() function."""

    def test_returns_cpu_when_cuda_unavailable(self):
        """Should return 'cpu' when CUDA is not available."""
        with patch("src.training.memory._is_cuda_available", return_value=False):
            device = get_device()
            assert device == "cpu"

    def test_returns_cuda_when_available(self):
        """Should return 'cuda' when CUDA is available."""
        with patch("src.training.memory._is_cuda_available", return_value=True):
            device = get_device()
            assert device == "cuda"

    def test_logs_warning_to_stderr_when_no_cuda(self, capsys):
        """Should print a warning to stderr when falling back to CPU."""
        with patch("src.training.memory._is_cuda_available", return_value=False):
            get_device()
            captured = capsys.readouterr()
            assert "WARNING" in captured.err
            assert "CUDA is not available" in captured.err
            assert "CPU" in captured.err


class TestCheckMemory:
    """Tests for the check_memory() function."""

    def test_returns_dummy_values_when_no_cuda(self):
        """Should return zeros when CUDA is unavailable."""
        with patch("src.training.memory._is_cuda_available", return_value=False):
            result = check_memory()
            assert result == {
                "allocated_gb": 0.0,
                "reserved_gb": 0.0,
                "peak_gb": 0.0,
            }

    def test_returns_dict_with_required_keys(self):
        """Should return dict with allocated_gb, reserved_gb, peak_gb."""
        with patch("src.training.memory._is_cuda_available", return_value=False):
            result = check_memory()
            assert "allocated_gb" in result
            assert "reserved_gb" in result
            assert "peak_gb" in result

    def test_returns_real_values_when_cuda_available(self):
        """Should query torch.cuda when CUDA is available."""
        mock_torch = MagicMock()
        mock_torch.cuda.memory_allocated.return_value = 2 * (1024**3)  # 2 GB
        mock_torch.cuda.memory_reserved.return_value = 4 * (1024**3)  # 4 GB
        mock_torch.cuda.max_memory_allocated.return_value = 3 * (1024**3)  # 3 GB

        with patch("src.training.memory._is_cuda_available", return_value=True):
            with patch.dict("sys.modules", {"torch": mock_torch}):
                result = check_memory()
                assert result["allocated_gb"] == 2.0
                assert result["reserved_gb"] == 4.0
                assert result["peak_gb"] == 3.0


class TestGetPeakMemory:
    """Tests for the get_peak_memory() function."""

    def test_returns_zero_when_no_cuda(self):
        """Should return 0.0 when CUDA is unavailable."""
        with patch("src.training.memory._is_cuda_available", return_value=False):
            result = get_peak_memory()
            assert result == 0.0

    def test_returns_peak_from_torch_when_cuda_available(self):
        """Should return torch.cuda.max_memory_allocated value in GB."""
        mock_torch = MagicMock()
        mock_torch.cuda.max_memory_allocated.return_value = 8 * (1024**3)  # 8 GB

        with patch("src.training.memory._is_cuda_available", return_value=True):
            with patch.dict("sys.modules", {"torch": mock_torch}):
                result = get_peak_memory()
                assert result == 8.0


class TestMemoryMonitor:
    """Tests for the MemoryMonitor class."""

    def test_init_sets_cuda_available_flag(self):
        """Should set cuda_available based on system state."""
        with patch("src.training.memory._is_cuda_available", return_value=False):
            monitor = MemoryMonitor()
            assert monitor.cuda_available is False

        with patch("src.training.memory._is_cuda_available", return_value=True):
            monitor = MemoryMonitor()
            assert monitor.cuda_available is True

    def test_init_warns_when_no_cuda(self, capsys):
        """Should warn to stderr when CUDA is unavailable."""
        with patch("src.training.memory._is_cuda_available", return_value=False):
            monitor = MemoryMonitor()
            captured = capsys.readouterr()
            assert "WARNING" in captured.err
            assert "CUDA is not available" in captured.err

    def test_check_returns_zero_stats_when_no_cuda(self):
        """Should return zeroed MemoryStats when CUDA is unavailable."""
        with patch("src.training.memory._is_cuda_available", return_value=False):
            monitor = MemoryMonitor()
            stats = monitor.check()
            assert stats.allocated_gb == 0.0
            assert stats.reserved_gb == 0.0
            assert stats.peak_gb == 0.0

    def test_check_appends_to_history(self):
        """Should track snapshots in history."""
        with patch("src.training.memory._is_cuda_available", return_value=False):
            monitor = MemoryMonitor()
            monitor.check()
            monitor.check()
            assert len(monitor.history) == 2

    def test_check_with_cuda_queries_torch(self):
        """Should query torch.cuda and build correct MemoryStats."""
        mock_torch = MagicMock()
        mock_torch.cuda.memory_allocated.return_value = 5 * (1024**3)
        mock_torch.cuda.memory_reserved.return_value = 8 * (1024**3)
        mock_torch.cuda.max_memory_allocated.return_value = 6 * (1024**3)

        with patch("src.training.memory._is_cuda_available", return_value=True):
            monitor = MemoryMonitor()
            with patch.dict("sys.modules", {"torch": mock_torch}):
                stats = monitor.check()
                assert stats.allocated_gb == 5.0
                assert stats.reserved_gb == 8.0
                assert stats.peak_gb == 6.0

    def test_check_updates_peak_observed(self):
        """Should track the highest allocated memory across checks."""
        mock_torch = MagicMock()

        with patch("src.training.memory._is_cuda_available", return_value=True):
            monitor = MemoryMonitor()

            # First check: 5 GB
            mock_torch.cuda.memory_allocated.return_value = 5 * (1024**3)
            mock_torch.cuda.memory_reserved.return_value = 8 * (1024**3)
            mock_torch.cuda.max_memory_allocated.return_value = 5 * (1024**3)
            with patch.dict("sys.modules", {"torch": mock_torch}):
                monitor.check()
            assert monitor.peak_observed_gb == 5.0

            # Second check: 10 GB — peak should update
            mock_torch.cuda.memory_allocated.return_value = 10 * (1024**3)
            mock_torch.cuda.memory_reserved.return_value = 12 * (1024**3)
            mock_torch.cuda.max_memory_allocated.return_value = 10 * (1024**3)
            with patch.dict("sys.modules", {"torch": mock_torch}):
                monitor.check()
            assert monitor.peak_observed_gb == 10.0

            # Third check: 3 GB — peak should NOT change
            mock_torch.cuda.memory_allocated.return_value = 3 * (1024**3)
            mock_torch.cuda.memory_reserved.return_value = 5 * (1024**3)
            mock_torch.cuda.max_memory_allocated.return_value = 10 * (1024**3)
            with patch.dict("sys.modules", {"torch": mock_torch}):
                monitor.check()
            assert monitor.peak_observed_gb == 10.0

    def test_check_and_enforce_raises_on_limit_exceeded(self):
        """Should raise MemoryError when memory exceeds limit."""
        mock_torch = MagicMock()
        mock_torch.cuda.memory_allocated.return_value = 25 * (1024**3)  # 25 GB
        mock_torch.cuda.memory_reserved.return_value = 26 * (1024**3)
        mock_torch.cuda.max_memory_allocated.return_value = 25 * (1024**3)

        with patch("src.training.memory._is_cuda_available", return_value=True):
            monitor = MemoryMonitor()
            with patch.dict("sys.modules", {"torch": mock_torch}):
                with pytest.raises(MemoryError) as exc_info:
                    monitor.check_and_enforce(max_memory_gb=24.0)
                assert "25.00 GB allocated" in str(exc_info.value)
                assert "limit is 24.00 GB" in str(exc_info.value)
                assert "Peak usage observed" in str(exc_info.value)

    def test_check_and_enforce_passes_within_limit(self):
        """Should not raise when memory is within limit."""
        mock_torch = MagicMock()
        mock_torch.cuda.memory_allocated.return_value = 20 * (1024**3)  # 20 GB
        mock_torch.cuda.memory_reserved.return_value = 22 * (1024**3)
        mock_torch.cuda.max_memory_allocated.return_value = 20 * (1024**3)

        with patch("src.training.memory._is_cuda_available", return_value=True):
            monitor = MemoryMonitor()
            with patch.dict("sys.modules", {"torch": mock_torch}):
                stats = monitor.check_and_enforce(max_memory_gb=24.0)
                assert stats.allocated_gb == 20.0

    def test_check_and_enforce_no_op_when_no_cuda(self):
        """Should not raise on CPU fallback (dummy values are 0.0)."""
        with patch("src.training.memory._is_cuda_available", return_value=False):
            monitor = MemoryMonitor()
            stats = monitor.check_and_enforce(max_memory_gb=24.0)
            assert stats.allocated_gb == 0.0

    def test_get_peak_returns_peak_observed(self):
        """Should return the tracked peak value."""
        with patch("src.training.memory._is_cuda_available", return_value=False):
            monitor = MemoryMonitor()
            assert monitor.get_peak() == 0.0

    def test_reset_clears_state(self):
        """Should clear history and reset peak on reset()."""
        with patch("src.training.memory._is_cuda_available", return_value=False):
            monitor = MemoryMonitor()
            monitor.check()
            monitor.check()
            monitor.peak_observed_gb = 5.0

            monitor.reset()
            assert len(monitor.history) == 0
            assert monitor.peak_observed_gb == 0.0

    def test_reset_calls_torch_reset_when_cuda(self):
        """Should call torch.cuda.reset_peak_memory_stats on reset."""
        mock_torch = MagicMock()

        with patch("src.training.memory._is_cuda_available", return_value=True):
            monitor = MemoryMonitor()
            with patch.dict("sys.modules", {"torch": mock_torch}):
                monitor.reset()
                mock_torch.cuda.reset_peak_memory_stats.assert_called_once()

    def test_summary_returns_correct_format(self):
        """Should return summary dict with expected keys."""
        with patch("src.training.memory._is_cuda_available", return_value=False):
            monitor = MemoryMonitor()
            monitor.check()
            summary = monitor.summary()
            assert summary == {
                "peak_allocated_gb": 0.0,
                "num_snapshots": 1,
                "cuda_available": False,
            }

    def test_custom_memory_limit(self):
        """Should enforce custom memory limit."""
        mock_torch = MagicMock()
        mock_torch.cuda.memory_allocated.return_value = 10 * (1024**3)  # 10 GB
        mock_torch.cuda.memory_reserved.return_value = 12 * (1024**3)
        mock_torch.cuda.max_memory_allocated.return_value = 10 * (1024**3)

        with patch("src.training.memory._is_cuda_available", return_value=True):
            monitor = MemoryMonitor()
            with patch.dict("sys.modules", {"torch": mock_torch}):
                # Should pass with 12 GB limit
                stats = monitor.check_and_enforce(max_memory_gb=12.0)
                assert stats.allocated_gb == 10.0

                # Should fail with 8 GB limit
                with pytest.raises(MemoryError):
                    monitor.check_and_enforce(max_memory_gb=8.0)
