"""Unit tests for the structured logging utility."""

import json
import logging
import os
from unittest.mock import patch

import pytest

from src.utils.logging import (
    ContextLogger,
    StructuredJSONFormatter,
    _get_log_level,
    get_logger,
)


class TestStructuredJSONFormatter:
    """Tests for the JSON log formatter."""

    def test_basic_format_produces_valid_json(self):
        formatter = StructuredJSONFormatter()
        record = logging.LogRecord(
            name="test.module",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=None,
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["level"] == "INFO"
        assert parsed["module"] == "test.module"
        assert parsed["message"] == "Test message"
        assert "timestamp" in parsed

    def test_timestamp_is_iso_format(self):
        formatter = StructuredJSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.DEBUG,
            pathname="",
            lineno=0,
            msg="msg",
            args=None,
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)

        # ISO 8601 format check — should end with +00:00 or Z
        timestamp = parsed["timestamp"]
        assert "T" in timestamp
        assert "+" in timestamp or "Z" in timestamp

    def test_extra_fields_included(self):
        formatter = StructuredJSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="msg",
            args=None,
            exc_info=None,
        )
        record.custom_field = "custom_value"
        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["custom_field"] == "custom_value"

    def test_context_fields_included(self):
        formatter = StructuredJSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="msg",
            args=None,
            exc_info=None,
        )
        record._context_fields = {"schema_id": "test_schema", "step": 5}
        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["schema_id"] == "test_schema"
        assert parsed["step"] == 5

    def test_exception_info_included(self):
        formatter = StructuredJSONFormatter()
        try:
            raise ValueError("test error")
        except ValueError:
            import sys

            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="error occurred",
            args=None,
            exc_info=exc_info,
        )
        output = formatter.format(record)
        parsed = json.loads(output)

        assert "exception" in parsed
        assert "ValueError: test error" in parsed["exception"]


class TestGetLogLevel:
    """Tests for log level configuration via environment variable."""

    def test_default_is_info(self):
        with patch.dict(os.environ, {}, clear=True):
            # Remove LOG_LEVEL if present
            os.environ.pop("LOG_LEVEL", None)
            assert _get_log_level() == logging.INFO

    def test_debug_level(self):
        with patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"}):
            assert _get_log_level() == logging.DEBUG

    def test_warning_level(self):
        with patch.dict(os.environ, {"LOG_LEVEL": "WARNING"}):
            assert _get_log_level() == logging.WARNING

    def test_error_level(self):
        with patch.dict(os.environ, {"LOG_LEVEL": "ERROR"}):
            assert _get_log_level() == logging.ERROR

    def test_case_insensitive(self):
        with patch.dict(os.environ, {"LOG_LEVEL": "debug"}):
            assert _get_log_level() == logging.DEBUG

    def test_invalid_level_defaults_to_info(self):
        with patch.dict(os.environ, {"LOG_LEVEL": "INVALID"}):
            assert _get_log_level() == logging.INFO


class TestGetLogger:
    """Tests for the get_logger factory function."""

    def setup_method(self):
        """Clear any existing loggers to avoid handler duplication."""
        logging.Logger.manager.loggerDict.clear()

    def test_returns_context_logger(self):
        logger = get_logger("test.module")
        assert isinstance(logger, ContextLogger)

    def test_schema_id_in_context(self, capsys):
        logger = get_logger("test.schema", schema_id="conference_talk_simple")
        logger.error("test message")

        captured = capsys.readouterr()
        parsed = json.loads(captured.err)
        assert parsed["schema_id"] == "conference_talk_simple"

    def test_step_in_context(self, capsys):
        logger = get_logger("test.step", step=42)
        logger.error("step message")

        captured = capsys.readouterr()
        parsed = json.loads(captured.err)
        assert parsed["step"] == 42

    def test_extra_context_fields(self, capsys):
        logger = get_logger("test.extra", batch_size=16, epoch=2)
        logger.error("training")

        captured = capsys.readouterr()
        parsed = json.loads(captured.err)
        assert parsed["batch_size"] == 16
        assert parsed["epoch"] == 2

    def test_with_context_creates_new_logger(self, capsys):
        logger = get_logger("test.ctx", schema_id="base")
        child = logger.with_context(step=10)

        child.error("child message")
        captured = capsys.readouterr()
        parsed = json.loads(captured.err)

        assert parsed["schema_id"] == "base"
        assert parsed["step"] == 10

    def test_log_level_respects_env(self, capsys):
        with patch.dict(os.environ, {"LOG_LEVEL": "ERROR"}):
            logger = get_logger("test.level")
            logger.logger.setLevel(_get_log_level())
            logger.info("should not appear")
            logger.error("should appear")

        captured = capsys.readouterr()
        lines = [l for l in captured.err.strip().split("\n") if l]
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["message"] == "should appear"

    def test_no_duplicate_handlers(self):
        logger1 = get_logger("test.dedup")
        logger2 = get_logger("test.dedup")
        # Both should reference the same underlying logger with one handler
        assert len(logger1.logger.handlers) == 1
        assert len(logger2.logger.handlers) == 1
