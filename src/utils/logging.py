"""Structured JSON logging utility for the Small Model Supremacy project.

Provides structured logging with JSON output for machine-readable logs.
Log level is configurable via the LOG_LEVEL environment variable
(DEBUG, INFO, WARNING, ERROR). Default is INFO.

Usage:
    from src.utils.logging import get_logger

    logger = get_logger(__name__, schema_id="conference_talk_simple")
    logger.info("Processing example", extra={"step": 1, "total": 100})
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Optional


class StructuredJSONFormatter(logging.Formatter):
    """Formats log records as structured JSON lines.

    Each log line includes:
    - timestamp (ISO 8601 UTC)
    - level
    - module
    - message
    - Any additional context fields passed via `extra`
    """

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "module": record.name,
            "message": record.getMessage(),
        }

        # Include context fields injected by ContextLogger or passed via extra
        context_fields = getattr(record, "_context_fields", {})
        if context_fields:
            log_entry.update(context_fields)

        # Include any extra fields passed directly in the log call
        # Standard LogRecord attributes to exclude from extras
        standard_attrs = {
            "name", "msg", "args", "created", "relativeCreated",
            "exc_info", "exc_text", "stack_info", "lineno", "funcName",
            "filename", "module", "pathname", "thread", "threadName",
            "process", "processName", "levelname", "levelno",
            "msecs", "message", "taskName", "_context_fields",
        }
        for key, value in record.__dict__.items():
            if key not in standard_attrs and not key.startswith("_"):
                log_entry[key] = value

        # Include exception info if present
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)


class ContextLogger(logging.LoggerAdapter):
    """Logger adapter that injects persistent context fields into every log record.

    Context fields (e.g., schema_id, step) are included automatically in all
    log entries produced by this logger instance.
    """

    def process(
        self, msg: str, kwargs: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        # Merge adapter context into the record via a custom attribute
        extra = kwargs.get("extra", {})
        extra["_context_fields"] = {**self.extra, **extra.pop("_context_fields", {})}
        kwargs["extra"] = extra
        return msg, kwargs

    def with_context(self, **kwargs: Any) -> "ContextLogger":
        """Create a new logger with additional context fields.

        Returns a new ContextLogger instance with merged context,
        useful for adding step-specific or schema-specific context.
        """
        merged = {**self.extra, **kwargs}
        return ContextLogger(self.logger, merged)


def _get_log_level() -> int:
    """Read log level from LOG_LEVEL environment variable.

    Supported values: DEBUG, INFO, WARNING, ERROR.
    Defaults to INFO if unset or invalid.
    """
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }
    return level_map.get(level_name, logging.INFO)


def get_logger(
    name: str,
    schema_id: Optional[str] = None,
    step: Optional[int] = None,
    **extra_context: Any,
) -> ContextLogger:
    """Create a structured JSON logger with optional context fields.

    Args:
        name: Logger name, typically __name__ of the calling module.
        schema_id: Optional schema identifier for context.
        step: Optional step number for context.
        **extra_context: Additional key-value pairs to include in every log entry.

    Returns:
        A ContextLogger instance that outputs structured JSON to stderr.

    Example:
        logger = get_logger(__name__, schema_id="product_listing_medium")
        logger.info("Validating example", extra={"example_idx": 42})
    """
    logger = logging.getLogger(name)

    # Only add handler if the logger doesn't already have one
    # (avoids duplicate handlers on repeated calls)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(StructuredJSONFormatter())
        logger.addHandler(handler)

    logger.setLevel(_get_log_level())

    # Build context fields
    context: dict[str, Any] = {}
    if schema_id is not None:
        context["schema_id"] = schema_id
    if step is not None:
        context["step"] = step
    context.update(extra_context)

    return ContextLogger(logger, context)
