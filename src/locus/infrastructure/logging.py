"""
logging.py

Structured JSON logging. Every record carries a fixed set of fields
(timestamp, level, component, event) plus optional domain fields
(asset, model, run_id) and any extra key-value pairs supplied by the
caller through `logger.info(..., extra={...})`.

Author: Marco Pérez Padilla
Date:   20-10-2026
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any


class StructuredFormatter(logging.Formatter):
    """Format `logging.LogRecord` instances as single-line JSON objects.

    Reserved LogRecord attributes are stripped out; everything else is
    preserved as top-level keys. Exceptions are serialised under the
    `exception` key so downstream tooling can parse them uniformly.
    """

    RESERVED: set[str] = {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "message",
    }

    def format(self, record: logging.LogRecord) -> str:
        """Render a log record as a JSON string.

        Fixed fields are always present; domain fields and extra
        attributes are included only when the caller supplied them.
        """
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "component": getattr(record, "component", record.name),
            "event": record.getMessage(),
        }

        for field in ("asset", "model", "run_id"):
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value

        for key, value in record.__dict__.items():
            if key not in self.RESERVED and not key.startswith("_"):
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False, default=str)


class LoggerFactory:
    """Build and cache structured loggers.

    Configuration is applied once (idempotently) to the root logger so
    every subsequent `get` call returns a logger writing JSON lines to
    stdout at the requested level.
    """

    _configured = False

    @classmethod
    def configure(cls, level: str = "INFO") -> None:
        """Attach a single JSON handler to the root logger at the given
        level. Subsequent calls are no-ops, which makes this safe to call
        from anywhere in the codebase without double handlers.
        """
        if cls._configured:
            return
        root = logging.getLogger()
        root.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(StructuredFormatter())
        root.addHandler(handler)
        cls._configured = True

    @classmethod
    def get(cls, component: str) -> logging.Logger:
        """Return a logger named after the given component, configuring
        the root handler on first use if it has not been configured yet.
        """
        cls.configure()
        logger = logging.getLogger(component)
        logger.setLevel(logging.getLogger().level)
        return logger

    @classmethod
    def set_level(cls, level: str) -> None:
        """Change the level of the root logger and every logger already
        created. Useful for CLI scripts that want to suppress INFO logs
        after configuration has happened."""
        numeric = getattr(logging, level.upper(), None)
        if not isinstance(numeric, int):
            raise ValueError(f"unknown log level: {level}")
        logging.getLogger().setLevel(numeric)
        for name in list(logging.Logger.manager.loggerDict):
            log = logging.getLogger(name)
            if isinstance(log, logging.Logger):
                log.setLevel(numeric)
