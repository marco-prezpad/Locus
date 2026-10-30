"""
test_logging.py

Verifies the structured JSON formatter emits the required fixed fields
(timestamp, level, component, event) plus any domain fields and extras
supplied by the caller.

Author: Marco Pérez Padilla
Date:   20-10-2026
"""

from __future__ import annotations

import json
import logging

import pytest

from locus.infrastructure.logging import LoggerFactory, StructuredFormatter


def test_structured_formatter_emits_json(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    LoggerFactory._configured = False
    log = LoggerFactory.get("test.component")
    log.info("test_event", extra={"asset": "BTC", "run_id": "run-1"})

    assert len(caplog.records) == 1
    formatter = StructuredFormatter()
    payload = json.loads(formatter.format(caplog.records[0]))

    assert payload["level"] == "INFO"
    assert payload["component"] == "test.component"
    assert payload["event"] == "test_event"
    assert payload["asset"] == "BTC"
    assert payload["run_id"] == "run-1"
    assert "timestamp" in payload


def test_structured_formatter_includes_extras(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    LoggerFactory._configured = False
    log = LoggerFactory.get("test.component")
    log.info("job_done", extra={"duration_ms": 1234, "status": "ok"})

    payload = json.loads(StructuredFormatter().format(caplog.records[0]))

    assert payload["duration_ms"] == 1234
    assert payload["status"] == "ok"


# --- set_level (lines 123-133) ---


def test_set_level_before_configure_configures_with_level(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """If set_level is called before any get(), it configures the layer."""
    LoggerFactory._configured = False
    LoggerFactory.set_level("DEBUG")
    assert LoggerFactory._configured is True
    assert logging.getLogger().level == logging.DEBUG


def test_set_level_after_configure_updates_root_and_children(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """After configure, set_level updates root and every registered logger."""
    LoggerFactory._configured = False
    log = LoggerFactory.get("test.child")
    LoggerFactory.set_level("WARNING")
    assert logging.getLogger().level == logging.WARNING
    assert log.level == logging.WARNING


def test_set_level_invalid_raises() -> None:
    LoggerFactory._configured = False
    with pytest.raises(ValueError, match="unknown log level"):
        LoggerFactory.set_level("NOT_A_LEVEL")


def test_set_level_case_insensitive(
    caplog: pytest.LogCaptureFixture,
) -> None:
    LoggerFactory._configured = False
    LoggerFactory.set_level("debug")  # lowercase
    assert logging.getLogger().level == logging.DEBUG


# --- format: exception and non-serializable values (line 77) ---


def test_formatter_serializes_exception(caplog: pytest.LogCaptureFixture) -> None:
    """exc_info is rendered under `exception` key."""
    caplog.set_level(logging.ERROR)
    LoggerFactory._configured = False
    log = LoggerFactory.get("test.exc")
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        log.exception("failed_event")

    record = caplog.records[0]
    payload = json.loads(StructuredFormatter().format(record))
    assert "exception" in payload
    assert "RuntimeError: boom" in payload["exception"]
    assert payload["event"] == "failed_event"


def test_formatter_serializes_non_serializable_extra(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Extras that are not JSON-native fall back to str() via default=str."""
    from datetime import UTC, datetime

    caplog.set_level(logging.INFO)
    LoggerFactory._configured = False
    log = LoggerFactory.get("test.default_str")
    when = datetime(2026, 10, 30, 12, 0, 0, tzinfo=UTC)
    log.info("event_with_datetime", extra={"when": when})

    payload = json.loads(StructuredFormatter().format(caplog.records[0]))
    # datetime is serialized as its str representation, not raising
    assert "2026-10-30" in payload["when"]


def test_formatter_includes_model_field(caplog: pytest.LogCaptureFixture) -> None:
    """`model` domain field is emitted when supplied (only asset/run_id
    are covered by the existing tests)."""
    caplog.set_level(logging.INFO)
    LoggerFactory._configured = False
    log = LoggerFactory.get("test.model")
    log.info("fit_started", extra={"model": "M1_logistic_l2"})

    payload = json.loads(StructuredFormatter().format(caplog.records[0]))
    assert payload["model"] == "M1_logistic_l2"


def test_formatter_omits_null_domain_fields(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """asset/model/run_id are omitted if not provided (getattr returns None)."""
    caplog.set_level(logging.INFO)
    LoggerFactory._configured = False
    log = LoggerFactory.get("test.no_domain")
    log.info("simple_event")

    payload = json.loads(StructuredFormatter().format(caplog.records[0]))
    assert "asset" not in payload
    assert "model" not in payload
    assert "run_id" not in payload


def test_formatter_strips_underscore_keys(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Keys starting with `_` are not propagated to the payload."""
    caplog.set_level(logging.INFO)
    LoggerFactory._configured = False
    log = LoggerFactory.get("test.private")
    log.info("event", extra={"_private": "secret", "public": "visible"})

    payload = json.loads(StructuredFormatter().format(caplog.records[0]))
    assert "_private" not in payload
    assert payload["public"] == "visible"
