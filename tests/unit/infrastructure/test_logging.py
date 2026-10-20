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
