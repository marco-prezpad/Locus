"""Tests for the P1.c microstructure capture module."""

from __future__ import annotations

from pathlib import Path

import pytest
from src.locus.data.capture.microstructure import GAP_FACTOR

from locus.data.capture.microstructure import (
    DEPTH_CADENCE_MS,
    TRADE_CADENCE_MS,
    Capture,
)


@pytest.fixture
def cap(tmp_path: Path) -> Capture:
    cap = Capture(tmp_path)
    cap._gaps_file.touch()  # noqa: SLF001
    return cap


# --- parsing ---


def test_parse_depth_valid(cap: Capture) -> None:
    data = {
        "s": "BTCUSDT",
        "E": 1_700_000_000_000,
        "b": [["100.0", "1.0"]],
        "a": [["101.0", "1.0"]],
    }
    recv_ns = 1_700_000_000_050 * 1_000_000
    row = cap.parse_depth("BTCUSDT", data, recv_ns)
    assert row is not None
    assert row["latency_ms"] == pytest.approx(50.0, abs=1e-3)
    assert "100.0" in row["bids_json"]
    assert "101.0" in row["asks_json"]


def test_parse_depth_missing_event_time(cap: Capture) -> None:
    row = cap.parse_depth("BTCUSDT", {"s": "BTCUSDT"}, 0)
    assert row is None


def test_parse_trade_valid(cap: Capture) -> None:
    data = {
        "s": "ETHUSDT",
        "E": 1_700_000_000_000,
        "p": "2000.5",
        "q": "0.01",
        "m": True,
    }
    recv_ns = 1_700_000_000_100 * 1_000_000
    row = cap.parse_trade("ETHUSDT", data, recv_ns)
    assert row is not None
    assert row["price"] == pytest.approx(2000.5)
    assert row["qty"] == pytest.approx(0.01)
    assert row["is_buyer_maker"] is True


def test_parse_trade_missing_event_time(cap: Capture) -> None:
    row = cap.parse_trade("ETHUSDT", {"s": "ETHUSDT"}, 0)
    assert row is None


# --- gap detection ---


def test_no_gap_when_cadence_normal(cap: Capture, tmp_path: Path) -> None:
    base = 1_700_000_000_000
    for i in range(5):
        cap.parse_depth("BTCUSDT", {"s": "BTCUSDT", "E": base + i * DEPTH_CADENCE_MS}, 0)
    gaps = (tmp_path / "gaps.log").read_text(encoding="utf-8")
    assert gaps == ""


def test_gap_logged_when_cadence_exceeded(cap: Capture, tmp_path: Path) -> None:
    # Trigger a gap by exceeding GAP_FACTOR * DEPTH_CADENCE_MS. Using the
    # constant keeps the test valid if GAP_FACTOR changes again.
    base = 1_700_000_000_000
    cap.parse_depth("BTCUSDT", {"s": "BTCUSDT", "E": base}, 0)
    cap.parse_depth(
        "BTCUSDT",
        {"s": "BTCUSDT", "E": base + int(DEPTH_CADENCE_MS * GAP_FACTOR) + 100},
        0,
    )
    gaps = (tmp_path / "gaps.log").read_text(encoding="utf-8")
    assert "BTCUSDT" in gaps
    assert str(int(DEPTH_CADENCE_MS * GAP_FACTOR) + 100) in gaps


def test_gap_state_counted(cap: Capture) -> None:
    # Trigger a gap by exceeding GAP_FACTOR * TRADE_CADENCE_MS.
    base = 1_700_000_000_000
    cap.parse_trade("BTCUSDT", {"s": "BTCUSDT", "E": base}, 0)
    cap.parse_trade(
        "BTCUSDT",
        {"s": "BTCUSDT", "E": base + int(TRADE_CADENCE_MS * GAP_FACTOR) + 1000},
        0,
    )
    state = cap._state[("aggTrade", "BTCUSDT")]  # noqa: SLF001
    assert state.n_gaps == 1
    assert state.n_messages == 2


# --- state file ---


def test_state_file_written(cap: Capture, tmp_path: Path) -> None:
    cap.parse_depth("BTCUSDT", {"s": "BTCUSDT", "E": 1_700_000_000_000}, 0)
    cap._write_state()  # noqa: SLF001
    text = (tmp_path / "state.json").read_text(encoding="utf-8")
    assert "BTCUSDT" in text
    assert "n_messages" in text
