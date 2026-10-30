"""Tests for the P1.c microstructure capture module."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
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


# --- _handle (dispatch + parsing paths) ---


def _depth_msg(symbol: str = "BTCUSDT", event_ms: int = 1_700_000_000_000) -> str:
    return json.dumps(
        {
            "stream": f"{symbol.lower()}@depth20@100ms",
            "data": {"s": symbol, "E": event_ms, "b": [["100.0", "1.0"]], "a": [["101.0", "1.0"]]},
        }
    )


def _trade_msg(symbol: str = "BTCUSDT", event_ms: int = 1_700_000_000_000) -> str:
    return json.dumps(
        {
            "stream": f"{symbol.lower()}@aggTrade",
            "data": {"s": symbol, "E": event_ms, "p": "100.0", "q": "0.01", "m": False},
        }
    )


def test_handle_depth_str(cap: Capture) -> None:
    asyncio.run(cap._handle(_depth_msg()))
    assert len(cap._buffers[("depth", "BTCUSDT")]) == 1
    assert "btcusdt@depth20@100ms" in cap._seen_streams


def test_handle_trade_str(cap: Capture) -> None:
    asyncio.run(cap._handle(_trade_msg()))
    assert len(cap._buffers[("aggTrade", "BTCUSDT")]) == 1


def test_handle_bytes(cap: Capture) -> None:
    asyncio.run(cap._handle(_depth_msg().encode("utf-8")))
    assert len(cap._buffers[("depth", "BTCUSDT")]) == 1


def test_handle_invalid_utf8(cap: Capture) -> None:
    asyncio.run(cap._handle(b"\xff\xfe not utf-8"))
    assert not cap._buffers


def test_handle_invalid_json(cap: Capture) -> None:
    asyncio.run(cap._handle("{not json"))
    assert not cap._buffers


def test_handle_missing_stream(cap: Capture) -> None:
    asyncio.run(cap._handle(json.dumps({"data": {"s": "BTCUSDT", "E": 1}})))
    assert not cap._buffers


def test_handle_missing_symbol(cap: Capture) -> None:
    asyncio.run(
        cap._handle(
            json.dumps(
                {
                    "stream": "btcusdt@depth20@100ms",
                    "data": {"E": 1_700_000_000_000},
                }
            )
        )
    )
    assert not cap._buffers


def test_handle_unknown_stream_type(cap: Capture) -> None:
    asyncio.run(
        cap._handle(
            json.dumps(
                {
                    "stream": "btcusdt@kline_1m",
                    "data": {"s": "BTCUSDT", "E": 1_700_000_000_000},
                }
            )
        )
    )
    assert not cap._buffers


def test_handle_seen_stream_logged_once(cap: Capture) -> None:
    asyncio.run(cap._handle(_depth_msg()))
    asyncio.run(cap._handle(_depth_msg(event_ms=1_700_000_000_100)))
    # Stream is recorded once, but messages keep appending
    assert "btcusdt@depth20@100ms" in cap._seen_streams
    assert len(cap._buffers[("depth", "BTCUSDT")]) == 2


# --- _flush_all ---


def test_flush_all_empty_only_state(cap: Capture, tmp_path: Path) -> None:
    asyncio.run(cap._flush_all())
    assert (tmp_path / "state.json").exists()
    assert len(list(tmp_path.rglob("*.parquet"))) == 0


def test_flush_all_writes_partitions(cap: Capture, tmp_path: Path) -> None:
    cap._buffers[("depth", "BTCUSDT")].append(
        {
            "event_time": datetime(2026, 1, 1, tzinfo=UTC),
            "reception_time_ns": 1,
            "latency_ms": 50.0,
            "bids_json": "[]",
            "asks_json": "[]",
        }
    )
    asyncio.run(cap._flush_all())
    files = list((tmp_path / "depth" / "BTCUSDT").rglob("*.parquet"))
    assert len(files) == 1
    assert cap._buffers[("depth", "BTCUSDT")] == []


# --- _writer_loop ---


def test_writer_loop_returns_immediately_on_stop(cap: Capture) -> None:
    cap.stop()
    asyncio.run(cap._writer_loop())


def test_writer_loop_flushes_before_stop(
    cap: Capture,
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "locus.data.capture.microstructure.FLUSH_INTERVAL_S",
        0.05,
    )
    cap._buffers[("depth", "BTCUSDT")].append(
        {
            "event_time": datetime(2026, 1, 1, tzinfo=UTC),
            "reception_time_ns": 1,
            "latency_ms": 50.0,
            "bids_json": "[]",
            "asks_json": "[]",
        }
    )

    async def _run() -> None:
        task = asyncio.create_task(cap._writer_loop())
        await asyncio.sleep(0.15)
        cap.stop()
        await task

    asyncio.run(_run())
    assert len(list((tmp_path / "depth" / "BTCUSDT").rglob("*.parquet"))) == 1


# --- run() and _connection_loop ---


class _FakeWS:
    """Minimal async context manager that yields nothing and closes."""

    def __init__(self, *, raise_on_enter: bool = False) -> None:
        self._raise_on_enter = raise_on_enter

    async def __aenter__(self) -> _FakeWS:
        if self._raise_on_enter:
            raise ConnectionError("fake connect failure")
        return self

    async def __aexit__(self, *args: object) -> None:
        pass

    def __aiter__(self) -> _FakeWS:
        return self

    async def __anext__(self) -> str:
        raise StopAsyncIteration


def test_run_stops_cleanly_when_connect_fails(
    cap: Capture,
    monkeypatch,
) -> None:
    attempts = {"n": 0}

    def fake_connect(*a: object, **kw: object) -> _FakeWS:
        attempts["n"] += 1
        return _FakeWS(raise_on_enter=True)

    monkeypatch.setattr(
        "locus.data.capture.microstructure.websockets.connect",
        fake_connect,
    )

    async def _run() -> None:
        task = asyncio.create_task(cap.run())
        await asyncio.sleep(0.1)
        cap.stop()
        await task

    asyncio.run(_run())
    # Both connections (depth, aggTrade) attempted at least once
    assert attempts["n"] >= 2
    assert (cap._out / "state.json").exists()


def test_run_stops_cleanly_when_ws_closes(
    cap: Capture,
    monkeypatch,
) -> None:
    def fake_connect(*a: object, **kw: object) -> _FakeWS:
        return _FakeWS(raise_on_enter=False)

    monkeypatch.setattr(
        "locus.data.capture.microstructure.websockets.connect",
        fake_connect,
    )

    async def _run() -> None:
        task = asyncio.create_task(cap.run())
        await asyncio.sleep(0.1)
        cap.stop()
        await task

    asyncio.run(_run())
    assert (cap._out / "state.json").exists()
