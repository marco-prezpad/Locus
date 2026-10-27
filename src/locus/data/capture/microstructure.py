"""
microstructure.py

P1.c — microstructure capture from Binance USD-M Futures.

Captures partial book depth (top 20 levels) and aggregated trades for
the U10 symbols. Continuous process, appends to daily Parquet
partitions, logs gaps, writes state periodically.

Design (plan v3.1 §9, P1.c):
- Single process, no orchestration.
- Two WebSocket connections: depth (/public) and aggTrade (/market).
  Binance migrated USD-M Futures WebSocket endpoints on 2026-04-23 into
  routed paths. The legacy combined URL only delivers /public streams;
  /market streams (@aggTrade, @markPrice, @kline) fail silently. See
  https://developers.binance.info/en/docs/products/caas/market-data-streams
- Reception timestamp + event timestamp + latency per message.
- Daily partitioning: data/raw/microstructure/{stream}/{symbol}/{YYYY-MM-DD}/.
- Auto-reconnect with exponential backoff per connection (cap 60 s).
- Gap log: if interval between consecutive messages on the same stream
  exceeds GAP_FACTOR * expected cadence, append a line to gaps.log.
- State file: uptime, message counts, gap counts, written every flush.

Note on latency: `latency_ms` measures (reception - event_time). For
depth streams Binance batches events before publishing, so 500-1000 ms
is normal. Slippage calibration in P6 uses `reception_time_ns`, not
`event_time`.

This module does NOT formulate hypotheses. Per plan §9, A4 is
pre-registered only after at least four weeks of clean data. Do not
analyze the captured data to decide what to test.

Streams:
    <symbol>@depth20@100ms  — top 20 levels, 100 ms updates  (/public)
    <symbol>@aggTrade       — aggregated trades, event-driven  (/market)

Entry point: `scripts/capture_microstructure.py` (argparse + signal
handling). This module is importable for testing and future consumers
(P6 slippage calibration, P13 dashboard status).

Author: Marco Pérez Padilla
Date:   27-10-2026
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import websockets

from locus.infrastructure.logging import LoggerFactory

# Binance USD-M Futures routed WebSocket endpoints (2026-04-23 migration).
# The legacy `wss://fstream.binance.com/stream?streams=...` only delivers
# /public streams; /market streams fail silently.
DEPTH_BASE_URL = "wss://fstream.binance.com/public/stream?streams="
TRADE_BASE_URL = "wss://fstream.binance.com/market/stream?streams="

SYMBOLS: list[str] = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "AVAXUSDT",
    "DOTUSDT",
    "LINKUSDT",
]

DEPTH_STREAM = "@depth20@100ms"
TRADE_STREAM = "@aggTrade"

DEPTH_CADENCE_MS = 100
TRADE_CADENCE_MS = 60_000

GAP_FACTOR = 5.0
FLUSH_INTERVAL_S = 30.0
OUTPUT_ROOT = Path("data/raw/microstructure")


class CaptureState:
    __slots__ = ("last_event_ms", "n_messages", "n_gaps")

    def __init__(self) -> None:
        self.last_event_ms: int | None = None
        self.n_messages: int = 0
        self.n_gaps: int = 0


class Capture:
    """WebSocket consumer that writes depth and trade partitions."""

    def __init__(
        self,
        output_root: Path,
        symbols: list[str] | None = None,
    ) -> None:
        self._log = LoggerFactory.get("data.capture.microstructure")
        self._out = output_root
        self._symbols = symbols if symbols is not None else list(SYMBOLS)
        self._buffers: dict[tuple[str, str], list[dict]] = defaultdict(list)
        self._state: dict[tuple[str, str], CaptureState] = defaultdict(CaptureState)
        self._seen_streams: set[str] = set()
        self._gaps_file = output_root / "gaps.log"
        self._state_file = output_root / "state.json"
        self._stop = asyncio.Event()
        self._started_at = time.time()

    def stop(self) -> None:
        self._stop.set()

    # --- parsing (testable without WebSocket) ---

    def parse_depth(
        self,
        symbol: str,
        data: dict,
        recv_ns: int,
    ) -> dict | None:
        event_ms = int(data.get("E", 0))
        if event_ms == 0:
            return None
        key = ("depth", symbol)
        self._check_gap(key, event_ms, DEPTH_CADENCE_MS)
        self._state[key].n_messages += 1
        self._state[key].last_event_ms = event_ms
        return {
            "event_time": datetime.fromtimestamp(event_ms / 1000, tz=UTC),
            "reception_time_ns": recv_ns,
            "latency_ms": recv_ns / 1e6 - event_ms,
            "bids_json": json.dumps(data.get("b", [])),
            "asks_json": json.dumps(data.get("a", [])),
        }

    def parse_trade(
        self,
        symbol: str,
        data: dict,
        recv_ns: int,
    ) -> dict | None:
        event_ms = int(data.get("E", 0))
        if event_ms == 0:
            return None
        key = ("aggTrade", symbol)
        self._check_gap(key, event_ms, TRADE_CADENCE_MS)
        self._state[key].n_messages += 1
        self._state[key].last_event_ms = event_ms
        return {
            "event_time": datetime.fromtimestamp(event_ms / 1000, tz=UTC),
            "reception_time_ns": recv_ns,
            "latency_ms": recv_ns / 1e6 - event_ms,
            "price": float(data.get("p", 0.0)),
            "qty": float(data.get("q", 0.0)),
            "is_buyer_maker": bool(data.get("m", False)),
        }

    def _check_gap(
        self,
        key: tuple[str, str],
        event_ms: int,
        expected_ms: int,
    ) -> None:
        state = self._state[key]
        if state.last_event_ms is None:
            return
        delta = event_ms - state.last_event_ms
        if delta > expected_ms * GAP_FACTOR:
            state.n_gaps += 1
            with self._gaps_file.open("a", encoding="utf-8") as f:
                f.write(f"{datetime.now(UTC).isoformat()},{key[0]},{key[1]},{delta}\n")

    # --- I/O ---

    def _write_partition(self, key: tuple[str, str], rows: list[dict]) -> None:
        stream, symbol = key
        df = pl.DataFrame(rows)
        # rows[].event_time is a datetime.datetime (set in parse_depth/parse_trade).
        # Computing the date from Python avoids Polars' union return type
        # for .min() (int | float | Decimal | timedelta | str | bool | bytes
        # | ndarray | list | None), which pyright cannot narrow.
        date_str = min(row["event_time"] for row in rows).strftime("%Y-%m-%d")
        out_dir = self._out / stream / symbol / date_str
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = datetime.now(UTC).strftime("%H-%M-%S-%f") + ".parquet"
        df.write_parquet(out_dir / fname, compression="snappy")

    def _write_state(self) -> None:
        state = {
            "started_at": datetime.fromtimestamp(
                self._started_at,
                tz=UTC,
            ).isoformat(),
            "uptime_s": int(time.time() - self._started_at),
            "streams": {
                f"{k[0]}/{k[1]}": {
                    "n_messages": v.n_messages,
                    "n_gaps": v.n_gaps,
                }
                for k, v in self._state.items()
            },
        }
        self._state_file.write_text(
            json.dumps(state, indent=2),
            encoding="utf-8",
        )

    async def _flush_all(self) -> None:
        for key, rows in list(self._buffers.items()):
            if not rows:
                continue
            self._buffers[key] = []
            await asyncio.to_thread(self._write_partition, key, rows)
        await asyncio.to_thread(self._write_state)

    async def _writer_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(
                    self._stop.wait(),
                    timeout=FLUSH_INTERVAL_S,
                )
                return
            except TimeoutError:
                await self._flush_all()

    # --- connection loop (one per routed path) ---

    async def _connection_loop(self, url: str, label: str) -> None:
        """Maintain a WebSocket connection to `url` until stop is set."""
        backoff = 1.0
        while not self._stop.is_set():
            try:
                async with websockets.connect(url, max_size=2**22) as ws:
                    self._log.info("ws_connected", extra={"group": label})
                    backoff = 1.0
                    async for raw in ws:
                        if self._stop.is_set():
                            break
                        await self._handle(raw)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._log.warning(
                    "ws_disconnected",
                    extra={
                        "group": label,
                        "error": type(exc).__name__,
                        "backoff_s": backoff,
                    },
                )
            if self._stop.is_set():
                break
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60.0)

    # --- main loop ---

    async def run(self) -> None:
        self._out.mkdir(parents=True, exist_ok=True)
        self._gaps_file.touch(exist_ok=True)

        depth_streams = [f"{s.lower()}{DEPTH_STREAM}" for s in self._symbols]
        trade_streams = [f"{s.lower()}{TRADE_STREAM}" for s in self._symbols]

        depth_url = DEPTH_BASE_URL + "/".join(depth_streams)
        trade_url = TRADE_BASE_URL + "/".join(trade_streams)

        self._log.info(
            "capture_starting",
            extra={
                "n_symbols": len(self._symbols),
                "depth_streams": len(depth_streams),
                "trade_streams": len(trade_streams),
            },
        )

        writer = asyncio.create_task(self._writer_loop())
        tasks = [
            asyncio.create_task(self._connection_loop(depth_url, "depth")),
            asyncio.create_task(self._connection_loop(trade_url, "aggTrade")),
        ]

        await self._stop.wait()

        for t in tasks:
            t.cancel()
        for t in tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await t

        writer.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await writer

        await self._flush_all()
        self._log.info("capture_stopped")

    async def _handle(self, raw: str | bytes) -> None:
        recv_ns = time.time_ns()
        if isinstance(raw, bytes):
            try:
                raw = raw.decode("utf-8")
            except UnicodeDecodeError:
                return
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return
        stream = msg.get("stream", "")
        data = msg.get("data", {})
        if not stream:
            return
        if stream not in self._seen_streams:
            self._seen_streams.add(stream)
            if self._log.isEnabledFor(logging.INFO):
                self._log.info("stream_seen", extra={"stream": stream})
        symbol = data.get("s", "")
        if not symbol:
            return

        if "@depth" in stream:
            row = self.parse_depth(symbol, data, recv_ns)
            if row is not None:
                self._buffers[("depth", symbol)].append(row)
        elif "@aggTrade" in stream:
            row = self.parse_trade(symbol, data, recv_ns)
            if row is not None:
                self._buffers[("aggTrade", symbol)].append(row)
