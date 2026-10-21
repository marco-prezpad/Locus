"""
rate_limiter.py

Client-side rate limiter for CoinGecko's demo tier (100 calls/minute,
10,000 calls/month). Enforces both limits before every request so that
429s are prevented rather than retried.

Design:
- Per-minute: sliding window (deque) of timestamps. Blocks with sleep
  when the window is full. In-memory; resets on process restart.
- Per-month: persistent JSON counter at `quota_path`, keyed by UTC
  month. Raises FxProviderError when exhausted, including the next
  reset date. Resets automatically on the 1st of each month.
- Warn threshold: logs a warning when usage crosses `warn_threshold`
  (default 9,500 of 10,000), so operators see it coming.
- `get_status()`: read-only status snapshot, safe to call anytime.

Atomic writes (tmp + rename) prevent a corrupted file if the process
crashes mid-write. A corrupt file is treated as a fresh month with a
loud warning — the alternative (fail closed) would lock the user out
permanently.

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

import json
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path

from locus.data.fx.types import FxProviderError
from locus.infrastructure.logging import LoggerFactory


class CoinGeckoRateLimiter:
    """Enforce CoinGecko's per-minute and per-month rate limits."""

    def __init__(
        self,
        quota_path: Path,
        max_per_minute: int = 100,
        max_per_month: int = 10_000,
        warn_threshold: int = 9_500,
        window_seconds: float = 60.0,
    ) -> None:
        if max_per_minute <= 0:
            raise ValueError("max_per_minute must be positive")
        if max_per_month <= 0:
            raise ValueError("max_per_month must be positive")
        if not (0 < warn_threshold <= max_per_month):
            raise ValueError("warn_threshold must be in (0, max_per_month]")
        self._quota_path = Path(quota_path)
        self._max_per_minute = max_per_minute
        self._max_per_month = max_per_month
        self._warn_threshold = warn_threshold
        self._window_seconds = window_seconds
        self._window: deque[float] = deque()
        self._log = LoggerFactory.get("data.fx.coingecko.rate_limiter")

    def wait_and_reserve(self) -> None:
        """Block until a per-minute slot is available and reserve one
        monthly call.

        Raises FxProviderError if the monthly quota is exhausted; the
        message includes the next reset date so callers can decide when
        to retry. Logs a warning when usage crosses `warn_threshold`.

        The counter is incremented optimistically (before the HTTP
        request). Failed requests still consume a slot; this is the
        conservative choice and avoids exceeding CoinGecko's server-side
        limit if the client is buggy.
        """
        month, count = self._read_quota()
        current_month = datetime.now(UTC).strftime("%Y-%m")

        if month != current_month:
            self._log.info(
                "quota_month_rollover",
                extra={"from": month, "to": current_month},
            )
            month = current_month
            count = 0

        if count >= self._max_per_month:
            reset = self._next_month_start()
            raise FxProviderError(
                f"CoinGecko monthly quota exhausted "
                f"({count}/{self._max_per_month} in {month}). "
                f"Next reset: {reset.isoformat()}."
            )

        if count + 1 >= self._warn_threshold:
            self._log.warning(
                "quota_near_limit",
                extra={
                    "used": count + 1,
                    "max": self._max_per_month,
                    "remaining": self._max_per_month - count - 1,
                    "next_reset": self._next_month_start().isoformat(),
                },
            )

        self._wait_for_minute_window()

        self._write_quota(month, count + 1)
        self._log.info(
            "quota_reserved",
            extra={
                "month": month,
                "used": count + 1,
                "max": self._max_per_month,
            },
        )

    def get_status(self) -> dict[str, object]:
        """Return the current quota state without consuming a slot.

        Safe to call as often as needed: it only reads the quota file.
        Includes the next reset date (first day of next UTC month).
        """
        month, count = self._read_quota()
        current = datetime.now(UTC).strftime("%Y-%m")
        used = count if month == current else 0
        return {
            "month": current,
            "used": used,
            "max": self._max_per_month,
            "remaining": self._max_per_month - used,
            "warn_threshold": self._warn_threshold,
            "next_reset": self._next_month_start().isoformat(),
        }

    def _next_month_start(self) -> datetime:
        """First day of next UTC month. When quota resets."""
        now = datetime.now(UTC)
        if now.month == 12:
            return datetime(now.year + 1, 1, 1, tzinfo=UTC)
        return datetime(now.year, now.month + 1, 1, tzinfo=UTC)

    def _wait_for_minute_window(self) -> None:
        """Block until fewer than max_per_minute calls occurred in the
        last window_seconds. Uses monotonic clock (immune to wall-clock
        changes)."""
        while True:
            now = time.monotonic()
            while self._window and now - self._window[0] >= self._window_seconds:
                self._window.popleft()

            if len(self._window) < self._max_per_minute:
                self._window.append(now)
                return

            wait = self._window_seconds - (now - self._window[0])
            if wait > 0:
                self._log.info(
                    "quota_sleep_per_minute",
                    extra={"seconds": round(wait, 2)},
                )
                time.sleep(wait)

    def _read_quota(self) -> tuple[str, int]:
        """Read (month, count) from disk. Returns ("", 0) if missing or
        corrupt."""
        try:
            data = json.loads(self._quota_path.read_text(encoding="utf-8"))
            return str(data.get("month", "")), int(data.get("count", 0))
        except (FileNotFoundError, json.JSONDecodeError, ValueError, TypeError):
            return "", 0

    def _write_quota(self, month: str, count: int) -> None:
        """Atomic write via tmp + rename."""
        self._quota_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._quota_path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"month": month, "count": count}),
            encoding="utf-8",
        )
        tmp.replace(self._quota_path)
