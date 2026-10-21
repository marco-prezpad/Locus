"""
test_rate_limiter.py

CoinGeckoRateLimiter: per-minute window, per-month quota, month
rollover, corrupt file handling, warn threshold, get_status.

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from locus.data.fx.rate_limiter import CoinGeckoRateLimiter
from locus.data.fx.types import FxProviderError


def _limiter(
    tmp_path: Path,
    max_per_minute: int = 100,
    max_per_month: int = 10_000,
    warn_threshold: int | None = None,
) -> CoinGeckoRateLimiter:
    # Derive warn_threshold from max_per_month when not explicit so
    # small-limit tests (e.g. max_per_month=2) don't trip the guard.
    if warn_threshold is None:
        warn_threshold = min(9_500, max_per_month)
    return CoinGeckoRateLimiter(
        quota_path=tmp_path / "quota.json",
        max_per_minute=max_per_minute,
        max_per_month=max_per_month,
        warn_threshold=warn_threshold,
    )


def test_reserve_increments_counter(tmp_path: Path) -> None:
    limiter = _limiter(tmp_path)
    limiter.wait_and_reserve()
    limiter.wait_and_reserve()
    data = json.loads((tmp_path / "quota.json").read_text())
    assert data["count"] == 2


def test_monthly_limit_raises_with_reset_date(tmp_path: Path) -> None:
    limiter = _limiter(tmp_path, max_per_month=2)
    limiter.wait_and_reserve()
    limiter.wait_and_reserve()
    with pytest.raises(FxProviderError) as exc_info:
        limiter.wait_and_reserve()
    msg = str(exc_info.value)
    assert "monthly quota exhausted" in msg
    assert "Next reset:" in msg


def test_month_rollover_resets(tmp_path: Path) -> None:
    (tmp_path / "quota.json").write_text(json.dumps({"month": "2024-01", "count": 9_999}))
    limiter = _limiter(tmp_path, max_per_month=10_000)
    with patch("locus.data.fx.rate_limiter.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2024, 2, 1, tzinfo=UTC)
        mock_dt.UTC = UTC
        limiter.wait_and_reserve()
    data = json.loads((tmp_path / "quota.json").read_text())
    assert data["month"] == "2024-02"
    assert data["count"] == 1


def test_per_minute_blocks(tmp_path: Path) -> None:
    limiter = _limiter(tmp_path, max_per_minute=2)
    with (
        patch("locus.data.fx.rate_limiter.time.sleep") as mock_sleep,
        patch(
            "locus.data.fx.rate_limiter.time.monotonic",
            side_effect=[0.0, 0.0, 0.0, 60.0, 60.0],
        ),
    ):
        limiter.wait_and_reserve()
        limiter.wait_and_reserve()
        limiter.wait_and_reserve()
    assert mock_sleep.called


def test_corrupt_file_starts_fresh(tmp_path: Path) -> None:
    (tmp_path / "quota.json").write_text("not valid json {{{")
    limiter = _limiter(tmp_path)
    limiter.wait_and_reserve()
    data = json.loads((tmp_path / "quota.json").read_text())
    assert data["count"] == 1


def test_missing_file_starts_fresh(tmp_path: Path) -> None:
    limiter = _limiter(tmp_path)
    limiter.wait_and_reserve()
    assert (tmp_path / "quota.json").exists()


def test_invalid_constructor_args(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        CoinGeckoRateLimiter(tmp_path / "q.json", max_per_minute=0)
    with pytest.raises(ValueError):
        CoinGeckoRateLimiter(tmp_path / "q.json", max_per_month=0)
    with pytest.raises(ValueError):
        CoinGeckoRateLimiter(tmp_path / "q.json", warn_threshold=0)
    with pytest.raises(ValueError):
        CoinGeckoRateLimiter(tmp_path / "q.json", max_per_month=100, warn_threshold=200)


def test_get_status_without_consuming(tmp_path: Path) -> None:
    limiter = _limiter(tmp_path, max_per_month=100, warn_threshold=90)
    limiter.wait_and_reserve()
    limiter.wait_and_reserve()
    status = limiter.get_status()
    assert status["used"] == 2
    assert status["remaining"] == 98
    assert status["max"] == 100
    assert status["warn_threshold"] == 90
    assert "next_reset" in status
    assert limiter.get_status()["used"] == 2  # get_status no consume


def test_warn_threshold_logs_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING)
    limiter = _limiter(tmp_path, max_per_month=100, warn_threshold=98)
    for _ in range(97):
        limiter.wait_and_reserve()
    assert "quota_near_limit" not in caplog.text
    limiter.wait_and_reserve()  # 98th, cruza el umbral
    assert "quota_near_limit" in caplog.text


def test_status_resets_after_month_change(tmp_path: Path) -> None:
    (tmp_path / "quota.json").write_text(json.dumps({"month": "2024-01", "count": 9_999}))
    limiter = _limiter(tmp_path, max_per_month=10_000)
    with patch("locus.data.fx.rate_limiter.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2024, 2, 15, tzinfo=UTC)
        mock_dt.UTC = UTC
        status = limiter.get_status()
    assert status["used"] == 0
    assert status["remaining"] == 10_000
    assert status["month"] == "2024-02"
