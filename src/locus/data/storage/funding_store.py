"""
funding_store.py

Two-layer store for funding rates:

- Processed: data/processed/funding/venue={v}/asset={a}/year={y}/rates.parquet
- Raw: data/raw/funding/{venue}/{asset}/raw.parquet

The raw layer preserves the literal API response per request plus
`fetched_at`, for audit and replay. First raw layer in the project.

Author: Marco Pérez Padilla
Date:   23-10-2026
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import polars as pl

from locus.data.providers.funding_base import FUNDING_SCHEMA

ParquetCompression = Literal["lz4", "uncompressed", "snappy", "gzip", "brotli", "zstd"]

_RAW_FUNDING_SCHEMA: dict[str, Any] = {
    "fetched_at": pl.Datetime("us", "UTC"),
    "symbol": pl.Utf8,
    "request_start_ms": pl.Int64,
    "request_end_ms": pl.Int64,
    "response_json": pl.Utf8,
}


class FundingStore:
    def __init__(
        self,
        processed_root: Path,
        raw_root: Path,
        compression: ParquetCompression = "snappy",
    ) -> None:
        self._processed = Path(processed_root) / "funding"
        self._raw = Path(raw_root) / "funding"
        self._compression: ParquetCompression = compression
        self._processed.mkdir(parents=True, exist_ok=True)
        self._raw.mkdir(parents=True, exist_ok=True)

    # ---- processed layer ----

    def write(self, df: pl.DataFrame) -> int:
        if df.height == 0:
            return 0
        if df.schema != FUNDING_SCHEMA:
            raise ValueError(f"schema mismatch: {df.schema}")

        by_year = df.with_columns(pl.col("timestamp").dt.year().alias("_year")).partition_by(
            "_year", as_dict=True
        )
        total = 0
        for year_key, chunk in by_year.items():
            year = int(year_key[0])
            asset_id = chunk["asset_id"][0]
            venue = chunk["venue"][0]
            target_dir = self._processed / f"venue={venue}" / f"asset={asset_id}" / f"year={year}"
            target_dir.mkdir(parents=True, exist_ok=True)
            chunk.drop("_year").sort("timestamp").write_parquet(
                target_dir / "rates.parquet",
                compression=self._compression,
            )
            total += chunk.height
        return total

    def read(self, venue: str, asset_id: str, start: datetime, end: datetime) -> pl.DataFrame:
        pattern = (
            self._processed / f"venue={venue}" / f"asset={asset_id}" / "year=*" / "rates.parquet"
        )
        if not list(pattern.parent.parent.glob("year=*/rates.parquet")):
            return pl.DataFrame(schema=FUNDING_SCHEMA)
        return (
            pl.scan_parquet(str(pattern))
            .filter((pl.col("timestamp") >= start) & (pl.col("timestamp") < end))
            .collect()
        )

    # ---- raw layer ----

    def write_raw(
        self,
        venue: str,
        asset_id: str,
        symbol: str,
        request_start_ms: int,
        request_end_ms: int,
        response_json: str,
    ) -> None:
        target_dir = self._raw / venue / asset_id
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / "raw.parquet"

        new_row = pl.DataFrame(
            {
                "fetched_at": [datetime.now(UTC)],
                "symbol": [symbol],
                "request_start_ms": [request_start_ms],
                "request_end_ms": [request_end_ms],
                "response_json": [response_json],
            },
            schema=_RAW_FUNDING_SCHEMA,
        )
        if target.exists():
            existing = pl.read_parquet(target)
            combined = pl.concat([existing, new_row]).sort("fetched_at")
        else:
            combined = new_row
        combined.write_parquet(target, compression=self._compression)

    def read_raw(self, venue: str, asset_id: str) -> pl.DataFrame:
        target = self._raw / venue / asset_id / "raw.parquet"
        if not target.exists():
            return pl.DataFrame(schema=_RAW_FUNDING_SCHEMA)
        return pl.read_parquet(target)
