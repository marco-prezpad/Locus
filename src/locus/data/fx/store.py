"""
store.py

Parquet store for FX rates. Layout:

    {root}/fx/pair={pair}/source={source}/year={YYYY}/rates.parquet

One file per (pair, source, year). Reads use polars' hive scanning.

The store enriches incoming DataFrames with `source` and `fetched_at`
(now UTC) on write; the provider output only carries pair, date, rate.

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import polars as pl

from locus.data.fx.base import FX_PROVIDER_SCHEMA

ParquetCompression = Literal["lz4", "uncompressed", "snappy", "gzip", "brotli", "zstd"]

_STORED_SCHEMA: dict[str, Any] = {
    "pair": pl.Utf8,
    "date": pl.Datetime("us", "UTC"),
    "rate": pl.Float64,
    "source": pl.Utf8,
    "fetched_at": pl.Datetime("us", "UTC"),
}


class FxRateStore:
    """Write and read FX rate DataFrames as partitioned Parquet."""

    def __init__(self, root: Path, compression: ParquetCompression = "snappy") -> None:
        self._root = Path(root)
        self._compression: ParquetCompression = compression
        self._root.mkdir(parents=True, exist_ok=True)

    def write(self, df: pl.DataFrame, source: str) -> int:
        """Enrich with source and fetched_at, then write partitioned by
        (pair, source, year). Overwrites per-year files. Returns rows
        written. Empty DataFrame is a no-op.
        """
        if df.height == 0:
            return 0
        if df.schema != FX_PROVIDER_SCHEMA:
            raise ValueError(f"schema mismatch: expected {FX_PROVIDER_SCHEMA}, got {df.schema}")

        enriched = df.with_columns(
            pl.lit(source).alias("source"),
            pl.lit(datetime.now(UTC)).alias("fetched_at"),
        ).with_columns(pl.col("date").dt.year().alias("_year"))

        by_pair = enriched.partition_by("pair", as_dict=True)
        total = 0
        for pair_key, pair_df in by_pair.items():
            pair = pair_key[0]
            for year_key, year_df in pair_df.partition_by("_year", as_dict=True).items():
                year = int(year_key[0])
                target_dir = (
                    self._root / "fx" / f"pair={pair}" / f"source={source}" / f"year={year}"
                )
                target_dir.mkdir(parents=True, exist_ok=True)
                target = target_dir / "rates.parquet"
                year_df.drop("_year").sort("date").select(
                    ["pair", "date", "rate", "source", "fetched_at"]
                ).write_parquet(target, compression=self._compression)
                total += year_df.height
        return total

    def read(
        self,
        pair: str,
        source: str,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        """Read rates for one (pair, source) in [start, end)."""
        pattern = (
            self._root / "fx" / f"pair={pair}" / f"source={source}" / "year=*" / "rates.parquet"
        )
        if not list(pattern.parent.parent.glob("year=*/rates.parquet")):
            return pl.DataFrame(schema=_STORED_SCHEMA)

        return (
            pl.scan_parquet(str(pattern))
            .filter((pl.col("date") >= start) & (pl.col("date") < end))
            .collect()
        )

    def available_sources(self, pair: str) -> list[str]:
        """List sources that have at least one stored rate for `pair`."""
        pair_dir = self._root / "fx" / f"pair={pair}"
        if not pair_dir.exists():
            return []
        return sorted(
            d.name.removeprefix("source=")
            for d in pair_dir.iterdir()
            if d.is_dir() and d.name.startswith("source=")
        )

    def available_pairs(self) -> list[str]:
        """List pairs that have at least one stored rate."""
        fx_root = self._root / "fx"
        if not fx_root.exists():
            return []
        return sorted(
            d.name.removeprefix("pair=")
            for d in fx_root.iterdir()
            if d.is_dir() and d.name.startswith("pair=")
        )
