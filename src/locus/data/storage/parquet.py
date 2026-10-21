"""
parquet.py

Parquet-based storage for canonical OHLCV DataFrames. Partitioned layout:

    {root}/market=crypto/venue={venue}/asset={asset}/timeframe={tf}/year={YYYY}/bars.parquet

One file per (venue, asset, timeframe, year). Reads use polars' hive
scanning so a range query transparently unions the relevant years.

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

import polars as pl

from locus.data.types import ProviderResponseError, Timeframe

ParquetCompression = Literal["lz4", "uncompressed", "snappy", "gzip", "brotli", "zstd"]


class ParquetBarStore:
    """Write and read canonical OHLCV DataFrames as partitioned Parquet."""

    def __init__(self, root: Path, compression: ParquetCompression = "snappy") -> None:
        self._root = Path(root)
        self._compression: ParquetCompression = compression
        self._root.mkdir(parents=True, exist_ok=True)

    def write(self, df: pl.DataFrame) -> int:
        """Write a canonical OHLCV DataFrame, partitioned by year.

        Groups rows by year (from the timestamp) and writes one Parquet
        file per year. Overwrites existing files for those years.
        Returns the number of rows written.
        """
        if df.height == 0:
            return 0

        for col in ("venue", "asset_id", "timeframe"):
            if df[col].n_unique() != 1:
                raise ProviderResponseError(f"write() expects a single {col} per call")

        venue = df["venue"][0]
        asset = df["asset_id"][0]
        tf = df["timeframe"][0]
        years = df.with_columns(pl.col("timestamp").dt.year().alias("_year")).partition_by(
            "_year", as_dict=True
        )

        total = 0
        for year_key, chunk in years.items():
            year_val = int(year_key[0])
            target_dir = (
                self._root
                / "market=crypto"
                / f"venue={venue}"
                / f"asset={asset}"
                / f"timeframe={tf}"
                / f"year={year_val}"
            )
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / "bars.parquet"
            chunk.drop("_year").sort("timestamp").write_parquet(
                target, compression=self._compression
            )
            total += chunk.height
        return total

    def read(
        self,
        venue: str,
        asset_id: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        """Read bars for one (venue, asset, timeframe) in [start, end).

        Returns an empty DataFrame with the canonical schema if no data
        is found for that range.
        """
        from locus.data.providers.base import BAR_SCHEMA

        pattern = (
            self._root
            / "market=crypto"
            / f"venue={venue}"
            / f"asset={asset_id}"
            / f"timeframe={timeframe.value}"
            / "year=*"
            / "bars.parquet"
        )
        if not list(pattern.parent.parent.glob("year=*/bars.parquet")):
            return pl.DataFrame(schema=BAR_SCHEMA)

        df = (
            pl.scan_parquet(str(pattern))
            .filter((pl.col("timestamp") >= start) & (pl.col("timestamp") < end))
            .collect()
        )
        return df
