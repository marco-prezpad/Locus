# Market Data Layer

Documentation for the market data layer (phases F2.A.1, F2.A.2, F2.A.3). Covers data sources, providers, validation, storage, universe reconstruction, and known limitations.

## 1. Sources

The layer fetches OHLCV bars from three public APIs. Each provider implements the `MarketDataProvider` Protocol and returns a Polars DataFrame with the canonical `BAR_SCHEMA` columns.

### Binance (primary)

- **Endpoint:** `GET /api/v3/klines`
- **History:** Full history since pair listing date.
- **Pagination:** `startTime` cursor, 1000 bars per request.
- **Volume:** Reported in base asset units (e.g., BTC for BTCUSDT).
- **Role:** Primary source for historical backtests.

### Kraken (fallback)

- **Endpoint:** `GET /0/public/OHLC`
- **History:** **Only the most recent 720 bars** (30 days for 1h, 2 years for 1d). The `since` parameter is ignored for older ranges.
- **Pagination:** `since` cursor, 720 bars per request.
- **Role:** Recent-data fallback; not usable for historical backtests.

The provider raises `ProviderNoDataError` when the returned range starts after `request.end`, so the fallback chain skips it.

### CoinGecko (last resort)

- **Endpoint:** `GET /coins/{id}/ohlc`
- **History:** Daily bars only. Intraday timeframes raise `ProviderResponseError`.
- **Volume:** Not exposed by the `/ohlc` endpoint; reported as `0`.
- **Rate limit:** 10–30 requests/minute on the free tier.
- **Role:** Emergency fallback when Binance and Kraken both fail.

## 2. Symbol Translation

`src/locus/data/symbols.py` maps canonical asset IDs to per-venue symbols. The mapping is **hardcoded on purpose**:

- There is no universal identifier between venues.
- Historical renames (`MATIC` → `POL`, `RNDR` → `RENDER`, `FTM` → `S`) and collisions (`UNI` could be Uniswap or a regional token) require human review.
- The universe of the final holdout is pre-registered (see plan §8.15); an auto-discovered mapper would silently change the universe between runs.

Three guardrails in `tests/unit/data/test_symbols.py`:

- `test_all_assets_mapped_for_all_venues`: Fails if an asset lacks an entry for one venue.
- `test_no_duplicate_symbols_within_a_venue`: Fails if two assets map to the same symbol in the same venue.
- `test_reverse_lookup_roundtrip`: Verifies `to_asset_id(to_symbol(x)) == x`.

**Adding a new asset:** Edit `_SYMBOLS` and run `make ci`. The first guard will tell you which venues you forgot.

## 3. Validation

`src/locus/data/validation/ohlcv.py` validates the whole DataFrame using vectorized operations before storage.

Checks:

- Schema matches `BAR_SCHEMA` exactly (columns and dtypes).
- No nulls.
- OHLC coherence:
  - `high >= max(open, close, low)`
  - `low <= min(open, close, high)`
  - All prices are `> 0`.
- `volume >= 0`.
- Timestamps are strictly sorted, contain no duplicates, and are timezone-aware UTC.

Validation failures raise `ProviderResponseError` and abort the ingestion run. A bad bar does not enter storage.

## 4. Storage

### Market Data (Parquet)

```text
data/processed/market=crypto/venue={venue}/asset={asset}/timeframe={tf}/year={yyyy}/bars.parquet
```

One file per `(venue, asset, timeframe, year)`. Reads use Polars' Hive scanning, so a range query transparently unions the relevant years.

### Universe Registry (Parquet)

```text
data/processed/universe/registry.parquet
```

Single file, one row per `(asset, venue)` with `first_tradable_ts` and `last_tradable_ts`. Small (~150 rows for the MVP). Overwritten on each discovery run.

### Raw vs. Processed

`data/raw/` and `data/interim/` are reserved but unused in F2.

If a provider response needs to be preserved verbatim (for audit, replay, or debugging), it goes to `data/raw/`. The current pipeline does not persist raw responses; the processed Parquet is the source of truth.

## 5. Universe Reconstruction

`src/locus/data/universe/builder.py` builds the set of tradable assets at any historical timestamp.

**No survivorship bias:** An asset is included only if it was listed at `t` and has been listed for at least `min_history_days`.

```python
snapshot = UniverseBuilder(registry).build(
    datetime(2024, 6, 1, tzinfo=UTC),
    UniverseConstraint(min_history_days=90, venues=["binance"]),
)
```

The builder is pure: it reads from the registry and never touches the network. Discovery populates the registry offline.

### Discovery

`src/locus/data/universe/discovery.py` populates the registry from Binance's `/api/v3/exchangeInfo`.

Listing dates come from `onboardDate` when present, and fall back to the timestamp of the first available 1d bar otherwise.

Binance's public `exchangeInfo` does not currently include `onboardDate`, so the fallback is the normal path in practice: approximately 50 assets × 1 request each, around 20 seconds total, cached afterwards.

### Known Limitations

- Delisted assets are not discoverable through public APIs. All registry entries are live assets in the MVP. The `last_tradable_ts` field supports delisted assets in the contract, but no data source populates it yet.
- Kraken and CoinGecko do not expose listing dates through their public APIs. Discovery is Binance-only.
- The first-bar fallback is approximate: It may be later than the real listing date if Binance started storing klines later, or earlier if a symbol was delisted and relisted. This limitation is documented and accepted.

## 6. Ingestion

### Single Asset

`MarketDataIngestor.ingest(request)` runs the full pipeline for one asset:

1. Fetch data.
2. Normalize `asset_id` and `timeframe`.
3. Validate the data.
4. Write the data to storage.

Returns an `IngestionResult`.

### Batch

`BatchIngestor.run(snapshot, timeframe, start, end)` iterates over a `UniverseSnapshot` and returns a `DownloadReport`.

Per-asset failures are captured in the report, not raised: a single delisted or missing asset does not abort the batch.

`sleep_between_assets_seconds` keeps the batch within the provider's rate limit (set to approximately 0.2 seconds for Binance in the smoke script).

## 7. Failure Semantics

| Error type | Meaning | Fallback behavior |
| :--- | :--- | :--- |
| `ProviderResponseError` | Malformed or semantically invalid response | Skip asset, log `provider_failed` |
| `ProviderRateLimitError` | Rate limit exhausted after retries | Skip asset, log `provider_failed` |
| `ProviderNoDataError` | Requested range has no data | Skip asset, log `provider_failed` |
| `httpx.HTTPError` | Transport error | Retried by Tenacity, then `ProviderResponseError` |
| Empty DataFrame, no error | Provider responded with no bars | Logged as `provider_empty`; does not fail the batch |

The distinction matters: `rows_written == 0` with no error is a valid outcome (an asset simply had no data in the range). Treating it as a failure would produce false alarms in batch reports.

## 8. Smoke Tests

- `scripts/smoke_f2a2.py`: Verifies the fallback chain `[Binance → Kraken → CoinGecko]` with a two-day recent window.
  - `--skip-binance` forces Kraken.
  - `--skip-binance-kraken` forces CoinGecko.
- `scripts/smoke_f2a3.py`: Discovers the universe, persists the registry, and optionally downloads seven days of H1 for the full universe (`--download`).

Neither script is collected by pytest because they live under `scripts/`, not `tests/`.

## 9. Adding a New Venue

To add a provider for a new exchange:

1. Add the venue symbol for every asset in `src/locus/data/symbols.py`. Run `make ci` — the guardrails will tell you which assets are missing.
2. Implement `MarketDataProvider` in `src/locus/data/providers/`.
3. Register it in the fallback chain where the batch is configured.
4. Add unit tests with mocked HTTP.
5. Document its limitations here (history depth, rate limits, symbol convention).
