# Market Data Layer

Documentation for the market data layer (phases F2.A.1, F2.A.2, F2.A.3,
F2.B.1). Covers OHLCV and FX data sources, providers, validation,
storage, universe reconstruction, quota enforcement, and known
limitations.

The **semantic** side of FX (settlement vs reporting, numeraire,
conversion rules, multi-currency accounting) is documented in
`docs/currency.md`. This document covers the data layer only: where
rates come from, how they are stored, and how they are rate-limited.
The FX layer is fully closed as of F2.B.2.

## 1. Sources

### 1.1 OHLCV (market data)

The layer fetches OHLCV bars from three public APIs. Each provider
implements the `MarketDataProvider` Protocol and returns a Polars
DataFrame with the canonical `BAR_SCHEMA` columns.

#### Binance (primary)

- **Endpoint:** `GET /api/v3/klines`
- **History:** Full history since pair listing date.
- **Pagination:** `startTime` cursor, 1000 bars per request.
- **Volume:** Reported in base asset units (e.g., BTC for BTCUSDT).
- **Role:** Primary source for historical backtests.

#### Kraken (fallback)

- **Endpoint:** `GET /0/public/OHLC`
- **History:** **Only the most recent 720 bars** (30 days for 1h, 2
  years for 1d). The `since` parameter is ignored for older ranges.
- **Pagination:** `since` cursor, 720 bars per request.
- **Role:** Recent-data fallback; not usable for historical backtests.

The provider raises `ProviderNoDataError` when the returned range
starts after `request.end`, so the fallback chain skips it.

#### CoinGecko (last resort)

- **Endpoint:** `GET /coins/{id}/ohlc`
- **History:** Daily bars only. Intraday timeframes raise
  `ProviderResponseError`.
- **Volume:** Not exposed by the `/ohlc` endpoint; reported as `0`.
- **Rate limit:** 10–30 requests/minute on the free tier.
- **Role:** Emergency fallback when Binance and Kraken both fail.

### 1.2 FX rates

The layer fetches daily FX rates from three public APIs. Each provider
implements the `FxRateProvider` Protocol and returns a Polars DataFrame
with the canonical `FX_PROVIDER_SCHEMA` columns (`pair`, `date`,
`rate`). Semantics: `1 unit of pair.base = rate units of pair.quote`.

#### Frankfurter / ECB (EUR/USD primary)

- **Endpoint:** `GET /{start}..{end}?from=EUR&to=USD` on
  `https://api.frankfurter.dev/v1` (the old `.app` host 301-redirects
  to `.dev/v1`).
- **History:** Full history from 1999. Inclusive on both ends; the
  provider translates the half-open `[start, end)` contract to
  `start..(end - 1 day)`.
- **API key:** None.
- **Role:** Primary source for EUR/USD. No fallback: if the ECB fails,
  the layer fails loudly rather than using an unofficial source.

#### Kraken (USDT/USD primary)

- **Endpoint:** `GET /0/public/OHLC?pair=USDTUSD&interval=1440`.
- **History:** **Only the most recent 720 daily bars** (~2 years). The
  `since` parameter is ignored for older ranges, same cap as the OHLCV
  Kraken provider.
- **API key:** None.
- **Role:** Primary source for USDT/USD.

#### CoinGecko (USDT/USD last resort)

- **Endpoint:** `GET /coins/tether/market_chart/range?vs_currency=usd&from=...&to=...`.
- **History:** Demo tier capped at **365 days per request**. Longer
  ranges raise `FxProviderError`; use Kraken or a wider window.
- **API key:** **Required**. Free demo key at
  <https://www.coingecko.com/en/developers/dashboard>, sent as header
  `x-cg-demo-api-key`.
- **Rate limit:** Enforced client-side (see section 4).
- **Role:** Emergency fallback for USDT/USD when Kraken fails.

### 1.3 Fallback chains

- **OHLCV:** `[Binance, Kraken, CoinGecko]` via
  `FallbackMarketDataProvider`. Picks the first non-empty result.
- **FX EUR/USD:** `[Frankfurter]` only. No fallback in the MVP.
- **FX USDT/USD:** `[Kraken, CoinGecko]` via `FallbackFxProvider`.
  CoinGecko is only called if Kraken fails or returns empty.

### 1.4 Funding rates (perpetual futures)

Funding rates are fetched from a **single venue**, chosen by
accessibility from the deployment region (Spain). See
`docs/g0_protocol_v3.md` §2 for the venue-selection test.

#### Binance Futures (primary and sole source for pivote B)

- **Endpoint:** `GET /fapi/v1/fundingRate` on `fapi.binance.com`.
- **History:** full since perpetual contract listing (2020 for BTC/ETH,
  later for smaller perps).
- **Cadence:** 8h for most perps (00:00, 08:00, 16:00 UTC). Some newer
  pairs use 4h or 1h; the provider captures whatever the exchange
  returns and does not assume 8h.
- **Pagination:** `startTime` cursor, 1000 funding events per request
  (~333 days at 8h cadence).
- **API key:** none required for the historical endpoint.
- **Role:** sole source for funding rates in pivote B. No fallback
  chain in this cycle; if Binance Futures becomes inaccessible, the
  cycle stops and pivots to a different venue (Bybit, OKX) only after
  re-running the accessibility test.

`FallbackFxProvider` **does not merge** results across providers: FX
rates from different sources disagree, and mixing them silently would
corrupt downstream PnL. The caller always sees one source's data,
tagged with `source`.

## 2. Symbol Translation

`src/locus/data/symbols.py` maps canonical asset IDs to per-venue
symbols. The mapping is **hardcoded on purpose**:

- There is no universal identifier between venues.
- Historical renames (`MATIC` → `POL`, `RNDR` → `RENDER`, `FTM` → `S`)
  and collisions (`UNI` could be Uniswap or a regional token) require
  human review.
- The universe of the final holdout is pre-registered (see plan §8.15);
  an auto-discovered mapper would silently change the universe between
  runs.

Three guardrails in `tests/unit/data/test_symbols.py`:

- `test_all_assets_mapped_for_all_venues`: Fails if an asset lacks an
  entry for one venue.
- `test_no_duplicate_symbols_within_a_venue`: Fails if two assets map
  to the same symbol in the same venue.
- `test_reverse_lookup_roundtrip`: Verifies `to_asset_id(to_symbol(x)) == x`.

**Adding a new asset:** Edit `_SYMBOLS` and run `make ci`. The first
guard will tell you which venues you forgot.

FX pairs are not part of `_SYMBOLS`: they are identified by canonical
name (`EURUSD`, `USDTUSD`) and each FX provider knows which pairs it
supports.

### 2.2 Perpetual futures symbols

Perpetual futures use the same canonical `asset_id` and the same
`BTCUSDT`-style symbol as spot on Binance. The `SymbolMapper` is
reused unchanged; no separate perp map is needed for the initial
universe (BTC, ETH, SOL, BNB, XRP).

**Caveat:** not every spot pair has a perpetual. Stablecoins, wrapped
tokens, and some newer listings have spot pairs but no perp. The
perp universe is a **strict subset** of the spot universe. The source
of truth for "which assets have a perp at `t`" is the
`PerpUniverseBuilder` (§5.2), not `_SYMBOLS`. Perp discovery reports
which assets from `_SYMBOLS` have a perp on the chosen venue.

## 3. Validation

`src/locus/data/validation/ohlcv.py` validates the whole DataFrame
using vectorized operations before storage.

Checks:

- Schema matches `BAR_SCHEMA` exactly (columns and dtypes).
- No nulls.
- OHLC coherence:
  - `high >= max(open, close, low)`
  - `low <= min(open, close, high)`
  - All prices are `> 0`.
- `volume >= 0`.
- Timestamps are strictly sorted, contain no duplicates, and are
  timezone-aware UTC.

Validation failures raise `ProviderResponseError` and abort the
ingestion run. A bad bar does not enter storage.

**FX validation:** `FxRate` enforces `rate > 0` and timezone-aware
`date` at construction. The store trusts the provider's filtering and
does not re-validate the DataFrame; this is acceptable because FX
providers are constrained to one pair and one granularity per call.

### 3.2 Funding validation

Funding data has its own schema, distinct from `BAR_SCHEMA`:

```text
FUNDING_SCHEMA:
asset_id: Utf8
venue: Utf8
symbol: Utf8
timestamp: Datetime("us", "UTC") # publication time, not period end
funding_rate: Float64 # signed
```


`validate_funding(df, expected_cadence=timedelta(hours=8))` checks:

- Exact schema (columns and dtypes).
- No nulls.
- `funding_rate` is finite. Typical range ±0.0075 per 8h period
  (±0.75%). Values outside ±5% are logged as warnings but not rejected
  (rare events do happen).
- Timestamps strictly sorted per asset, no duplicates.
- **Continuity**: spacing between consecutive timestamps per asset is
  consistent. A gap > 1.5× the asset's median spacing raises
  `ProviderResponseError`. Reuses the same `check_continuity` logic as
  OHLCV, parameterized by expected cadence.

Validation failures abort the ingestion of that asset; the rest of the
batch continues. A bad funding event does not enter storage.

## 4. Storage

### Market Data (Parquet)

```text
data/processed/market=crypto/venue={venue}/asset={asset}/timeframe={tf}/year={yyyy}/bars.parquet
```

One file per `(venue, asset, timeframe, year)`. Reads use Polars' Hive scanning, so a range query transparently unions the relevant years.

### FX Rates (Parquet)

```text
data/processed/fx/pair={pair}/source={source}/year={yyyy}/rates.parquet
```

One file per (pair, source, year). Each row carries `pair`, `date`, `rate`, `source`, `fetched_at`. All sources for a pair coexist; the caller selects one via `source=` on read.

### Funding Rates (Parquet)

```text
data/processed/funding/venue={venue}/asset={asset}/year={yyyy}/rates.parquet
```

One file per (venue, asset, year). Same Hive-scanning read pattern as market data. Fields: `asset_id`, `venue`, `symbol`, `timestamp`, `funding_rate`. No source column: funding is single-source per venue by construction (one API per venue).

### Funding Raw (Parquet) — first raw layer in the project

```text
data/raw/funding/{venue}/{asset}/raw.parquet
```

Literal API response per request, plus a `fetched_at` timestamp. Never mutated after write. Used for audit and replay.

#### Why a raw layer for funding and not for OHLCV:

- The funding response shape has changed historically (Binance migrated the `/fapi/v1/fundingRate` payload fields in 2022).
- Funding history is small (~3,285 events per asset over 3 years), so the raw dump is cheap.
- A pre-registered G0-v3 study needs byte-for-byte reproducible input.

OHLCV still writes only to processed. The funding raw layer is the precedent for a later refactor of the OHLCV raw layer, not a prerequisite.

### CoinGecko Quota State (JSON)

```text
data/processed/fx/_quota/coingecko.json
```

Single JSON file: `{"month": "YYYY-MM", "count": N}`. Keyed by UTC month; resets automatically on the 1st. Written atomically (tmp + rename). Corrupt or missing file is treated as a fresh month with a loud warning.

### Universe Registry (Parquet)

```text
data/processed/universe/registry.parquet
```

Single file, one row per (asset, venue) with `first_tradable_ts` and `last_tradable_ts`. Small (~150 rows for the MVP). Overwritten on each discovery run.

### Raw vs. Processed

`data/raw/` and `data/interim/` are reserved but unused in F2.

If a provider response needs to be preserved verbatim (for audit, replay, or debugging), it goes to `data/raw/`. The current pipeline does not persist raw responses; the processed Parquet is the source of truth.

---

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

Listing dates come from `onboardDate` when present, and fall back to the timestamp of the first available `1d` bar otherwise.

Binance's public `exchangeInfo` does not currently include `onboardDate`, so the fallback is the normal path in practice: approximately 50 assets × 1 request each, around 20 seconds total, cached afterwards.

### Known Limitations

- **Delisted assets are not discoverable through public APIs.** All registry entries are live assets in the MVP. The `last_tradable_ts` field supports delisted assets in the contract, but no data source populates it yet.
- **Kraken and CoinGecko do not expose listing dates** through their public APIs. Discovery is Binance-only.
- **The first-bar fallback is approximate:** It may be later than the real listing date if Binance started storing klines later, or earlier if a symbol was delisted and relisted. This limitation is documented and accepted.

### 5.2 Perp universe (point-in-time)

```text
data/processed/universe_perps/registry.parquet
```

Single file, one row per (asset, venue):

- `asset_id: Utf8`
- `venue: Utf8`
- `list_date: Datetime("us", "UTC")` — first funding timestamp available
- `delist_date: Datetime("us", "UTC")` | null — last funding timestamp, or null if still live

A perp is in the universe at `t` if `list_date <= t < delist_date`. `PerpUniverseBuilder.build(t, constraints)` is the perp equivalent of `UniverseBuilder`, pure function of the registry and `t`.

**Discovery**: `/fapi/v1/exchangeInfo` filtered by
`contractType == "PERPETUAL"` and `status == "TRADING"` provides the currently-live perps with `onboardDate` where present (falls back to
first funding timestamp otherwise). Historical delistings are **not discoverable** through the live API; they are populated manually
for the initial universe (BTC, ETH, SOL, BNB, XRP are all live, no delistings in the 3-year window). This is documented honestly and revisited if the universe expands.

**Why this matters**: pivote B is a cross-sectional strategy. A perp that lists mid-window and is treated as present from the start, or a
delisted perp valued at 0, would contaminate the cross-sectional ranking. The point-in-time universe is a **prerequisite**, not a follow-up.

---

## 6. Ingestion

### Single Asset

`MarketDataIngestor.ingest(request)` runs the full pipeline for one asset:

1. Fetch data.
2. Normalize `asset_id` and `timeframe`.
3. Validate the data.
4. Write the data to storage.
5. Returns an `IngestionResult`.

### Batch

`BatchIngestor.run(snapshot, timeframe, start, end)` iterates over a `UniverseSnapshot` and returns a `DownloadReport`.

Per-asset failures are captured in the report, not raised: a single delisted or missing asset does not abort the batch.

`sleep_between_assets_seconds` keeps the batch within the provider's rate limit (set to approximately 0.2 seconds for Binance in the smoke script).

### FX Ingestion

`FxRateIngestor` exposes two methods:

- **`run_batch(start, end)`**: downloads the full range for every pair,
  ignoring the cache. Used once at setup and for full refreshes.
- **`run_incremental(lookback_days=7)`**: fetches the last N days per
  pair, **skipping pairs whose store already has a rate for the
  current UTC day** (from any source). This makes daily runs
  idempotent: re-running on the same day returns `skipped` for
  everything and does not call any provider.

Both are invoked **manually or by an external cron** (OS scheduler or
GitHub Actions). Neither auto-triggers. Recommended cadence: once per
day at 01:00 UTC, after Kraken's daily close and Frankfurter's daily
update. A built-in scheduler is out of scope for the MVP and belongs to
B6/B10.

Per-pair outcomes are captured in `FxIngestionReport` as
`FxPairIngestionResult`:

- `written`: rows written to the store.
- `skipped`: the pair was up to date (no provider called).
- `reason`: error message or `"empty"` when a provider returned no
  data for the requested range.

The ingestor never aborts the batch on a per-pair failure: the
remaining pairs proceed and the report lists which failed and why.

The `FxConverter` (in `src/locus/data/fx/converter.py`) consumes the
store to convert amounts between currencies. Its semantics
(staleness, routing, audit trail) are documented in
`docs/currency.md`, not here.

### Funding ingestion

`FundingIngestor` exposes two methods:

- **`run_batch(start, end)`**: for each asset in the perp universe at
  `end`, fetch funding in `[max(list_date, start), min(delist_date or end, end))`.
  Skip assets listed after `end` or delisted before `start`. Per-asset
  outcomes captured in a `FundingIngestionReport`.
- **`run_incremental(lookback_days=7)`**: fetch only the last N days of
  funding. Cache skip: if the store already has the most recent funding
  timestamp for an asset, skip the fetch. Idempotent: re-running on the
  same day returns `skipped` for all assets.

Both are invoked **manually or by an external cron**. Neither
auto-triggers. Same as FX (§6.3). Recommended cadence: once per day
at 01:00 UTC.

**Rate limits (Binance Futures):** weight 1 per request, 2400 weight/min
general limit. 5 assets × 1 request/asset ≈ 5 requests per batch.
**~480× headroom.** No monthly quota. Retries: same `tenacity` policy
as OHLCV (5 attempts, exponential backoff).

**Failure semantics:** same `ProviderError` hierarchy as OHLCV.
- `ProviderNoDataError` → skip asset, log `funding_empty`.
- `ProviderResponseError` → skip asset, log `funding_failed`.
- Empty DataFrame with no error → valid outcome (asset not yet listed
  or just delisted), log `funding_skipped`.
- Missing price for a position during funding accrual →
  `MissingPriceError` from `PortfolioAccounting`. Engine fails loudly.
  Same contract as OHLCV.

**Reproducibility:** all funding ingestion and downstream scripts accept
`--end-date` with a fixed default (`2026-10-01`). See
`docs/g0_protocol_v3.md` §4.3.

---

## 7. Rate Limits and Quotas

### Per-request retries (all providers)

`tenacity` retries `httpx.HTTPError` with exponential backoff. Retries consume quota when the provider enforces one (CoinGecko). Configured per provider.

### CoinGecko demo tier

`CoinGeckoRateLimiter` enforces both limits client-side, before every HTTP request:

| Limit | Behavior |
|---|---|
| 100 calls/minute | In-memory sliding window; blocks with sleep until a slot frees. Never fails. |
| 10,000 calls/month | Persistent JSON counter at `data/processed/fx/_quota/coingecko.json`, keyed by UTC month. Resets on the 1st. Raises `FxProviderError` when exhausted. |
| Warning at 9,500 | Logged (`quota_near_limit`) before hitting the hard limit. |

The exhausted-quota error includes the next reset date:

```text
CoinGecko monthly quota exhausted (10000/10000 in 2026-10).
Next reset: 2026-11-01T00:00:00+00:00.
```

`get_status()` returns `{month, used, max, remaining, warn_threshold, next_reset}` without consuming quota. Safe to call anytime; used by the smoke script and (in F2.B.2) by `FxRateIngestor` as a pre-check.

**Counter semantics:** incremented *before* the HTTP request. Failed requests still consume a slot. This is the conservative choice and avoids exceeding CoinGecko's server-side limit if the client is buggy.

**No retries on quota exhaustion:** `FxProviderError` is not an `httpx.HTTPError`, so tenacity does not retry it. The caller decides whether to skip the pair, back off until `next_reset`, or fail the batch.

### Frequency in practice

FX rates update once per business day (ECB) or once per day (Kraken, CoinGecko). Calling any provider more than once per day returns the same data and wastes quota. With the fallback chain (`[Kraken, CoinGecko]`), CoinGecko sees 0 calls/day in normal operation and ~30 calls/month only in the worst case (Kraken down every day). The 10,000/month limit is ~300× the expected usage.

---

## 8. Failure Semantics

### OHLCV

| Error type | Meaning | Fallback behavior |
|---|---|---|
| `ProviderResponseError` | Malformed or semantically invalid response | Skip asset, log `provider_failed` |
| `ProviderRateLimitError` | Rate limit exhausted after retries | Skip asset, log `provider_failed` |
| `ProviderNoDataError` | Requested range has no data | Skip asset, log `provider_failed` |
| `httpx.HTTPError` | Transport error | Retried by Tenacity, then `ProviderResponseError` |
| Empty DataFrame, no error | Provider responded with no bars | Logged as `provider_empty`; does not fail the batch |

The distinction matters: `rows_written == 0` with no error is a valid outcome (an asset simply had no data in the range). Treating it as a failure would produce false alarms in batch reports.

### FX

| Error type | Meaning | Fallback behavior |
|---|---|---|
| `FxProviderError` | Provider failed (network, parse, unsupported pair, quota exhausted) | Fallback chain tries next; if last, re-raises to caller |
| Empty DataFrame, no error | Provider responded with no rates for the range | Fallback chain tries next |

### API key

`CoinGeckoUSDTUSDProvider` fails at construction if the API key is missing. The error names the environment variable and points to the dashboard. No HTTP request is attempted.

---

## 9. Smoke Tests

### OHLCV

- `scripts/smoke_f2a2.py`: Verifies the fallback chain `[Binance → Kraken → CoinGecko]` with a two-day recent window.
  - `--skip-binance` forces Kraken.
  - `--skip-binance-kraken` forces CoinGecko.
- `scripts/smoke_f2a3.py`: Discovers the universe, persists the registry, and optionally downloads seven days of H1 for the full universe (`--download`).

### FX

- **`scripts/smoke_f2b1.py`**: fetches EUR/USD from Frankfurter and
  USDT/USD from Kraken and CoinGecko, writes all three to the FX
  store, then prints the available sources per pair and the
  CoinGecko quota status. Skips CoinGecko if the API key is not
  configured.
- **`scripts/smoke_f2b2.py`**: runs `run_incremental(lookback_days=7)`,
  prints per-pair outcomes (`written` / `skipped` / `failed`), then
  executes four conversions with `FxConverter.convert_with_details`
  (`EUR→USDT`, `USDT→EUR`, `EUR→USD`, `USDT→USD`). Verifies that
  two-hop conversions route correctly, that source per pair is
  logged, and that the incremental cache skip works on re-runs.

### Funding

- **`scripts/smoke_b1.py`**: fetches 3 years of funding for BTC, verifies
  continuity and schema, does a store roundtrip, prints per-asset stats
  (count, min, max, mean of `funding_rate`, null count). Fails loudly if
  the API is inaccessible (HTTP 451/403) or if continuity is violated.
- **`scripts/check_backfill_funding.py`**: quick coverage check per
  asset (first/last timestamp, count, cadence). Analogous to
  `check_backfill.py` for OHLCV.

---

## 10. Adding a New Venue

To add a provider for a new exchange:

### OHLCV

1. Add the venue symbol for every asset in `src/locus/data/symbols.py`. Run `make ci` — the guardrails will tell you which assets are missing.
2. Implement `MarketDataProvider` in `src/locus/data/providers/`.
3. Register it in the fallback chain where the batch is configured.
4. Add unit tests with mocked HTTP.
5. Document its limitations here (history depth, rate limits, symbol convention).

### FX

1. Implement `FxRateProvider` in `src/locus/data/fx/`. The interface is `fetch_rates(pair, start, end) -> pl.DataFrame` with `FX_PROVIDER_SCHEMA`.
2. If the source has a quota (per-minute, per-month), add a `RateLimiter` and enforce it before each request. Exhaustion must raise `FxProviderError` with the next reset date.
3. Register it in the fallback chain: `FallbackFxProvider([...])`.
4. Add unit tests with mocked HTTP (including the quota paths).
5. Document its limitations here (history depth, rate limits, API key requirement, supported pairs).

### Funding

1. Add the asset to `_SYMBOLS` if not present (reuses the spot symbol).
2. Run `PerpUniverseDiscovery` to update
   `universe_perps/registry.parquet`.
3. Run `FundingIngestor.run_batch(start, end)` for the new asset.
4. Verify schema, continuity, and store roundtrip.
5. Document any perp-specific limitation (list_date known? delisted?
   cadence 8h or different?).

Adding a new **venue** for funding (Bybit, OKX) requires a new
`FundingProvider` implementing the same interface, plus a new raw and
processed layout under `venue=<new>`. The `FundingStore` partitions by
`venue` from day 1, so no storage migration is needed.

---

## 11. Adding a New FX Pair

To add a pair (e.g., GBPUSD):

1. Add it to `data.fx.pairs` in `configs/base.yaml`.
2. Add a provider that supports it. Frankfurter supports any pair with EUR as base; a GBPUSD provider would need a different source.
3. Wire it into the fallback chain in `FxRateIngestor` (F2.B.2).
4. Document its sources and limitations here.

The `FxPair.parse()` helper assumes a 3-character base and a quote of the remainder (`EURUSD` → `EUR/USD`, `USDTUSD` → `USDT/USD`). A general currency registry is needed for pairs like `BTCUSDT` (4-char base) if they ever enter FX scope; for the MVP, all pairs fit the 3-char-base convention.

---

## 12. Downstream: Feature Layer

Feature engineering (F3) consumes OHLCV bars from this layer and
produces a feature dataset. The feature layer does not modify or
persist OHLCV; it reads from `ParquetBarStore` and returns a DataFrame
with the canonical `BAR_SCHEMA` columns plus one column per feature.

Feature definitions, formulas, families, leakage guarantees, and the
config-driven registry are documented in **`docs/features.md`**. The
feature pipeline assumes:

- Bars are sorted by `(asset_id, timestamp)` and free of duplicates
  (guaranteed by `validate_ohlcv`).
- Timestamps are timezone-aware UTC.
- The DataFrame has exactly `BAR_SCHEMA` columns before features are
  applied.

Features are backward-looking by contract: at row `t`, a feature's
value depends only on data at timestamps `<= t` within the same asset.
This is enforced by `tests/unit/features/test_leakage.py`. Any change
to this layer that breaks the sort, dedup, or UTC guarantees will
silently invalidate the leakage tests; if you add a provider that
violates any of them, extend `validate_ohlcv` before shipping.


## 13. Downstream: Dataset Layer

Feature engineering (F3) and dataset assembly (F4) consume OHLCV bars
from this layer but do not modify them. The dataset layer builds
labels, temporal splits, purge gaps, and uniqueness weights; its
contract and semantics are documented in **`docs/datasets.md`**.

The dataset layer inherits all guarantees from this one: sorted bars,
no duplicates, UTC timestamps. If you add a provider that violates any
of them, extend `validate_ohlcv` before shipping, or the splitter and
label builder will produce silently wrong results.


## 14. Downstream: Backtest Layer

The event-driven backtester (F6) consumes OHLCV bars from this layer
plus the `sigma_daily` and `daily_volume_quote` per-bar estimates from
`backtest/market_state.py` (rolling, backward-looking). It does not
modify the data; it computes fills, costs, and equity curves.

Execution model, capital-aware constraints, cost scenarios, and the
`CapitalScalingExperiment` are documented in **`docs/backtest.md`**.
The backtester inherits all guarantees from this layer: sorted bars,
no duplicates, UTC timestamps, no nulls in required columns. Any
change that breaks them (new provider, validation relaxation)
invalidates the backtest results silently; extend `validate_ohlcv`
before shipping.

## 15. Downstream: Funding Features

The funding feature layer (B.2) consumes the funding store plus OHLCV
prices (for valuation). Feature definitions, strategy semantics,
market-neutrality constraints, and the pre-registered criterion are in
**`docs/g0_protocol_v3.md`**, not here. This section records only the
contract the data layer provides to that layer:

- Funding timestamps are **publication time**, not period end. A funding
  event at `t` is available for any price bar with `timestamp >= t`.
- The perp universe is **point-in-time** (list/delist dates from §5.2).
- Continuity is enforced at ingestion (§3.2) and again at feature
  computation. Gaps fail loudly, not silently.
- Any feature that reads funding at `t` reads only events with
  `timestamp <= t`. Leakage tests in `tests/unit/features/test_funding_leakage.py`
  verify this.

Any change to this layer that breaks point-in-time discipline, point-in-
time universe, or continuity **invalidates the pre-registration** and
requires a new protocol version (`v3.1`, `v4`, …). The pre-registration
is the contract; the data layer is its foundation.

### Known coverage gaps (snapshot 2026-10-01)

| Assets | Bars | Range |
|---|---|---|
| BTC, ETH, SOL, BNB, XRP | 43,799 | 2021-10-02 → 2026-09-30 |
| ADA, AVAX, DOGE, DOT, LINK | 17,021 | 2024-10-21 → 2026-10-01 |

The common timestamp range across all ten assets is
**2024-10-21 20:00 UTC → 2026-09-30 23:00 UTC** (17,020 hourly bars).
Scripts that run a portfolio-level backtest across the full universe must
align to this range (`scripts/smoke_b3.py` does it automatically).

Funding data is present only for **BTC, ETH, SOL, BNB, XRP** in this
snapshot. Portfolio-level funding metrics therefore reflect only the covered
subset of capital (50% under equal weight). Extending funding coverage to the
remaining five assets is tracked as a follow-up.

---

## 16. Cost assumptions (P4)

Labels use a fixed round-trip cost assumption derived from the same
scenario as the backtest. The three scenarios from `configs/base.yaml`:

| Scenario | Fee | Spread | Slippage | Impact | Round-trip |
|---|---|---|---|---|---|
| optimistic | 5 | 2 | 2 | 0.5 | 2 × 8.75 = 17.5 bps |
| central | 10 | 5 | 5 | 1 | 2 × 18.5 = 37 bps |
| pessimistic | 20 | 10 | 10 | 2 | 2 × 33 = 66 bps |

The central scenario is the default for label construction (P4) and
walk-forward (P6). T1 pre-registration documents which scenario the
labels and verdict use; changing it mid-experiment is a methodological
change per plan §4.1 and requires a new trial.

The pessimistic round-trip (66 bps) is what E8 in plan §7.4 tests: a
strategy that survives 66 bps per round-trip is robust to cost
assumptions. A strategy that only survives the optimistic scenario
(17.5 bps) is fragile and should not be promoted.

### `impact_bps` vs `impact_k`

`configs/base.yaml` carries two impact fields under `backtest:`:

    impact_model: "sqrt_volume"
    impact_k:     1.0   # engine: impact ∝ impact_k * sqrt(size / daily_volume)
    impact_bps:   1.0   # labels: fixed bps added to the round-trip cost

The engine uses `impact_k` because it knows the position size and the
daily volume at execution time. Labels use `impact_bps` because at label
time neither is known: the model's future weights depend on training,
which depends on the labels. A fixed assumption is the only defensible
choice (plan §7.3 proposal: 1 bps per side).

In the central scenario they coincide (1.0). Other scenarios can differ:
a pessimistic scenario may set `impact_k: 2.0` (engine) and
`impact_bps: 2.0` (labels), or diverge deliberately if the engine's
model is deemed unrealistic at small sizes.

### Labels path

After running `scripts/build_labels.py`, labels land in
`data/processed/labels/{ASSET}.parquet` with one row per (timestamp,
asset) and columns:

- `future_return_{h}h` (gross, diagnostics)
- `future_return_{h}h_net` (net, training target)
- `probability_positive` (binary)

Files are per-asset, one per U10 asset. Regenerating with a different
cost scenario is a new file, not a modification.

Because label files preserve the bar schema, `build_labels.py` and
`verify_splits.py` exclude the `labels` directory explicitly when
discovering bars. This avoids re-processing labels as if they were raw
OHLCV.

## Coverage snapshot for label construction (2026-10-30)

Labels are built on the aligned U10 bars, using the same bar discovery
and alignment as the backtest:

| Item | Value |
|---|---|
| Assets | 10 (BTC, ETH, SOL, BNB, XRP, ADA, DOGE, AVAX, DOT, LINK) |
| Bars (aligned) | 170,200 (17,020 timestamps × 10 assets) |
| Range | 2024-10-21 20:00 UTC → 2026-09-30 23:00 UTC |
| Label rows | 169,950 (tail rows without exit price dropped) |
| Horizon | 24h (T1), 168h (diagnostics) |
| Cost scenario | central (37 bps round-trip) |

The bars themselves come from `scripts/backfill.py` (spot OHLCV from
Binance). Funding for the same U10 universe comes from
`scripts/backfill_funding.py` (perp funding from Binance Futures, 21,270
events). Both datasets share the same aligned timestamp range; any
mismatch means the bars backfill needs to run again before building
labels.
