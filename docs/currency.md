# Currency, Numeraire, and Multi-Asset Accounting

Documentation for how the system handles currency: which one it settles
in, which one it reports in, how FX rates are fetched and converted, and
what is deliberately out of scope for the MVP.

This document is the semantic counterpart of `docs/data.md`. Data.md
covers *where rates come from*; this file covers *what the rates mean
for PnL, portfolio accounting, and reporting*.

## 1. Three concepts that must not be confused

| Concept | Definition | MVP value |
| :--- | :--- | :--- |
| **Settlement currency** | The currency the exchange settles trades in and holds as balance. | `USDT` |
| **Reporting currency** | The currency the backtest metrics (Sharpe, drawdown, PnL) are expressed in. | `USDT` (primary); `EUR` (display only) |
| **Numeraire** | The unit of account against which "wealth" is measured. | `USDT` |

The three are independent. The MVP uses the same currency (USDT) for
all three, which simplifies everything. Post-MVP, the reporting
currency may become `EUR` if the user is based in the eurozone.

## 2. Four currency layers

The system operates across four layers. Not all are modeled in the MVP;
the unmodeled ones are documented here and revisited in B5/B9.

### 2.1 Asset risk (modeled)

The asset's price moves against its quote currency. `BTCUSDT` rises or
falls in USDT. This is what the price series measures and what the
backtester simulates.

### 2.2 Peg risk (documented, not modeled)

`USDT` is a stablecoin pegged to `USD`. The peg has broken twice
historically (2018, 2022), briefly, by up to a few percent. The MVP
treats `1 USDT = 1 USD` for PnL purposes and **does not** model
depeg. It is a known limitation.

### 2.3 FX risk (documented, not modeled in the backtest)

`EUR/USD` and `USDT/USD` move. A PnL of `+10% USDT` is not necessarily
`+10% EUR`. If the euro appreciates 5% against the dollar, the same
USDT PnL is worth ~5% less in EUR. The MVP reports PnL in USDT and
converts to EUR for display using the FX layer. The conversion does
**not** feed back into the backtest metrics.

### 2.4 Conversion cost (documented, not modeled)

Depositing EUR and converting to USDT on a CEX costs 10–30 bps in
spread. Withdrawing and converting back costs another 10–30 bps. The
MVP assumes the settlement balance is already in USDT and does not
model the cost of getting there. It is a known limitation.

## 3. MVP decisions

### 3.1 Settlement, reporting, numeraire = USDT

All three are USDT. This is the simplest possible configuration and
matches the settlement currency of every major crypto exchange's spot
market for pairs against USDT.

### 3.2 Universe: only pairs against USDT

`BTCUSDT`, `ETHUSDT`, etc. Crypto/crypto pairs (`ETHBTC`) are out of
scope for the MVP. See section 7.

### 3.3 Cash: single currency (USDT)

`PortfolioState.cash` is a single `float` in the settlement currency.
Multi-currency cash holdings are out of scope for the MVP and will be
added in B5 if needed.

### 3.4 EUR as display only

The dashboard (F16) converts PnL from USDT to EUR using the FX layer
for display. The conversion is deterministic and reversible: it uses
the historical rate at the timestamp of each observation, not a single
rate for the whole period. The backtest metrics themselves are USDT.

### 3.5 No scheduler

`FxRateIngestor.run_incremental()` is invoked manually or by an external
cron (OS scheduler or GitHub Actions). Neither method auto-triggers. A
built-in scheduler is out of scope for the MVP (B6/B10).

Recommended cadence: **once per day at 01:00 UTC**. After Kraken's
daily close and Frankfurter's daily update. Running more often wastes
quota and returns the same data.

## 4. FX sources

| Pair | Primary | Fallback | Notes |
| :--- | :--- | :--- | :--- |
| `EURUSD` | Frankfurter (ECB) | none | No unofficial fallback; if ECB fails, the layer fails loudly. |
| `USDTUSD` | Kraken (public OHLC) | CoinGecko (demo API key) | CoinGecko requires a free API key; see `docs/data.md`. |

`FallbackFxProvider` tries providers in order and returns the first
non-empty result. It **does not merge** across providers: FX rates from
different sources disagree, and mixing them silently would corrupt
downstream PnL.

## 5. Conversion rules (hard)

These are the rules the `FxConverter` enforces. They are the semantic
counterpart of the numeric behaviour.

### 5.1 Report in `CapitalScenario.currency`

Every backtest result is reported in the currency specified by the
`CapitalScenario`. For the MVP, that currency is `USDT`. No metric
mixes currencies.

### 5.2 EUR is display only

Any conversion to EUR is for presentation. It does not change Sharpe,
drawdown, or any decision. The dashboard renders both: primary in
USDT, secondary in EUR, with the FX rate used and its date.

### 5.3 Record the rate and its date

When a value is converted to EUR, the rate used and its date are
logged. This is required for audit: a converted PnL without a rate
and date is not reproducible.

### 5.4 Cash is single-currency (USDT)

`PortfolioState.cash` is a `float` in the settlement currency. Multi-
currency cash is post-MVP.

### 5.5 Conversion cost is not modeled

The spread for converting EUR↔USDT on a CEX is a known limitation. It
does not appear in backtest PnL. The user is expected to size their
initial capital with this cost in mind.

## 6. Using `FxConverter`

```python
from locus.data.fx.converter import FxConverter
from locus.data.fx.store import FxRateStore

store = FxRateStore(Path("data/processed"))
converter = FxConverter(
    store=store,
    stale_days_threshold=7,
    source_preferences={
        "EURUSD": ["frankfurter"],
        "USDTUSD": ["kraken", "coingecko"],
    },
)

# Simple conversion
pnl_eur = converter.convert(
    amount=1000.0,
    from_currency="USDT",
    to_currency="EUR",
    date=datetime(2024, 6, 1, tzinfo=UTC),
)

# Auditable conversion
result = converter.convert_with_details(
    amount=1000.0,
    from_currency="USDT",
    to_currency="EUR",
    date=datetime(2024, 6, 1, tzinfo=UTC),
)
print(result.rate, result.date_used, result.sources)
# 0.92, 2024-05-31 (Friday's rate for a Saturday), {"USDTUSD": "kraken", "EURUSD": "frankfurter"}

# Weekend and Holiday Handling

Daily FX rates are not published on weekends or holidays. If the requested date has no rate, the converter falls back to the most recent rate within `stale_days_threshold` (default: 7 days). Older rates raise `FxRateNotFoundError`. The actual `date_used` is returned in the audit details.

---

# Routing

The converter builds a currency graph from the available pairs and finds the shortest path between the source and target currencies. For the MVP (`EURUSD`, `USDTUSD`), any pair of currencies is at most 2 hops:

- **EUR → USD**: `EURUSD` × rate
- **EUR → USDT**: `EURUSD` × rate, then `USDTUSD` ÷ rate
- **USDT → EUR**: `USDTUSD` × rate, then `EURUSD` ÷ rate

Adding a new pair (e.g. `GBPUSD`) automatically extends the graph.

---

# 7. Out of Scope for the MVP

## 7.1 Crypto/crypto pairs

`ETHBTC`, `SOLETH`, etc. are excluded. Reasoning:

- A second numeraire (`BTC`, `ETH`) fluctuates against USDT, so attribution and Sharpe become ambiguous.
- Liquidity is 10–100× lower than USDT pairs; the impact model would need recalibration per pair.
- Triangular arbitrage detection, feature deduplication by underlying asset, and correlation-adjusted DSR all become research problems, not implementation details.

Crypto/crypto pairs are deferred to B9 (Multi-market).

## 7.2 Multi-currency holdings

`PortfolioState` holds a single cash balance. Multi-currency holdings (BTC + ETH + USDT simultaneously, with per-currency PnL attribution) are B5.

## 7.3 Conversion cost and depeg

The spread of EUR↔USDT conversion and the depeg of USDT from USD are documented as known limitations. They are modeled in B5 if the system goes live.

## 7.4 Fiat on/off ramp

Bank transfers, SEPA, Open Banking are handled by the exchange's fiat ramp. The system only sees the resulting USDT balance. Direct bank integration would require PSD2/AISP licensing and is out of scope.

## 7.5 On-chain wallets and DEX

Self-custody wallets and DEX execution use a fundamentally different execution model (private keys, gas, nonces, AMM routing, MEV). They would need a dedicated adapter and order state machine (B11 if it ever lands).

---

# 8. Reporting Example

A backtest runs from 2024-01-01 to 2024-12-31 with `CapitalScenario` in USDT. The results:

- **Sharpe:** 1.42 (USDT)
- **Max drawdown:** -18.3% (USDT)
- **CAGR:** 24.7% (USDT)
- **PnL:** +2,470 USDT on 10,000 USDT capital

For the dashboard, the same period converted to EUR (display only):

- **PnL:** +2,215 EUR (using historical EUR/USD at each date)
- **CAGR:** +21.8% (EUR, using historical EUR/USD at the boundaries)

The two sets of numbers are not the same, and that is the point: the strategy made 24.7% in USDT and 21.8% in EUR because the euro appreciated against the dollar over the year. The USDT numbers are the strategy's performance; the EUR numbers are what a euro-based user would have experienced.

The report always states both, with the FX rate used and its date.

---

# 9. Frequency and Scheduling

FX rates update:

- **EUR/USD**: once per business day (ECB, ~16:00 CET).
- **USDT/USD**: continuously, but the daily close is at 00:00 UTC.

`FxRateIngestor.run_incremental(lookback_days=7)` should run once per day, after 01:00 UTC. Recommended trigger: GitHub Actions schedule, OS cron, or a manual `make fx-incremental`. No built-in scheduler.

Running more often than once per day returns the same data and wastes CoinGecko quota. In normal operation, CoinGecko is not called at all (Kraken succeeds); it is a fallback for outages, not a regular source.

Quota behavior, next reset date, and warnings are documented in `docs/data.md` section 7.

---

# 10. Cross-references

- `docs/data.md` — FX data sources, storage layout, quota enforcement.
- Plan §5.8 — original specification of the currency layer.
- Plan §21 — documented risks (depeg, conversion cost, crypto/crypto).
- `src/locus/data/fx/converter.py` — the converter implementation.
- `src/locus/data/ingestion/fx.py` — the ingestor implementation.
