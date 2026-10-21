# Market Data Sources

## Historical OHLCV (Binance)
- Endpoint: `GET /api/v3/klines`
- History: full since pair listing date.
- Pagination: `startTime` cursor, 1000 bars per request.
- Use: primary source for historical backtests.

## Fallback OHLCV (Kraken)
- Endpoint: `GET /0/public/OHLC`
- **Limitation**: returns only the most recent 720 bars (30 days for 1h,
  2 years for 1d). `since` is ignored for older ranges.
- Pagination: `since` cursor, 720 bars per request.
- Use: recent data, live execution fallback (B5), NOT historical backtests.

## Last-resort OHLCV (CoinGecko)
- Endpoint: `GET /coins/{id}/ohlc`
- **Limitation**: daily bars only (no 1h/4h). Volume reported as 0.
- Rate limit: 10–30 req/min (free tier).
- Use: emergency fallback when Binance and Kraken both fail.

## Symbol translation
- See `src/locus/data/symbols.py`.
- BTC → `BTCUSDT` (Binance), `XBTUSDT` (Kraken), `bitcoin` (CoinGecko).
