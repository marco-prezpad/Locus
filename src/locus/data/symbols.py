"""
symbols.py

Cross-venue symbol translation. Each asset has one canonical identifier
(the key of the mapping) and one symbol per venue. Keeps providers
decoupled from each other: Binance does not need to know that Kraken
calls BTC "XBT".

Extending to a new asset: add one line to `_SYMBOLS`. Extending to a new
venue: add one entry per asset under the venue key (a runtime check
will flag missing venues when a provider asks for a symbol).

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

from locus.data.types import ProviderResponseError

_SYMBOLS: dict[str, dict[str, str]] = {
    "BTC": {"binance": "BTCUSDT", "kraken": "XBTUSDT", "coingecko": "bitcoin"},
    "ETH": {"binance": "ETHUSDT", "kraken": "ETHUSDT", "coingecko": "ethereum"},
    "SOL": {"binance": "SOLUSDT", "kraken": "SOLUSDT", "coingecko": "solana"},
    "BNB": {"binance": "BNBUSDT", "kraken": "BNBUSDT", "coingecko": "binancecoin"},
    "XRP": {"binance": "XRPUSDT", "kraken": "XRPUSDT", "coingecko": "ripple"},
    "ADA": {"binance": "ADAUSDT", "kraken": "ADAUSDT", "coingecko": "cardano"},
    "DOGE": {"binance": "DOGEUSDT", "kraken": "DOGEUSDT", "coingecko": "dogecoin"},
    "AVAX": {"binance": "AVAXUSDT", "kraken": "AVAXUSDT", "coingecko": "avalanche-2"},
    "DOT": {"binance": "DOTUSDT", "kraken": "DOTUSDT", "coingecko": "polkadot"},
    "LINK": {"binance": "LINKUSDT", "kraken": "LINKUSDT", "coingecko": "chainlink"},
    "LTC": {"binance": "LTCUSDT", "kraken": "LTCUSDT", "coingecko": "litecoin"},
    "UNI": {"binance": "UNIUSDT", "kraken": "UNIUSDT", "coingecko": "uniswap"},
    "ATOM": {"binance": "ATOMUSDT", "kraken": "ATOMUSDT", "coingecko": "cosmos"},
    "XLM": {"binance": "XLMUSDT", "kraken": "XLMUSDT", "coingecko": "stellar"},
    "ALGO": {"binance": "ALGOUSDT", "kraken": "ALGOUSDT", "coingecko": "algorand"},
    "FIL": {"binance": "FILUSDT", "kraken": "FILUSDT", "coingecko": "filecoin"},
    "NEAR": {"binance": "NEARUSDT", "kraken": "NEARUSDT", "coingecko": "near"},
    "APT": {"binance": "APTUSDT", "kraken": "APTUSDT", "coingecko": "aptos"},
    "ARB": {"binance": "ARBUSDT", "kraken": "ARBUSDT", "coingecko": "arbitrum"},
    "OP": {"binance": "OPUSDT", "kraken": "OPUSDT", "coingecko": "optimism"},
    "SUI": {"binance": "SUIUSDT", "kraken": "SUIUSDT", "coingecko": "sui"},
    "TIA": {"binance": "TIAUSDT", "kraken": "TIAUSDT", "coingecko": "celestia"},
    "INJ": {"binance": "INJUSDT", "kraken": "INJUSDT", "coingecko": "injective-protocol"},
    "AAVE": {"binance": "AAVEUSDT", "kraken": "AAVEUSDT", "coingecko": "aave"},
    "MKR": {"binance": "MKRUSDT", "kraken": "MKRUSDT", "coingecko": "maker"},
    "SNX": {"binance": "SNXUSDT", "kraken": "SNXUSDT", "coingecko": "havven"},
    "CRV": {"binance": "CRVUSDT", "kraken": "CRVUSDT", "coingecko": "curve-dao-token"},
    "COMP": {"binance": "COMPUSDT", "kraken": "COMPUSDT", "coingecko": "compound-governance-token"},
    "LDO": {"binance": "LDOUSDT", "kraken": "LDOUSDT", "coingecko": "lido-dao"},
    "FTM": {"binance": "FTMUSDT", "kraken": "FTMUSDT", "coingecko": "fantom"},
    "GRT": {"binance": "GRTUSDT", "kraken": "GRTUSDT", "coingecko": "the-graph"},
    "IMX": {"binance": "IMXUSDT", "kraken": "IMXUSDT", "coingecko": "immutable-x"},
    "SAND": {"binance": "SANDUSDT", "kraken": "SANDUSDT", "coingecko": "the-sandbox"},
    "MANA": {"binance": "MANAUSDT", "kraken": "MANAUSDT", "coingecko": "decentraland"},
    "GALA": {"binance": "GALAUSDT", "kraken": "GALAUSDT", "coingecko": "gala"},
    "RNDR": {"binance": "RNDRUSDT", "kraken": "RNDRUSDT", "coingecko": "render-token"},
    "STX": {"binance": "STXUSDT", "kraken": "STXUSDT", "coingecko": "blockstack"},
    "JUP": {"binance": "JUPUSDT", "kraken": "JUPUSDT", "coingecko": "jupiter-exchange-solana"},
    "PYTH": {"binance": "PYTHUSDT", "kraken": "PYTHUSDT", "coingecko": "pyth-network"},
    "WIF": {"binance": "WIFUSDT", "kraken": "WIFUSDT", "coingecko": "dogwifcoin"},
    "BONK": {"binance": "BONKUSDT", "kraken": "BONKUSDT", "coingecko": "bonk"},
    "PEPE": {"binance": "PEPEUSDT", "kraken": "PEPEUSDT", "coingecko": "pepe"},
    "SHIB": {"binance": "SHIBUSDT", "kraken": "SHIBUSDT", "coingecko": "shiba-inu"},
    "ORDI": {"binance": "ORDIUSDT", "kraken": "ORDIUSDT", "coingecko": "ordinals"},
    "SEI": {"binance": "SEIUSDT", "kraken": "SEIUSDT", "coingecko": "sei-network"},
    "MATIC": {"binance": "MATICUSDT", "kraken": "MATICUSDT", "coingecko": "matic-network"},
    "ETC": {"binance": "ETCUSDT", "kraken": "ETCUSDT", "coingecko": "ethereum-classic"},
    "RUNE": {"binance": "RUNEUSDT", "kraken": "RUNEUSDT", "coingecko": "thorchain"},
    "RPL": {"binance": "RPLUSDT", "kraken": "RPLUSDT", "coingecko": "rocket-pool"},
}

_VENUES: frozenset[str] = frozenset(venue for venue_map in _SYMBOLS.values() for venue in venue_map)


class SymbolMapper:
    """Translate between canonical asset ids and per-venue symbols.

    All methods are static and stateless. Unknown assets or venues raise
    ProviderResponseError so failures surface early rather than producing
    invalid requests downstream.
    """

    @staticmethod
    def to_symbol(asset_id: str, venue: str) -> str:
        """Return the venue-specific symbol for the given canonical asset.

        Raises ProviderResponseError if the asset or venue is unknown.
        """
        if asset_id not in _SYMBOLS:
            raise ProviderResponseError(f"unknown asset: {asset_id}")
        venue_map = _SYMBOLS[asset_id]
        if venue not in venue_map:
            raise ProviderResponseError(f"asset {asset_id} not mapped for venue {venue}")
        return venue_map[venue]

    @staticmethod
    def to_asset_id(symbol: str, venue: str) -> str:
        """Reverse lookup: given a venue-specific symbol, return the
        canonical asset id.

        Raises ProviderResponseError if the symbol is not found for the
        given venue.
        """
        for asset_id, venue_map in _SYMBOLS.items():
            if venue_map.get(venue) == symbol:
                return asset_id
        raise ProviderResponseError(f"symbol {symbol!r} not mapped for venue {venue!r}")

    @staticmethod
    def supported_assets() -> list[str]:
        """Return the canonical ids of all mapped assets."""
        return sorted(_SYMBOLS.keys())

    @staticmethod
    def supported_venues() -> list[str]:
        """Return the venue names that appear in at least one mapping."""
        return sorted(_VENUES)
