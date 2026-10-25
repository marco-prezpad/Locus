"""
config.py

Centralised configuration loading: YAML files merged with environment
variables and validated through Pydantic Settings. Environment variables
override YAML values, which override field defaults. Provides typed access
to application, data, model, portfolio, risk, backtest, and
signal-execution configuration, plus a helper to load cost scenarios from
disk.

Author: Marco Pérez Padilla
Date:   20-10-2026
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Literal

import yaml
from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


class AppConfig(BaseSettings):
    """Application-level settings: name, environment, and log verbosity."""

    name: str = "locus"
    environment: Literal["local", "dev", "staging", "prod"] = "local"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"


class KrakenProviderConfig(BaseSettings):
    """Kraken public REST API settings."""

    base_url: str = "https://api.kraken.com"
    max_bars_per_request: int = 720
    timeout_seconds: int = 30
    max_retries: int = 5


class CoinGeckoProviderConfig(BaseSettings):
    """CoinGecko public API settings (last-resort fallback)."""

    base_url: str = "https://api.coingecko.com/api/v3"
    timeout_seconds: int = 30
    max_retries: int = 3
    default_vs_currency: str = "usd"


class BinanceProviderConfig(BaseSettings):
    """Binance public REST API settings."""

    base_url: str = "https://api.binance.com"
    max_bars_per_request: int = 1000
    requests_per_minute: int = 1000
    timeout_seconds: int = 30
    max_retries: int = 5


class ProvidersConfig(BaseSettings):
    """Per-venue provider settings."""

    binance: BinanceProviderConfig = Field(default_factory=BinanceProviderConfig)
    kraken: KrakenProviderConfig = Field(default_factory=KrakenProviderConfig)
    coingecko: CoinGeckoProviderConfig = Field(default_factory=CoinGeckoProviderConfig)


class UniverseConfig(BaseSettings):
    """Universe reconstruction parameters."""

    discovery_source: str = "binance"
    min_history_days: int = 30
    default_venues: list[str] = Field(default_factory=lambda: ["binance"])


class FxConfig(BaseSettings):
    """FX conversion configuration: pairs, numeraire, staleness, CoinGecko quota."""

    pairs: list[str] = Field(default_factory=lambda: ["EURUSD", "USDTUSD"])
    numeraire: str = "USDT"
    reporting_currency: str = "EUR"
    stale_days_threshold: int = 7
    coingecko_api_key: str = ""
    coingecko_max_per_minute: int = 100
    coingecko_max_per_month: int = 10_000
    coingecko_warn_threshold: int = 9_500
    source_preferences: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "EURUSD": ["frankfurter"],
            "USDTUSD": ["kraken", "coingecko"],
        }
    )


class FeatureSpecConfig(BaseSettings):
    """One feature specification from config."""

    name: str
    family: str
    params: dict[str, Any] = Field(default_factory=dict)


class FeaturesConfig(BaseSettings):
    """Active feature set. Order matters only for logging; the pipeline
    enforces name uniqueness."""

    active: list[FeatureSpecConfig] = Field(default_factory=list)


class LabelSpecConfig(BaseSettings):
    """One label specification from config."""

    name: str
    bars: int


class LabelsConfig(BaseSettings):
    """Active label set and the primary label used by models/backtester."""

    horizons_bars: list[LabelSpecConfig] = Field(default_factory=list)
    probability_positive_label: str = "future_return_24h"
    primary_label: str = "future_return_24h"


class TemporalSplitConfig(BaseSettings):
    """Single temporal split ratios (train/val/test)."""

    train_ratio: float = 0.70
    val_ratio: float = 0.15


class WalkForwardSplitConfig(BaseSettings):
    """Walk-forward split parameters."""

    n_splits: int = 5
    horizon_bars: int = 24
    lookback_bars: int = 50
    rebalance_freq_bars: int = 24


class SplitsConfig(BaseSettings):
    """Split strategy: 'temporal' (single split) or 'walk_forward' (K folds)."""

    mode: Literal["temporal", "walk_forward"] = "temporal"
    temporal: TemporalSplitConfig = Field(default_factory=TemporalSplitConfig)
    walk_forward: WalkForwardSplitConfig = Field(default_factory=WalkForwardSplitConfig)


class FundingConfig(BaseSettings):
    """Funding rate data configuration."""

    venue: str = "binance_futures"
    base_url: str = "https://fapi.binance.com"
    max_events_per_request: int = 1000
    timeout_seconds: int = 30
    max_retries: int = 5
    expected_cadence_hours: int = 8
    default_end_date: str = "2026-10-01"
    universe_assets: list[str] = Field(default_factory=lambda: ["BTC", "ETH", "SOL", "BNB", "XRP"])


class DataConfig(BaseSettings):
    """Filesystem locations and storage options for market data."""

    raw_dir: Path = Path("data/raw")
    interim_dir: Path = Path("data/interim")
    processed_dir: Path = Path("data/processed")
    parquet_compression: str = "snappy"
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    fx: FxConfig = Field(default_factory=FxConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    labels: LabelsConfig = Field(default_factory=LabelsConfig)
    splits: SplitsConfig = Field(default_factory=SplitsConfig)
    funding: FundingConfig = Field(default_factory=FundingConfig)


class ModelConfig(BaseSettings):
    """Global model defaults: forecast horizon and random seed."""

    default_horizon_hours: int = 24
    random_seed: int = 42


class PortfolioConfig(BaseSettings):
    """Portfolio construction constraints enforced at allocation time."""

    max_weight_per_asset: float = 0.10
    max_total_exposure: float = 1.0
    min_cash_weight: float = 0.05
    max_turnover: float = 0.30


class RiskConfig(BaseSettings):
    """Risk-management limits and kill-switch configuration."""

    max_daily_loss_pct: float = 0.02
    max_drawdown_pct: float = 0.15
    kill_switch_enabled: bool = True


class BacktestConfig(BaseSettings):
    """Execution-cost assumptions used by the event-driven backtester."""

    fee_bps: float = 10.0
    spread_bps: float = 5.0
    slippage_bps: float = 5.0
    impact_model: str = "sqrt_volume"
    impact_k: float = 1.0
    initial_capital: float = 10_000.0
    fee_schedule_version: str = "binance-spot-2026-01"


class SignalExecutionConfig(BaseSettings):
    """Signal-to-execution contract enforcing the one-bar-delay invariant.

    Features are computed on the close of bar T, the signal becomes available
    at the close of bar T, and the earliest possible execution is the open of
    bar T+1. Intermediate signals (intra-rebalance) are recorded for research
    but never used for execution in the MVP.
    """

    feature_compute_time: Literal["bar_close_T"] = "bar_close_T"
    signal_availability_time: Literal["bar_close_T"] = "bar_close_T"
    earliest_execution_time: Literal["bar_open_T+1"] = "bar_open_T+1"
    rebalance_cadence_hours: int = 24
    rebalance_anchor: str = "00:00 UTC"
    holding_period_hours: int = 24
    signal_used_at_rebalance: Literal["close_T-1"] = "close_T-1"
    intermediate_signals_used_for_execution: bool = False
    hysteresis_bps: float = 50.0
    execution_price: Literal["bar_open_T+1"] = "bar_open_T+1"


class _YamlSettingsSource(PydanticBaseSettingsSource):
    """Read settings from a YAML file at runtime.

    Registered after `env_settings` and `dotenv_settings` in the priority
    chain, so environment variables still override YAML values, while YAML
    overrides field defaults.
    """

    def __init__(self, settings_cls: type[BaseSettings], yaml_path: Path) -> None:
        super().__init__(settings_cls)
        self._yaml_path = yaml_path

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        """Unused: this source returns the whole mapping in `__call__`."""
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        """Return the parsed YAML content, or an empty mapping if the file
        does not exist (so defaults apply cleanly)."""
        if not self._yaml_path.exists():
            return {}
        with open(self._yaml_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}


class Config(BaseSettings):
    """Root configuration aggregating all sub-sections.

    Resolution order (highest to lowest priority): explicit constructor
    arguments, environment variables (nested with double underscore, e.g.
    APP__LOG_LEVEL), variables from `.env`, YAML loaded by `load_config`,
    and finally the field defaults.
    """

    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    _yaml_path: ClassVar[Path] = Path("configs/base.yaml")

    app: AppConfig = Field(default_factory=AppConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    signal_execution: SignalExecutionConfig = Field(default_factory=SignalExecutionConfig)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Insert a YAML source between `.env` and secrets, so env vars
        override YAML but YAML still overrides field defaults."""
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            _YamlSettingsSource(settings_cls, cls._yaml_path),
            file_secret_settings,
        )


def load_config(yaml_path: str | Path = "configs/base.yaml") -> Config:
    """Load configuration by pointing the YAML source at `yaml_path` and
    building a validated `Config` instance.

    Environment variables (nested with `__`) override YAML values; YAML
    overrides field defaults; missing YAML files are tolerated.
    """
    Config._yaml_path = Path(yaml_path)
    return Config()


def load_cost_scenario(
    name: Literal["optimistic", "central", "pessimistic"],
) -> dict:
    """Load one of the three pre-registered cost scenarios (optimistic,
    central, pessimistic) from `configs/costs/<name>.yaml`.

    Raises `FileNotFoundError` if the scenario file does not exist, so that
    a missing scenario is caught early rather than silently defaulting.
    """
    path = Path(f"configs/costs/{name}.yaml")
    if not path.exists():
        raise FileNotFoundError(f"Cost scenario not found: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
