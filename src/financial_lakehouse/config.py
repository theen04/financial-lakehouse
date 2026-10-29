from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
LAKEHOUSE_DIR = DATA_DIR / "lakehouse"

BRONZE_DIR = LAKEHOUSE_DIR / "bronze"
SILVER_DIR = LAKEHOUSE_DIR / "silver"
GOLD_DIR = LAKEHOUSE_DIR / "gold"

# Delta tables, grouped by medallion layer.
BRONZE_PRICES = BRONZE_DIR / "market_prices"

SILVER_PRICES = SILVER_DIR / "market_prices"
SILVER_QUARANTINE = SILVER_DIR / "quarantine"
SILVER_TICKER_REFERENCE = SILVER_DIR / "ticker_reference"

GOLD_FEATURES = GOLD_DIR / "daily_features"
GOLD_TIME_SERIES = GOLD_DIR / "time_series"
GOLD_CLASSIFICATION = GOLD_DIR / "classification"
GOLD_BASELINE_METRICS = GOLD_DIR / "baseline_metrics"

# Fixed history window so runs are reproducible.
DEFAULT_START_DATE = "2016-01-01"
DEFAULT_END_DATE = "2026-01-01"

# Sector-diverse ticker universe (ticker -> GICS sector).
TICKER_SECTORS = {
    "NVDA": "Technology",
    "AAPL": "Technology",
    "MSFT": "Technology",
    "AVGO": "Technology",
    "ORCL": "Technology",
    "JPM": "Financials",
    "BRK-B": "Financials",
    "BAC": "Financials",
    "GOOGL": "Communication Services",
    "META": "Communication Services",
    "AMZN": "Consumer Discretionary",
    "TSLA": "Consumer Discretionary",
    "LLY": "Health Care",
    "JNJ": "Health Care",
    "CAT": "Industrials",
    "GE": "Industrials",
    "WMT": "Consumer Staples",
    "PG": "Consumer Staples",
    "XOM": "Energy",
    "NEE": "Utilities",
}

DEFAULT_TICKERS = list(TICKER_SECTORS)

# Forward horizons (trading days) used for forecasting targets and labels.
TARGET_HORIZONS = (1, 5, 20)

# A 5-day move counts as "up"/"down" only when it exceeds this many
# horizon-scaled standard deviations of recent daily volatility.
TREND_LABEL_HORIZON = 5
TREND_THRESHOLD_SIGMA = 0.5

# Chronological splits. Rows whose label window crosses a split boundary
# are purged from the classification dataset to prevent leakage.
TRAIN_END_DATE = "2022-12-31"
VALIDATION_END_DATE = "2023-12-31"
