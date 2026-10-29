from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from financial_lakehouse.bronze import to_bronze_dataframe
from financial_lakehouse.silver import build_silver, build_ticker_reference
from financial_lakehouse.spark import get_spark


@pytest.fixture(scope="session")
def spark():
    session = get_spark(app_name="financial-lakehouse-tests", master="local[2]", shuffle_partitions=2)
    yield session
    session.stop()


def make_market_data(
    tickers=("AAPL", "XOM"),
    start="2022-01-03",
    periods=600,
    seed=0,
    ingested_at=datetime(2026, 1, 1, tzinfo=UTC),
) -> pd.DataFrame:
    """Synthetic daily OHLCV data shaped like the ingestion output."""

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=periods).date
    frames = []

    for ticker in tickers:
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, periods)))
        open_ = close * np.exp(rng.normal(0, 0.005, periods))
        high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, periods))
        low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, periods))

        frames.append(pd.DataFrame({
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            # Constant dividend adjustment factor.
            "adj_close": close * 0.98,
            "volume": rng.integers(1_000_000, 5_000_000, periods),
            "ticker": ticker,
        }))

    df = pd.concat(frames, ignore_index=True)
    df["source"] = "synthetic"
    df["ingestion_id"] = ingested_at.strftime("%Y%m%dT%H%M%SZ")
    df["ingested_at"] = ingested_at

    return df


@pytest.fixture(scope="session")
def market_data() -> pd.DataFrame:
    return make_market_data()


@pytest.fixture(scope="session")
def bronze_df(spark, market_data):
    return to_bronze_dataframe(spark, market_data)


@pytest.fixture(scope="session")
def silver_df(spark, bronze_df):
    silver, _ = build_silver(bronze_df, build_ticker_reference(spark))
    return silver.cache()
