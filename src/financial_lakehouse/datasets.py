from pathlib import Path

from pyspark.sql import DataFrame

from .config import CLASSIFICATION_DIR, TIME_SERIES_DIR


TIME_SERIES_COLUMNS = [
    "Date",
    "ticker",
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
    "daily_return",
    "log_return",
    "price_change",
    "ma_5",
    "ma_20",
    "ma_50",
    "price_vs_ma5",
    "price_vs_ma20",
    "price_vs_ma50",
    "volatility_5",
    "volatility_20",
    "volatility_50",
    "avg_volume_20",
    "momentum_20",
]


CLASSIFICATION_COLUMNS = [
    "Date",
    "ticker",
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
    "daily_return",
    "log_return",
    "price_change",
    "ma_5",
    "ma_20",
    "ma_50",
    "price_vs_ma5",
    "price_vs_ma20",
    "price_vs_ma50",
    "volatility_5",
    "volatility_20",
    "volatility_50",
    "avg_volume_20",
    "momentum_20",
    "next_day_up",
]


def build_time_series_dataset(df: DataFrame) -> DataFrame:
    """Build a time-series dataset from the Gold feature layer."""

    return (
        df
        .dropna(
            subset=[
                "daily_return",
                "log_return",
                "price_change",
                "ma_5",
                "ma_20",
                "ma_50",
                "price_vs_ma5",
                "price_vs_ma20",
                "price_vs_ma50",
                "volatility_5",
                "volatility_20",
                "volatility_50",
                "avg_volume_20",
                "momentum_20",
            ]
        )
        .select(TIME_SERIES_COLUMNS)
        .orderBy("ticker", "Date")
    )


def build_classification_dataset(df: DataFrame) -> DataFrame:
    """Build a classification dataset from the Gold feature layer."""

    return (
        df
        .dropna(
            subset=[
                "daily_return",
                "log_return",
                "price_change",
                "ma_5",
                "ma_20",
                "ma_50",
                "price_vs_ma5",
                "price_vs_ma20",
                "price_vs_ma50",
                "volatility_5",
                "volatility_20",
                "volatility_50",
                "avg_volume_20",
                "momentum_20",
                "next_close",
            ]
        )
        .select(CLASSIFICATION_COLUMNS)
        .orderBy("ticker", "Date")
    )


def write_time_series_dataset(
    df: DataFrame,
    output_dir: Path = TIME_SERIES_DIR,
) -> None:
    """Write the time-series dataset to Parquet."""

    (
        df.write
        .mode("overwrite")
        .partitionBy("ticker")
        .parquet(str(output_dir))
    )


def write_classification_dataset(
    df: DataFrame,
    output_dir: Path = CLASSIFICATION_DIR,
) -> None:
    """Write the classification dataset to Parquet."""

    (
        df.write
        .mode("overwrite")
        .partitionBy("ticker")
        .parquet(str(output_dir))
    )