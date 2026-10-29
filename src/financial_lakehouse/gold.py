from pathlib import Path

from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F

from .config import GOLD_FEATURES, TARGET_HORIZONS
from .storage import write_table


# Feature columns produced by build_gold. Every feature uses only data
# available at the close of `date`; price levels are excluded so the
# features are comparable across tickers and over time.
FEATURE_COLUMNS = [
    # Returns
    "log_return_1d",
    "return_lag_1",
    "return_lag_2",
    "return_lag_3",
    "return_lag_4",
    "return_lag_5",
    # Momentum
    "momentum_5",
    "momentum_20",
    "momentum_60",
    # Trend
    "price_vs_ma_5",
    "price_vs_ma_20",
    "price_vs_ma_50",
    "ma_5_vs_ma_20",
    "ma_20_vs_ma_50",
    # Risk
    "volatility_5",
    "volatility_20",
    "volatility_50",
    "volatility_ratio_5_20",
    "intraday_range",
    "overnight_gap",
    "rsi_14",
    # Volume
    "relative_volume_20",
    # Market context
    "market_log_return_1d",
    "excess_log_return_1d",
    "sector_log_return_1d",
    # Calendar
    "day_of_week",
    "month_of_year",
]

TARGET_COLUMNS = [f"fwd_log_return_{h}d" for h in TARGET_HORIZONS]


def _ticker_window() -> Window:
    return Window.partitionBy("ticker").orderBy("date")


def rolling(column: str, periods: int, agg) -> Column:
    """
    Aggregate a trailing window, returning null until the window is full.

    Spark computes partial windows at the start of each series; requiring
    a full window avoids silently mixing 3-day and 50-day statistics.
    """
    window = _ticker_window().rowsBetween(-(periods - 1), 0)

    return F.when(
        F.count(column).over(window) == periods,
        agg(column).over(window),
    )


def add_return_features(df: DataFrame) -> DataFrame:
    """Add return, lagged return, and momentum features from adjusted prices."""

    w = _ticker_window()

    df = df.withColumn(
        "log_return_1d",
        F.log(F.col("adj_close") / F.lag("adj_close").over(w)),
    )

    for lag in range(1, 6):
        df = df.withColumn(
            f"return_lag_{lag}",
            F.lag("log_return_1d", lag).over(w),
        )

    for periods in (5, 20, 60):
        df = df.withColumn(
            f"momentum_{periods}",
            F.log(F.col("adj_close") / F.lag("adj_close", periods).over(w)),
        )

    return df


def add_trend_features(df: DataFrame) -> DataFrame:
    """Add moving-average trend features, expressed relative to price."""

    for periods in (5, 20, 50):
        df = df.withColumn(f"_ma_{periods}", rolling("adj_close", periods, F.avg))
        df = df.withColumn(
            f"price_vs_ma_{periods}",
            F.col("adj_close") / F.col(f"_ma_{periods}") - 1,
        )

    return (
        df
        .withColumn("ma_5_vs_ma_20", F.col("_ma_5") / F.col("_ma_20") - 1)
        .withColumn("ma_20_vs_ma_50", F.col("_ma_20") / F.col("_ma_50") - 1)
        .drop("_ma_5", "_ma_20", "_ma_50")
    )


def add_risk_features(df: DataFrame) -> DataFrame:
    """Add volatility, range, gap, and RSI features."""

    w = _ticker_window()

    for periods in (5, 20, 50):
        df = df.withColumn(
            f"volatility_{periods}",
            rolling("log_return_1d", periods, F.stddev),
        )

    # Raw OHLC is not dividend adjusted; scale it onto the adjusted series
    # so gaps are not distorted by ex-dividend dates.
    adjustment = F.col("adj_close") / F.col("close")

    df = (
        df
        .withColumn(
            "volatility_ratio_5_20",
            F.col("volatility_5") / F.col("volatility_20"),
        )
        .withColumn(
            "intraday_range",
            (F.col("high") - F.col("low")) / F.col("close"),
        )
        .withColumn(
            "overnight_gap",
            F.log(F.col("open") * adjustment / F.lag("adj_close").over(w)),
        )
        .withColumn("_gain", F.greatest(F.col("log_return_1d"), F.lit(0.0)))
        .withColumn("_loss", F.greatest(-F.col("log_return_1d"), F.lit(0.0)))
        .withColumn("_avg_gain", rolling("_gain", 14, F.avg))
        .withColumn("_avg_loss", rolling("_loss", 14, F.avg))
        .withColumn(
            "rsi_14",
            F.when(F.col("_avg_loss") == 0, F.lit(100.0)).otherwise(
                100 - 100 / (1 + F.col("_avg_gain") / F.col("_avg_loss"))
            ),
        )
        .drop("_gain", "_loss", "_avg_gain", "_avg_loss")
    )

    return df


def add_volume_features(df: DataFrame) -> DataFrame:
    """Add volume relative to its trailing 20-day average."""

    return df.withColumn(
        "relative_volume_20",
        F.col("volume") / rolling("volume", 20, F.avg),
    )


def add_market_features(df: DataFrame) -> DataFrame:
    """Add equal-weighted market and sector returns for each date."""

    market = Window.partitionBy("date")
    sector = Window.partitionBy("date", "sector")

    return (
        df
        .withColumn("market_log_return_1d", F.avg("log_return_1d").over(market))
        .withColumn(
            "excess_log_return_1d",
            F.col("log_return_1d") - F.col("market_log_return_1d"),
        )
        .withColumn("sector_log_return_1d", F.avg("log_return_1d").over(sector))
    )


def add_calendar_features(df: DataFrame) -> DataFrame:
    """Add day-of-week (1 = Monday) and month features."""

    return (
        df
        .withColumn("day_of_week", F.weekday("date") + 1)
        .withColumn("month_of_year", F.month("date"))
    )


def add_targets(df: DataFrame) -> DataFrame:
    """Add forward log returns for each target horizon."""

    w = _ticker_window()

    for horizon in TARGET_HORIZONS:
        df = df.withColumn(
            f"fwd_log_return_{horizon}d",
            F.log(F.lead("adj_close", horizon).over(w) / F.col("adj_close")),
        )

    return df


def build_gold(silver_df: DataFrame) -> DataFrame:
    """Build the Gold daily feature table from Silver prices."""

    df = silver_df.select(
        "ticker", "date", "sector",
        "open", "high", "low", "close", "adj_close", "volume",
    )

    df = add_return_features(df)
    df = add_trend_features(df)
    df = add_risk_features(df)
    df = add_volume_features(df)
    df = add_market_features(df)
    df = add_calendar_features(df)
    df = add_targets(df)

    feature_complete = F.lit(True)
    for column in FEATURE_COLUMNS:
        feature_complete = feature_complete & F.col(column).isNotNull()

    return df.withColumn("feature_complete", feature_complete)


def write_gold(
    df: DataFrame,
    output_dir: Path = GOLD_FEATURES,
) -> None:
    """
    Write the Gold feature table.

    Rolling features depend on full history, so Gold is recomputed and
    overwritten from Silver on every run; Delta keeps prior versions.
    """
    write_table(df, output_dir, partition_by=["ticker"])
