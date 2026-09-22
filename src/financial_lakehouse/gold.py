from pyspark.sql import DataFrame
from pyspark.sql import Window
from pyspark.sql import functions as F

from pathlib import Path

from pyspark.sql import DataFrame

from financial_lakehouse.config import GOLD_DIR


def build_gold(df: DataFrame) -> DataFrame:
    """Build the Gold analytical feature layer from Silver data."""

    # Base window ordered by date for each ticker.
    time_window = (
        Window
        .partitionBy("ticker")
        .orderBy("Date")
    )

    # Rolling windows for technical metrics.
    rolling_window_5 = (
        Window
        .partitionBy("ticker")
        .orderBy("Date")
        .rowsBetween(-4, 0)
    )

    rolling_window_20 = (
        Window
        .partitionBy("ticker")
        .orderBy("Date")
        .rowsBetween(-19, 0)
    )

    rolling_window_50 = (
        Window
        .partitionBy("ticker")
        .orderBy("Date")
        .rowsBetween(-49, 0)
    )

    # Price features.
    df = (
        df
        .withColumn(
            "previous_close",
            F.lag("Close").over(time_window),
        )
        .withColumn(
            "daily_return",
            (F.col("Close") - F.col("previous_close"))
            / F.col("previous_close"),
        )
        .withColumn(
            "log_return",
            F.log(F.col("Close") / F.col("previous_close")),
        )
        .withColumn(
            "price_change",
            F.col("Close") - F.col("previous_close"),
        )
    )

    # Trend features.
    df = (
        df
        .withColumn(
            "ma_5",
            F.avg("Close").over(rolling_window_5),
        )
        .withColumn(
            "ma_20",
            F.avg("Close").over(rolling_window_20),
        )
        .withColumn(
            "ma_50",
            F.avg("Close").over(rolling_window_50),
        )
        .withColumn(
            "price_vs_ma5",
            F.col("Close") / F.col("ma_5"),
        )
        .withColumn(
            "price_vs_ma20",
            F.col("Close") / F.col("ma_20"),
        )
        .withColumn(
            "price_vs_ma50",
            F.col("Close") / F.col("ma_50"),
        )
    )

    # Risk features.
    df = (
        df
        .withColumn(
            "volatility_5",
            F.stddev("daily_return").over(rolling_window_5),
        )
        .withColumn(
            "volatility_20",
            F.stddev("daily_return").over(rolling_window_20),
        )
        .withColumn(
            "volatility_50",
            F.stddev("daily_return").over(rolling_window_50),
        )
    )

    # Volume trend.
    df = df.withColumn(
        "avg_volume_20",
        F.avg("Volume").over(rolling_window_20),
    )

    # Momentum.
    df = df.withColumn(
        "momentum_20",
        (F.col("Close") / F.lag("Close", 20).over(time_window)) - 1,
    )

    # Prediction target.
    df = (
        df
        .withColumn(
            "next_close",
            F.lead("Close").over(time_window),
        )
        .withColumn(
            "next_day_up",
            F.when(F.col("next_close") > F.col("Close"), 1)
            .otherwise(0),
        )
    )

    # previous_close is an intermediate calculation and is not
    # part of the reusable Gold feature layer.
    df = df.drop("previous_close")

    return df


def write_gold(
    df: DataFrame,
    output_dir: Path = GOLD_DIR,
) -> None:
    """Write the Gold feature layer to partitioned Parquet."""

    (
        df.write
        .mode("overwrite")
        .partitionBy("ticker", "year", "month")
        .parquet(str(output_dir))
    )