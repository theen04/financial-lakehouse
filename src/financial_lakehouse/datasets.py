import math
from pathlib import Path

from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F

from .config import (
    GOLD_CLASSIFICATION,
    GOLD_TIME_SERIES,
    TARGET_HORIZONS,
    TRAIN_END_DATE,
    TREND_LABEL_HORIZON,
    TREND_THRESHOLD_SIGMA,
    VALIDATION_END_DATE,
)
from .gold import FEATURE_COLUMNS, TARGET_COLUMNS
from .storage import write_table


ID_COLUMNS = ["date", "ticker", "sector"]

DIRECTION_LABELS = [f"direction_{h}d" for h in TARGET_HORIZONS]
TREND_LABEL = f"trend_{TREND_LABEL_HORIZON}d"

# trend_5d class encoding.
TREND_CLASSES = {0: "down", 1: "flat", 2: "up"}

LABEL_COLUMNS = [*DIRECTION_LABELS, TREND_LABEL]

SPLITS = ("train", "validation", "test")


def split_for(date_column: Column) -> Column:
    """Assign a chronological split to each date."""

    return (
        F.when(date_column.isNull(), None)
        .when(date_column <= F.lit(TRAIN_END_DATE).cast("date"), "train")
        .when(date_column <= F.lit(VALIDATION_END_DATE).cast("date"), "validation")
        .otherwise("test")
    )


def add_split(df: DataFrame) -> DataFrame:
    """Add the chronological split column."""

    return df.withColumn("split", split_for(F.col("date")))


def add_labels(df: DataFrame) -> DataFrame:
    """
    Add classification labels from forward returns.

    direction_{h}d: 1 if the h-day forward return is positive.
    trend_5d: 0/1/2 (down/flat/up) using a dead-band of
        TREND_THRESHOLD_SIGMA * volatility_20 * sqrt(horizon), so moves
        that are small relative to the stock's own volatility are "flat".
    """
    for horizon in TARGET_HORIZONS:
        df = df.withColumn(
            f"direction_{horizon}d",
            (F.col(f"fwd_log_return_{horizon}d") > 0).cast("int"),
        )

    forward = F.col(f"fwd_log_return_{TREND_LABEL_HORIZON}d")
    threshold = (
        F.lit(TREND_THRESHOLD_SIGMA)
        * F.col("volatility_20")
        * F.lit(math.sqrt(TREND_LABEL_HORIZON))
    )

    return df.withColumn(
        TREND_LABEL,
        F.when(forward > threshold, 2)
        .when(forward < -threshold, 0)
        .otherwise(1),
    )


def purge_split_boundaries(df: DataFrame, horizon: int = max(TARGET_HORIZONS)) -> DataFrame:
    """
    Drop rows whose label window extends into a later split.

    A training row dated just before the cut-off has a forward return that
    is realised during validation; keeping it leaks validation prices into
    training labels. Rows at the end of a series have no label yet and are
    also dropped.
    """
    w = Window.partitionBy("ticker").orderBy("date")
    label_end_split = split_for(F.lead("date", horizon).over(w))

    return (
        df
        .withColumn("_label_end_split", label_end_split)
        .filter(F.col("split") == F.col("_label_end_split"))
        .drop("_label_end_split")
    )


def build_time_series_dataset(gold_df: DataFrame) -> DataFrame:
    """
    Build the time-series forecasting panel.

    One contiguous row per ticker and trading day once all features are
    available. The adjusted close and volume are kept for classical
    univariate models; forward returns are the forecasting targets. The
    final rows of each series have null targets and act as forecast origins.
    """
    return (
        add_split(gold_df.filter("feature_complete"))
        .select(
            *ID_COLUMNS,
            "adj_close",
            "volume",
            *FEATURE_COLUMNS,
            *TARGET_COLUMNS,
            "split",
        )
    )


def build_classification_dataset(gold_df: DataFrame) -> DataFrame:
    """
    Build the classification dataset.

    Contains scale-free features, direction and trend labels, and a
    chronological split with boundary purging. Rows without a complete
    feature vector or label are excluded.
    """
    df = add_labels(gold_df)
    df = add_split(df)
    df = purge_split_boundaries(df)

    return (
        df
        .filter("feature_complete")
        .dropna(subset=TARGET_COLUMNS)
        .select(
            *ID_COLUMNS,
            *FEATURE_COLUMNS,
            *TARGET_COLUMNS,
            *LABEL_COLUMNS,
            "split",
        )
    )


def write_time_series_dataset(
    df: DataFrame,
    output_dir: Path = GOLD_TIME_SERIES,
) -> None:
    """Write the time-series dataset to Delta."""

    write_table(df, output_dir, partition_by=["ticker"])


def write_classification_dataset(
    df: DataFrame,
    output_dir: Path = GOLD_CLASSIFICATION,
) -> None:
    """Write the classification dataset to Delta."""

    write_table(df, output_dir, partition_by=["split"])
