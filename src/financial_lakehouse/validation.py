from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from .datasets import LABEL_COLUMNS, SPLITS
from .gold import FEATURE_COLUMNS


class DatasetValidationError(Exception):
    """Raised when a Gold dataset fails a validation check."""


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise DatasetValidationError(message)


def _duplicate_keys(df: DataFrame) -> int:
    return df.groupBy("ticker", "date").count().filter("count > 1").count()


def _null_counts(df: DataFrame, columns: list[str]) -> dict[str, int]:
    row = df.select([
        F.sum(F.col(c).isNull().cast("int")).alias(c) for c in columns
    ]).first()

    return {c: row[c] for c in columns if row[c]}


def validate_time_series(df: DataFrame) -> dict:
    """Validate the time-series dataset and return a summary."""

    _check(_duplicate_keys(df) == 0, "time_series: duplicate (ticker, date) rows")

    nulls = _null_counts(df, FEATURE_COLUMNS)
    _check(not nulls, f"time_series: null features {nulls}")

    per_ticker = (
        df.groupBy("ticker")
        .agg(
            F.count("*").alias("rows"),
            F.min("date").alias("start"),
            F.max("date").alias("end"),
        )
        .orderBy("ticker")
        .collect()
    )

    return {
        "rows": df.count(),
        "tickers": len(per_ticker),
        "start": min(r["start"] for r in per_ticker),
        "end": max(r["end"] for r in per_ticker),
    }


def validate_classification(df: DataFrame) -> dict:
    """Validate the classification dataset and return a summary."""

    _check(_duplicate_keys(df) == 0, "classification: duplicate (ticker, date) rows")

    nulls = _null_counts(df, [*FEATURE_COLUMNS, *LABEL_COLUMNS])
    _check(not nulls, f"classification: null features or labels {nulls}")

    split_rows = {
        r["split"]: r["count"] for r in df.groupBy("split").count().collect()
    }
    _check(
        all(split_rows.get(s, 0) > 0 for s in SPLITS),
        f"classification: empty split {split_rows}",
    )

    # Splits must not overlap in time.
    bounds = {
        r["split"]: (r["start"], r["end"])
        for r in df.groupBy("split")
        .agg(F.min("date").alias("start"), F.max("date").alias("end"))
        .collect()
    }
    for earlier, later in zip(SPLITS, SPLITS[1:]):
        _check(
            bounds[earlier][1] < bounds[later][0],
            f"classification: {earlier} overlaps {later}",
        )

    label_balance = {
        label: {
            r[label]: r["count"]
            for r in df.groupBy(label).count().orderBy(label).collect()
        }
        for label in LABEL_COLUMNS
    }

    return {
        "rows": sum(split_rows.values()),
        "split_rows": {s: split_rows[s] for s in SPLITS},
        "split_bounds": {s: bounds[s] for s in SPLITS},
        "label_balance": label_balance,
    }
