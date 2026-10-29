import math

from pyspark.sql import functions as F

from financial_lakehouse.datasets import (
    LABEL_COLUMNS,
    TREND_LABEL,
    build_classification_dataset,
    build_time_series_dataset,
)
from financial_lakehouse.gold import FEATURE_COLUMNS, build_gold
from financial_lakehouse.validation import validate_classification, validate_time_series


def test_time_series_dataset_is_complete_and_contiguous(silver_df):
    ts = build_time_series_dataset(build_gold(silver_df))
    summary = validate_time_series(ts)

    assert summary["tickers"] == 2
    # 600 days minus 60 warm-up rows per ticker; tail rows are kept as forecast origins.
    assert summary["rows"] == 2 * 540
    assert ts.filter("fwd_log_return_20d IS NULL").count() == 2 * 20


def test_classification_labels(silver_df):
    clf = build_classification_dataset(build_gold(silver_df))
    pdf = clf.toPandas()

    assert ((pdf["fwd_log_return_5d"] > 0).astype(int) == pdf["direction_5d"]).all()

    threshold = 0.5 * pdf["volatility_20"] * math.sqrt(5)
    up = pdf["fwd_log_return_5d"] > threshold
    down = pdf["fwd_log_return_5d"] < -threshold
    expected = up * 2 + (~up & ~down) * 1
    assert (expected == pdf[TREND_LABEL]).all()
    assert set(pdf[TREND_LABEL]) == {0, 1, 2}


def test_classification_splits_are_purged(silver_df):
    clf = build_classification_dataset(build_gold(silver_df))
    summary = validate_classification(clf)

    # Synthetic data spans 2022-01-03 .. ~2024-04; no training label may
    # be realised after the training cut-off.
    train_end = summary["split_bounds"]["train"][1]
    assert str(train_end) < "2022-12-31"
    assert clf.filter(F.col("split") == "train").count() > 0

    nulls = clf.select([F.sum(F.col(c).isNull().cast("int")) for c in [*FEATURE_COLUMNS, *LABEL_COLUMNS]]).first()
    assert not any(nulls)
