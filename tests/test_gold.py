import numpy as np
from pyspark.sql import functions as F

from financial_lakehouse.gold import FEATURE_COLUMNS, build_gold


def _ticker_pdf(gold_df, ticker="AAPL"):
    return gold_df.filter(F.col("ticker") == ticker).orderBy("date").toPandas()


def test_rolling_features_require_full_window(silver_df):
    pdf = _ticker_pdf(build_gold(silver_df))

    # volatility_50 needs 50 returns, i.e. 51 prices.
    assert pdf["volatility_50"].iloc[:50].isna().all()
    assert pdf["volatility_50"].iloc[50:].notna().all()
    # momentum_60 is the longest lookback.
    assert pdf["feature_complete"].iloc[:60].eq(False).all()
    assert pdf["feature_complete"].iloc[60:].all()


def test_features_match_pandas_reference(silver_df):
    pdf = _ticker_pdf(build_gold(silver_df))
    log_return = np.log(pdf["adj_close"] / pdf["adj_close"].shift())

    np.testing.assert_allclose(pdf["log_return_1d"].iloc[1:], log_return.iloc[1:])
    np.testing.assert_allclose(
        pdf["volatility_20"].iloc[21:],
        log_return.rolling(20).std().iloc[21:],
    )
    np.testing.assert_allclose(
        pdf["price_vs_ma_50"].iloc[49:],
        (pdf["adj_close"] / pdf["adj_close"].rolling(50).mean() - 1).iloc[49:],
    )


def test_targets_look_forward_and_features_do_not(silver_df):
    gold = build_gold(silver_df)
    base = _ticker_pdf(gold)

    # Perturbing the final price must change earlier targets but no features.
    last_date = base["date"].iloc[-1]
    shocked = silver_df.withColumn(
        "adj_close",
        F.when((F.col("ticker") == "AAPL") & (F.col("date") == last_date), F.col("adj_close") * 2)
        .otherwise(F.col("adj_close")),
    )
    after = _ticker_pdf(build_gold(shocked))

    earlier = slice(0, len(base) - 1)
    for column in FEATURE_COLUMNS:
        if column in ("market_log_return_1d", "excess_log_return_1d", "sector_log_return_1d"):
            continue
        np.testing.assert_array_equal(base[column].iloc[earlier], after[column].iloc[earlier])

    assert base["fwd_log_return_1d"].iloc[-2] != after["fwd_log_return_1d"].iloc[-2]
    assert base["fwd_log_return_20d"].iloc[-20:].isna().all()


def test_excess_returns_sum_to_zero_across_market(silver_df):
    totals = (
        build_gold(silver_df)
        .groupBy("date")
        .agg(F.sum("excess_log_return_1d").alias("total"))
        .filter(F.col("total").isNotNull())
    )

    assert totals.filter(F.abs("total") > 1e-12).count() == 0
