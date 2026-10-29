from datetime import UTC, datetime

from pyspark.sql import functions as F

from financial_lakehouse.bronze import to_bronze_dataframe
from financial_lakehouse.silver import (
    build_silver,
    build_ticker_reference,
    deduplicate_latest,
    quality_report,
)
from financial_lakehouse.storage import merge_table, read_table

from conftest import make_market_data


def test_deduplicate_keeps_latest_ingestion(spark):
    old = make_market_data(tickers=("AAPL",), periods=10, ingested_at=datetime(2026, 1, 1, tzinfo=UTC))
    new = make_market_data(tickers=("AAPL",), periods=10, ingested_at=datetime(2026, 2, 1, tzinfo=UTC))
    new["close"] = new["close"] + 1

    df = to_bronze_dataframe(spark, old).unionByName(to_bronze_dataframe(spark, new))
    deduped = deduplicate_latest(df)

    assert deduped.count() == 10
    assert deduped.select("ingestion_id").distinct().first()[0] == "20260201T000000Z"


def test_invalid_records_are_quarantined(spark):
    pdf = make_market_data(tickers=("AAPL",), periods=10)
    pdf.loc[0, "close"] = -1.0
    pdf.loc[1, "low"] = pdf.loc[1, "high"] + 1
    pdf.loc[2, "adj_close"] = None

    valid, quarantined = build_silver(to_bronze_dataframe(spark, pdf), build_ticker_reference(spark))

    assert valid.count() == 7
    failed = {r["date"]: set(r["failed_rules"]) for r in quarantined.collect()}
    assert failed[pdf.loc[0, "date"]] >= {"prices_positive"}
    assert failed[pdf.loc[1, "date"]] >= {"high_gte_low"}
    assert failed[pdf.loc[2, "date"]] == {"required_fields_present"}


def test_quality_report_counts_violations(spark):
    pdf = make_market_data(tickers=("AAPL",), periods=10)
    pdf.loc[0, "volume"] = -5

    report = {r["rule"]: r["violations"] for r in quality_report(to_bronze_dataframe(spark, pdf))}

    assert report["volume_non_negative"] == 1
    assert report["prices_positive"] == 0


def test_silver_joins_sector(silver_df):
    sectors = {r["ticker"]: r["sector"] for r in silver_df.select("ticker", "sector").distinct().collect()}

    assert sectors == {"AAPL": "Technology", "XOM": "Energy"}


def test_merge_only_replaces_with_newer_records(spark, tmp_path):
    path = tmp_path / "silver"
    reference = build_ticker_reference(spark)

    def silver_batch(ingested_at, bump):
        pdf = make_market_data(tickers=("AAPL",), periods=5, ingested_at=ingested_at)
        pdf["adj_close"] = pdf["adj_close"] + bump
        return build_silver(to_bronze_dataframe(spark, pdf), reference)[0]

    merge_table(spark, silver_batch(datetime(2026, 2, 1, tzinfo=UTC), 1), path, keys=["ticker", "date"],
                update_condition="s.ingested_at > t.ingested_at")
    # Replaying an older batch must not overwrite newer data.
    merge_table(spark, silver_batch(datetime(2026, 1, 1, tzinfo=UTC), 0), path, keys=["ticker", "date"],
                update_condition="s.ingested_at > t.ingested_at")

    df = read_table(spark, path)
    assert df.count() == 5
    assert df.select(F.max("ingested_at")).first()[0] == df.select(F.min("ingested_at")).first()[0]
