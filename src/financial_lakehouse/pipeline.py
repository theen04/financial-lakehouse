import logging

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from .bronze import to_bronze_dataframe, write_bronze
from .config import (
    BRONZE_PRICES,
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    DEFAULT_TICKERS,
    GOLD_CLASSIFICATION,
    GOLD_FEATURES,
    GOLD_TIME_SERIES,
    SILVER_PRICES,
    TICKER_SECTORS,
)
from .datasets import (
    build_classification_dataset,
    build_time_series_dataset,
    write_classification_dataset,
    write_time_series_dataset,
)
from .gold import build_gold, write_gold
from .ingestion import download_market_data
from .modeling import evaluate_baselines, write_baseline_metrics
from .silver import build_silver, build_ticker_reference, quality_report, write_silver
from .storage import read_table, table_exists
from .validation import validate_classification, validate_time_series


logger = logging.getLogger(__name__)


def run_pipeline(
    spark: SparkSession,
    tickers: list[str] = DEFAULT_TICKERS,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    ingest: bool = True,
    evaluate: bool = False,
) -> dict:
    """
    Run the financial market data lakehouse pipeline.

    Args:
        ingest: Download a new batch into Bronze. When False, Silver is
            rebuilt by replaying all Bronze history for the tickers.
        evaluate: Train and score baseline classifiers on the result.

    Returns:
        Summary of data quality and dataset validation results.
    """
    unknown = sorted(set(tickers) - set(TICKER_SECTORS))
    if unknown:
        raise ValueError(f"Tickers missing sector reference data: {unknown}")

    summary: dict = {}

    # Ingestion -> Bronze (append-only raw history).
    if ingest:
        market_data = download_market_data(tickers, start_date, end_date)
        batch_df = to_bronze_dataframe(spark, market_data)
        write_bronze(batch_df)

        ingestion_id = market_data["ingestion_id"].iloc[0]
        bronze_df = read_table(spark, BRONZE_PRICES).filter(
            F.col("ingestion_id") == ingestion_id
        )
        logger.info("Bronze: appended batch %s", ingestion_id)
    else:
        if not table_exists(spark, BRONZE_PRICES):
            raise FileNotFoundError("Bronze table not found; run with ingestion first.")
        bronze_df = read_table(spark, BRONZE_PRICES)

    bronze_df = bronze_df.filter(F.col("ticker").isin(tickers))

    # Bronze -> Silver (deduplicate, validate, quarantine, MERGE).
    summary["quality"] = quality_report(bronze_df)
    reference_df = build_ticker_reference(spark)
    silver_batch, quarantined = build_silver(bronze_df, reference_df)
    write_silver(spark, silver_batch, quarantined, reference_df)

    for rule in summary["quality"]:
        if rule["violations"]:
            logger.warning("Quality rule %s: %d violations quarantined",
                           rule["rule"], rule["violations"])

    # Silver -> Gold features.
    silver_df = (
        read_table(spark, SILVER_PRICES)
        .filter(F.col("ticker").isin(tickers))
        .filter(F.col("date").between(start_date, end_date))
    )
    gold_df = build_gold(silver_df)
    write_gold(gold_df)
    logger.info("Gold: daily features written")

    # Gold -> ML-ready datasets.
    gold_df = read_table(spark, GOLD_FEATURES)

    time_series_df = build_time_series_dataset(gold_df)
    write_time_series_dataset(time_series_df)

    classification_df = build_classification_dataset(gold_df)
    write_classification_dataset(classification_df)

    summary["time_series"] = validate_time_series(read_table(spark, GOLD_TIME_SERIES))
    summary["classification"] = validate_classification(
        read_table(spark, GOLD_CLASSIFICATION)
    )
    logger.info("Datasets written and validated")

    if evaluate:
        metrics_df = evaluate_baselines(spark, read_table(spark, GOLD_CLASSIFICATION))
        write_baseline_metrics(metrics_df)
        summary["baseline_metrics"] = [row.asDict() for row in metrics_df.collect()]

    return summary
