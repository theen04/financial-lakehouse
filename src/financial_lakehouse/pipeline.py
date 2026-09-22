from pyspark.sql import SparkSession

from .bronze import write_market_data_to_bronze
from .config import BRONZE_DIR
from .datasets import (
    build_classification_dataset,
    build_time_series_dataset,
    write_classification_dataset,
    write_time_series_dataset,
)
from .gold import build_gold, write_gold
from .ingestion import download_market_data
from .silver import build_silver, write_silver


def run_pipeline(
    spark: SparkSession,
    tickers: list[str],
    start_date: str,
    end_date: str,
) -> None:
    """Run the complete financial market data lakehouse pipeline."""

    print("Starting financial market data lakehouse pipeline.")

    # data ingestion

    market_data = download_market_data(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
    )

    # bronze layer

    write_market_data_to_bronze(market_data)

    # read bronze layer

    bronze_df = spark.read.parquet(str(BRONZE_DIR))

    # build and write silver

    silver_df = build_silver(bronze_df)
    write_silver(silver_df)

    # build and write gold

    gold_df = build_gold(silver_df)
    write_gold(gold_df)

    # downstream datasets

    time_series_df = build_time_series_dataset(gold_df)
    classification_df = build_classification_dataset(gold_df)

    write_time_series_dataset(time_series_df)
    write_classification_dataset(classification_df)

    print("Pipeline completed successfully.")