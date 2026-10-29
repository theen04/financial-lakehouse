from pathlib import Path

import pandas as pd
import pyarrow as pa
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import types as T

from .config import BRONZE_PRICES
from .storage import write_table


# Bronze keeps the source values as delivered. Prices are stored as
# doubles and nothing is filtered, deduplicated, or corrected here.
BRONZE_SCHEMA = T.StructType([
    T.StructField("date", T.DateType()),
    T.StructField("open", T.DoubleType()),
    T.StructField("high", T.DoubleType()),
    T.StructField("low", T.DoubleType()),
    T.StructField("close", T.DoubleType()),
    T.StructField("adj_close", T.DoubleType()),
    T.StructField("volume", T.LongType()),
    T.StructField("ticker", T.StringType()),
    T.StructField("source", T.StringType()),
    T.StructField("ingestion_id", T.StringType()),
    T.StructField("ingested_at", T.TimestampType()),
])


def to_bronze_dataframe(
    spark: SparkSession,
    market_data: pd.DataFrame,
) -> DataFrame:
    """Convert downloaded market data to a Spark DataFrame with the Bronze schema."""

    pdf = market_data[BRONZE_SCHEMA.fieldNames()].copy()

    # yfinance returns float volume when values are missing; Bronze keeps
    # nulls as nulls rather than dropping the record.
    pdf["volume"] = pdf["volume"].astype("Int64")

    # Hand Spark an Arrow table directly; pandas 3 Arrow-backed columns
    # otherwise fail the pandas -> Arrow fast path.
    return spark.createDataFrame(
        pa.Table.from_pandas(pdf, preserve_index=False),
        schema=BRONZE_SCHEMA,
    )


def write_bronze(
    df: DataFrame,
    output_dir: Path = BRONZE_PRICES,
) -> None:
    """
    Append a raw ingestion batch to the Bronze Delta table.

    Bronze is append-only: every ingestion is retained with its
    ingestion_id, giving a full replayable history of the source.
    """
    write_table(
        df,
        output_dir,
        mode="append",
        partition_by=["ticker"],
    )
