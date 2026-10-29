from functools import reduce

from pyspark.sql import Column, DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from .config import (
    SILVER_PRICES,
    SILVER_QUARANTINE,
    SILVER_TICKER_REFERENCE,
    TICKER_SECTORS,
)
from .storage import merge_table, write_table


PRICE_COLUMNS = ["open", "high", "low", "close", "adj_close"]
KEY_COLUMNS = ["ticker", "date"]

REQUIRED_COLUMNS = [*KEY_COLUMNS, *PRICE_COLUMNS, "volume"]


def quality_rules() -> dict[str, Column]:
    """
    Data quality rules for Silver.

    Each rule describes a valid record. Records failing any rule are
    quarantined instead of being written to Silver.
    """
    return {
        "required_fields_present": reduce(
            lambda left, right: left & right,
            [F.col(c).isNotNull() for c in REQUIRED_COLUMNS],
        ),
        "prices_positive": F.least(*PRICE_COLUMNS) > 0,
        "volume_non_negative": F.col("volume") >= 0,
        "high_gte_low": F.col("high") >= F.col("low"),
        "high_gte_open_close": F.col("high") >= F.greatest("open", "close"),
        "low_lte_open_close": F.col("low") <= F.least("open", "close"),
    }


def build_ticker_reference(spark: SparkSession) -> DataFrame:
    """Build the ticker reference (dimension) table from configuration."""

    return spark.createDataFrame(
        list(TICKER_SECTORS.items()),
        schema="ticker string, sector string",
    )


def deduplicate_latest(df: DataFrame) -> DataFrame:
    """Keep the most recently ingested record for each ticker and date."""

    latest_first = (
        Window
        .partitionBy(*KEY_COLUMNS)
        .orderBy(F.col("ingested_at").desc())
    )

    return (
        df
        .withColumn("_rank", F.row_number().over(latest_first))
        .filter(F.col("_rank") == 1)
        .drop("_rank")
    )


def apply_quality_rules(df: DataFrame) -> DataFrame:
    """Add a failed_rules array column listing every rule a record violates."""

    failures = [
        # A null rule result means a field was missing; treat it as a failure.
        F.when(F.coalesce(condition, F.lit(False)), None).otherwise(F.lit(name))
        for name, condition in quality_rules().items()
    ]

    return df.withColumn(
        "failed_rules",
        F.filter(F.array(*failures), lambda rule: rule.isNotNull()),
    )


def split_valid_records(df: DataFrame) -> tuple[DataFrame, DataFrame]:
    """Split records into (valid, quarantined) using the quality rules."""

    checked = apply_quality_rules(df)

    valid = (
        checked
        .filter(F.size("failed_rules") == 0)
        .drop("failed_rules")
    )
    quarantined = (
        checked
        .filter(F.size("failed_rules") > 0)
        .withColumn("quarantined_at", F.current_timestamp())
    )

    return valid, quarantined


def build_silver(
    bronze_df: DataFrame,
    reference_df: DataFrame,
) -> tuple[DataFrame, DataFrame]:
    """
    Apply Silver transformations to a Bronze batch.

    Returns:
        Tuple of (clean Silver records, quarantined records).
    """
    df = deduplicate_latest(bronze_df)
    valid, quarantined = split_valid_records(df)

    valid = (
        valid
        .join(F.broadcast(reference_df), on="ticker", how="left")
        .select(
            *KEY_COLUMNS,
            "sector",
            *PRICE_COLUMNS,
            "volume",
            "source",
            "ingestion_id",
            "ingested_at",
        )
    )

    return valid, quarantined


def quality_report(bronze_df: DataFrame) -> list[dict]:
    """Count violations of each quality rule in a Bronze batch."""

    checked = apply_quality_rules(bronze_df)
    rule_names = list(quality_rules())

    counts = checked.select(
        F.count("*").alias("records"),
        *[
            F.sum(F.array_contains("failed_rules", name).cast("int")).alias(name)
            for name in rule_names
        ],
    ).first()

    return [
        {"rule": name, "violations": counts[name] or 0, "records": counts["records"]}
        for name in rule_names
    ]


def write_silver(
    spark: SparkSession,
    silver_df: DataFrame,
    quarantined_df: DataFrame,
    reference_df: DataFrame,
) -> None:
    """
    Upsert clean records into Silver and append rejects to quarantine.

    The MERGE only replaces an existing row when the incoming record was
    ingested later, so replaying older Bronze batches is a no-op.
    """
    write_table(reference_df, SILVER_TICKER_REFERENCE)

    merge_table(
        spark,
        silver_df,
        SILVER_PRICES,
        keys=KEY_COLUMNS,
        update_condition="s.ingested_at > t.ingested_at",
        partition_by=["ticker"],
    )

    if not quarantined_df.isEmpty():
        write_table(quarantined_df, SILVER_QUARANTINE, mode="append")
