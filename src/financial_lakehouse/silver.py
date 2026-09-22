from pathlib import Path
from .config import SILVER_DIR

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def check_nulls(df: DataFrame) -> DataFrame:
    """Return null counts for each column."""

    return df.select([
        F.sum(
            F.when(F.col(column).isNull(), 1).otherwise(0)
        ).alias(column)
        for column in df.columns
    ])


def remove_duplicates(df: DataFrame) -> DataFrame:
    """Remove duplicate records."""

    return df.dropDuplicates()


def standardize_schema(df: DataFrame) -> DataFrame:
    """Cast market data to the Silver layer schema."""

    return df.select(
        F.to_date(F.col("Date")).alias("Date"),
        F.col("Adj Close").cast("double").alias("Adj Close"),
        F.col("Open").cast("double").alias("Open"),
        F.col("High").cast("double").alias("High"),
        F.col("Low").cast("double").alias("Low"),
        F.col("Close").cast("double").alias("Close"),
        F.col("Volume").cast("long").alias("Volume"),
        F.col("download_timestamp"),
        F.col("ticker"),
        F.col("year"),
        F.col("month"),
    )


def check_data_quality(df: DataFrame):
    """Evaluate Silver data quality rules."""

    rules = {
        "Closing Price > 0": F.col("Close") > 0,
        "Volume >= 0": F.col("Volume") >= 0,
        "High >= Low": F.col("High") >= F.col("Low"),
    }

    results = []

    for rule_name, condition in rules.items():
        violations = df.filter(~condition).count()

        results.append({
            "rule": rule_name,
            "violations": violations,
        })

    return results

def build_silver(df: DataFrame) -> DataFrame:
    """Apply Silver layer transformations."""

    df = remove_duplicates(df)
    df = standardize_schema(df)

    return df


def write_silver(
    df: DataFrame,
    output_dir: Path = SILVER_DIR,
) -> None:
    """Write the Silver DataFrame to partitioned Parquet."""

    (
        df.write
        .mode("overwrite")
        .partitionBy("ticker", "year", "month")
        .parquet(str(output_dir))
    )