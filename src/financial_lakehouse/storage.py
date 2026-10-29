from pathlib import Path

from delta.tables import DeltaTable
from pyspark.sql import DataFrame, SparkSession


def table_exists(spark: SparkSession, path: Path) -> bool:
    """Return True if a Delta table exists at the given path."""

    return DeltaTable.isDeltaTable(spark, str(path))


def read_table(spark: SparkSession, path: Path) -> DataFrame:
    """Read the current version of a Delta table."""

    return spark.read.format("delta").load(str(path))


def read_table_version(spark: SparkSession, path: Path, version: int) -> DataFrame:
    """Read a Delta table as of an earlier version (time travel)."""

    return (
        spark.read.format("delta")
        .option("versionAsOf", version)
        .load(str(path))
    )


def write_table(
    df: DataFrame,
    path: Path,
    mode: str = "overwrite",
    partition_by: list[str] | None = None,
) -> None:
    """Write a DataFrame to a Delta table."""

    writer = df.write.format("delta").mode(mode)

    if mode == "overwrite":
        writer = writer.option("overwriteSchema", "true")

    if partition_by:
        writer = writer.partitionBy(*partition_by)

    writer.save(str(path))


def merge_table(
    spark: SparkSession,
    df: DataFrame,
    path: Path,
    keys: list[str],
    update_condition: str | None = None,
    partition_by: list[str] | None = None,
) -> None:
    """
    Upsert a DataFrame into a Delta table on the given keys.

    Creates the table on first write.
    """
    if not table_exists(spark, path):
        write_table(df, path, partition_by=partition_by)
        return

    condition = " AND ".join(f"t.`{key}` = s.`{key}`" for key in keys)

    (
        DeltaTable.forPath(spark, str(path))
        .alias("t")
        .merge(df.alias("s"), condition)
        .whenMatchedUpdateAll(condition=update_condition)
        .whenNotMatchedInsertAll()
        .execute()
    )


def table_history(spark: SparkSession, path: Path) -> DataFrame:
    """Return the Delta transaction log for a table."""

    return (
        DeltaTable.forPath(spark, str(path))
        .history()
        .select("version", "timestamp", "operation", "operationMetrics")
    )
