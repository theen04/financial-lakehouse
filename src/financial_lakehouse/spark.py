import os
import sys

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession


def get_spark(
    app_name: str = "Financial Lakehouse",
    master: str = "local[*]",
    shuffle_partitions: int = 8,
) -> SparkSession:
    """Create (or reuse) a local Spark session with Delta Lake enabled."""

    # Run Python workers on the driver's interpreter; otherwise Spark uses
    # whatever `python3` is first on PATH, which may be a different version.
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)

    builder = (
        SparkSession.builder
        .appName(app_name)
        .master(master)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.ui.showConsoleProgress", "false")
    )

    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    return spark
