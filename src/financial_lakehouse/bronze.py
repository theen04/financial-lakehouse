from pathlib import Path

import pandas as pd

from .config import BRONZE_DIR


def write_ticker_to_bronze(
    df: pd.DataFrame,
    ticker: str,
    output_dir: Path = BRONZE_DIR,
) -> tuple[str, int]:
    """
    Write ticker data to partitioned Bronze Parquet.

    Partitions are organized by ticker, year, and month.

    Returns:
        Tuple containing the ticker and number of rows written.
    """
    df = df.copy()

    # Convert date for partitioning and downstream processing.
    df["Date"] = pd.to_datetime(df["Date"])

    # Create partition columns.
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month.astype(str).str.zfill(2)

    for (year, month), partition_df in df.groupby(["year", "month"]):
        partition_path = (
            output_dir
            / f"ticker={ticker}"
            / f"year={year}"
            / f"month={month}"
        )

        partition_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_file = partition_path / "data.parquet"

        (
            partition_df
            .drop(columns=["ticker", "year", "month"])
            .to_parquet(
                output_file,
                index=False,
                engine="pyarrow",
            )
        )

    return ticker, len(df)


def write_market_data_to_bronze(
    market_data: dict[str, pd.DataFrame],
    output_dir: Path = BRONZE_DIR,
) -> list[tuple[str, int]]:
    """
    Write multiple ticker DataFrames to partitioned Bronze Parquet.

    Returns:
        List of ticker and row-count tuples.
    """
    return [
        write_ticker_to_bronze(
            df=df,
            ticker=ticker,
            output_dir=output_dir,
        )
        for ticker, df in market_data.items()
    ]