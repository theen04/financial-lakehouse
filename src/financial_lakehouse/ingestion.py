from datetime import UTC, datetime

import pandas as pd
import yfinance as yf


def download_ticker(
    ticker: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Download historical market data for a ticker.

    Returns:
        DataFrame containing the downloaded market data and ingestion metadata.
    """
    print(f"Downloading {ticker}...")

    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        raise ValueError(f"No market data returned for {ticker}.")

    # Flatten yfinance MultiIndex columns if present.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    # Add ingestion metadata.
    df["ticker"] = ticker
    df["download_timestamp"] = datetime.now(UTC)

    print(f"Downloaded {ticker}: {len(df):,} rows")

    return df


def download_market_data(
    tickers: list[str],
    start_date: str,
    end_date: str,
) -> dict[str, pd.DataFrame]:
    """
    Download historical market data for multiple tickers.

    Returns:
        Dictionary mapping ticker symbols to their DataFrames.
    """
    return {
        ticker: download_ticker(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
        )
        for ticker in tickers
    }