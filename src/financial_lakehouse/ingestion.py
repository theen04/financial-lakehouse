import logging
from datetime import UTC, datetime

import pandas as pd
import yfinance as yf


logger = logging.getLogger(__name__)

# yfinance column names -> lakehouse column names. Delta does not allow
# spaces in column names, so names are conformed at ingestion.
SOURCE_COLUMNS = {
    "Date": "date",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}


def download_ticker(
    ticker: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Download historical daily market data for a ticker.

    Returns:
        DataFrame with conformed column names and the ticker symbol.
    """
    logger.info("Downloading %s...", ticker)

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

    df = (
        df
        .reset_index()
        .rename(columns=SOURCE_COLUMNS)
        [list(SOURCE_COLUMNS.values())]
    )

    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["ticker"] = ticker

    logger.info("Downloaded %s: %s rows", ticker, f"{len(df):,}")

    return df


def download_market_data(
    tickers: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Download historical market data for multiple tickers.

    Returns:
        A single long-format DataFrame with ingestion metadata attached.
    """
    ingested_at = datetime.now(UTC)

    df = pd.concat(
        [
            download_ticker(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
            )
            for ticker in tickers
        ],
        ignore_index=True,
    )

    # Ingestion metadata: lineage for every raw record.
    df["source"] = "yfinance"
    df["ingestion_id"] = ingested_at.strftime("%Y%m%dT%H%M%SZ")
    df["ingested_at"] = ingested_at

    return df
