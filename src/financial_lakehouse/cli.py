import argparse
import logging
from pprint import pformat

from .config import DEFAULT_END_DATE, DEFAULT_START_DATE, DEFAULT_TICKERS
from .pipeline import run_pipeline
from .spark import get_spark


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="financial-lakehouse",
        description="Run the Bronze -> Silver -> Gold financial lakehouse pipeline.",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=DEFAULT_TICKERS,
        help="Ticker symbols to process (default: 20-ticker sector universe).",
    )
    parser.add_argument("--start", default=DEFAULT_START_DATE, help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", default=DEFAULT_END_DATE, help="End date, exclusive (YYYY-MM-DD).")
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Do not download; rebuild Silver and Gold from existing Bronze history.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Train baseline classifiers on the classification dataset.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # yfinance and py4j are noisy at INFO.
    logging.getLogger("py4j").setLevel(logging.WARNING)
    logging.getLogger("yfinance").setLevel(logging.WARNING)

    spark = get_spark()

    try:
        summary = run_pipeline(
            spark,
            tickers=args.tickers,
            start_date=args.start,
            end_date=args.end,
            ingest=not args.skip_ingest,
            evaluate=args.evaluate,
        )
    finally:
        spark.stop()

    logging.getLogger(__name__).info("Pipeline summary:\n%s", pformat(summary, sort_dicts=False))


if __name__ == "__main__":
    main()
