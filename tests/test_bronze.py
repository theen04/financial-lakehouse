from financial_lakehouse.bronze import BRONZE_SCHEMA, write_bronze
from financial_lakehouse.storage import read_table, table_history

from conftest import make_market_data


def test_bronze_schema(bronze_df, market_data):
    assert bronze_df.schema == BRONZE_SCHEMA
    assert bronze_df.count() == len(market_data)


def test_bronze_is_append_only(spark, bronze_df, tmp_path):
    path = tmp_path / "bronze"

    write_bronze(bronze_df, path)
    write_bronze(bronze_df, path)

    assert read_table(spark, path).count() == 2 * bronze_df.count()
    operations = [r["operation"] for r in table_history(spark, path).collect()]
    assert operations == ["WRITE", "WRITE"]


def test_nullable_volume_is_preserved(spark):
    from financial_lakehouse.bronze import to_bronze_dataframe

    pdf = make_market_data(tickers=("AAPL",), periods=5)
    pdf["volume"] = pdf["volume"].astype("float")
    pdf.loc[2, "volume"] = float("nan")

    df = to_bronze_dataframe(spark, pdf)

    assert df.filter("volume IS NULL").count() == 1
