from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
LAKEHOUSE_DIR = DATA_DIR / "lakehouse"

BRONZE_DIR = LAKEHOUSE_DIR / "bronze"
SILVER_DIR = LAKEHOUSE_DIR / "silver"
GOLD_DIR = LAKEHOUSE_DIR / "gold"

DEFAULT_START_DATE = "2016-01-01"
DEFAULT_END_DATE = "2026-01-01"