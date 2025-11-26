"""
Data ingestion module for Retailrocket dataset.

EN: Loads raw CSVs, standardizes types, and saves cleaned outputs.
FA: بارگذاری فایل‌های CSV خام، استانداردسازی انواع داده و ذخیره خروجی پاکسازی شده.
"""

from pathlib import Path
from typing import List
import pandas as pd

from src.utils.io import ensure_dir

# EN: Define default filenames
# FA: نام فایل‌های پیش‌فرض را مشخص می‌کنیم
EVENTS_FILE = "events.csv"
CATEGORY_TREE_FILE = "category_tree.csv"
ITEM_PROPERTIES_PATTERN = "item_properties_*.csv"


def _resolve_dataset_dir() -> Path:
    """
    Determine the dataset directory.

    EN: Prefer a sibling Dataset/ directory next to the project.
    FA: ابتدا پوشه Dataset در کنار پروژه را جستجو می‌کند.
    """
    project_root = Path(__file__).resolve().parents[2]
    # EN: Typical layout expects Dataset alongside the project folder
    # FA: ساختار رایج شامل پوشه Dataset در کنار پوشه اصلی پروژه است
    sibling_dataset = project_root.parent / "Dataset"
    internal_dataset = project_root / "Dataset"
    if sibling_dataset.exists():
        return sibling_dataset
    return internal_dataset


def load_raw_events(dataset_dir: Path) -> pd.DataFrame:
    """
    Load raw events data.

    EN: Reads the events CSV and converts timestamp to datetime.
    FA: فایل events را خوانده و ستون زمان را به datetime تبدیل می‌کند.
    """
    events_path = dataset_dir / EVENTS_FILE
    # EN: Load with minimal dtype coercion to keep IDs as strings
    # FA: داده‌ها را طوری می‌خوانیم که شناسه‌ها به صورت رشته باقی بمانند
    df = pd.read_csv(events_path)
    # EN: Normalize timestamp column (UNIX seconds to datetime)
    # FA: ستون زمان را از ثانیه یونیکس به datetime در پانداس تبدیل می‌کنیم
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
    # EN: Cast identifier columns to string for consistency
    # FA: ستون‌های شناسه را برای یکپارچگی به رشته تبدیل می‌کنیم
    for col in ["visitorid", "itemid"]:
        if col in df.columns:
            df[col] = df[col].astype("string")
    return df


def load_raw_item_properties(dataset_dir: Path) -> pd.DataFrame:
    """
    Load and concatenate all item_properties parts.

    EN: Aggregates multiple part files into one DataFrame.
    FA: چند فایل تکه‌ای item_properties را در یک دیتافریم ادغام می‌کند.
    """
    paths: List[Path] = sorted(dataset_dir.glob(ITEM_PROPERTIES_PATTERN))
    frames = []
    for path in paths:
        # EN: Read each part and append
        # FA: هر بخش را می‌خوانیم و به لیست اضافه می‌کنیم
        frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    # EN: Normalize timestamp to datetime
    # FA: ستون زمان را به datetime تبدیل می‌کنیم
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
    # EN: Cast identifier columns to string
    # FA: ستون‌های شناسه را به رشته تبدیل می‌کنیم
    for col in ["itemid", "categoryid"]:
        if col in df.columns:
            df[col] = df[col].astype("string")
    return df


def load_raw_category_tree(dataset_dir: Path) -> pd.DataFrame:
    """
    Load category tree data.

    EN: Reads the category tree CSV.
    FA: فایل درخت دسته‌بندی را می‌خواند.
    """
    path = dataset_dir / CATEGORY_TREE_FILE
    df = pd.read_csv(path)
    # EN: Cast identifiers to string for consistency
    # FA: شناسه‌ها را برای یکپارچگی به رشته تبدیل می‌کنیم
    for col in df.columns:
        df[col] = df[col].astype("string")
    return df


def _print_basic_stats(df: pd.DataFrame, name: str) -> None:
    """Print quick stats for a dataframe."""
    # EN: Display row count and a sample
    # FA: تعداد ردیف‌ها و نمونه‌ای کوچک را چاپ می‌کنیم
    print(f"{name}: {len(df):,} rows")
    print(df.head(3))
    print("-" * 40)


def ingest_all(dataset_dir: Path | None = None, processed_dir: Path | None = None) -> None:
    """
    Run the full ingestion pipeline.

    EN: Loads raw CSVs, cleans them, and saves standardized outputs.
    FA: کل فرآیند دریافت داده را اجرا می‌کند، داده‌ها را پاکسازی و ذخیره می‌کند.
    """
    # EN: Resolve directories with sensible defaults
    # FA: مسیرها را با مقادیر پیش‌فرض منطقی تعیین می‌کنیم
    dataset_dir = dataset_dir or _resolve_dataset_dir()
    project_root = Path(__file__).resolve().parents[2]
    processed_dir = processed_dir or (project_root / "data" / "processed")

    # EN: Ensure output directory exists
    # FA: مطمئن می‌شویم پوشه خروجی وجود دارد
    ensure_dir(processed_dir)

    print(f"Loading data from: {dataset_dir}")
    print(f"Saving cleaned data to: {processed_dir}")

    # EN: Load each dataset
    # FA: هر دیتاست را بارگذاری می‌کنیم
    events_df = load_raw_events(dataset_dir)
    item_props_df = load_raw_item_properties(dataset_dir)
    category_df = load_raw_category_tree(dataset_dir)

    # EN: Drop rows with missing critical IDs
    # FA: ردیف‌هایی با شناسه‌های ضروری مفقود را حذف می‌کنیم
    events_df = events_df.dropna(subset=["visitorid", "itemid"])
    item_props_df = item_props_df.dropna(subset=["itemid"])

    # EN: Print basic stats for sanity check
    # FA: آمار اولیه برای اطمینان از صحت داده
    _print_basic_stats(events_df, "Events")
    _print_basic_stats(item_props_df, "Item Properties")
    _print_basic_stats(category_df, "Category Tree")

    # EN: Save cleaned versions to Parquet for efficiency
    # FA: نسخه‌های پاکسازی شده را برای کارایی به صورت Parquet ذخیره می‌کنیم
    events_path = processed_dir / "events_clean.parquet"
    item_props_path = processed_dir / "item_properties_clean.parquet"
    category_path = processed_dir / "category_tree_clean.parquet"

    events_df.to_parquet(events_path, index=False)
    item_props_df.to_parquet(item_props_path, index=False)
    category_df.to_parquet(category_path, index=False)

    print("Ingestion completed.")
    print(f"Events saved to: {events_path}")
    print(f"Item properties saved to: {item_props_path}")
    print(f"Category tree saved to: {category_path}")


if __name__ == "__main__":
    # EN: Allow running as a script for quick ingestion
    # FA: امکان اجرای مستقیم اسکریپت برای دریافت سریع داده
    ingest_all()
