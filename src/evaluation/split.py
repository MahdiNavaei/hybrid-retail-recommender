"""
User-centric train/test split for offline evaluation.

EN: Splits interactions per user into chronological train/test partitions.
FA: تعاملات هر کاربر را به‌صورت زمانی به مجموعه آموزش و آزمون تقسیم می‌کند.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from src.data.preprocess import EVENT_WEIGHTS
from src.utils.io import ensure_dir


def build_train_test_split(
    processed_dir: Path,
    test_fraction: float = 0.2,
    min_items_in_test: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build train/test interactions split per user.

    EN: Splits each user's history chronologically, keeping the last part as test.
    FA: تاریخچه هر کاربر را به ترتیب زمان تقسیم کرده و بخش انتهایی را برای آزمون نگه می‌دارد.
    """
    events_path = processed_dir / "events_clean.parquet"
    if not events_path.exists():
        raise FileNotFoundError(f"Missing events file: {events_path}")

    events = pd.read_parquet(events_path)

    # EN: Keep only known event types and attach weight without aggregation
    # FA: فقط رویدادهای شناخته‌شده را نگه داشته و وزن را بدون تجمیع اضافه می‌کنیم
    events = events[events["event"].isin(EVENT_WEIGHTS.keys())].copy()
    events["score"] = events["event"].map(EVENT_WEIGHTS)
    interactions = events[["visitorid", "itemid", "score", "timestamp"]].copy()

    train_parts = []
    test_parts = []

    # EN: Group by user and split chronologically
    # FA: بر اساس کاربر گروه‌بندی کرده و به صورت زمانی تقسیم می‌کنیم
    for user_id, user_df in interactions.groupby("visitorid"):
        user_df = user_df.sort_values("timestamp")
        n = len(user_df)
        if n < (min_items_in_test + 1):
            # EN: Skip users without enough interactions to split
            # FA: کاربرانی که تعامل کافی ندارند حذف می‌شوند
            continue
        test_size = max(min_items_in_test, int(n * test_fraction))
        test_slice = user_df.tail(test_size)
        train_slice = user_df.head(n - test_size)
        if len(train_slice) == 0 or len(test_slice) == 0:
            continue
        train_parts.append(train_slice)
        test_parts.append(test_slice)

    if not train_parts or not test_parts:
        raise ValueError("No users with sufficient interactions for splitting.")

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    # EN: Save to processed directory for reuse
    # FA: برای استفاده‌های بعدی در مسیر processed ذخیره می‌کنیم
    ensure_dir(processed_dir)
    train_path = processed_dir / "train_interactions.parquet"
    test_path = processed_dir / "test_interactions.parquet"
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    print(f"Train interactions: {len(train_df):,}")
    print(f"Test interactions:  {len(test_df):,}")
    return train_df, test_df


if __name__ == "__main__":
    # EN: Build train/test split and save to processed folder
    # FA: ساخت و ذخیره داده‌های آموزش/آزمون در پوشه processed
    project_root = Path(__file__).resolve().parents[2]
    processed = project_root / "data" / "processed"
    build_train_test_split(processed_dir=processed)
