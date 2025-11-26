"""
Preprocessing module for building implicit feedback interaction matrix.

EN: Cleans events, applies weights, filters, and creates sparse user-item matrix.
FA: داده‌های رویداد را پاکسازی کرده، وزن‌دهی و فیلتر می‌کند و ماتریس تنک کاربر-کالا می‌سازد.
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

from src.utils.io import ensure_dir, save_pickle

# EN: Event weights reflecting implicit feedback strength
# FA: وزن رویدادها که شدت بازخورد ضمنی را نشان می‌دهد
EVENT_WEIGHTS: Dict[str, float] = {
    "view": 1.0,
    "addtocart": 2.0,
    "transaction": 3.0,
}

# EN: Minimum thresholds to reduce noise
# FA: آستانه‌های حداقلی برای کاهش نویز
MIN_USER_EVENTS = 5
MIN_ITEM_EVENTS = 5


def load_clean_events(processed_dir: Path) -> pd.DataFrame:
    """
    Load cleaned events parquet.

    EN: Reads the already-cleaned events file.
    FA: فایل رویداد پاکسازی شده را می‌خواند.
    """
    events_path = processed_dir / "events_clean.parquet"
    df = pd.read_parquet(events_path)
    return df


def apply_event_weights(events: pd.DataFrame) -> pd.DataFrame:
    """
    Attach implicit feedback weights to events.

    EN: Maps event types to numeric weights and aggregates per user-item.
    FA: نوع رویداد را به وزن عددی نگاشت کرده و در سطح کاربر-کالا تجمیع می‌کند.
    """
    # EN: Keep only known event types for consistency
    # FA: فقط رویدادهای شناخته‌شده را نگه می‌داریم تا سازگار بماند
    events = events[events["event"].isin(EVENT_WEIGHTS.keys())].copy()
    # EN: Map each event string to its configured weight
    # FA: هر رویداد را به وزن تعریف‌شده‌اش نگاشت می‌کنیم
    events["weight"] = events["event"].map(EVENT_WEIGHTS)
    # EN: Aggregate weights per (visitorid, itemid) pair
    # FA: وزن‌ها را در سطح (کاربر، کالا) تجمیع می‌کنیم
    aggregated = (
        events.groupby(["visitorid", "itemid"], as_index=False)["weight"]
        .sum()
        .rename(columns={"weight": "score"})
    )
    return aggregated


def filter_minimums(interactions: pd.DataFrame) -> pd.DataFrame:
    """
    Filter users and items with too few interactions.

    EN: Removes sparse users/items to improve matrix quality.
    FA: کاربران و کالاهای بسیار کم‌تعامل را حذف می‌کند تا کیفیت ماتریس بهبود یابد.
    """
    # EN: Count interactions per user
    # FA: تعداد تعاملات هر کاربر را می‌شماریم
    user_counts = interactions["visitorid"].value_counts()
    valid_users = user_counts[user_counts >= MIN_USER_EVENTS].index

    # EN: Filter out users below threshold
    # FA: کاربران پایین‌تر از آستانه حذف می‌شوند
    filtered = interactions[interactions["visitorid"].isin(valid_users)].copy()

    # EN: Count interactions per item (unique users)
    # FA: تعداد کاربران یکتا برای هر کالا را می‌شماریم
    item_counts = filtered.groupby("itemid")["visitorid"].nunique()
    valid_items = item_counts[item_counts >= MIN_ITEM_EVENTS].index

    # EN: Keep items that meet the threshold
    # FA: کالاهای واجد شرایط آستانه نگه داشته می‌شوند
    filtered = filtered[filtered["itemid"].isin(valid_items)].copy()
    return filtered


def build_mappings(interactions: pd.DataFrame) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Create integer index mappings for users and items.

    EN: Maps string IDs to contiguous integer indices.
    FA: شناسه‌های متنی را به اندیس‌های صحیح متوالی نگاشت می‌کند.
    """
    unique_users = interactions["visitorid"].unique()
    unique_items = interactions["itemid"].unique()

    # EN: Build dictionaries for fast lookup
    # FA: دیکشنری‌های نگاشت سریع می‌سازیم
    user_mapping = {user_id: idx for idx, user_id in enumerate(unique_users)}
    item_mapping = {item_id: idx for idx, item_id in enumerate(unique_items)}
    return user_mapping, item_mapping


def build_sparse_matrix(
    interactions: pd.DataFrame, user_mapping: Dict[str, int], item_mapping: Dict[str, int]
) -> sparse.coo_matrix:
    """
    Construct a COO sparse matrix from interactions.

    EN: Uses aggregated scores to populate the matrix.
    FA: ماتریس تنک را با امتیازهای تجمیع‌شده پر می‌کند.
    """
    # EN: Map string IDs to integer indices
    # FA: شناسه‌های متنی را به اندیس عددی تبدیل می‌کنیم
    user_indices = interactions["visitorid"].map(user_mapping).to_numpy()
    item_indices = interactions["itemid"].map(item_mapping).to_numpy()
    scores = interactions["score"].to_numpy()

    # EN: Build sparse matrix (users x items)
    # FA: ساخت ماتریس تنک با ابعاد (کاربر × کالا)
    matrix = sparse.coo_matrix(
        (scores, (user_indices, item_indices)),
        shape=(len(user_mapping), len(item_mapping)),
        dtype=np.float32,
    )
    return matrix


def build_interaction_matrix(processed_dir: Path | None = None) -> None:
    """
    Main entry point to generate user-item interaction matrix.

    EN: Loads cleaned events, applies weights/filters, saves matrix and mappings.
    FA: رویدادهای پاکسازی‌شده را بارگذاری کرده، وزن و فیلتر اعمال کرده و ماتریس و نگاشت‌ها را ذخیره می‌کند.
    """
    project_root = Path(__file__).resolve().parents[2]
    processed_dir = processed_dir or (project_root / "data" / "processed")

    # EN: Ensure output directory exists
    # FA: مطمئن می‌شویم پوشه خروجی وجود دارد
    ensure_dir(processed_dir)

    print(f"Loading cleaned events from: {processed_dir}")
    events = load_clean_events(processed_dir)

    # EN: Apply weights to implicit events
    # FA: وزن‌دهی به رویدادهای ضمنی
    interactions = apply_event_weights(events)

    # EN: Filter out very sparse users/items
    # FA: حذف کاربران/کالاهای بسیار کم‌تعامل
    interactions = filter_minimums(interactions)

    # EN: Build ID-to-index mappings
    # FA: ساخت نگاشت شناسه به اندیس
    user_mapping, item_mapping = build_mappings(interactions)

    # EN: Create sparse interaction matrix
    # FA: ایجاد ماتریس تنک تعاملی
    interaction_matrix = build_sparse_matrix(interactions, user_mapping, item_mapping)

    # EN: Save artifacts
    # FA: ذخیره مصنوعات پردازش
    matrix_path = processed_dir / "user_item_interactions.npz"
    user_map_path = processed_dir / "user_mapping.pkl"
    item_map_path = processed_dir / "item_mapping.pkl"

    ensure_dir(processed_dir)
    sparse.save_npz(matrix_path, interaction_matrix)
    save_pickle(user_mapping, user_map_path)
    save_pickle(item_mapping, item_map_path)

    # EN: Print summary statistics
    # FA: چاپ آمار خلاصه
    num_users, num_items = interaction_matrix.shape
    nnz = interaction_matrix.nnz
    total_possible = num_users * num_items if num_users and num_items else 1
    sparsity = 1 - (nnz / total_possible)

    print(f"Users: {num_users:,} | Items: {num_items:,}")
    print(f"Non-zero interactions: {nnz:,}")
    print(f"Sparsity: {sparsity:.4f}")
    print(f"Matrix saved to: {matrix_path}")
    print(f"User mapping saved to: {user_map_path}")
    print(f"Item mapping saved to: {item_map_path}")


if __name__ == "__main__":
    # EN: Allow running as a standalone preprocessing step
    # FA: امکان اجرای مستقل گام پیش‌پردازش
    build_interaction_matrix()
