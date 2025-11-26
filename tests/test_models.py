"""
Quick smoke tests for Phase 2 models.

EN: Trains CF and content models on a small sample and prints example recommendations.
FA: مدل‌های CF و محتوایی را روی نمونه کوچک آموزش داده و توصیه‌های نمونه چاپ می‌کند.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

import sys

# EN: Add project root to sys.path so `src` can be imported when running as a script
# FA: برای ایمپورت پوشه `src` هنگام اجرای مستقیم، ریشه پروژه را به sys.path اضافه می‌کنیم
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.collaborative import CollaborativeRecommender, load_processed_mappings
from src.models.content_based import ContentBasedRecommender
from src.models.hybrid import HybridRecommender


def load_artifacts(project_root: Path) -> tuple[sparse.csr_matrix, dict, dict, pd.DataFrame]:
    """
    Load processed interaction matrix, mappings, and item properties.

    EN: Reads artifacts generated in Phase 1 preprocessing.
    FA: مصنوعات ساخته‌شده در پیش‌پردازش فاز ۱ را بارگذاری می‌کند.
    """
    processed = project_root / "data" / "processed"
    matrix_path = processed / "user_item_interactions.npz"
    user_map_path = processed / "user_mapping.pkl"
    item_map_path = processed / "item_mapping.pkl"
    item_props_path = processed / "item_properties_clean.parquet"

    # EN: Validate presence of required artifacts
    # FA: وجود فایل‌های ضروری را بررسی می‌کنیم
    missing = [p for p in [matrix_path, user_map_path, item_map_path, item_props_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing processed artifacts: {missing}. "
            "Run Phase 1 preprocessing (ingest.py and preprocess.py) first."
        )

    interaction_matrix = sparse.load_npz(matrix_path).tocsr()
    user_mapping, item_mapping = load_processed_mappings(project_root)
    item_props = pd.read_parquet(item_props_path)
    return interaction_matrix, user_mapping, item_mapping, item_props


def sample_interactions(matrix: sparse.csr_matrix, max_users: int = 2000, max_items: int = 3000) -> sparse.csr_matrix:
    """
    Take a smaller submatrix for quick testing.

    EN: Limits users/items to keep ALS training fast in tests.
    FA: برای سرعت در تست، تعداد کاربران/اقلام را محدود می‌کند.
    """
    num_users, num_items = matrix.shape
    u = min(max_users, num_users)
    i = min(max_items, num_items)
    return matrix[:u, :i].copy()


def main() -> None:
    """
    Run lightweight training and print sample outputs.

    EN: Demonstrates CF, content, and hybrid recommendations.
    FA: نحوه کار مدل‌های CF، محتوایی و هایبرید را نشان می‌دهد.
    """
    project_root = Path(__file__).resolve().parents[1]
    interactions, user_map, item_map, item_props = load_artifacts(project_root)

    # EN: Reduce matrix size for a quick smoke test
    # FA: ماتریس را برای تست سریع کوچک می‌کنیم
    interactions_sample = sample_interactions(interactions, max_users=1500, max_items=2000)

    # EN: Train collaborative model
    # FA: آموزش مدل مشارکتی
    cf_model = CollaborativeRecommender(factors=20, iterations=4, regularization=0.1)
    cf_model.fit(interactions_sample, user_mapping=user_map, item_mapping=item_map)

    # EN: Train content model
    # FA: آموزش مدل محتوایی
    content_model = ContentBasedRecommender(max_features=3000, ngram_range=(1, 2), min_df=2)
    content_model.fit(item_props)

    # EN: Build hybrid model with alpha=0.7
    # FA: ساخت مدل هایبرید با آلفای 0.7
    hybrid_model = HybridRecommender(
        model_cf=cf_model,
        model_content=content_model,
        alpha=0.7,
        interactions=interactions_sample,
        user_mapping=user_map,
        item_mapping=item_map,
    )

    # EN: Pick sample user/item for demonstration
    # FA: انتخاب یک کاربر/کالای نمونه برای نمایش
    example_user = next(iter(user_map.keys()))
    example_item = next(iter(item_map.keys()))

    print(f"\nSample user id: {example_user}")
    print(f"Sample item id: {example_item}")

    print("\n=== Collaborative recommendations ===")
    print(cf_model.recommend(example_user, top_k=5))

    print("\n=== Content-based similar items ===")
    print(content_model.similar_items(example_item, top_k=5))

    print("\n=== Hybrid recommendations ===")
    print(hybrid_model.recommend_for_user(example_user, top_k=5))


if __name__ == "__main__":
    # EN: Execute smoke test when run directly
    # FA: در صورت اجرای مستقیم، تست دود اجرا می‌شود
    main()
