"""
Offline evaluation runner for recommender models.

EN: Trains baseline, CF, and hybrid models on train split and evaluates on test split.
FA: مدل‌های پایه، CF و هایبرید را روی داده آموزش تمرین داده و روی داده آزمون ارزیابی می‌کند.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy import sparse

from src.evaluation.metrics import evaluate_user_ranking
from src.models.baseline import PopularityRecommender
from src.models.collaborative import CollaborativeRecommender
from src.models.content_based import ContentBasedRecommender
from src.models.hybrid import HybridRecommender
from src.utils.io import ensure_dir, load_pickle


def load_yaml_safe(path: Path) -> Dict:
    """
    Load YAML config if exists, else return empty dict.

    EN: Safe helper to avoid errors when config is missing.
    FA: تابع کمکی برای خواندن YAML در صورت وجود و جلوگیری از خطا در نبود فایل.
    """
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_train_matrix(
    interactions: pd.DataFrame, user_mapping: Dict[str, int], item_mapping: Dict[str, int]
) -> sparse.csr_matrix:
    """
    Build sparse user-item matrix from train interactions using existing mappings.

    EN: Recreates a COO/CSR matrix restricted to known users/items.
    FA: ماتریس تنک را با نگاشت‌های موجود برای کاربران/کالاهای شناخته‌شده می‌سازد.
    """
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    for row in interactions.itertuples():
        if row.visitorid not in user_mapping or row.itemid not in item_mapping:
            # EN: Skip unknown IDs to stay consistent with mappings
            # FA: شناسه‌های ناشناخته را برای سازگاری حذف می‌کنیم
            continue
        rows.append(user_mapping[row.visitorid])
        cols.append(item_mapping[row.itemid])
        data.append(row.score)
    num_users = max(user_mapping.values()) + 1
    num_items = max(item_mapping.values()) + 1
    coo = sparse.coo_matrix((data, (rows, cols)), shape=(num_users, num_items))
    return coo.tocsr()


def evaluate_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    processed_dir: Path,
    results_dir: Path,
    k: int = 10,
) -> None:
    """
    Train/evaluate baseline, CF, and hybrid models.

    EN: Loops through users in test set, computes ranking metrics, saves summaries.
    FA: برای کاربران موجود در آزمون، متریک‌های رتبه‌بندی را محاسبه و خلاصه‌ها را ذخیره می‌کند.
    """
    # EN: Load mappings and item properties
    # FA: نگاشت‌ها و ویژگی‌های اقلام را بارگذاری می‌کنیم
    user_mapping = load_pickle(processed_dir / "user_mapping.pkl")
    item_mapping = load_pickle(processed_dir / "item_mapping.pkl")
    item_props = pd.read_parquet(processed_dir / "item_properties_clean.parquet")

    # EN: Build train matrix from train interactions to avoid leakage from test
    # FA: برای جلوگیری از نشت داده آزمون، ماتریس را فقط از تعاملات آموزش می‌سازیم
    train_matrix = build_train_matrix(train_df, user_mapping, item_mapping)

    # EN: Load configs if present
    # FA: در صورت وجود، تنظیمات را می‌خوانیم
    project_root = processed_dir.parent.parent
    cf_cfg = load_yaml_safe(project_root / "configs" / "model_cf.yaml")
    hybrid_cfg = load_yaml_safe(project_root / "configs" / "hybrid.yaml")

    # EN: Train models
    # FA: آموزش مدل‌ها
    pop_model = PopularityRecommender().fit(train_df)
    cf_model = CollaborativeRecommender(
        factors=cf_cfg.get("factors", 40),
        regularization=cf_cfg.get("regularization", 0.1),
        iterations=cf_cfg.get("iterations", 10),
        random_state=cf_cfg.get("random_state", 42),
    ).fit(train_matrix, user_mapping=user_mapping, item_mapping=item_mapping)

    content_model = ContentBasedRecommender().fit(item_props)
    hybrid_model = HybridRecommender(
        model_cf=cf_model,
        model_content=content_model,
        alpha=hybrid_cfg.get("alpha", 0.7),
        interactions=train_matrix,
        user_mapping=user_mapping,
        item_mapping=item_mapping,
        filter_seen=hybrid_cfg.get("filter_seen", True),
    )

    models = {
        "baseline": lambda uid: pop_model.recommend(uid, top_k=k),
        "cf": lambda uid: cf_model.recommend(uid, top_k=k),
        "hybrid": lambda uid: hybrid_model.recommend_for_user(uid, top_k=k),
    }

    user_metrics_rows: List[Dict] = []
    evaluated_users = 0

    # EN: Group test interactions by user to evaluate per-user ground truth
    # FA: تعاملات آزمون را بر اساس کاربر گروه‌بندی می‌کنیم تا حقیقت زمین را بسازیم
    for user_id, group in test_df.groupby("visitorid"):
        if user_id not in user_mapping:
            continue
        relevant = set(group["itemid"])
        if not relevant:
            continue
        evaluated_users += 1
        for model_name, recommender in models.items():
            try:
                recs = recommender(user_id)
                rec_ids = [iid for iid, _ in recs]
            except ValueError:
                # EN: If model cannot serve the user, skip
                # FA: در صورت ناتوانی مدل برای کاربر، رد می‌شود
                continue
            metrics = evaluate_user_ranking(rec_ids, relevant, k)
            row = {"user_id": user_id, "model": model_name}
            row.update(metrics)
            user_metrics_rows.append(row)

    user_metrics_df = pd.DataFrame(user_metrics_rows)
    if user_metrics_df.empty:
        raise ValueError("No users were evaluated; check data and mappings.")

    # EN: Aggregate metrics by model (mean over users)
    # FA: متریک‌ها را به‌صورت میانگین روی کاربران به ازای هر مدل تجمیع می‌کنیم
    agg_rows = []
    for model_name, model_df in user_metrics_df.groupby("model"):
        means = model_df.mean(numeric_only=True)
        for metric_name, value in means.items():
            agg_rows.append({"model": model_name, "metric": metric_name, "value": float(value)})
    agg_df = pd.DataFrame(agg_rows)

    ensure_dir(results_dir)
    agg_path = results_dir / "model_comparison.csv"
    user_path = results_dir / "user_metrics.parquet"
    summary_path = results_dir / "summary.json"

    agg_df.to_csv(agg_path, index=False)
    user_metrics_df.to_parquet(user_path, index=False)

    summary = {
        "users_evaluated": evaluated_users,
        "k": k,
        "metrics": agg_rows,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # EN: Print concise summary
    # FA: خلاصه‌ای کوتاه چاپ می‌کنیم
    print(f"Model comparison (K={k}, users evaluated = {evaluated_users:,})")
    for model_name in sorted(models.keys()):
        line_parts = [model_name.ljust(8)]
        for metric in [f"precision@{k}", f"recall@{k}", f"ndcg@{k}", f"map@{k}"]:
            val = agg_df.loc[(agg_df.model == model_name) & (agg_df.metric == metric), "value"]
            if not val.empty:
                line_parts.append(f"{metric}={val.iloc[0]:.4f}")
        print("  ".join(line_parts))


def main() -> None:
    """
    Entry point for running offline evaluation.

    EN: Loads train/test splits, trains models, and writes metrics.
    FA: داده‌های آموزش/آزمون را می‌خواند، مدل‌ها را آموزش داده و متریک‌ها را ذخیره می‌کند.
    """
    project_root = Path(__file__).resolve().parents[2]
    processed_dir = project_root / "data" / "processed"
    results_dir = project_root / "results"

    train_path = processed_dir / "train_interactions.parquet"
    test_path = processed_dir / "test_interactions.parquet"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Train/test splits not found. Run src.evaluation.split first.")

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    evaluate_models(train_df, test_df, processed_dir, results_dir, k=10)


if __name__ == "__main__":
    main()
