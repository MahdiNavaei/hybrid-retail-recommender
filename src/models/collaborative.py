"""
Collaborative filtering model for implicit feedback data.

EN: Implements a lightweight ALS-style matrix factorization without external dependencies.
FA: یک مدل فیلترینگ مشارکتی سبک با فاکتورگیری ماتریس (ALS) بدون وابستگی خارجی پیاده‌سازی می‌کند.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy import sparse

from src.utils.io import load_pickle, save_pickle, ensure_dir


class CollaborativeRecommender:
    """
    Collaborative filtering recommender using implicit feedback matrix factorization.

    EN: Trains a matrix factorization model and provides top-K recommendations.
    FA: یک مدل فاکتورگیری ماتریس آموزش داده و پیشنهادهای برتر را ارائه می‌کند.
    """

    def __init__(
        self,
        factors: int = 40,
        regularization: float = 0.1,
        iterations: int = 10,
        random_state: int = 42,
        user_mapping: Optional[Dict[str, int]] = None,
        item_mapping: Optional[Dict[str, int]] = None,
    ) -> None:
        # EN: Hyperparameters and mappings
        # FA: هایپرپارامترها و نگاشت‌ها
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.random_state = random_state
        self.user_mapping: Dict[str, int] = user_mapping or {}
        self.item_mapping: Dict[str, int] = item_mapping or {}
        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None
        self._interactions: Optional[sparse.csr_matrix] = None

    @staticmethod
    def _als_step(
        ratings: sparse.csr_matrix, fixed_vecs: np.ndarray, reg: float
    ) -> np.ndarray:
        """
        Perform one ALS update step for either user or item factors.

        EN: Solves normal equations for each row independently.
        FA: معادلات نرمال را برای هر ردیف به صورت مستقل حل می‌کند.
        """
        num_rows, factors = ratings.shape[0], fixed_vecs.shape[1]
        updated = np.zeros((num_rows, factors), dtype=np.float64)

        # EN: Precompute Gram matrix for stability
        # FA: ماتریس گرم را برای پایداری پیش‌محاسبه می‌کنیم
        gram = fixed_vecs.T @ fixed_vecs
        reg_identity = reg * np.eye(factors)

        for i in range(num_rows):
            row = ratings.getrow(i)
            if row.nnz == 0:
                # EN: If no interactions, keep zeros
                # FA: در صورت نبود تعامل، صفر باقی می‌ماند
                continue
            indices = row.indices
            values = row.data
            fixed_subset = fixed_vecs[indices]
            A = gram + reg_identity
            b = fixed_subset.T @ values
            updated[i] = np.linalg.solve(A, b)
        return updated

    def fit(
        self,
        interaction_matrix: sparse.csr_matrix | sparse.coo_matrix,
        user_mapping: Optional[Dict[str, int]] = None,
        item_mapping: Optional[Dict[str, int]] = None,
    ) -> "CollaborativeRecommender":
        """
        Train collaborative filtering model.

        EN: Train a collaborative filtering model using implicit feedback.
        FA: آموزش مدل فیلترینگ مشارکتی با استفاده از داده‌های ضمنی.
        """
        if user_mapping is not None:
            self.user_mapping = user_mapping
        if item_mapping is not None:
            self.item_mapping = item_mapping

        # EN: Convert to CSR for efficient row access
        # FA: برای دسترسی سریع به ردیف‌ها، ماتریس را به CSR تبدیل می‌کنیم
        interactions_csr = interaction_matrix.tocsr()
        self._interactions = interactions_csr

        num_users, num_items = interactions_csr.shape
        # EN: Restrict mappings to the available matrix dimensions to stay in-bounds
        # FA: نگاشت‌ها را به ابعاد موجود ماتریس محدود می‌کنیم تا خطای خارج از دامنه نداشته باشیم
        self.user_mapping = {uid: idx for uid, idx in self.user_mapping.items() if idx < num_users}
        self.item_mapping = {iid: idx for iid, idx in self.item_mapping.items() if idx < num_items}

        rng = np.random.default_rng(self.random_state)
        self.user_factors = rng.normal(scale=0.01, size=(num_users, self.factors))
        self.item_factors = rng.normal(scale=0.01, size=(num_items, self.factors))

        for it in range(self.iterations):
            # EN: Update user factors fixing item factors
            # FA: با ثابت نگه داشتن فاکتورهای کالا، فاکتورهای کاربر را به‌روزرسانی می‌کنیم
            self.user_factors = self._als_step(
                interactions_csr, self.item_factors, self.regularization
            )
            # EN: Update item factors fixing user factors
            # FA: با ثابت نگه داشتن فاکتورهای کاربر، فاکتورهای کالا را به‌روزرسانی می‌کنیم
            self.item_factors = self._als_step(
                interactions_csr.T.tocsr(), self.user_factors, self.regularization
            )
            print(
                f"ALS iteration {it + 1}/{self.iterations} completed "
                f"(users: {num_users}, items: {num_items})"
            )

        print(
            f"Trained CF model | users: {num_users:,}, items: {num_items:,}, "
            f"factors: {self.factors}, reg: {self.regularization}, iters: {self.iterations}"
        )
        return self

    def _user_index(self, user_id: str) -> int:
        """
        Map user_id to matrix index or raise error.

        EN: Ensures recommendations only for known users.
        FA: اطمینان می‌دهد تنها برای کاربران شناخته‌شده توصیه ارائه شود.
        """
        if user_id not in self.user_mapping:
            raise ValueError(f"Unknown user_id: {user_id}")
        return self.user_mapping[user_id]

    def _item_ids_from_indices(self, indices: Iterable[int]) -> List[str]:
        """
        Convert item indices back to original IDs.

        EN: Reverse mapping for returning human-readable IDs.
        FA: نگاشت معکوس برای برگرداندن شناسه‌های قابل فهم انسان.
        """
        inverse_map = {idx: iid for iid, idx in self.item_mapping.items()}
        return [inverse_map[i] for i in indices if i in inverse_map]

    def recommend(
        self, user_id: str, top_k: int = 10, exclude_interacted: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Recommend top-K items for a given user.

        EN: Computes dot-product scores and returns item_ids with scores.
        FA: امتیاز حاصل از ضرب داخلی را محاسبه کرده و شناسه کالاها را همراه با امتیاز برمی‌گرداند.
        """
        if self.user_factors is None or self.item_factors is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        user_idx = self._user_index(user_id)

        user_vector = self.user_factors[user_idx]
        scores = self.item_factors @ user_vector

        # EN: Optionally remove items the user has already interacted with
        # FA: در صورت نیاز اقلامی که کاربر قبلاً تعامل داشته حذف می‌شود
        if exclude_interacted and self._interactions is not None:
            interacted_items = self._interactions[user_idx].indices
            scores[interacted_items] = -np.inf

        top_indices = np.argpartition(-scores, kth=min(top_k, len(scores) - 1))[:top_k]
        top_sorted = top_indices[np.argsort(-scores[top_indices])]
        top_item_ids = self._item_ids_from_indices(top_sorted)
        top_scores = scores[top_sorted]
        return list(zip(top_item_ids, top_scores.tolist()))

    def save(self, path: Path | str) -> None:
        """
        Save model factors and metadata.

        EN: Stores factors in NPZ and mappings in pickle alongside.
        FA: فاکتورها را در NPZ و نگاشت‌ها را در پیکل ذخیره می‌کند.
        """
        target = Path(path)
        ensure_dir(target.parent)
        np.savez_compressed(
            target,
            user_factors=self.user_factors,
            item_factors=self.item_factors,
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=self.random_state,
        )
        save_pickle(
            {"user_mapping": self.user_mapping, "item_mapping": self.item_mapping},
            target.with_suffix(".mappings.pkl"),
        )

    @classmethod
    def load(cls, path: Path | str) -> "CollaborativeRecommender":
        """
        Load model from disk.

        EN: Restores factors and mappings.
        FA: فاکتورها و نگاشت‌ها را از دیسک بازیابی می‌کند.
        """
        target = Path(path)
        data = np.load(target, allow_pickle=True)
        mappings = load_pickle(target.with_suffix(".mappings.pkl"))

        model = cls(
            factors=int(data["factors"]),
            regularization=float(data["regularization"]),
            iterations=int(data["iterations"]),
            random_state=int(data["random_state"]),
            user_mapping=mappings.get("user_mapping", {}),
            item_mapping=mappings.get("item_mapping", {}),
        )
        model.user_factors = data["user_factors"]
        model.item_factors = data["item_factors"]
        return model


def load_processed_mappings(project_root: Path | None = None) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Helper to load user/item mappings from processed artifacts.

    EN: Reads pickled mappings produced in Phase 1 preprocessing.
    FA: نگاشت‌های کاربر/کالا تولید شده در پیش‌پردازش فاز اول را بارگذاری می‌کند.
    """
    root = project_root or Path(__file__).resolve().parents[2]
    processed = root / "data" / "processed"
    user_mapping = load_pickle(processed / "user_mapping.pkl")
    item_mapping = load_pickle(processed / "item_mapping.pkl")
    return user_mapping, item_mapping
