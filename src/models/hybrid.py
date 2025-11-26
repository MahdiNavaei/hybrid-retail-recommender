"""
Hybrid recommender combining collaborative filtering and content-based signals.

EN: Blends CF scores with content similarity for robust recommendations.
FA: برای ارائه توصیه‌های پایدار، امتیازهای CF را با شباهت محتوایی ترکیب می‌کند.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse

from src.models.collaborative import CollaborativeRecommender
from src.models.content_based import ContentBasedRecommender
from src.utils.io import ensure_dir, save_pickle, load_pickle


class HybridRecommender:
    """
    Hybrid recommender that linearly blends CF and content scores.

    EN: score = alpha * score_cf + (1 - alpha) * score_content
    FA: امتیاز نهایی = آلفا * امتیاز CF + (۱-آلفا) * امتیاز محتوایی
    """

    def __init__(
        self,
        model_cf: CollaborativeRecommender,
        model_content: ContentBasedRecommender,
        alpha: float = 0.7,
        interactions: Optional[sparse.csr_matrix] = None,
        user_mapping: Optional[Dict[str, int]] = None,
        item_mapping: Optional[Dict[str, int]] = None,
        filter_seen: bool = True,
    ) -> None:
        # EN: Store sub-models and blend weight
        # FA: ذخیره زیرمدل‌ها و وزن ترکیب
        self.model_cf = model_cf
        self.model_content = model_content
        self.alpha = alpha
        self.interactions = interactions
        self.user_mapping = user_mapping or getattr(model_cf, "user_mapping", {})
        self.item_mapping = item_mapping or getattr(model_cf, "item_mapping", {})
        self.filter_seen = filter_seen

    def _user_history(self, user_id: str) -> List[str]:
        """
        Retrieve items a user has interacted with.

        EN: Uses the interactions matrix if available.
        FA: در صورت وجود ماتریس تعاملات، اقلام تعامل‌شده کاربر را برمی‌گرداند.
        """
        if self.interactions is None:
            return []
        if user_id not in self.user_mapping:
            return []
        user_idx = self.user_mapping[user_id]
        item_indices = self.interactions[user_idx].indices
        inverse_item_map = {idx: iid for iid, idx in self.item_mapping.items()}
        return [inverse_item_map[i] for i in item_indices if i in inverse_item_map]

    def _content_scores_for_candidates(
        self, seed_items: Sequence[str], candidates: Sequence[str]
    ) -> Dict[str, float]:
        """
        Compute content-based scores for candidate items given seed items.

        EN: Uses max similarity to any seed as content score.
        FA: بیشینه شباهت هر نامزد به اقلام مبنا را به‌عنوان امتیاز محتوایی می‌گیرد.
        """
        if not seed_items:
            return {cid: 0.0 for cid in candidates}

        # EN: Build matrices for seeds and candidates
        # FA: ماتریس اقلام مبنا و نامزد را می‌سازیم
        item_features = self.model_content.item_features
        if item_features is None:
            return {cid: 0.0 for cid in candidates}

        idx_map = self.model_content.item_index
        seed_idx = [idx_map[s] for s in seed_items if s in idx_map]
        candidate_pairs = [(c, idx_map[c]) for c in candidates if c in idx_map]
        if not seed_idx or not candidate_pairs:
            return {cid: 0.0 for cid in candidates}

        seed_matrix = item_features[seed_idx]
        cand_matrix = item_features[[idx for _, idx in candidate_pairs]]

        # EN: Cosine similarity via matrix multiplication
        # FA: شباهت کسینوسی با ضرب ماتریس‌ها
        sims = cand_matrix @ seed_matrix.T  # shape: (candidates, seeds)
        # EN: Convert to dense array to safely take row-wise maxima
        # FA: برای گرفتن ماکسیمم ردیفی، به آرایه چگال تبدیل می‌کنیم
        sims_array = sims.toarray()
        max_sims = sims_array.max(axis=1)

        content_scores = {candidate_pairs[i][0]: float(max_sims[i]) for i in range(len(candidate_pairs))}
        # EN: Fill missing candidates with zero
        # FA: برای نامزدهای بدون شباهت، صفر می‌گذاریم
        for cid in candidates:
            content_scores.setdefault(cid, 0.0)
        return content_scores

    def _fallback_content(self, top_k: int, seed_items: Sequence[str]) -> List[Tuple[str, float]]:
        """
        Fallback recommendations purely from content model.

        EN: Uses similar items to seeds if available, otherwise top items by norm.
        FA: اگر اقلام مبنا باشد، مشابهات را برمی‌گرداند وگرنه اقلام با نُرم بالاتر را می‌دهد.
        """
        if seed_items:
            # EN: Aggregate top similar items from first seed
            # FA: مشابه‌ترین اقلام به اولین مبنا را می‌گیریم
            return self.model_content.similar_items(seed_items[0], top_k=top_k)
        return self.model_content.top_items_by_norm(top_k=top_k)

    def recommend_for_user(self, user_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Recommend items for a user using hybrid scoring.

        EN: Blends CF scores with content similarity, optionally filtering seen items.
        FA: امتیاز ترکیبی CF و شباهت محتوایی را اعمال کرده و در صورت نیاز اقلام دیده‌شده را حذف می‌کند.
        """
        # EN: Determine user history for filtering/content weighting
        # FA: سابقه کاربر را برای فیلتر و وزن‌دهی محتوایی استخراج می‌کنیم
        seen_items = set(self._user_history(user_id))

        try:
            cf_recs = self.model_cf.recommend(
                user_id, top_k=top_k * 3, exclude_interacted=self.filter_seen
            )
        except ValueError:
            # EN: CF unavailable for this user, fallback to content
            # FA: در صورت ناموجود بودن CF برای کاربر، به محتوایی سقوط می‌کنیم
            return self._fallback_content(top_k, list(seen_items))

        candidate_ids = [iid for iid, _ in cf_recs]
        content_scores = self._content_scores_for_candidates(seen_items, candidate_ids)

        hybrid_scores: List[Tuple[str, float]] = []
        for iid, cf_score in cf_recs:
            cont_score = content_scores.get(iid, 0.0)
            final_score = self.alpha * cf_score + (1 - self.alpha) * cont_score
            hybrid_scores.append((iid, final_score))

        # EN: Optionally filter already seen items
        # FA: در صورت نیاز اقلام دیده‌شده حذف می‌شوند
        if self.filter_seen and seen_items:
            hybrid_scores = [(iid, sc) for iid, sc in hybrid_scores if iid not in seen_items]

        # EN: Sort and truncate
        # FA: مرتب‌سازی و برش به K مورد
        hybrid_scores.sort(key=lambda x: x[1], reverse=True)
        if not hybrid_scores:
            return self._fallback_content(top_k, list(seen_items))
        return hybrid_scores[:top_k]

    def similar_items(self, item_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Delegate similar items to content-based model.

        EN: Content similarity is more appropriate for item-to-item queries.
        FA: برای پرس‌وجوی آیتم-به-آیتم از مدل محتوایی استفاده می‌شود.
        """
        return self.model_content.similar_items(item_id, top_k=top_k)

    def save(self, path: Path | str) -> None:
        """
        Save hybrid configuration.

        EN: Stores alpha, flags, and metadata paths if needed.
        FA: آلفا و تنظیمات مربوط به فیلتر را ذخیره می‌کند.
        """
        target = Path(path)
        ensure_dir(target.parent)
        save_pickle(
            {
                "alpha": self.alpha,
                "filter_seen": self.filter_seen,
            },
            target,
        )

    @classmethod
    def load(
        cls,
        path: Path | str,
        model_cf: CollaborativeRecommender,
        model_content: ContentBasedRecommender,
        interactions: Optional[sparse.csr_matrix] = None,
        user_mapping: Optional[Dict[str, int]] = None,
        item_mapping: Optional[Dict[str, int]] = None,
    ) -> "HybridRecommender":
        """
        Load hybrid model configuration.

        EN: Reconstructs HybridRecommender using existing sub-models.
        FA: با استفاده از زیرمدل‌های موجود، هایبرید را بازسازی می‌کند.
        """
        cfg = load_pickle(path)
        return cls(
            model_cf=model_cf,
            model_content=model_content,
            alpha=cfg.get("alpha", 0.7),
            interactions=interactions,
            user_mapping=user_mapping,
            item_mapping=item_mapping,
            filter_seen=cfg.get("filter_seen", True),
        )
