"""
Popularity-based baseline recommender.

EN: Recommends globally popular items as a simple baseline.
FA: اقلام محبوب سراسری را به عنوان خط مبنا پیشنهاد می‌دهد.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import pandas as pd


class PopularityRecommender:
    """
    Global popularity recommender.

    EN: Scores items by overall interaction count or summed score.
    FA: اقلام را بر اساس تعداد/امتیاز کلی تعاملات رتبه‌بندی می‌کند.
    """

    def __init__(self) -> None:
        # EN: item_scores holds popularity values
        # FA: محبوبیت اقلام در item_scores ذخیره می‌شود
        self.item_scores: Dict[str, float] = {}
        # EN: user->items mapping for exclusion if needed
        # FA: نگاشت کاربر به اقلام برای حذف اقلام دیده‌شده
        self.user_history: Dict[str, set[str]] = defaultdict(set)

    def fit(self, train_interactions: pd.DataFrame) -> "PopularityRecommender":
        """
        Fit popularity scores from training interactions.

        EN: Aggregates scores per item and builds user histories.
        FA: امتیازها را در سطح آیتم تجمیع کرده و تاریخچه کاربران را می‌سازد.
        """
        if not {"itemid", "visitorid", "score"} <= set(train_interactions.columns):
            raise ValueError("train_interactions must contain itemid, visitorid, score columns.")

        # EN: Sum scores per item as popularity measure
        # FA: مجموع امتیازها به‌عنوان محبوبیت آیتم استفاده می‌شود
        self.item_scores = (
            train_interactions.groupby("itemid")["score"].sum().sort_values(ascending=False).to_dict()
        )

        # EN: Build per-user history for exclusion
        # FA: تاریخچه کاربران برای حذف اقلام دیده‌شده ساخته می‌شود
        for row in train_interactions.itertuples():
            self.user_history[row.visitorid].add(row.itemid)

        print(f"Fitted popularity baseline on {len(self.item_scores):,} items.")
        return self

    def recommend(
        self, user_id: str, top_k: int = 10, exclude_interacted: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Recommend top-K popular items.

        EN: Optionally removes items already seen by the user.
        FA: در صورت نیاز اقلامی که کاربر قبلاً دیده است حذف می‌شوند.
        """
        if not self.item_scores:
            raise RuntimeError("Model not fitted. Call fit() first.")

        seen = self.user_history.get(user_id, set()) if exclude_interacted else set()

        # EN: Iterate through popularity ranking and skip seen items
        # FA: در لیست محبوبیت حرکت کرده و اقلام دیده‌شده را رد می‌کنیم
        recs: List[Tuple[str, float]] = []
        for item_id, score in self.item_scores.items():
            if item_id in seen:
                continue
            recs.append((item_id, score))
            if len(recs) >= top_k:
                break
        return recs


__all__ = ["PopularityRecommender"]
