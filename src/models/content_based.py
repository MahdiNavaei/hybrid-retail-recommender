"""
Content-based recommender using item properties.

EN: Builds TF-IDF features over item property text and serves similarity queries.
FA: با استفاده از ویژگی‌های اقلام، ماتریس TF-IDF ساخته و جستجوی شباهت انجام می‌دهد.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils.io import ensure_dir, save_pickle, load_pickle


class ContentBasedRecommender:
    """
    Content-based recommender using TF-IDF item representations.

    EN: Represents each item as a TF-IDF vector built from its properties.
    FA: هر کالا را با بردار TF-IDF ساخته‌شده از ویژگی‌هایش نمایش می‌دهد.
    """

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        stop_words: Optional[str] = "english",
    ) -> None:
        # EN: Hyperparameters for TF-IDF
        # FA: هایپرپارامترهای TF-IDF
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.stop_words = stop_words

        self.vectorizer: Optional[TfidfVectorizer] = None
        self.item_features: Optional[sparse.csr_matrix] = None
        self.item_ids: List[str] = []
        self.item_index: Dict[str, int] = {}

    @staticmethod
    def _build_item_corpus(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Build corpus strings per item from properties.

        EN: Combines property name and value into a text blob per item.
        FA: نام ویژگی و مقدار آن را برای هر کالا در یک متن ترکیب می‌کند.
        """
        required_cols = {"itemid", "property", "value"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns for content model: {missing}")
        df = df.copy()
        df["itemid"] = df["itemid"].astype("string")
        # EN: Concatenate property and value as tokens
        # FA: ویژگی و مقدار را به صورت کلمه به کلمه به هم می‌چسبانیم
        df["property_text"] = (
            df["property"].astype("string").fillna("")
            + " "
            + df["value"].astype("string").fillna("")
        )
        grouped = df.groupby("itemid")["property_text"].apply(
            lambda vals: " ".join(vals)
        )
        item_ids = grouped.index.tolist()
        corpus = grouped.tolist()
        return item_ids, corpus

    def fit(self, item_properties_df: pd.DataFrame) -> "ContentBasedRecommender":
        """
        Fit TF-IDF features for all items.

        EN: Builds the vectorizer and sparse item-feature matrix.
        FA: بردارساز و ماتریس ویژگی‌های کالا را ایجاد می‌کند.
        """
        item_ids, corpus = self._build_item_corpus(item_properties_df)
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            stop_words=self.stop_words,
        )
        # EN: Fit and transform corpus into sparse matrix
        # FA: بدست آوردن ماتریس تنک از روی کورپوس
        self.item_features = self.vectorizer.fit_transform(corpus).tocsr()
        self.item_ids = item_ids
        self.item_index = {iid: idx for idx, iid in enumerate(item_ids)}

        print(
            f"Fitted content model for {len(self.item_ids):,} items "
            f"with {self.item_features.shape[1]:,} features. "
            f"(max_features={self.max_features}, ngram_range={self.ngram_range}, min_df={self.min_df})"
        )
        return self

    def _vector_for_item(self, item_id: str) -> sparse.csr_matrix:
        """
        Get TF-IDF vector for a specific item.

        EN: Returns a 1xF sparse row for similarity calculations.
        FA: یک سطر ۱xF تنک برای محاسبات شباهت برمی‌گرداند.
        """
        if self.item_features is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if item_id not in self.item_index:
            raise ValueError(f"Unknown item_id: {item_id}")
        idx = self.item_index[item_id]
        return self.item_features[idx]

    def similar_items(self, item_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Return top-K similar items based on cosine similarity.

        EN: Uses TF-IDF cosine similarity to rank items.
        FA: با استفاده از شباهت کسینوسی بر اساس TF-IDF اقلام مشابه را رتبه‌بندی می‌کند.
        """
        if self.item_features is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        query_vec = self._vector_for_item(item_id)
        # EN: Cosine similarity via dot product (vectors are L2-normalized by TF-IDF)
        # FA: شباهت کسینوسی با ضرب نقطه‌ای (بردارها توسط TF-IDF نرمال شده‌اند)
        sims = self.item_features @ query_vec.T
        sims = np.asarray(sims.todense()).ravel()

        query_idx = self.item_index[item_id]
        sims[query_idx] = -np.inf  # EN/FA: حذف همان آیتم از نتایج

        top_indices = np.argpartition(-sims, kth=min(top_k, len(sims) - 1))[:top_k]
        top_sorted = top_indices[np.argsort(-sims[top_indices])]
        top_scores = sims[top_sorted]
        top_item_ids = [self.item_ids[i] for i in top_sorted]
        return list(zip(top_item_ids, top_scores.tolist()))

    def top_items_by_norm(self, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Get globally strong items using vector L2 norms.

        EN: Useful as a fallback when no seed item is available.
        FA: برای حالت نبود آیتم مبنا، از اقلام با نُرم بالاتر استفاده می‌شود.
        """
        if self.item_features is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        norms = sparse.linalg.norm(self.item_features, axis=1)
        norms = np.asarray(norms).ravel()
        top_indices = np.argpartition(-norms, kth=min(top_k, len(norms) - 1))[:top_k]
        top_sorted = top_indices[np.argsort(-norms[top_indices])]
        return [(self.item_ids[i], float(norms[i])) for i in top_sorted]

    def save(self, features_path: Path | str, vectorizer_path: Path | str) -> None:
        """
        Save feature matrix and vectorizer.

        EN: Persists sparse matrix as NPZ and vectorizer via pickle.
        FA: ماتریس تنک را به NPZ و بردارساز را به صورت پیکل ذخیره می‌کند.
        """
        if self.item_features is None or self.vectorizer is None:
            raise RuntimeError("Model not fitted. Nothing to save.")

        features_path = Path(features_path)
        vectorizer_path = Path(vectorizer_path)
        ensure_dir(features_path.parent)
        ensure_dir(vectorizer_path.parent)

        sparse.save_npz(features_path, self.item_features)
        save_pickle(
            {
                "vectorizer": self.vectorizer,
                "item_ids": self.item_ids,
                "item_index": self.item_index,
                "params": {
                    "max_features": self.max_features,
                    "ngram_range": self.ngram_range,
                    "min_df": self.min_df,
                    "stop_words": self.stop_words,
                },
            },
            vectorizer_path,
        )

    @classmethod
    def load(cls, features_path: Path | str, vectorizer_path: Path | str) -> "ContentBasedRecommender":
        """
        Load feature matrix and vectorizer from disk.

        EN: Restores TF-IDF artifacts for inference.
        FA: مصنوعات TF-IDF را برای استنتاج بازیابی می‌کند.
        """
        meta = load_pickle(vectorizer_path)
        model = cls(**meta.get("params", {}))
        model.vectorizer = meta["vectorizer"]
        model.item_ids = meta["item_ids"]
        model.item_index = meta["item_index"]
        model.item_features = sparse.load_npz(features_path).tocsr()
        return model
