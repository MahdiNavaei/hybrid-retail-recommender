"""
Ranking metrics for recommender evaluation.

EN: Implements Precision@K, Recall@K, NDCG@K, and MAP@K.
FA: متریک‌های دقت، یادآوری، NDCG و MAP در برش K را پیاده‌سازی می‌کند.
"""

from __future__ import annotations

import math
from typing import Dict, List, Set


def precision_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
    """
    Compute Precision@K.

    EN: Fraction of recommended items in top-K that are relevant.
    FA: نسبت اقلام مرتبط در بین K پیشنهاد اول.
    """
    if k <= 0:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k


def recall_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
    """
    Compute Recall@K.

    EN: Fraction of all relevant items that appear in top-K.
    FA: نسبت کل اقلام مرتبط که در K پیشنهاد اول حضور دارند.
    """
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
    """
    Compute NDCG@K (Normalized Discounted Cumulative Gain).

    EN: Rewards relevant items at higher ranks with log discount.
    FA: اقلام مرتبط در رتبه‌های بالاتر را با ضریب لگاریتمی تشویق می‌کند.
    """
    top_k = recommended[:k]
    dcg = 0.0
    for idx, item in enumerate(top_k, start=1):
        if item in relevant:
            dcg += 1.0 / math.log2(idx + 1)

    # EN: Ideal DCG assumes all relevant items are at the top
    # FA: DCG ایده‌آل فرض می‌کند تمام اقلام مرتبط در ابتدای لیست باشند
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def average_precision_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
    """
    Compute MAP-style Average Precision at K.

    EN: Mean of precision values at each rank where the item is relevant.
    FA: میانگین دقت در رتبه‌هایی که آیتم مربوطه مرتبط است.
    """
    top_k = recommended[:k]
    ap_sum = 0.0
    hit_count = 0
    for idx, item in enumerate(top_k, start=1):
        if item in relevant:
            hit_count += 1
            ap_sum += hit_count / idx
    if hit_count == 0:
        return 0.0
    return ap_sum / hit_count


def evaluate_user_ranking(recommended: List[str], relevant: Set[str], k: int) -> Dict[str, float]:
    """
    Compute all ranking metrics for a single user.

    EN: Returns a dictionary with precision, recall, ndcg, and map at K.
    FA: دیکشنری شامل متریک‌های دقت، یادآوری، NDCG و MAP در برش K را برمی‌گرداند.
    """
    return {
        f"precision@{k}": precision_at_k(recommended, relevant, k),
        f"recall@{k}": recall_at_k(recommended, relevant, k),
        f"ndcg@{k}": ndcg_at_k(recommended, relevant, k),
        f"map@{k}": average_precision_at_k(recommended, relevant, k),
    }
