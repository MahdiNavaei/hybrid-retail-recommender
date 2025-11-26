"""
Pydantic schemas for API requests and responses.

EN: Defines request/response models for the recommender API.
FA: مدل‌های درخواست/پاسخ برای API سیستم پیشنهاددهنده را تعریف می‌کند.
"""

from __future__ import annotations

from typing import List, Optional, Dict

from pydantic import BaseModel


class ItemMetadata(BaseModel):
    """
    EN: Metadata for an item (used in responses).
    FA: فراداده مربوط به یک کالا (در پاسخ‌ها استفاده می‌شود).
    """

    item_id: str
    title: Optional[str] = None
    category: Optional[str] = None
    score: Optional[float] = None  # EN/FA: امتیاز توصیه در پاسخ
    attributes: Optional[List[Dict[str, str]]] = None  # EN/FA: ویژگی‌های اضافی کالا


class RecommendationRequest(BaseModel):
    """
    EN: Request body for user recommendations.
    FA: بدنه درخواست برای دریافت توصیه‌های کاربر.
    """

    user_id: str
    top_k: int = 10
    model: str = "hybrid"  # EN/FA: مدل انتخابی (baseline, cf, hybrid)


class RecommendationResponse(BaseModel):
    """
    EN: Response payload for recommendations.
    FA: پاسخ شامل لیست توصیه‌ها.
    """

    user_id: str
    model: str
    top_k: int
    items: List[ItemMetadata]


class SimilarItemsRequest(BaseModel):
    """
    EN: Request body for similar items lookup.
    FA: بدنه درخواست برای جستجوی اقلام مشابه.
    """

    item_id: str
    top_k: int = 10


class SimilarItemsResponse(BaseModel):
    """
    EN: Response payload for similar items.
    FA: پاسخ شامل اقلام مشابه.
    """

    item_id: str
    top_k: int
    items: List[ItemMetadata]


class HealthResponse(BaseModel):
    """
    EN: Health check response.
    FA: پاسخ بررسی سلامت سرویس.
    """

    status: str
    detail: str


__all__ = [
    "ItemMetadata",
    "RecommendationRequest",
    "RecommendationResponse",
    "SimilarItemsRequest",
    "SimilarItemsResponse",
    "HealthResponse",
]
