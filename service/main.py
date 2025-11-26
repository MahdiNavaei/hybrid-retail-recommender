"""
FastAPI service exposing the hybrid recommender.

EN: Loads models once at startup and serves recommendation endpoints.
FA: مدل‌ها را در شروع بارگذاری کرده و API توصیه را ارائه می‌دهد.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from scipy import sparse

from src.data.preprocess import EVENT_WEIGHTS
from src.models.baseline import PopularityRecommender
from src.models.collaborative import CollaborativeRecommender
from src.models.content_based import ContentBasedRecommender
from src.models.hybrid import HybridRecommender
from src.utils.io import ensure_dir, load_pickle
from service.config import ServiceSettings, load_service_settings
from service.schemas import (
    HealthResponse,
    ItemMetadata,
    RecommendationRequest,
    RecommendationResponse,
    SimilarItemsRequest,
    SimilarItemsResponse,
)

# EN: Configure logger to JSON lines file
# FA: پیکربندی لاگر برای نوشتن به فایل JSONL
logger = logging.getLogger("api")
logger.setLevel(logging.INFO)


def setup_logger(log_dir: Path) -> None:
    """
    EN: Initialize logger with JSONL file handler.
    FA: لاگر را با فایل JSONL مقداردهی اولیه می‌کنیم.
    """
    ensure_dir(log_dir)
    fh = logging.FileHandler(log_dir / "api_requests.jsonl", encoding="utf-8")
    fmt = logging.Formatter("%(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)


def log_recommendation_request(
    user_id: str,
    model: str,
    top_k: int,
    num_returned: int,
    path: str,
    status_code: int,
) -> None:
    """
    EN: Log structured info about recommendation calls.
    FA: اطلاعات ساخت‌یافته درباره فراخوانی توصیه را لاگ می‌کند.
    """
    log_line = {
        "event": "recommendation",
        "user_id": user_id,
        "model": model,
        "top_k": top_k,
        "returned": num_returned,
        "path": path,
        "timestamp": datetime.utcnow().isoformat(),
        "status_code": status_code,
    }
    logger.info(json.dumps(log_line, ensure_ascii=False))


def log_similarity_request(
    item_id: str,
    top_k: int,
    num_returned: int,
    path: str,
    status_code: int,
) -> None:
    """
    EN: Log structured info about similar-items calls.
    FA: اطلاعات ساخت‌یافته درباره فراخوانی اقلام مشابه را لاگ می‌کند.
    """
    log_line = {
        "event": "similar_items",
        "item_id": item_id,
        "top_k": top_k,
        "returned": num_returned,
        "path": path,
        "timestamp": datetime.utcnow().isoformat(),
        "status_code": status_code,
    }
    logger.info(json.dumps(log_line, ensure_ascii=False))


class ModelRegistry:
    """
    EN: Holds loaded models and metadata in memory.
    FA: مدل‌ها و فراداده بارگذاری‌شده را در حافظه نگه می‌دارد.
    """

    def __init__(self, settings: ServiceSettings) -> None:
        self.settings = settings
        self.user_mapping: Dict[str, int] = {}
        self.item_mapping: Dict[str, int] = {}
        self.item_props: pd.DataFrame = pd.DataFrame()
        self.pop_model: PopularityRecommender | None = None
        self.cf_model: CollaborativeRecommender | None = None
        self.content_model: ContentBasedRecommender | None = None
        self.hybrid_model: HybridRecommender | None = None
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """
        EN: Load mappings, item properties, and train models (fit if artifacts missing).
        FA: نگاشت‌ها، ویژگی‌های اقلام و مدل‌ها را بارگذاری (یا در صورت نبود، آموزش) می‌کند.
        """
        processed = self.settings.processed_dir
        models_dir = self.settings.models_dir
        fast_mode = os.getenv("FAST_TEST", "0") == "1"
        self.user_mapping = load_pickle(processed / "user_mapping.pkl")
        self.item_mapping = load_pickle(processed / "item_mapping.pkl")
        self.item_props = pd.read_parquet(processed / "item_properties_clean.parquet")

        # EN: Fit popularity model from train interactions if available; else from events
        # FA: مدل محبوبیت را از داده آموزش (در صورت وجود) یا events می‌سازیم
        train_path = processed / "train_interactions.parquet"
        if train_path.exists():
            train_df = pd.read_parquet(train_path)
        else:
            events = pd.read_parquet(processed / "events_clean.parquet")
            # EN: Apply implicit weights for popularity when train split is absent
            # FA: در صورت نبود برش آموزش، وزن رویدادها را برای محبوبیت اعمال می‌کنیم
            events = events[events["event"].isin(EVENT_WEIGHTS.keys())].copy()
            events["score"] = events["event"].map(EVENT_WEIGHTS)
            train_df = events[["visitorid", "itemid", "score"]].copy()
        self.pop_model = PopularityRecommender().fit(train_df)

        # EN: Load interaction matrix for CF/Hybrid
        # FA: ماتریس تعاملات برای CF/Hybrid بارگذاری می‌شود
        interactions = sparse.load_npz(processed / "user_item_interactions.npz").tocsr()
        if fast_mode:
            # EN: Subsample for faster unit tests
            # FA: برای تست سریع، زیرنمونه کوچک استفاده می‌کنیم
            interactions = interactions[:500, :800]

        # EN: Try to load CF model if artifact exists, otherwise fit
        # FA: اگر آرتیفکت CF موجود باشد بارگذاری، وگرنه آموزش می‌دهیم
        cf_path = models_dir / "cf_model.npz"
        if cf_path.exists():
            self.cf_model = CollaborativeRecommender.load(cf_path)
        else:
            ensure_dir(models_dir)
            self.cf_model = CollaborativeRecommender().fit(
                interactions, user_mapping=self.user_mapping, item_mapping=self.item_mapping
            )
            self.cf_model.save(cf_path)

        # EN: Try to load content model artifacts if available
        # FA: اگر آرتیفکت مدل محتوایی موجود باشد آن را بارگذاری می‌کنیم
        content_features = models_dir / "content_features.npz"
        content_vec = models_dir / "tfidf_vectorizer.pkl"
        if content_features.exists() and content_vec.exists():
            self.content_model = ContentBasedRecommender.load(content_features, content_vec)
        else:
            ensure_dir(models_dir)
            content_df = self.item_props
            if fast_mode:
                # EN: Limit rows for faster fitting during tests
                # FA: برای تست سریع، تعداد ردیف‌ها را محدود می‌کنیم
                content_df = content_df.head(50000)
            self.content_model = ContentBasedRecommender().fit(content_df)
            self.content_model.save(content_features, content_vec)

        # EN: Build hybrid model using trained CF and content models
        # FA: مدل هایبرید را با CF و محتوایی ساخته‌شده می‌سازیم
        self.hybrid_model = HybridRecommender(
            model_cf=self.cf_model,
            model_content=self.content_model,
            alpha=0.7,
            interactions=interactions,
            user_mapping=self.user_mapping,
            item_mapping=self.item_mapping,
        )

    def recommend(self, user_id: str, model_name: str, top_k: int) -> List[Tuple[str, float]]:
        """
        EN: Route recommendation request to the correct model.
        FA: درخواست توصیه را به مدل مناسب هدایت می‌کند.
        """
        if model_name == "baseline":
            if self.pop_model is None:
                raise RuntimeError("Popularity model not available.")
            return self.pop_model.recommend(user_id, top_k=top_k)
        if model_name == "cf":
            if self.cf_model is None:
                raise RuntimeError("CF model not available.")
            return self.cf_model.recommend(user_id, top_k=top_k)
        if model_name == "hybrid":
            if self.hybrid_model is None:
                raise RuntimeError("Hybrid model not available.")
            return self.hybrid_model.recommend_for_user(user_id, top_k=top_k)
        raise ValueError(f"Unknown model: {model_name}")

    def similar_items(self, item_id: str, top_k: int) -> List[Tuple[str, float]]:
        """
        EN: Use content-based similarity for item-to-item queries.
        FA: برای شباهت آیتم به آیتم از مدل محتوایی استفاده می‌کند.
        """
        if self.content_model is None:
            raise RuntimeError("Content model not available.")
        return self.content_model.similar_items(item_id, top_k=top_k)

    def item_metadata(self, item_id: str) -> ItemMetadata:
        """
        EN: Fetch metadata for a single item from item properties.
        FA: فراداده یک کالا را از ویژگی‌های آیتم استخراج می‌کند.
        """
        if self.item_props.empty:
            return ItemMetadata(item_id=item_id)
        rows = self.item_props[self.item_props["itemid"] == item_id]
        category = None
        attributes = []
        if not rows.empty and "property" in rows.columns and "value" in rows.columns:
            def looks_numeric(text: str) -> bool:
                return text.replace(".", "").replace("-", "").isdigit()

            for _, r in rows.iterrows():
                prop = str(r["property"])
                val = str(r["value"])
                # EN: Prefer explicit category fields
                # FA: فیلدهای دسته‌بندی صریح را در اولویت قرار می‌دهیم
                if category is None and "category" in prop.lower():
                    category = val
                    continue
                # EN: Skip noisy long or purely numeric values
                # FA: مقادیر بسیار طولانی یا تماماً عددی حذف می‌شوند
                if len(val) > 40 or looks_numeric(val):
                    continue
                attributes.append({"name": prop, "value": val[:40]})
                if len(attributes) >= 2:
                    break

            # EN: Fallback category from first row if none found
            # FA: در صورت نبود دسته صریح، اولین مقدار به عنوان دسته قرار می‌گیرد
            if category is None and not rows.empty:
                category = str(rows.iloc[0]["value"])
        return ItemMetadata(item_id=item_id, category=category, attributes=attributes or None)


settings = load_service_settings()
setup_logger(settings.project_root / "logs")
app = FastAPI(title="Hybrid Retail Recommender API")

# EN: Enable CORS for allowed origins
# FA: فعال کردن CORS برای مبداهای مجاز
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event() -> None:
    """
    EN: Load models and metadata once at service startup.
    FA: بارگذاری مدل‌ها و فراداده در هنگام راه‌اندازی سرویس.
    """
    app.state.registry = ModelRegistry(settings)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """
    EN: Health check endpoint.
    FA: اندپوینت بررسی سلامت سرویس.
    """
    return HealthResponse(status="ok", detail="service is running")


@app.post("/recommendations", response_model=RecommendationResponse)
async def recommend(req: RecommendationRequest, request: Request) -> RecommendationResponse:
    """
    EN: Recommend items for a given user using the selected model.
    FA: برای کاربر مشخص‌شده با مدل انتخابی، اقلام پیشنهادی را برمی‌گرداند.
    """
    registry: ModelRegistry = app.state.registry
    model_name = req.model.lower()
    top_k = req.top_k or settings.default_top_k

    if model_name not in {"baseline", "cf", "hybrid"}:
        raise HTTPException(status_code=400, detail="Invalid model name / مدل نامعتبر است.")

    # EN: Validate user existence to return clear 404 if unknown
    # FA: برای کاربران ناشناخته ۴۰۴ برگردانده شود
    registry: ModelRegistry = app.state.registry
    if req.user_id not in registry.user_mapping:
        raise HTTPException(status_code=404, detail="User not found / کاربر یافت نشد.")

    try:
        recs = registry.recommend(req.user_id, model_name, top_k)
    except ValueError:
        raise HTTPException(status_code=404, detail="User not found / کاربر یافت نشد.")
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    items = []
    for item_id, score in recs:
        meta = registry.item_metadata(item_id)
        meta.score = score
        items.append(meta)

    log_recommendation_request(
        user_id=req.user_id,
        model=model_name,
        top_k=top_k,
        num_returned=len(items),
        path=str(request.url.path),
        status_code=200,
    )

    return RecommendationResponse(user_id=req.user_id, model=model_name, top_k=top_k, items=items)


@app.post("/similar-items", response_model=SimilarItemsResponse)
async def similar_items(req: SimilarItemsRequest, request: Request) -> SimilarItemsResponse:
    """
    EN: Return similar items using content-based similarity.
    FA: اقلام مشابه را با شباهت محتوایی برمی‌گرداند.
    """
    registry: ModelRegistry = app.state.registry
    registry: ModelRegistry = app.state.registry
    # EN: Validate item existence
    # FA: بررسی وجود کالا
    if req.item_id not in registry.item_mapping:
        raise HTTPException(status_code=404, detail="Item not found / کالا یافت نشد.")

    try:
        sims = registry.similar_items(req.item_id, req.top_k)
    except ValueError:
        raise HTTPException(status_code=404, detail="Item not found / کالا یافت نشد.")
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    items = []
    for item_id, score in sims:
        meta = registry.item_metadata(item_id)
        meta.score = score
        items.append(meta)

    log_similarity_request(
        item_id=req.item_id,
        top_k=req.top_k,
        num_returned=len(items),
        path=str(request.url.path),
        status_code=200,
    )

    return SimilarItemsResponse(item_id=req.item_id, top_k=req.top_k, items=items)


@app.get("/items/{item_id}", response_model=ItemMetadata)
async def get_item(item_id: str) -> ItemMetadata:
    """
    EN: Fetch metadata for a single item.
    FA: فراداده یک کالا را برمی‌گرداند.
    """
    registry: ModelRegistry = app.state.registry
    return registry.item_metadata(item_id)


if __name__ == "__main__":
    uvicorn.run("service.main:app", host=settings.api_host, port=settings.api_port, reload=True)
