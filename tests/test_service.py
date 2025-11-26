"""
API tests for the FastAPI service.

EN: Verifies health, recommendation, similarity, and error handling.
FA: صحت سلامت، توصیه، اقلام مشابه و مدیریت خطا را بررسی می‌کند.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from service.main import app


def _sample_user_and_item() -> tuple[str, str]:
    """
    EN: Read processed data to pick a known user and item for tests.
    FA: برای تست، یک کاربر و کالا از داده‌های پردازش‌شده برمی‌دارد.
    """
    project_root = Path(__file__).resolve().parents[1]
    processed = project_root / "data" / "processed"
    user_map_path = processed / "user_mapping.pkl"
    item_map_path = processed / "item_mapping.pkl"
    if not (user_map_path.exists() and item_map_path.exists()):
        raise RuntimeError("Mappings not found; run preprocessing first.")
    user_map = pd.read_pickle(user_map_path)
    item_map = pd.read_pickle(item_map_path)
    # EN: pick first user/item in mapping to ensure they exist in model factors
    # FA: اولین کاربر/کالا در نگاشت انتخاب می‌شود تا در مدل وجود داشته باشد
    sample_user = next(iter(user_map.keys()))
    sample_item = next(iter(item_map.keys()))
    return str(sample_user), str(sample_item)


SAMPLE_USER, SAMPLE_ITEM = _sample_user_and_item()


@pytest.fixture(scope="module")
def api_client():
    """
    EN: Use TestClient context to ensure startup events run.
    FA: از TestClient با کانتکست استفاده می‌کنیم تا رویداد startup اجرا شود.
    """
    with TestClient(app) as c:
        yield c


def test_health_endpoint_works(api_client):
    """
    EN: Ensure /health responds with status ok.
    FA: اطمینان از اینکه /health وضعیت ok را برمی‌گرداند.
    """
    resp = api_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


def test_recommendations_returns_items_for_known_user(api_client):
    """
    EN: Recommendations should return items for a known user.
    FA: برای کاربر شناخته‌شده باید اقلامی برگردد.
    """
    resp = api_client.post("/recommendations", json={"user_id": SAMPLE_USER, "top_k": 5, "model": "hybrid"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["user_id"] == SAMPLE_USER
    assert len(data["items"]) > 0


def test_similar_items_returns_items_for_known_item(api_client):
    """
    EN: Similar-items endpoint should return items for a known item.
    FA: اندپوینت اقلام مشابه باید برای کالای شناخته‌شده خروجی دهد.
    """
    resp = api_client.post("/similar-items", json={"item_id": SAMPLE_ITEM, "top_k": 5})
    assert resp.status_code == 200
    data = resp.json()
    assert data["item_id"] == SAMPLE_ITEM
    assert len(data["items"]) > 0


def test_invalid_model_returns_400(api_client):
    """
    EN: Invalid model names should yield HTTP 400.
    FA: نام مدل نامعتبر باید خطای ۴۰۰ ایجاد کند.
    """
    resp = api_client.post("/recommendations", json={"user_id": SAMPLE_USER, "top_k": 5, "model": "unknown"})
    assert resp.status_code == 400
