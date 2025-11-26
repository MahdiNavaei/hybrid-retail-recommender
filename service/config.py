"""
Service configuration for the FastAPI backend.

EN: Centralizes settings with env/YAML overrides.
FA: تنظیمات سرویس را با امکان override توسط محیط و YAML متمرکز می‌کند.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic_settings import BaseSettings


class ServiceSettings(BaseSettings):
    """
    EN: Backend service settings with sensible defaults.
    FA: تنظیمات سرویس بک‌اند با مقادیر پیش‌فرض منطقی.
    """

    project_root: Path = Path(__file__).resolve().parents[1]
    processed_dir: Path = project_root / "data" / "processed"
    models_dir: Path = processed_dir / "models"
    default_top_k: int = 10
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    allowed_origins: list[str] = ["*"]  # EN/FA: مجوز CORS؛ در آینده محدود می‌شود
    default_model: str = "hybrid"  # EN/FA: مقدار پیش‌فرض مدل توصیه

    class Config:
        # EN: Allow overriding via env vars with RETAIL_REC_ prefix
        # FA: امکان override با متغیرهای محیطی دارای پیشوند RETAIL_REC_
        env_prefix = "RETAIL_REC_"


def load_service_settings(config_path: Optional[Path] = None) -> ServiceSettings:
    """
    Load settings from YAML (if present) plus environment overrides.

    EN: Reads configs/service.yaml when available, then applies env overrides.
    FA: در صورت وجود فایل YAML، تنظیمات را می‌خواند و سپس متغیرهای محیطی را اعمال می‌کند.
    """
    project_root = Path(__file__).resolve().parents[1]
    default_path = project_root / "configs" / "service.yaml"
    path = config_path or default_path

    # EN: Start with YAML values if file exists
    # FA: اگر فایل YAML وجود داشته باشد از همان مقادیر شروع می‌کنیم
    yaml_data = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}

    # EN: Pydantic BaseSettings will merge defaults, YAML, and env overrides
    # FA: Pydantic ترکیبی از پیش‌فرض، YAML و متغیر محیطی را اعمال می‌کند
    return ServiceSettings(**yaml_data)


__all__ = ["ServiceSettings", "load_service_settings"]
