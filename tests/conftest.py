"""
EN: Ensure project root is on sys.path for imports in tests.
FA: ریشه پروژه را برای ایمپورت‌ها به sys.path اضافه می‌کند.
"""

import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# EN: Speed up API model loading during tests by enabling fast mode
# FA: برای سریع‌تر شدن بارگذاری مدل در تست‌ها، حالت سریع فعال می‌شود
os.environ.setdefault("FAST_TEST", "1")
