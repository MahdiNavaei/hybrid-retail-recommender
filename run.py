"""
Utility runner to start backend (FastAPI) and frontend (Vite) together.

EN: Spawns uvicorn for the API and `npm run dev` for the UI.
FA: برای اجرای هم‌زمان بک‌اند و فرانت‌اند (uvicorn و npm dev) استفاده می‌شود.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import List
import shutil


ROOT = Path(__file__).resolve().parent
UI_DIR = ROOT / "ui"


def ensure_npm_install() -> None:
    """
    EN: Install frontend dependencies if node_modules is missing.
    FA: در صورت نبود node_modules وابستگی‌های فرانت را نصب می‌کند.
    """
    node_modules = UI_DIR / "node_modules"
    if node_modules.exists():
        return
    # EN: Verify npm is available before trying to install
    # FA: قبل از نصب، در دسترس بودن npm بررسی می‌شود
    if shutil.which("npm") is None:
        raise RuntimeError(
            "npm not found. Please install Node.js and ensure npm is on PATH.\n"
            "npm پیدا نشد؛ لطفاً Node.js را نصب و npm را به PATH اضافه کنید."
        )
    print("Installing frontend dependencies...")
    subprocess.check_call(["npm", "install"], cwd=UI_DIR)


def start_process(cmd: List[str], cwd: Path | None = None) -> subprocess.Popen:
    """
    EN: Start a subprocess and return the handle.
    FA: یک پردازش فرزند راه‌اندازی کرده و هندل آن را برمی‌گرداند.
    """
    return subprocess.Popen(cmd, cwd=cwd)


def main() -> None:
    # EN: Ensure UI deps are installed before starting dev server
    # FA: قبل از اجرای سرور توسعه فرانت، وابستگی‌ها را نصب کنید
    ensure_npm_install()

    # EN: Start backend (uvicorn) and frontend (npm run dev)
    # FA: بک‌اند (uvicorn) و فرانت‌اند (npm run dev) را اجرا می‌کنیم
    backend_cmd = [sys.executable, "-m", "uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8000"]
    frontend_cmd = ["npm", "run", "dev"]

    print("Starting backend (uvicorn)...")
    backend_proc = start_process(backend_cmd, cwd=ROOT)

    env = os.environ.copy()
    env.setdefault("VITE_API_BASE_URL", "http://localhost:8000")
    print("Starting frontend (Vite)...")
    frontend_proc = subprocess.Popen(frontend_cmd, cwd=UI_DIR, env=env)

    try:
        # EN: Wait for child processes; Ctrl+C will trigger cleanup
        # FA: منتظر می‌مانیم؛ با Ctrl+C پاک‌سازی انجام می‌شود
        backend_proc.wait()
    except KeyboardInterrupt:
        print("Stopping processes...")
    finally:
        for proc in [backend_proc, frontend_proc]:
            if proc and proc.poll() is None:
                proc.terminate()


if __name__ == "__main__":
    main()
