"""Utility helpers for file system and serialization operations."""

from pathlib import Path
from typing import Any, Union
import pickle


def ensure_dir(path: Union[str, Path]) -> None:
    """
    Create a directory (and parents) if it does not already exist.

    EN: Ensures the provided directory exists.
    FA: اطمینان حاصل می‌کند که پوشه مورد نظر وجود داشته باشد.
    """
    dir_path = Path(path)
    # EN: Create the directory tree if missing
    # FA: اگر پوشه وجود نداشت، مسیر کامل را می‌سازد
    dir_path.mkdir(parents=True, exist_ok=True)


def save_pickle(obj: Any, path: Union[str, Path]) -> None:
    """
    Save a Python object to disk using pickle.

    EN: Serializes an object to a binary file.
    FA: یک شیٔ پایتونی را به صورت باینری روی دیسک ذخیره می‌کند.
    """
    file_path = Path(path)
    # EN: Ensure parent directory exists before writing the pickle
    # FA: قبل از نوشتن فایل پیکل، پوشه والد را ایجاد/بررسی می‌کنیم
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Union[str, Path]) -> Any:
    """
    Load a Python object from a pickle file.

    EN: Deserializes an object from disk.
    FA: شیٔ ذخیره شده را از فایل پیکل بازیابی می‌کند.
    """
    file_path = Path(path)
    # EN: Open the binary file and load the object
    # FA: فایل باینری را باز کرده و شیٔ را بارگذاری می‌کنیم
    with file_path.open("rb") as f:
        return pickle.load(f)


__all__ = ["ensure_dir", "save_pickle", "load_pickle"]
