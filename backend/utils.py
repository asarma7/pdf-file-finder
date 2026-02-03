import hashlib
import os
import shutil
import uuid
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
COLLECTIONS_DIR = DATA_DIR / "collections"
MODELS_DIR = DATA_DIR / "models"
UPLOADS_DIR = DATA_DIR / "uploads"


def ensure_data_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    COLLECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def sha256_file(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def sha256_str(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_ocr_name(path: str, mtime: float) -> str:
    token = sha256_str(f"{path}:{mtime}")
    return f"{token}.pdf"


def has_tool(tool_name: str) -> bool:
    return shutil.which(tool_name) is not None


def new_upload_dir() -> Path:
    ensure_data_dirs()
    token = uuid.uuid4().hex
    upload_dir = UPLOADS_DIR / token
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir
