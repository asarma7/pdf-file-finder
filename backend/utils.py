import hashlib
import os
import re
import shutil
import uuid
from pathlib import Path

# Substitutions for redaction/OCR in email headers (applied before email regex).
# Order matters: replace longer patterns first to avoid partial hits.
EMAIL_HEADER_SUBSTITUTIONS = [
    ("©", "@"),
    ("eevacation", "jeevacation"),  # common OCR/typo for jeevacation
    ("e stein", "epstein"),         # "E stein" / "e stein" → Epstein
    ("ftmaitcom", "gmail.com"),
    ("enmail", "gmail"),
    ("aigmail", "gmail"),
]


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


def normalize_header_for_email_extraction(value: str) -> str:
    """Apply substitutions so the email regex can find addresses (e.g. © → @).
    Name/email variants like 'e stein'→'epstein', 'eevacation'→'jeevacation' use
    case-insensitive replace so we store and match the canonical form."""
    if not value:
        return ""
    s = value
    for old, new in EMAIL_HEADER_SUBSTITUTIONS:
        s = re.sub(re.escape(old), new, s, flags=re.IGNORECASE)
    return s


def normalize_email_for_lookup(addr: str) -> str:
    """Canonical form of an email for alias lookup (same substitutions, lowercased)."""
    if not addr or not addr.strip():
        return ""
    s = addr.strip().lower()
    for old, new in EMAIL_HEADER_SUBSTITUTIONS:
        s = s.replace(old, new)
    return s


# Used to extract one email from a string (e.g. when from_addr is empty but from_name has garbage).
_EMAIL_CANDIDATE = re.compile(
    r"\b([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})\b", re.IGNORECASE
)


def extract_canonical_email_from_text(text: str) -> str:
    """Return first plausible-looking email in text after normalization, or ''."""
    if not text:
        return ""
    s = normalize_header_for_email_extraction(text)
    for m in _EMAIL_CANDIDATE.finditer(s):
        candidate = m.group(1).lower()
        if len(candidate) >= 6 and "." in candidate.split("@")[-1]:
            return normalize_email_for_lookup(candidate)
    return ""
