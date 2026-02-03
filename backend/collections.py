import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

from .utils import COLLECTIONS_DIR, DATA_DIR, ensure_data_dirs


COLLECTIONS_DB = DATA_DIR / "collections.db"


def get_registry_connection() -> sqlite3.Connection:
    ensure_data_dirs()
    conn = sqlite3.connect(COLLECTIONS_DB)
    conn.row_factory = sqlite3.Row
    return conn


def init_registry_db() -> None:
    conn = get_registry_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS collections (
            id TEXT PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            root_path TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )
    conn.commit()
    conn.close()


def create_collection(name: str, root_path: str) -> dict:
    init_registry_db()
    collection_id = uuid.uuid4().hex
    created_at = datetime.utcnow().isoformat()
    conn = get_registry_connection()
    conn.execute(
        "INSERT INTO collections (id, name, root_path, created_at) VALUES (?, ?, ?, ?);",
        (collection_id, name, root_path, created_at),
    )
    conn.commit()
    conn.close()
    return {"id": collection_id, "name": name, "root_path": root_path}


def update_collection_root(collection_id: str, root_path: str) -> None:
    conn = get_registry_connection()
    conn.execute(
        "UPDATE collections SET root_path = ? WHERE id = ?;",
        (root_path, collection_id),
    )
    conn.commit()
    conn.close()


def list_collections() -> list[dict]:
    init_registry_db()
    conn = get_registry_connection()
    rows = conn.execute("SELECT * FROM collections ORDER BY created_at;").fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_collection_by_id(collection_id: str) -> dict | None:
    init_registry_db()
    conn = get_registry_connection()
    row = conn.execute(
        "SELECT * FROM collections WHERE id = ?;", (collection_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_collection_by_name(name: str) -> dict | None:
    init_registry_db()
    conn = get_registry_connection()
    row = conn.execute("SELECT * FROM collections WHERE name = ?;", (name,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_collection_dir(collection_id: str) -> Path:
    ensure_data_dirs()
    path = COLLECTIONS_DIR / collection_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_collection_db_path(collection_id: str) -> Path:
    return get_collection_dir(collection_id) / "app.db"


def get_collection_faiss_path(collection_id: str) -> Path:
    return get_collection_dir(collection_id) / "faiss.index"


def get_collection_ocr_cache_dir(collection_id: str) -> Path:
    path = get_collection_dir(collection_id) / "ocr_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_collection_files_dir(collection_id: str) -> Path:
    path = get_collection_dir(collection_id) / "files"
    path.mkdir(parents=True, exist_ok=True)
    return path
