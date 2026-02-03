import sqlite3
from pathlib import Path


def get_connection(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS docs (
            id INTEGER PRIMARY KEY,
            rel_path TEXT UNIQUE NOT NULL,
            filename TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            mtime REAL NOT NULL,
            size INTEGER NOT NULL,
            pages_count INTEGER NOT NULL
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS pages (
            id INTEGER PRIMARY KEY,
            doc_id INTEGER NOT NULL,
            page_num INTEGER NOT NULL,
            text TEXT NOT NULL,
            FOREIGN KEY(doc_id) REFERENCES docs(id)
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            doc_id INTEGER NOT NULL,
            page_num INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            start_char INTEGER NOT NULL,
            end_char INTEGER NOT NULL,
            FOREIGN KEY(doc_id) REFERENCES docs(id)
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunk_vectors (
            chunk_id INTEGER PRIMARY KEY,
            faiss_row_id INTEGER NOT NULL
        );
        """
    )
    conn.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS pages_fts
        USING fts5(
            text,
            doc_id UNINDEXED,
            page_num UNINDEXED
        );
        """
    )
    conn.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
        USING fts5(
            text,
            chunk_id UNINDEXED,
            doc_id UNINDEXED,
            page_num UNINDEXED,
            chunk_index UNINDEXED
        );
        """
    )
    conn.commit()


def reset_db(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM chunk_vectors;")
    conn.execute("DELETE FROM chunks_fts;")
    conn.execute("DELETE FROM pages_fts;")
    conn.execute("DELETE FROM chunks;")
    conn.execute("DELETE FROM pages;")
    conn.execute("DELETE FROM docs;")
    conn.commit()


def get_doc_by_rel_path(conn: sqlite3.Connection, rel_path: str) -> sqlite3.Row | None:
    cur = conn.execute("SELECT * FROM docs WHERE rel_path = ?;", (rel_path,))
    return cur.fetchone()


def get_doc_by_id(conn: sqlite3.Connection, doc_id: int) -> sqlite3.Row | None:
    cur = conn.execute("SELECT * FROM docs WHERE id = ?;", (doc_id,))
    return cur.fetchone()


def get_chunk_ids_for_doc(conn: sqlite3.Connection, doc_id: int) -> list[int]:
    rows = conn.execute("SELECT id FROM chunks WHERE doc_id = ?;", (doc_id,)).fetchall()
    return [row["id"] for row in rows]


def delete_doc_data(conn: sqlite3.Connection, doc_id: int) -> None:
    conn.execute("DELETE FROM chunk_vectors WHERE chunk_id IN (SELECT id FROM chunks WHERE doc_id = ?);", (doc_id,))
    conn.execute("DELETE FROM chunks_fts WHERE doc_id = ?;", (doc_id,))
    conn.execute("DELETE FROM pages_fts WHERE doc_id = ?;", (doc_id,))
    conn.execute("DELETE FROM chunks WHERE doc_id = ?;", (doc_id,))
    conn.execute("DELETE FROM pages WHERE doc_id = ?;", (doc_id,))


def upsert_doc(
    conn: sqlite3.Connection,
    *,
    rel_path: str,
    filename: str,
    sha256: str,
    mtime: float,
    size: int,
    pages_count: int,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO docs (rel_path, filename, sha256, mtime, size, pages_count)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(rel_path) DO UPDATE SET
            filename = excluded.filename,
            sha256 = excluded.sha256,
            mtime = excluded.mtime,
            size = excluded.size,
            pages_count = excluded.pages_count
        RETURNING id;
        """,
        (rel_path, filename, sha256, mtime, size, pages_count),
    )
    row = cur.fetchone()
    if row:
        return row["id"]
    row = conn.execute("SELECT id FROM docs WHERE rel_path = ?;", (rel_path,)).fetchone()
    return row["id"]


def insert_pages(conn: sqlite3.Connection, doc_id: int, pages: list[str]) -> None:
    page_rows = [(doc_id, idx, text) for idx, text in enumerate(pages)]
    conn.executemany(
        "INSERT INTO pages (doc_id, page_num, text) VALUES (?, ?, ?);", page_rows
    )
    fts_rows = [(text, doc_id, idx) for idx, text in enumerate(pages)]
    conn.executemany(
        "INSERT INTO pages_fts (text, doc_id, page_num) VALUES (?, ?, ?);", fts_rows
    )


def insert_chunks(
    conn: sqlite3.Connection,
    doc_id: int,
    chunks: list[tuple[int, int, int, str, int, int]],
) -> list[tuple[int, str]]:
    conn.executemany(
        """
        INSERT INTO chunks (doc_id, page_num, chunk_index, text, start_char, end_char)
        VALUES (?, ?, ?, ?, ?, ?);
        """,
        chunks,
    )
    rows = conn.execute(
        "SELECT id, text FROM chunks WHERE doc_id = ? ORDER BY id;",
        (doc_id,),
    ).fetchall()
    return [(row["id"], row["text"]) for row in rows]


def insert_chunks_fts(conn: sqlite3.Connection, doc_id: int) -> None:
    rows = conn.execute(
        "SELECT id, doc_id, page_num, chunk_index, text FROM chunks WHERE doc_id = ?;",
        (doc_id,),
    ).fetchall()
    fts_rows = [
        (row["text"], row["id"], row["doc_id"], row["page_num"], row["chunk_index"])
        for row in rows
    ]
    conn.executemany(
        "INSERT INTO chunks_fts (text, chunk_id, doc_id, page_num, chunk_index) VALUES (?, ?, ?, ?, ?);",
        fts_rows,
    )
