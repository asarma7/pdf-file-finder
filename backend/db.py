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
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS email_headers (
            id INTEGER PRIMARY KEY,
            doc_id INTEGER NOT NULL,
            page_num INTEGER NOT NULL,
            from_addr TEXT,
            to_addr TEXT,
            cc_addr TEXT,
            from_name TEXT,
            to_name TEXT,
            cc_name TEXT,
            subject TEXT,
            date TEXT,
            snippet TEXT,
            FOREIGN KEY(doc_id) REFERENCES docs(id)
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS email_aliases (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            UNIQUE(name, email)
        );
        """
    )
    conn.commit()


def reset_db(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM chunk_vectors;")
    conn.execute("DELETE FROM chunks_fts;")
    conn.execute("DELETE FROM pages_fts;")
    conn.execute("DELETE FROM email_headers;")
    conn.execute("DELETE FROM email_aliases;")
    conn.execute("DELETE FROM chunks;")
    conn.execute("DELETE FROM pages;")
    conn.execute("DELETE FROM docs;")
    conn.commit()


def clear_email_data(conn: sqlite3.Connection) -> None:
    """Remove all email_headers and email_aliases rows. Leaves docs, pages, chunks, etc. unchanged."""
    conn.execute("DELETE FROM email_headers;")
    conn.execute("DELETE FROM email_aliases;")
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
    conn.execute("DELETE FROM email_headers WHERE doc_id = ?;", (doc_id,))
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


def insert_email_headers(conn: sqlite3.Connection, rows: list[tuple]) -> None:
    if not rows:
        return
    conn.executemany(
        """
        INSERT INTO email_headers (
            doc_id,
            page_num,
            from_addr,
            to_addr,
            cc_addr,
            from_name,
            to_name,
            cc_name,
            subject,
            date,
            snippet
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        rows,
    )


def add_email_aliases(conn: sqlite3.Connection, name: str, emails: list[str]) -> int:
    if not name or not emails:
        return 0
    rows = [(name.lower().strip(), email.lower().strip()) for email in emails if email]
    conn.executemany(
        "INSERT OR IGNORE INTO email_aliases (name, email) VALUES (?, ?);",
        rows,
    )
    return conn.total_changes


def get_alias_emails(conn: sqlite3.Connection, name: str) -> list[str]:
    """Exact match: emails for this alias name."""
    if not name:
        return []
    rows = conn.execute(
        "SELECT email FROM email_aliases WHERE name = ?;",
        (name.lower().strip(),),
    ).fetchall()
    return [row["email"] for row in rows]


def get_alias_emails_for_term(conn: sqlite3.Connection, term: str) -> list[str]:
    """Emails for exact name match, or for any alias whose name contains the term (e.g. 'epstein' -> 'jeffrey epstein')."""
    if not term:
        return []
    term = term.lower().strip()
    exact = get_alias_emails(conn, term)
    if exact:
        return exact
    rows = conn.execute(
        "SELECT DISTINCT email FROM email_aliases WHERE name LIKE ?;",
        (f"%{term}%",),
    ).fetchall()
    return [row["email"] for row in rows]


def get_known_email_contacts(conn: sqlite3.Connection) -> list[dict]:
    """Distinct names and emails from headers (from/to/cc) for dropdowns."""
    rows = conn.execute(
        """
        SELECT DISTINCT display, addr FROM (
            SELECT from_name AS display, from_addr AS addr FROM email_headers WHERE from_addr != '' OR from_name != ''
            UNION
            SELECT to_name AS display, to_addr AS addr FROM email_headers WHERE to_addr != '' OR to_name != ''
            UNION
            SELECT cc_name AS display, cc_addr AS addr FROM email_headers WHERE cc_addr != '' OR cc_name != ''
        ) WHERE display != '' OR addr != ''
        ORDER BY LOWER(COALESCE(display, addr));
        """
    ).fetchall()
    contacts: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        display = (row["display"] or "").strip() or (row["addr"] or "").strip()
        addr = (row["addr"] or "").strip()
        if not display and not addr:
            continue
        key = (display.lower(), addr.lower())
        if key in seen:
            continue
        seen.add(key)
        contacts.append({"name": display or addr, "email": addr or display})
    return contacts


def get_alias_names_and_emails(conn: sqlite3.Connection) -> list[dict]:
    """Alias name -> list of emails for UI."""
    rows = conn.execute(
        "SELECT name, email FROM email_aliases ORDER BY name, email;"
    ).fetchall()
    by_name: dict[str, list[str]] = {}
    for row in rows:
        name = (row["name"] or "").strip()
        email = (row["email"] or "").strip()
        if not name:
            continue
        if name not in by_name:
            by_name[name] = []
        if email and email not in by_name[name]:
            by_name[name].append(email)
    return [{"name": k, "emails": v} for k, v in sorted(by_name.items())]


def query_email_headers(
    conn: sqlite3.Connection,
    *,
    sender_terms: list[str],
    recipient_terms: list[str],
    strict_direction: bool,
    subject_contains: str | None,
    date_from: str | None,
    date_to: str | None,
) -> list[sqlite3.Row]:
    clauses: list[str] = []
    params: list[str] = []

    def term_clause(term: str, columns: list[str]) -> tuple[str, list[str]]:
        sub = []
        local_params = []
        pattern = f"%{term}%"
        for col in columns:
            sub.append(f"LOWER(COALESCE({col}, '')) LIKE LOWER(?)")
            local_params.append(pattern)
        return "(" + " OR ".join(sub) + ")", local_params

    if subject_contains:
        clauses.append("LOWER(COALESCE(subject, '')) LIKE LOWER(?)")
        params.append(f"%{subject_contains.lower().strip()}%")
    if date_from:
        clauses.append("date >= ?")
        params.append(date_from)
    if date_to:
        clauses.append("date <= ?")
        params.append(date_to)

    # When only sender: match FROM field (emails from that person). When only recipient: match TO/CC.
    from_columns = ["from_addr", "from_name"]
    to_cc_columns = ["to_addr", "to_name", "cc_addr", "cc_name"]

    sender_from_clause = ""
    sender_from_params: list[str] = []
    if sender_terms:
        parts = []
        for term in sender_terms:
            clause, p = term_clause(term, from_columns)
            parts.append(clause)
            sender_from_params.extend(p)
        sender_from_clause = "(" + " OR ".join(parts) + ")"

    sender_any_clause = ""
    sender_any_params: list[str] = []
    if sender_terms:
        parts = []
        for term in sender_terms:
            clause, p = term_clause(term, ["from_addr", "from_name", "to_addr", "to_name", "cc_addr", "cc_name"])
            parts.append(clause)
            sender_any_params.extend(p)
        sender_any_clause = "(" + " OR ".join(parts) + ")"

    recipient_to_clause = ""
    recipient_to_params: list[str] = []
    if recipient_terms:
        parts = []
        for term in recipient_terms:
            clause, p = term_clause(term, to_cc_columns)
            parts.append(clause)
            recipient_to_params.extend(p)
        recipient_to_clause = "(" + " OR ".join(parts) + ")"

    recipient_any_clause = ""
    recipient_any_params: list[str] = []
    if recipient_terms:
        parts = []
        for term in recipient_terms:
            clause, p = term_clause(term, ["from_addr", "from_name", "to_addr", "to_name", "cc_addr", "cc_name"])
            parts.append(clause)
            recipient_any_params.extend(p)
        recipient_any_clause = "(" + " OR ".join(parts) + ")"

    if sender_terms and recipient_terms and strict_direction:
        from_parts = []
        from_params: list[str] = []
        for term in sender_terms:
            clause, p = term_clause(term, from_columns)
            from_parts.append(clause)
            from_params.extend(p)
        to_parts = []
        to_params: list[str] = []
        for term in recipient_terms:
            clause, p = term_clause(term, to_cc_columns)
            to_parts.append(clause)
            to_params.extend(p)
        clauses.append("(" + " OR ".join(from_parts) + ")")
        clauses.append("(" + " OR ".join(to_parts) + ")")
        params.extend(from_params + to_params)
    else:
        if sender_terms and not recipient_terms:
            clauses.append(sender_from_clause)
            params.extend(sender_from_params)
        elif recipient_terms and not sender_terms:
            clauses.append(recipient_to_clause)
            params.extend(recipient_to_params)
        else:
            if sender_any_clause:
                clauses.append(sender_any_clause)
                params.extend(sender_any_params)
            if recipient_any_clause:
                clauses.append(recipient_any_clause)
                params.extend(recipient_any_params)

    if not clauses:
        return []
    where_sql = " AND ".join(clauses)
    return conn.execute(
        f"""
        SELECT * FROM email_headers
        WHERE {where_sql}
        ORDER BY doc_id, page_num;
        """,
        params,
    ).fetchall()
