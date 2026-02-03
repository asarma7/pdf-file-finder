import re
import sqlite3

from .safety import redact_text, trim_text


def _sanitize_fts_query(query: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", " ", query)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _extract_terms(query: str) -> list[str]:
    terms = []
    for phrase, word in re.findall(r'"([^"]+)"|(\S+)', query):
        terms.append(phrase or word)
    return [t for t in terms if t]


def fts_search(
    conn: sqlite3.Connection,
    query: str,
    *,
    limit: int = 50,
    case_sensitive: bool = False,
    redact: bool = False,
) -> list[dict]:
    fts_query = _sanitize_fts_query(query)
    if not fts_query:
        return []
    rows = conn.execute(
        """
        SELECT
            chunks.doc_id AS doc_id,
            docs.rel_path AS rel_path,
            docs.filename AS filename,
            chunks.page_num AS page_num,
            chunks.text AS text,
            snippet(chunks_fts, 0, '<mark>', '</mark>', '...', 12) AS snippet,
            bm25(chunks_fts) AS rank
        FROM chunks_fts
        JOIN chunks ON chunks_fts.chunk_id = chunks.id
        JOIN docs ON docs.id = chunks.doc_id
        WHERE chunks_fts MATCH ?
        ORDER BY rank, docs.rel_path, chunks.page_num
        LIMIT ?;
        """,
        (fts_query, limit * 5 if case_sensitive else limit),
    ).fetchall()

    terms = _extract_terms(query)
    results = []
    for row in rows:
        text = row["text"] or ""
        if case_sensitive and terms:
            if not all(term in text for term in terms):
                continue
        snippet = row["snippet"] or ""
        snippet = trim_text(snippet, 200)
        if redact:
            snippet = redact_text(snippet)
        results.append(
            {
                "doc_id": row["doc_id"],
                "rel_path": row["rel_path"],
                "filename": row["filename"],
                "page_num": row["page_num"] + 1,
                "snippet": snippet,
                "matched_terms": terms,
            }
        )
        if len(results) >= limit:
            break
    return results


def regex_search(
    conn: sqlite3.Connection,
    query: str,
    *,
    limit: int = 50,
    case_sensitive: bool = False,
    redact: bool = False,
) -> list[dict]:
    flags = 0 if case_sensitive else re.IGNORECASE
    pattern = re.compile(query, flags=flags)
    rows = conn.execute(
        """
        SELECT
            chunks.doc_id AS doc_id,
            docs.rel_path AS rel_path,
            docs.filename AS filename,
            chunks.page_num AS page_num,
            chunks.text AS text
        FROM chunks
        JOIN docs ON docs.id = chunks.doc_id
        ORDER BY docs.rel_path, chunks.page_num;
        """
    ).fetchall()

    results = []
    for row in rows:
        text = row["text"] or ""
        match = pattern.search(text)
        if not match:
            continue
        start = max(0, match.start() - 200)
        end = min(len(text), match.end() + 200)
        snippet = text[start:end]
        snippet = snippet.replace(match.group(0), f"<mark>{match.group(0)}</mark>", 1)
        snippet = trim_text(snippet, 200)
        if redact:
            snippet = redact_text(snippet)
        results.append(
            {
                "doc_id": row["doc_id"],
                "rel_path": row["rel_path"],
                "filename": row["filename"],
                "page_num": row["page_num"] + 1,
                "snippet": snippet,
                "matched_terms": [match.group(0)],
            }
        )
        if len(results) >= limit:
            break
    return results
