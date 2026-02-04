import re
import sqlite3

from .safety import redact_text, trim_text


def _normalize_term(term: str) -> str:
    cleaned = term.strip().strip('"').strip("'")
    return re.sub(r"\s+", " ", cleaned)


def _build_patterns(term: str, aliases: list[str]) -> list[re.Pattern]:
    terms = [_normalize_term(term)] + [_normalize_term(a) for a in aliases]
    deduped: list[str] = []
    seen = set()
    for value in terms:
        if not value or value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    patterns = []
    for value in deduped:
        pattern = re.compile(re.escape(value), flags=re.IGNORECASE)
        patterns.append(pattern)
    return patterns


def count_mentions(
    conn: sqlite3.Connection,
    term: str,
    *,
    aliases: list[str] | None = None,
    redact: bool = False,
    limit: int = 10,
    debug: bool = False,
) -> dict:
    aliases = aliases or []
    patterns = _build_patterns(term, aliases)
    if not patterns:
        return {"total_hits": 0, "unique_pages": 0, "unique_docs": 0, "contexts": []}

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

    total_hits = 0
    pages = set()
    docs = set()
    contexts: list[dict] = []
    debug_samples: list[dict] = []

    for row in rows:
        text = row["text"] or ""
        hits = 0
        match = None
        for pattern in patterns:
            matches = list(pattern.finditer(text))
            hits += len(matches)
            if match is None and matches:
                match = matches[0]
        if hits == 0:
            continue
        if debug and len(debug_samples) < 5 and match:
            debug_samples.append(
                {
                    "filename": row["filename"],
                    "page_num": row["page_num"] + 1,
                    "match": match.group(0),
                }
            )
        total_hits += hits
        pages.add((row["doc_id"], row["page_num"]))
        docs.add(row["doc_id"])

        if len(contexts) < limit and match:
            start = max(0, match.start() - 200)
            end = min(len(text), match.end() + 200)
            snippet = text[start:end]
            snippet = snippet.replace(match.group(0), f"<mark>{match.group(0)}</mark>", 1)
            snippet = trim_text(snippet, 200)
            if redact:
                snippet = redact_text(snippet)
            contexts.append(
                {
                    "doc_id": row["doc_id"],
                    "rel_path": row["rel_path"],
                    "filename": row["filename"],
                    "page_num": row["page_num"] + 1,
                    "snippet": snippet,
                }
            )

    result = {
        "total_hits": total_hits,
        "unique_pages": len(pages),
        "unique_docs": len(docs),
        "contexts": contexts,
    }
    if debug:
        result["debug"] = {
            "term": term,
            "aliases": aliases,
            "patterns": [p.pattern for p in patterns],
            "sample_matches": debug_samples,
        }
    return result
