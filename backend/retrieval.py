import sqlite3
import logging
import re
import time
from . import embeddings
from .safety import redact_text, trim_text

logger = logging.getLogger("docqa")

ABS_MIN = 0.25
DELTA = 0.08
HIGH_CONF = 0.42
BOILERPLATE = [
    "no evidence",
    "conspiracy",
    "speculation",
    "debunk",
    "hoax",
    "rumor",
    "rumour",
    "unsubstantiated",
]
STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "if",
    "then",
    "than",
    "so",
    "of",
    "to",
    "in",
    "on",
    "for",
    "by",
    "with",
    "about",
    "from",
    "as",
    "at",
    "into",
    "who",
    "what",
    "when",
    "where",
    "why",
    "how",
    "which",
    "whom",
    "whose",
    "do",
    "does",
    "did",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "this",
    "that",
    "these",
    "those",
    "there",
    "their",
    "it",
    "its",
    "he",
    "she",
    "they",
    "them",
    "his",
    "her",
    "we",
    "you",
    "i",
    "document",
    "documents",
    "file",
    "files",
    "dataset",
    "page",
    "pages",
    "pdf",
    "email",
    "emails",
}


def _sanitize_fts_query(query: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", " ", query)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


def _stem(token: str) -> str:
    for suffix in ("ing", "edly", "edly", "edly", "ed", "ly", "es", "s"):
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            return token[: -len(suffix)]
    return token


def _strip_question_template(query: str) -> str:
    lowered = query.strip().lower()
    templates = [
        r"^(?:who|what|when|where|why|how|which|whom|whose)\s+(?:is|are|was|were)\s+(.+)$",
        r"^tell\s+me\s+about\s+(.+)$",
        r"^tell\s+us\s+about\s+(.+)$",
    ]
    for pattern in templates:
        match = re.match(pattern, lowered)
        if match:
            span = match.group(1)
            span = re.split(r"(?:\?|$| and |, )", span)[0]
            if span:
                return span
    role_match = re.match(r"^what\s+was\s+(.+?)\s+role\b.*", lowered)
    if role_match:
        return role_match.group(1)
    return lowered


def _extract_anchors(query: str) -> list[str]:
    focused = _strip_question_template(query)
    tokens = _tokenize(focused)
    filtered = [t for t in tokens if len(t) >= 3 and t not in STOPWORDS]
    anchors = []
    stemmed = [_stem(token) for token in filtered]
    anchors.extend(stemmed)
    bigrams = []
    for i in range(len(stemmed) - 1):
        bigrams.append(f"{stemmed[i]} {stemmed[i+1]}")
    anchors.extend(bigrams)
    seen = set()
    deduped = []
    for a in anchors:
        if a in seen:
            continue
        seen.add(a)
        deduped.append(a)
    return deduped[:12]


def _anchor_overlap(text: str, anchors: list[str]) -> int:
    if not anchors:
        return 0
    token_list = [_stem(t) for t in _tokenize(text)]
    tokens = set(token_list)
    bigrams = {f"{token_list[i]} {token_list[i+1]}" for i in range(len(token_list) - 1)}
    overlap = 0
    for a in anchors:
        if " " in a:
            if a in bigrams:
                overlap += 1
        else:
            if _stem(a) in tokens:
                overlap += 1
    return overlap


def _has_boilerplate(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in BOILERPLATE)


def _score_stats(scores: list[float]) -> dict:
    if not scores:
        return {"min": None, "median": None, "max": None}
    ordered = sorted(scores)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 0:
        median = (ordered[mid - 1] + ordered[mid]) / 2.0
    else:
        median = ordered[mid]
    return {"min": ordered[0], "median": median, "max": ordered[-1]}


def _apply_post_filters(
    *,
    items: list[dict],
    query: str,
    mode: str,
    abs_min: float = ABS_MIN,
    delta: float = DELTA,
    high_conf: float = HIGH_CONF,
) -> tuple[list[dict], dict]:
    start_time = time.perf_counter()
    anchors = _extract_anchors(query)
    anchor_time = time.perf_counter()
    raw_scores = [float(item.get("score", 0.0)) for item in items]
    raw_stats = _score_stats(raw_scores)
    top_scores = sorted(raw_scores, reverse=True)[:10]
    semantic_scores = [
        float(item["semantic_score"])
        for item in items
        if item.get("semantic_score") is not None
    ]
    s0 = max(semantic_scores) if semantic_scores else None
    cutoff = max(abs_min, s0 - delta) if s0 is not None else None
    stats_time = time.perf_counter()
    logger.info("retrieve:anchors %s", ",".join(anchors))
    logger.info(
        "retrieve:raw_scores count=%s top10=%s stats=%s",
        len(raw_scores),
        top_scores,
        raw_stats,
    )

    boilerplate_penalized = 0
    anchor_kept = 0
    anchor_dropped = 0
    cutoff_kept = 0
    cutoff_dropped = 0
    filtered: list[dict] = []

    for item in items:
        overlap = _anchor_overlap(item.get("chunk_text", ""), anchors)
        semantic_score = item.get("semantic_score")
        has_boiler = _has_boilerplate(item.get("chunk_text", ""))

        if has_boiler and overlap == 0:
            boilerplate_penalized += 1
            if semantic_score is None:
                if mode == "keyword":
                    continue
            else:
                semantic_score *= 0.5
                item["semantic_score"] = semantic_score
                if mode != "keyword":
                    item["score"] = float(item.get("score", 0.0)) * 0.5

        if mode == "keyword":
            filtered.append(item)
            continue

        if overlap >= 1:
            anchor_kept += 1
        else:
            anchor_dropped += 1
            continue

        if semantic_score is None:
            filtered.append(item)
            continue

        if cutoff is not None and semantic_score < cutoff:
            cutoff_dropped += 1
            continue
        cutoff_kept += 1
        filtered.append(item)
    loop_time = time.perf_counter()

    if mode in ("semantic", "hybrid"):
        filtered.sort(key=lambda x: x.get("score", 0.0), reverse=True)

    diagnostics = {
        "raw_count": len(items),
        "raw_scores_top10": top_scores,
        "raw_score_stats": raw_stats,
        "anchors": anchors,
        "boilerplate_penalized": boilerplate_penalized,
        "anchor_gate_kept": anchor_kept,
        "anchor_gate_dropped": anchor_dropped,
        "cutoff_threshold": cutoff,
        "cutoff_kept": cutoff_kept,
        "cutoff_dropped": cutoff_dropped,
        "final_count": len(filtered),
        "params": {"abs_min": abs_min, "delta": delta, "high_conf": high_conf},
    }
    logger.info(
        "retrieve:filters raw=%s boilerplate=%s anchor_keep=%s anchor_drop=%s cutoff_keep=%s cutoff_drop=%s final=%s",
        diagnostics["raw_count"],
        diagnostics["boilerplate_penalized"],
        diagnostics["anchor_gate_kept"],
        diagnostics["anchor_gate_dropped"],
        diagnostics["cutoff_kept"],
        diagnostics["cutoff_dropped"],
        diagnostics["final_count"],
    )
    logger.info(
        "retrieve:timing anchors=%.3fs stats=%.3fs filter_loop=%.3fs total=%.3fs",
        anchor_time - start_time,
        stats_time - anchor_time,
        loop_time - stats_time,
        loop_time - start_time,
    )
    return filtered, diagnostics


def _log_top(results: list[dict], anchors: list[str], label: str, limit: int = 30) -> None:
    top = sorted(results, key=lambda x: x["score"], reverse=True)[:limit]
    for idx, item in enumerate(top, start=1):
        overlap = _anchor_overlap(item.get("chunk_text", ""), anchors)
        logger.info(
            "retrieve:%s #%s file=%s page=%s score=%.4f overlap=%s",
            label,
            idx,
            item.get("filename"),
            item.get("page_num"),
            item.get("score", 0.0),
            overlap,
        )


def keyword_search(
    conn: sqlite3.Connection,
    query: str,
    limit: int,
) -> list[dict]:
    logger.info("retrieve:keyword query_len=%s limit=%s", len(query), limit)
    fts_query = _sanitize_fts_query(query)
    if not fts_query:
        return []
    rows = conn.execute(
        """
        SELECT
            chunks.id AS chunk_id,
            chunks.doc_id AS doc_id,
            docs.filename AS filename,
            docs.rel_path AS rel_path,
            chunks.page_num AS page_num,
            chunks.text AS text,
            snippet(chunks_fts, 0, '<mark>', '</mark>', '...', 12) AS snippet,
            bm25(chunks_fts) AS rank
        FROM chunks_fts
        JOIN chunks ON chunks_fts.chunk_id = chunks.id
        JOIN docs ON docs.id = chunks.doc_id
        WHERE chunks_fts MATCH ?
        ORDER BY rank
        LIMIT ?;
        """,
        (fts_query, limit),
    ).fetchall()
    results = []
    for row in rows:
        snippet = row["snippet"] or ""
        snippet = trim_text(snippet, 200)
        results.append(
            {
                "chunk_id": row["chunk_id"],
                "doc_id": row["doc_id"],
                "filename": row["filename"],
                "rel_path": row["rel_path"],
                "page_num": row["page_num"] + 1,
                "chunk_text": row["text"],
                "snippet": snippet,
                "score": float(row["rank"]),
            }
        )
    return results


def semantic_search(
    conn: sqlite3.Connection,
    query: str,
    index,
    top_k: int,
    embeddings_device: str | None,
    hf_token: str | None,
    engine: str | None,
    use_worker: bool,
) -> list[dict]:
    logger.info("retrieve:semantic query_len=%s top_k=%s device=%s", len(query), top_k, embeddings_device)
    if index is None:
        return []
    vectors = embeddings.embed_texts(
        [query],
        device=embeddings_device,
        hf_token=hf_token,
        engine=engine,
        use_worker=use_worker,
    )
    scores, ids = index.search(vectors, top_k)
    id_list = [int(i) for i in ids[0] if i != -1]
    if not id_list:
        return []
    placeholders = ",".join(["?"] * len(id_list))
    rows = conn.execute(
        f"""
        SELECT
            chunks.id AS chunk_id,
            chunks.doc_id AS doc_id,
            docs.filename AS filename,
            docs.rel_path AS rel_path,
            chunks.page_num AS page_num,
            chunks.text AS text
        FROM chunks
        JOIN docs ON docs.id = chunks.doc_id
        WHERE chunks.id IN ({placeholders});
        """,
        tuple(id_list),
    ).fetchall()
    row_map = {row["chunk_id"]: row for row in rows}
    results = []
    for idx, chunk_id in enumerate(id_list):
        row = row_map.get(chunk_id)
        if not row:
            continue
        results.append(
            {
                "chunk_id": row["chunk_id"],
                "doc_id": row["doc_id"],
                "filename": row["filename"],
                "rel_path": row["rel_path"],
                "page_num": row["page_num"] + 1,
                "chunk_text": row["text"],
                "snippet": trim_text(row["text"], 200),
                "score": float(scores[0][idx]),
            }
        )
    return results


def rrf_merge(keyword: list[dict], semantic: list[dict], k: int = 60) -> list[dict]:
    scores = {}
    for rank, item in enumerate(keyword, start=1):
        scores[item["chunk_id"]] = scores.get(item["chunk_id"], 0.0) + 1.0 / (k + rank)
    for rank, item in enumerate(semantic, start=1):
        scores[item["chunk_id"]] = scores.get(item["chunk_id"], 0.0) + 1.0 / (k + rank)
    merged = {item["chunk_id"]: item for item in keyword + semantic}
    ranked = sorted(merged.values(), key=lambda x: scores.get(x["chunk_id"], 0.0), reverse=True)
    for item in ranked:
        item["score"] = scores.get(item["chunk_id"], 0.0)
    return ranked


def retrieve(
    conn: sqlite3.Connection,
    query: str,
    mode: str,
    top_k: int,
    index,
    redact: bool,
    embeddings_device: str | None,
    hf_token: str | None,
    engine: str | None,
    use_worker: bool,
    debug: bool = False,
) -> tuple[list[dict], dict]:
    logger.info("retrieve:mode=%s top_k=%s", mode, top_k)
    debug_info: dict = {}
    if mode == "keyword":
        raw = keyword_search(conn, query, top_k)
        filtered, diagnostics = _apply_post_filters(items=raw, query=query, mode=mode)
        debug_info = diagnostics
        results = filtered
    elif mode == "semantic":
        try:
            raw = semantic_search(
                conn, query, index, top_k, embeddings_device, hf_token, engine, use_worker
            )
        except Exception as exc:
            logger.info("retrieve:semantic_error %s (fallback to keyword)", exc)
            raw = []
        for item in raw:
            item["semantic_score"] = item.get("score")
        filtered, diagnostics = _apply_post_filters(items=raw, query=query, mode=mode)
        debug_info = diagnostics
        results = filtered
    else:
        kw = keyword_search(conn, query, max(50, top_k * 2))
        try:
            sem = semantic_search(
                conn, query, index, max(50, top_k * 2), embeddings_device, hf_token, engine, use_worker
            )
        except Exception as exc:
            logger.info("retrieve:semantic_error %s (fallback to keyword)", exc)
            sem = []
        for item in sem:
            item["semantic_score"] = item.get("score")
        for item in kw:
            item["semantic_score"] = None
        merged = rrf_merge(kw, sem)
        filtered, diagnostics = _apply_post_filters(items=merged, query=query, mode=mode)
        debug_info = diagnostics
        results = filtered
    if redact:
        for item in results:
            item["snippet"] = redact_text(item["snippet"])
            item["chunk_text"] = redact_text(item["chunk_text"])
    if not debug:
        debug_info = {}
    return results, debug_info
