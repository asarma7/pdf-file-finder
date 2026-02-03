import sqlite3
import logging
import re
from . import embeddings
from .safety import redact_text, trim_text

logger = logging.getLogger("docqa")

ABS_MIN = 0.25
DELTA = 0.08
RATIO = 0.75
ANCHOR_SIM_MIN = 0.32
ANCHOR_SIM_BOOST = 0.12
BOILERPLATE = [
    "no evidence",
    "conspiracy",
    "speculation",
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
    "we",
    "you",
    "i",
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


def _extract_anchors(query: str) -> list[str]:
    tokens = _tokenize(query)
    filtered = [t for t in tokens if len(t) >= 3 and t not in STOPWORDS]
    anchors = []
    for token in filtered:
        anchors.append(_stem(token))
    bigrams = []
    for i in range(len(filtered) - 1):
        bigrams.append(f"{filtered[i]} {filtered[i+1]}")
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
    tokens = {_stem(t) for t in _tokenize(text)}
    overlap = 0
    for a in anchors:
        if " " in a:
            if a in text.lower():
                overlap += 1
        else:
            if _stem(a) in tokens:
                overlap += 1
    return overlap


def _anchor_sim_scores(
    items: list[dict],
    anchors: list[str],
    embeddings_device: str | None,
    hf_token: str | None,
    engine: str | None,
    use_worker: bool,
) -> list[float]:
    if not items or not anchors:
        return [0.0 for _ in items]
    anchor_vecs = embeddings.embed_texts(
        anchors,
        device=embeddings_device,
        hf_token=hf_token,
        engine=engine,
        use_worker=use_worker,
    )
    chunk_texts = [item["chunk_text"] for item in items]
    chunk_vecs = embeddings.embed_texts(
        chunk_texts,
        device=embeddings_device,
        hf_token=hf_token,
        engine=engine,
        use_worker=use_worker,
    )
    sims = chunk_vecs @ anchor_vecs.T
    return [float(sims[i].max()) for i in range(sims.shape[0])]


def _has_boilerplate(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in BOILERPLATE)


def _apply_semantic_filters(
    items: list[dict],
    anchors: list[str],
    embeddings_device: str | None,
    hf_token: str | None,
    engine: str | None,
    use_worker: bool,
) -> list[dict]:
    if not items:
        return []
    top_score = max(item["score"] for item in items)
    logger.info(
        "retrieve:score top=%.4f abs_min=%.2f delta=%.2f ratio=%.2f",
        top_score,
        ABS_MIN,
        DELTA,
        RATIO,
    )
    logger.info("retrieve:anchors %s", ",".join(anchors))
    anchor_sims = _anchor_sim_scores(items, anchors, embeddings_device, hf_token, engine, use_worker)
    filtered = []
    for item, anchor_sim in zip(items, anchor_sims):
        overlap = _anchor_overlap(item["chunk_text"], anchors)
        score = item["score"]
        if _has_boilerplate(item["chunk_text"]) and overlap == 0:
            score *= 0.5
        if anchor_sim >= ANCHOR_SIM_MIN:
            score += ANCHOR_SIM_BOOST * anchor_sim
        elif overlap > 0:
            score += 0.02 * min(1.0, overlap / max(1, len(anchors)))
        item["score"] = score
        keep = score >= max(ABS_MIN, top_score - DELTA) or score >= top_score * RATIO
        if keep:
            filtered.append(item)
        else:
            logger.info(
                "retrieve:drop file=%s page=%s score=%.4f overlap=%s anchor_sim=%.3f boilerplate=%s",
                item.get("filename"),
                item.get("page_num"),
                score,
                overlap,
                anchor_sim,
                _has_boilerplate(item.get("chunk_text", "")),
            )
    return filtered


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
) -> list[dict]:
    logger.info("retrieve:mode=%s top_k=%s", mode, top_k)
    if mode == "keyword":
        results = keyword_search(conn, query, top_k)
        if redact:
            for item in results:
                item["snippet"] = redact_text(item["snippet"])
                item["chunk_text"] = redact_text(item["chunk_text"])
        return results
    elif mode == "semantic":
        try:
            results = semantic_search(
                conn, query, index, top_k, embeddings_device, hf_token, engine, use_worker
            )
        except Exception as exc:
            logger.info("retrieve:semantic_error %s (fallback to keyword)", exc)
            results = []
    else:
        kw = keyword_search(conn, query, max(50, top_k * 2))
        try:
            sem = semantic_search(
                conn, query, index, max(50, top_k * 2), embeddings_device, hf_token, engine, use_worker
            )
        except Exception as exc:
            logger.info("retrieve:semantic_error %s (fallback to keyword)", exc)
            sem = []
        results = rrf_merge(kw, sem)
    results = results[:top_k]
    if redact:
        for item in results:
            item["snippet"] = redact_text(item["snippet"])
            item["chunk_text"] = redact_text(item["chunk_text"])
    return results
