import math
from typing import Iterable

import numpy as np

from . import embeddings, retrieval, rag


SUMMARY_TOKENS = ("summary", "overview", "themes", "worst", "accusations", "allegations")


def _expand_queries(query: str) -> list[str]:
    base = query.strip()
    if not base:
        return []
    expansions = [base]
    lowered = base.lower()
    if any(token in lowered for token in ("summary", "overview", "themes", "worst", "accusation", "allegation")):
        expansions.extend(
            [
                "allegations",
                "accusations",
                "abuse",
                "assault",
                "trafficking",
                "coercion",
                "misconduct",
                "exploitation",
            ]
        )
    seen = set()
    deduped = []
    for q in expansions:
        q = q.strip()
        if not q or q in seen:
            continue
        seen.add(q)
        deduped.append(q)
    return deduped[:8]


def _kmeans(vectors: np.ndarray, k: int, iters: int = 8) -> tuple[np.ndarray, np.ndarray]:
    if vectors.size == 0:
        return np.array([]), np.array([])
    k = max(1, min(k, len(vectors)))
    centroids = vectors[:k].copy()
    for _ in range(iters):
        sims = vectors @ centroids.T
        labels = np.argmax(sims, axis=1)
        new_centroids = []
        for idx in range(k):
            members = vectors[labels == idx]
            if len(members) == 0:
                new_centroids.append(centroids[idx])
            else:
                centroid = members.mean(axis=0)
                norm = np.linalg.norm(centroid)
                if norm == 0:
                    norm = 1.0
                new_centroids.append(centroid / norm)
        centroids = np.vstack(new_centroids)
    sims = vectors @ centroids.T
    labels = np.argmax(sims, axis=1)
    return labels, centroids


def _select_top_by_cluster(
    items: list[dict], vectors: np.ndarray, labels: np.ndarray, centroids: np.ndarray, per_cluster: int
) -> list[list[dict]]:
    clusters: list[list[dict]] = [[] for _ in range(len(centroids))]
    for idx, item in enumerate(items):
        cluster_id = int(labels[idx])
        clusters[cluster_id].append((item, float(vectors[idx] @ centroids[cluster_id])))
    grouped: list[list[dict]] = []
    for cluster in clusters:
        cluster.sort(key=lambda x: x[1], reverse=True)
        grouped.append([item for item, _score in cluster[:per_cluster]])
    return grouped


def _flatten(items: Iterable[list[dict]]) -> list[dict]:
    flattened: list[dict] = []
    for group in items:
        flattened.extend(group)
    return flattened


def corpus_summary(
    *,
    query: str,
    conn,
    index,
    top_k: int,
    embeddings_device: str,
    hf_token: str | None,
    embeddings_engine: str,
    embeddings_worker: bool,
    llm_provider: str,
    llm_model: str,
    llm_base_url: str,
    llm_api_key: str,
) -> dict:
    expanded_queries = _expand_queries(query)
    if not expanded_queries:
        return {"themes": [], "answer_markdown": "", "stats": {"expanded_queries": []}}

    pool: dict[int, dict] = {}
    per_query_k = max(30, min(80, top_k))
    for q in expanded_queries:
        results, _debug = retrieval.retrieve(
            conn,
            q,
            "hybrid",
            per_query_k,
            index,
            True,
            embeddings_device,
            hf_token,
            embeddings_engine,
            embeddings_worker,
            False,
            None,
            None,
            None,
            None,
            debug=False,
            disable_anchor_gate=True,
        )
        for item in results:
            pool[item["chunk_id"]] = item

    items = list(pool.values())
    if not items:
        return {"themes": [], "answer_markdown": "", "stats": {"expanded_queries": expanded_queries}}

    max_pool = min(300, max(60, top_k * 6))
    items = sorted(items, key=lambda x: x.get("score", 0.0), reverse=True)[:max_pool]
    texts = [item.get("chunk_text", "") for item in items]
    vectors = embeddings.embed_texts(
        texts,
        device=embeddings_device,
        hf_token=hf_token,
        engine=embeddings_engine,
        use_worker=embeddings_worker,
    )
    clusters = max(2, min(6, math.ceil(len(items) / 40)))
    labels, centroids = _kmeans(vectors, clusters)
    grouped = _select_top_by_cluster(items, vectors, labels, centroids, per_cluster=3)

    themes = []
    answer_blocks = []
    for idx, cluster_items in enumerate(grouped, start=1):
        if not cluster_items:
            continue
        summary = ""
        if llm_provider and llm_provider != "none":
            summary = rag.generate_answer(
                f"Summarize the theme for: {query}",
                cluster_items,
                "summary",
                {
                    "provider": llm_provider,
                    "model": llm_model,
                    "base_url": llm_base_url,
                    "api_key": llm_api_key,
                },
            ).get("answer_markdown", "")
        title = f"Theme {idx}"
        citations = [
            {
                "doc_id": item["doc_id"],
                "filename": item["filename"],
                "rel_path": item["rel_path"],
                "page_num": item["page_num"],
                "quote": item["chunk_text"],
                "snippet": item.get("snippet") or item.get("chunk_text", "")[:200],
            }
            for item in cluster_items
        ]
        themes.append({"title": title, "summary": summary, "citations": citations})
        if summary:
            answer_blocks.append(f"{title}: {summary}")
    return {
        "themes": themes,
        "answer_markdown": "\n\n".join(answer_blocks),
        "stats": {"expanded_queries": expanded_queries, "pool_size": len(items)},
    }


def _is_summary_like(query: str) -> bool:
    lowered = query.lower()
    return any(token in lowered for token in SUMMARY_TOKENS)


def _is_refusal(text: str) -> bool:
    """True if the summary looks like a short refusal (e.g. 'not found in the indexed documents'), not a substantive summary."""
    if not text or len(text.strip()) < 20:
        return True
    lowered = text.strip().lower()
    # Only treat as refusal when response is short and clearly a non-answer (avoid matching "does not provide" inside real summaries)
    if len(lowered) > 150:
        return False
    refusal_phrases = (
        "not found in the indexed documents",
        "not found in the documents",
        "no relevant information",
        "cannot find",
        "no content",
    )
    return any(phrase in lowered for phrase in refusal_phrases)


def email_filtered_summary(
    *,
    query: str,
    conn,
    allowed_pages: list[tuple[int, int]],
    priority_pages: list[tuple[int, int]] | None = None,
    secondary_terms: list[str] | None = None,
    embeddings_device: str,
    hf_token: str | None,
    embeddings_engine: str,
    embeddings_worker: bool,
    llm_provider: str,
    llm_model: str,
    llm_base_url: str,
    llm_api_key: str,
    max_pages: int = 300,
) -> dict:
    if not allowed_pages:
        return {"summary_type": "none", "answer_markdown": "", "themes": [], "sources": [], "stats": {}}
    priority_pages = priority_pages or []
    secondary_terms = [t for t in (secondary_terms or []) if t]
    pages = (priority_pages + [p for p in allowed_pages if p not in priority_pages])[:max_pages]
    clauses = []
    params: list[int] = []
    for doc_id, page_num in pages:
        clauses.append("(chunks.doc_id = ? AND chunks.page_num = ?)")
        params.extend([doc_id, page_num])
    where_pages = " OR ".join(clauses)
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
        WHERE {where_pages}
        ORDER BY chunks.doc_id, chunks.page_num, chunks.chunk_index;
        """,
        params,
    ).fetchall()
    items = [
        {
            "chunk_id": row["chunk_id"],
            "doc_id": row["doc_id"],
            "filename": row["filename"],
            "rel_path": row["rel_path"],
            "page_num": row["page_num"] + 1,
            "chunk_text": row["text"],
            "snippet": (row["text"] or "")[:200],
        }
        for row in rows
    ]
    if not items:
        return {"summary_type": "none", "answer_markdown": "", "themes": [], "sources": [], "stats": {}}

    texts = [item.get("chunk_text", "") for item in items]
    vectors = embeddings.embed_texts(
        texts,
        device=embeddings_device,
        hf_token=hf_token,
        engine=embeddings_engine,
        use_worker=embeddings_worker,
    )
    query_vec = embeddings.embed_texts(
        [query],
        device=embeddings_device,
        hf_token=hf_token,
        engine=embeddings_engine,
        use_worker=embeddings_worker,
    )[0]
    scores = vectors @ query_vec
    priority_set = set(priority_pages)
    boosted = []
    for score, item in zip(scores, items):
        boost = 0.0
        if (item["doc_id"], item["page_num"] - 1) in priority_set:
            boost += 0.05
        if secondary_terms and any(term.lower() in (item["chunk_text"] or "").lower() for term in secondary_terms):
            boost += 0.02
        boosted.append((score + boost, item))
    ranked = [item for _score, item in sorted(boosted, key=lambda x: x[0], reverse=True)]

    if _is_summary_like(query):
        cluster_items = ranked[:120]
        cluster_texts = [item.get("chunk_text", "") for item in cluster_items]
        cluster_vectors = embeddings.embed_texts(
            cluster_texts,
            device=embeddings_device,
            hf_token=hf_token,
            engine=embeddings_engine,
            use_worker=embeddings_worker,
        )
        # Allow 1 theme when few chunks; avoid forcing 2 when not applicable
        clusters = max(1, min(6, math.ceil(len(cluster_items) / 40)))
        labels, centroids = _kmeans(cluster_vectors, clusters)
        grouped = _select_top_by_cluster(cluster_items, cluster_vectors, labels, centroids, per_cluster=3)
        themes = []
        answer_blocks = []
        theme_idx = 0
        for group in grouped:
            if not group:
                continue
            summary_text = ""
            if llm_provider and llm_provider != "none":
                summary_text = (
                    rag.generate_answer(
                        f"Summarize the theme for: {query}",
                        group,
                        "summary",
                        {
                            "provider": llm_provider,
                            "model": llm_model,
                            "base_url": llm_base_url,
                            "api_key": llm_api_key,
                        },
                    ).get("answer_markdown", "") or ""
                ).strip()
            # Skip themes with no content or refusal-style response
            if not summary_text or _is_refusal(summary_text):
                continue
            theme_idx += 1
            citations = [
                {
                    "doc_id": item["doc_id"],
                    "filename": item["filename"],
                    "rel_path": item["rel_path"],
                    "page_num": item["page_num"],
                    "quote": item["chunk_text"],
                    "snippet": item.get("snippet", ""),
                }
                for item in group
            ]
            title = f"Theme {theme_idx}"
            themes.append({"title": title, "summary": summary_text, "citations": citations})
            answer_blocks.append(f"{title}: {summary_text}")
        if themes:
            return {
                "summary_type": "themes",
                "answer_markdown": "\n\n".join(answer_blocks),
                "themes": themes,
                "sources": [],
                "stats": {"page_pool": len(pages), "chunk_pool": len(items)},
            }
        # Fallback: all theme summaries were empty or refused; still return a single summary so user gets an answer
        if items and llm_provider and llm_provider != "none":
            top_items = ranked[:12]
            answer_markdown = rag.generate_answer(
                query,
                top_items,
                "summary",
                {
                    "provider": llm_provider,
                    "model": llm_model,
                    "base_url": llm_base_url,
                    "api_key": llm_api_key,
                },
            ).get("answer_markdown", "")
            return {
                "summary_type": "single",
                "answer_markdown": answer_markdown,
                "themes": [],
                "sources": top_items,
                "stats": {"page_pool": len(pages), "chunk_pool": len(items)},
            }

    top_items = ranked[:12]
    answer_markdown = ""
    if llm_provider and llm_provider != "none":
        answer_markdown = rag.generate_answer(
            query,
            top_items,
            "summary",
            {
                "provider": llm_provider,
                "model": llm_model,
                "base_url": llm_base_url,
                "api_key": llm_api_key,
            },
        ).get("answer_markdown", "")
    return {
        "summary_type": "single",
        "answer_markdown": answer_markdown,
        "themes": [],
        "sources": top_items,
        "stats": {"page_pool": len(pages), "chunk_pool": len(items)},
    }
