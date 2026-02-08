import math
from typing import Iterable

import numpy as np

from . import embeddings, retrieval, rag


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
