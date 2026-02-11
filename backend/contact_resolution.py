"""
Embedding-based entity resolution for email contacts.

Maps raw header display names (e.g. "; jeffrey epstein", "je vacation") to a
canonical name (e.g. "Jeffrey Epstein") by embedding all names and matching
each raw name to the nearest canonical name when similarity is above a threshold.
Uses the same embedding model as retrieval (e.g. FastEmbed) so no extra deps.
"""
import logging
import os

import numpy as np

from . import embeddings
from .utils import (
    extract_canonical_email_from_text,
    normalize_email_for_lookup,
)

logger = logging.getLogger("docqa")

# Minimum cosine similarity to treat a raw name as the same entity as a canonical name.
DEFAULT_SIMILARITY_THRESHOLD = 0.72


def resolve_raw_to_canonical(
    raw_names: list[str],
    canonical_names: list[str],
    *,
    threshold: float | None = None,
    device: str | None = None,
    engine: str | None = None,
) -> dict[str, str]:
    """
    For each raw name, if its embedding is close enough to a canonical name, map it to that canonical.
    Returns dict: raw_name -> canonical_name only for pairs above threshold.
    """
    if not raw_names or not canonical_names:
        return {}
    threshold = threshold or float(os.getenv("CONTACT_RESOLUTION_THRESHOLD", DEFAULT_SIMILARITY_THRESHOLD))
    # Dedupe and keep order
    raw_unique = list(dict.fromkeys(raw_names))
    canon_unique = list(dict.fromkeys(canonical_names))
    all_texts = raw_unique + canon_unique
    try:
        vecs = embeddings.embed_texts(
            all_texts,
            device=device or os.getenv("EMBEDDINGS_DEVICE", "cpu"),
            engine=engine or os.getenv("EMBEDDINGS_ENGINE", "fastembed"),
            use_worker=False,
        )
    except Exception as e:
        logger.warning("contact_resolution: embed failed %s", e)
        return {}
    n_raw = len(raw_unique)
    raw_vecs = vecs[:n_raw]
    canon_vecs = vecs[n_raw:]
    # Cosine similarity = dot product (embeddings are normalized)
    sims = np.dot(raw_vecs, canon_vecs.T)
    mapping: dict[str, str] = {}
    for i, raw in enumerate(raw_unique):
        j = int(np.argmax(sims[i]))
        sim = float(sims[i, j])
        if sim >= threshold:
            mapping[raw] = canon_unique[j]
    return mapping


def get_canonical_to_variant_terms(conn) -> dict[str, list[str]]:
    """
    For each canonical (alias) name, return all term strings that resolve to it:
    the canonical name, its alias emails, and every raw (display, addr) from headers
    that resolved to this canonical (email or embedding). Used at ask time to expand
    sender_terms/recipient_terms so LIKE matches all variants.
    """
    from . import db

    raw_contacts = db.get_raw_email_contacts(conn)
    alias_by_email = db.get_alias_email_to_name_map(conn)
    aliases = db.get_alias_names_and_emails(conn)
    canonical_names = [a["name"] for a in aliases]
    alias_emails_by_name: dict[str, list[str]] = {a["name"]: list(a["emails"]) for a in aliases}

    # Resolve each raw contact to a canonical name (email first, then embedding)
    raw_to_canonical: dict[str, str] = {}
    raw_names_for_embedding: list[str] = []
    for r in raw_contacts:
        name, addr = r["name"], r["email"]
        canonical_email = normalize_email_for_lookup(addr) if addr else extract_canonical_email_from_text(name)
        canonical_name = alias_by_email.get(canonical_email) if canonical_email else None
        if canonical_name:
            raw_to_canonical[name] = canonical_name
        else:
            raw_names_for_embedding.append(name)

    if raw_names_for_embedding and canonical_names:
        mapping = resolve_raw_to_canonical(raw_names_for_embedding, canonical_names)
        for raw, canon in mapping.items():
            raw_to_canonical[raw] = canon

    # Build canonical -> list of (raw name, raw email) that resolved to it
    canonical_to_raws: dict[str, list[tuple[str, str]]] = {}
    for r in raw_contacts:
        canon = raw_to_canonical.get(r["name"])
        if canon:
            if canon not in canonical_to_raws:
                canonical_to_raws[canon] = []
            canonical_to_raws[canon].append((r["name"], r["email"]))

    # For each canonical, terms = [canonical] + alias emails + all variant names/emails
    result: dict[str, list[str]] = {}
    for canon in set(alias_emails_by_name) | set(canonical_to_raws):
        terms: list[str] = [canon]
        terms.extend(alias_emails_by_name.get(canon, []))
        for name, addr in canonical_to_raws.get(canon, []):
            if name and name not in terms:
                terms.append(name)
            if addr and addr not in terms:
                terms.append(addr)
        result[canon] = terms
    return result
