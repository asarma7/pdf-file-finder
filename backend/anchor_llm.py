import json
import logging
import os
import re
from typing import Any

from .rag import call_ollama, call_openai_compat

logger = logging.getLogger("docqa")


GENERIC_ANCHORS = {
    "mention",
    "mentions",
    "mentioned",
    "mentioning",
    "file",
    "files",
    "document",
    "documents",
    "dataset",
    "page",
    "pages",
    "pdf",
    "email",
    "emails",
}


def _extract_subject_only(text: str) -> dict[str, Any]:
    """Try multiple patterns for subject_anchors only (used by extract_subjects)."""
    payload = _extract_json(text)
    if payload.get("subject_anchors"):
        return payload
    subject_match = re.search(
        r"(?:subject_anchors?|subjects?|entities?)\s*[:=]\s*\[([^\]]*)\]",
        text,
        flags=re.IGNORECASE,
    )
    if subject_match:
        payload["subject_anchors"] = _split_anchor_items(subject_match.group(1))
        return payload
    subject_match = re.search(
        r"(?:subject_anchors?|subjects?|entities?)\s*[:=]\s*([^\n\.]+)",
        text,
        flags=re.IGNORECASE,
    )
    if subject_match:
        payload["subject_anchors"] = _split_anchor_items(subject_match.group(1).strip())
        return payload
    names = _extract_names_from_text(text)
    if names:
        payload["subject_anchors"] = names
    return payload


def _extract_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return _extract_anchor_lists(text)
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return _extract_anchor_lists(text)


def _extract_anchor_lists(text: str) -> dict[str, Any]:
    subject_match = re.search(r"subject_anchors\s*[:=]\s*\[([^\]]*)\]", text, flags=re.IGNORECASE)
    descriptor_match = re.search(
        r"descriptor_anchors\s*[:=]\s*\[([^\]]*)\]", text, flags=re.IGNORECASE
    )
    subject_items = _split_anchor_items(subject_match.group(1) if subject_match else "")
    descriptor_items = _split_anchor_items(descriptor_match.group(1) if descriptor_match else "")
    if not subject_items and not descriptor_items:
        subject_items = _extract_names_from_text(text)
    return {"subject_anchors": subject_items, "descriptor_anchors": descriptor_items}


def _split_anchor_items(raw: str) -> list[str]:
    if not raw:
        return []
    items: list[str] = []
    for part in re.split(r"[,;]|\s+and\s+", raw, flags=re.IGNORECASE):
        value = part.strip().strip("'\"")
        if value:
            items.append(value)
    return items


def _extract_names_from_text(text: str) -> list[str]:
    """Fallback: look for quoted strings or capitalized name-like phrases in LLM output."""
    out: list[str] = []
    quoted = re.findall(r'"([^"]+)"', text)
    for q in quoted:
        q = q.strip()
        if len(q) > 1 and q.lower() not in GENERIC_ANCHORS:
            out.append(q)
    if out:
        return out
    capitalized = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text)
    seen = set()
    for name in capitalized:
        n = name.strip().lower()
        if len(n) < 2 or n in GENERIC_ANCHORS or n in seen:
            continue
        seen.add(n)
        out.append(n)
    return out[:5]


def _normalize_anchor_list(values: list[str]) -> list[str]:
    cleaned: list[str] = []
    for raw in values:
        value = re.sub(r"[^A-Za-z0-9_ ]+", " ", raw).strip().lower()
        value = re.sub(r"\s+", " ", value).strip()
        if not value:
            continue
        if value in GENERIC_ANCHORS:
            continue
        cleaned.append(value)
    seen = set()
    deduped = []
    for value in cleaned:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def build_anchor_messages(query: str) -> list[dict]:
    system = (
        "You extract concise search anchors from a question. "
        "Return JSON only with keys subject_anchors and descriptor_anchors. "
        "subject_anchors should be the main person/org/place/entity names (max 3). "
        "descriptor_anchors should be topical modifiers (max 6). "
        "Do not include generic terms like mention, epstein, file(s), document(s), dataset, page(s). "
        "Return ONLY valid JSON. No code, no explanations, no Markdown."
    )
    example = (
        "Example JSON:\n"
        '{"subject_anchors":["bill cosby"],"descriptor_anchors":["island"]}'
    )
    user = f"Question: {query}\n{example}\nReturn JSON only."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_subject_messages(query: str) -> list[dict]:
    system = (
        "Extract the main subject entities (people, organizations, places) from the question. "
        "Reply with a single line of valid JSON: {\"subject_anchors\": [\"name1\", \"name2\"]}. "
        "Use 1-3 entity names. No explanations, no markdown, no code blocks."
    )
    example = '{"subject_anchors": ["entity1", "entity2"]}'
    user = f"Question: {query}\nReply with one JSON line: {example}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def extract_anchors(
    query: str,
    *,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> dict:
    provider = provider or os.getenv("ANCHOR_LLM_PROVIDER", "ollama")
    if provider == "llama_cpp":
        provider = "openai_compat"
    model = model or os.getenv("ANCHOR_LLM_MODEL", "llama3.1:8b")
    if provider == "openai_compat":
        base_url = base_url or os.getenv("ANCHOR_LLM_BASE_URL", "https://api.openai.com")
    else:
        base_url = base_url or os.getenv("ANCHOR_LLM_BASE_URL", "http://127.0.0.1:11434")
    api_key = api_key or os.getenv("ANCHOR_LLM_API_KEY", "")

    messages = build_anchor_messages(query)
    if provider == "ollama":
        text = call_ollama(messages, model, base_url)
    elif provider == "openai_compat":
        text = call_openai_compat(messages, model, base_url, api_key)
    else:
        return {"subject_anchors": [], "descriptor_anchors": [], "raw": "", "provider": provider}

    payload = _extract_json(text)
    subject = _normalize_anchor_list(payload.get("subject_anchors", []) or [])
    descriptor = _normalize_anchor_list(payload.get("descriptor_anchors", []) or [])
    return {
        "subject_anchors": subject,
        "descriptor_anchors": descriptor,
        "raw": text,
        "provider": provider,
        "model": model,
    }


def extract_subjects(
    query: str,
    *,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> dict:
    provider = provider or os.getenv("ANCHOR_LLM_PROVIDER", "ollama")
    if provider == "llama_cpp":
        provider = "openai_compat"
    model = model or os.getenv("ANCHOR_LLM_MODEL", "llama3.1:8b")
    if provider == "openai_compat":
        base_url = base_url or os.getenv("ANCHOR_LLM_BASE_URL", "https://api.openai.com")
    else:
        base_url = base_url or os.getenv("ANCHOR_LLM_BASE_URL", "http://127.0.0.1:11434")
    api_key = api_key or os.getenv("ANCHOR_LLM_API_KEY", "")

    messages = build_subject_messages(query)
    if provider == "ollama":
        text = call_ollama(messages, model, base_url)
    elif provider == "openai_compat":
        text = call_openai_compat(messages, model, base_url, api_key)
    else:
        return {"subject_anchors": [], "raw": "", "provider": provider}

    payload = _extract_subject_only(text)
    subject = _normalize_anchor_list(payload.get("subject_anchors", []) or [])
    logger.info(
        "anchor_llm.extract_subjects raw_len=%s raw_preview=%s subject_anchors=%s",
        len(text),
        (text[:300] + "..." if len(text) > 300 else text),
        subject,
    )
    return {
        "subject_anchors": subject,
        "raw": text,
        "provider": provider,
        "model": model,
    }
