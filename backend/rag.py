import os
import httpx

from .safety import redact_text


def build_prompt(
    query: str,
    excerpts: list[dict],
    answer_mode: str,
    low_evidence: bool = False,
    memory: list[dict] | None = None,
) -> list[dict]:
    intro = (
        "You are a cautious assistant. Use only the provided excerpts. "
        "Every factual claim must include a citation like [filename p.X]. "
        "If the answer is not supported, say: Not found in the indexed documents."
    )
    if low_evidence:
        intro = (
            intro
            + " Evidence is weak; if not supported, say: Not found in the indexed documents."
        )
    if answer_mode == "summary":
        intro = intro + " Provide a brief summary."
    if memory:
        intro = intro + " Use prior conversation context if referenced."
    context_lines = []
    for item in excerpts:
        label = f"{item['filename']} p.{item['page_num']}"
        context_lines.append(f"[{label}] {item['chunk_text']}")
    context = "\n\n".join(context_lines)
    memory_text = ""
    if memory:
        memory_blocks = []
        for idx, item in enumerate(memory, start=1):
            q = item.get("question", "")
            a = item.get("answer", "")
            memory_blocks.append(f"Turn {idx} Q: {q}\nTurn {idx} A: {a}")
        memory_text = "Previous conversation:\n" + "\n\n".join(memory_blocks) + "\n\n"
    user_text = f"{memory_text}Question: {query}\n\nExcerpts:\n{context}"
    return [
        {"role": "system", "content": intro},
        {"role": "user", "content": user_text},
    ]


def call_openai_compat(messages: list[dict], model: str, base_url: str, api_key: str | None) -> str:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 700,
    }
    with httpx.Client(timeout=300) as client:
        resp = client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


def call_ollama(messages: list[dict], model: str, base_url: str) -> str:
    url = base_url.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.2},
    }
    with httpx.Client(timeout=300) as client:
        resp = client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"]


def generate_answer(
    query: str, excerpts: list[dict], answer_mode: str, options: dict | None = None
) -> dict:
    options = options or {}
    provider = options.get("provider") or os.getenv("LLM_PROVIDER", "none")
    if provider == "none" or answer_mode in ("sources_only", "evidence_view"):
        return {"answer_markdown": "", "provider": "none"}
    messages = build_prompt(
        query,
        excerpts,
        answer_mode,
        low_evidence=bool(options.get("low_evidence")),
        memory=options.get("memory"),
    )
    if provider == "llama_cpp":
        model = options.get("model") or os.getenv("LLM_MODEL", "model")
        base_url = options.get("base_url") or os.getenv(
            "LLM_BASE_URL", "http://127.0.0.1:8080"
        )
        api_key = options.get("api_key") or os.getenv("LLM_API_KEY")
        text = call_openai_compat(messages, model, base_url, api_key)
    elif provider == "openai_compat":
        model = options.get("model") or os.getenv("LLM_MODEL", "model")
        base_url = options.get("base_url") or os.getenv(
            "LLM_BASE_URL", "https://api.openai.com"
        )
        api_key = options.get("api_key") or os.getenv("LLM_API_KEY")
        text = call_openai_compat(messages, model, base_url, api_key)
    elif provider == "ollama":
        model = options.get("model") or os.getenv("OLLAMA_MODEL", "llama3")
        base_url = options.get("base_url") or os.getenv(
            "OLLAMA_URL", "http://127.0.0.1:11434"
        )
        text = call_ollama(messages, model, base_url)
    else:
        text = ""
    return {"answer_markdown": redact_text(text), "provider": provider}
