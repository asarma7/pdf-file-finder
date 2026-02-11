import os
import threading
import logging
import re
from pathlib import Path

import httpx
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from . import anchor_llm, collections, contact_resolution, count, db, embeddings, indexer, rag, retrieval, search, safety, summary
from .utils import DATA_DIR, MODELS_DIR, ensure_data_dirs, normalize_email_for_lookup


app = FastAPI(title="Document Q&A")
logger = logging.getLogger("docqa")
logging.basicConfig(level=logging.INFO)

BASE_DIR = DATA_DIR.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "backend" / "templates"))
app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "backend" / "static")),
    name="static",
)


UPLOAD_STATUS: dict[str, dict] = {}
UPLOAD_LOCK = threading.Lock()
MEMORY_LOCK = threading.Lock()


def _extract_topic_from_query(query: str) -> str | None:
    """If the query explicitly mentions a topic (e.g. 'regarding food', 'about travel'), return it.
    Avoids relying on the LLM to return the topic as second anchor (e.g. it may return a wrong person)."""
    if not query or len(query.strip()) < 4:
        return None
    q = query.strip().lower()
    # Match "regarding X", "about X", "related to X", "on the topic of X", "concerning X", "involving X"
    for pattern in (
        r"\bregarding\s+([a-z][a-z0-9\s]{0,40}?)(?:\s+[\.\?]|\s*$|\.)",
        r"\babout\s+([a-z][a-z0-9\s]{0,40}?)(?:\s+emails|\s+[\.\?]|\s*$|\.)",
        r"\brelated\s+to\s+([a-z][a-z0-9\s]{0,40}?)(?:\s+[\.\?]|\s*$|\.)",
        r"\b(?:on\s+the\s+topic\s+of|topic\s+of)\s+([a-z][a-z0-9\s]{0,40}?)(?:\s+[\.\?]|\s*$|\.)",
        r"\bconcerning\s+([a-z][a-z0-9\s]{0,40}?)(?:\s+[\.\?]|\s*$|\.)",
        r"\binvolving\s+([a-z][a-z0-9\s]{0,40}?)(?:\s+[\.\?]|\s*$|\.)",
    ):
        m = re.search(pattern, q, re.IGNORECASE)
        if m:
            topic = m.group(1).strip()
            if len(topic) >= 2 and len(topic) <= 50:
                return topic
    return None
MEMORY_STORE: dict[str, list[dict]] = {}
CACHE_LOCK = threading.Lock()
CACHE_STORE: dict[str, list[dict]] = {}


def _remember_turn(session_id: str, question: str, answer: str, sources: list[dict]) -> None:
    if not session_id:
        return
    with MEMORY_LOCK:
        turns = MEMORY_STORE.get(session_id, [])
        turns.append({"question": question, "answer": answer, "sources": sources})
        MEMORY_STORE[session_id] = turns[-3:]


def _load_memory(session_id: str) -> list[dict]:
    if not session_id:
        return []
    with MEMORY_LOCK:
        return list(MEMORY_STORE.get(session_id, []))


def _cache_key(session_id: str, query: str) -> str:
    return f"{session_id}:{query.strip().lower()}"


def _load_cache(session_id: str, query: str) -> list[dict]:
    if not session_id:
        return []
    key = _cache_key(session_id, query)
    with CACHE_LOCK:
        return list(CACHE_STORE.get(key, []))


def _append_cache(session_id: str, query: str, results: list[dict]) -> None:
    if not session_id or not results:
        return
    key = _cache_key(session_id, query)
    with CACHE_LOCK:
        existing = CACHE_STORE.get(key, [])
        existing_ids = {item.get("chunk_id") for item in existing}
        for item in results:
            chunk_id = item.get("chunk_id")
            if chunk_id is None or chunk_id in existing_ids:
                continue
            existing.append(item)
            existing_ids.add(chunk_id)
        CACHE_STORE[key] = existing


def _clear_cache(session_id: str, collection_id: str | None = None, query: str | None = None) -> None:
    if not session_id:
        return
    prefix = f"{session_id}:"
    query_key = query.strip().lower() if query else None
    with CACHE_LOCK:
        keys = list(CACHE_STORE.keys())
        for key in keys:
            if not key.startswith(prefix):
                continue
            if query_key and not key.endswith(f":{query_key}"):
                continue
            if collection_id:
                items = CACHE_STORE.get(key, [])
                if not any(item.get("collection_id") == collection_id for item in items):
                    continue
            CACHE_STORE.pop(key, None)


@app.on_event("startup")
def startup() -> None:
    ensure_data_dirs()
    collections.init_registry_db()


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/view", response_class=HTMLResponse)
def view(
    request: Request,
    collection_id: str = Query(...),
    doc_id: int = Query(...),
    page: int = Query(1),
    term: str = Query(""),
):
    collection = collections.get_collection_by_id(collection_id)
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")
    conn = db.get_connection(collections.get_collection_db_path(collection_id))
    doc = db.get_doc_by_id(conn, doc_id)
    conn.close()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return templates.TemplateResponse(
        "view.html",
        {
            "request": request,
            "collection_id": collection_id,
            "doc_id": doc_id,
            "page": page,
            "term": term,
        },
    )


@app.get("/file/{collection_id}/{doc_id}")
def file(collection_id: str, doc_id: int):
    collection = collections.get_collection_by_id(collection_id)
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")
    conn = db.get_connection(collections.get_collection_db_path(collection_id))
    doc = db.get_doc_by_id(conn, doc_id)
    conn.close()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    path = os.path.join(collection["root_path"], doc["rel_path"])
    if not os.path.exists(path):
        default_root = str(collections.get_collection_files_dir(collection_id))
        default_path = os.path.join(default_root, doc["rel_path"])
        if os.path.exists(default_path):
            collections.update_collection_root(collection_id, default_root)
            path = default_path
        else:
            override_root = os.getenv("DOCQA_ROOT_OVERRIDE", "").strip()
            if override_root:
                override_path = os.path.join(override_root, doc["rel_path"])
                if os.path.exists(override_path):
                    collections.update_collection_root(collection_id, override_root)
                    path = override_path
    return FileResponse(path, media_type="application/pdf", filename=doc["filename"])


@app.get("/search")
def search_endpoint(
    collection_id: str = Query(...),
    q: str = Query("", min_length=0),
    limit: int = Query(50, ge=1, le=200),
    case_sensitive: bool = Query(False),
    mode: str = Query("fts"),
    redact: bool = Query(False),
):
    query = q.strip()
    if not query:
        return {"query": q, "results": []}

    collection = collections.get_collection_by_id(collection_id)
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")
    conn = db.get_connection(collections.get_collection_db_path(collection_id))
    if mode == "regex":
        results = search.regex_search(
            conn,
            query,
            limit=limit,
            case_sensitive=case_sensitive,
            redact=redact,
        )
    else:
        results = search.fts_search(
            conn,
            query,
            limit=limit,
            case_sensitive=case_sensitive,
            redact=redact,
        )
    conn.close()
    return {"query": q, "results": results, "collection_id": collection_id}


class CreateCollectionRequest(BaseModel):
    name: str
    root_path: str


class RetrieveRequest(BaseModel):
    query: str
    collection_id: str
    mode: str = "hybrid"
    top_k: int = 10
    use_rerank: bool = False
    redact: bool = True
    embeddings_device: str = "auto"
    hf_token: str = ""
    embeddings_engine: str = "fastembed"
    embeddings_worker: bool = True
    anchor_llm_enabled: bool = False
    llm_provider: str = "none"
    llm_model: str = ""
    llm_base_url: str = ""
    llm_api_key: str = ""


class AskRequest(BaseModel):
    query: str
    collection_id: str
    mode: str = "hybrid"
    top_k: int = 10
    offset: int = 0
    use_rerank: bool = False
    answer_mode: str = "summary"
    redact: bool = True
    embeddings_device: str = "auto"
    hf_token: str = ""
    embeddings_engine: str = "fastembed"
    embeddings_worker: bool = True
    anchor_llm_enabled: bool = False
    llm_provider: str = "none"
    llm_model: str = ""
    llm_base_url: str = ""
    llm_api_key: str = ""
    session_id: str = ""


class QueryRequest(BaseModel):
    query: str
    collection_id: str
    mode: str = "auto"
    top_k: int = 10
    offset: int = 0
    answer_mode: str = "summary"
    redact: bool = True
    embeddings_device: str = "auto"
    hf_token: str = ""
    embeddings_engine: str = "fastembed"
    embeddings_worker: bool = True
    anchor_llm_enabled: bool = False
    llm_provider: str = "none"
    llm_model: str = ""
    llm_base_url: str = ""
    llm_api_key: str = ""
    session_id: str = ""
    aliases: list[str] = []
    sender: str = ""
    recipient: str = ""
    subject_contains: str = ""
    date_from: str = ""
    date_to: str = ""
    email_summary_mode: str = "auto"


class EmailAliasRequest(BaseModel):
    collection_id: str
    name: str
    emails: list[str]


def _route_query(query: str, mode: str) -> dict:
    normalized = query.strip().lower()
    if mode and mode != "auto":
        return {"mode": mode}
    if not normalized:
        return {"mode": "ask"}
    if re.search(r"\b(summary|overview|themes|worst|accusations|allegations)\b", normalized):
        return {"mode": "corpus_summary"}
    if re.search(r"\b(how many|count|number of|frequency)\b", normalized):
        return {"mode": "count"}
    if re.search(r"\b(where is|show all|list all|find all|every instance)\b", normalized):
        return {"mode": "search"}
    return {"mode": "ask"}


@app.get("/collections")
def list_collections():
    return {"collections": collections.list_collections()}


# Directories to scan for local LLM model files (e.g. .gguf for llama.cpp)
_LLM_MODEL_DIRS = (Path(DATA_DIR.parent) / "models", MODELS_DIR)


@app.get("/llm-models")
def list_llm_models(
    provider: str = Query(..., description="llama_cpp, ollama, or openai_compat"),
    base_url: str = Query(""),
):
    """Return list of model names/ids for the given LLM provider. Populates the Model dropdown."""
    base_url = (base_url or "").strip().rstrip("/")
    api_key = (os.getenv("LLM_API_KEY") or "").strip()
    models: list[dict] = []
    if provider == "llama_cpp":
        seen: set[str] = set()
        for dir_path in _LLM_MODEL_DIRS:
            if not dir_path.exists() or not dir_path.is_dir():
                continue
            for f in sorted(dir_path.iterdir()):
                if not f.is_file():
                    continue
                suf = f.suffix.lower()
                if suf == ".gguf" or f.name.endswith(".gguf.bin"):
                    name = f.stem if suf == ".gguf" else f.name
                    if name not in seen:
                        seen.add(name)
                        models.append({"value": name, "label": name})
        if not models:
            models = [
                {"value": "model", "label": "model (add .gguf files to project models/ or data/models/)"},
            ]
    elif provider == "ollama":
        url = (base_url or "http://127.0.0.1:11434") + "/api/tags"
        try:
            with httpx.Client(timeout=10) as client:
                resp = client.get(url)
                resp.raise_for_status()
                data = resp.json()
                for m in data.get("models") or []:
                    name = (m.get("name") or m.get("model", "")).strip()
                    if name:
                        models.append({"value": name, "label": name})
        except Exception as e:
            logger.info("llm-models:ollama list failed %s", e)
            models = [
                {"value": "llama3.1:8b", "label": "llama3.1:8b (Ollama not reachable)"},
            ]
    elif provider == "openai_compat":
        url = (base_url or "https://api.openai.com") + "/v1/models"
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        try:
            with httpx.Client(timeout=15) as client:
                resp = client.get(url, headers=headers or None)
                resp.raise_for_status()
                data = resp.json()
                for m in data.get("data") or []:
                    mid = (m.get("id") or m.get("model") or "").strip()
                    if mid and not mid.startswith("embedding"):
                        models.append({"value": mid, "label": mid})
        except Exception as e:
            logger.info("llm-models:openai_compat list failed %s", e)
            models = [
                {"value": "gpt-4o-mini", "label": "gpt-4o-mini (list failed)"},
            ]
    else:
        return {"models": []}
    return {"models": models}


@app.post("/collections")
def create_collection(request: CreateCollectionRequest):
    item = collections.create_collection(request.name, request.root_path)
    return {"collection": item}


@app.post("/collections/{collection_id}/reindex")
def reindex_collection(collection_id: str):
    stats = indexer.index_collection(collection_id, reindex=True)
    return {"collection_id": collection_id, "stats": stats}


@app.post("/collections/{collection_id}/index")
def index_collection(collection_id: str):
    stats = indexer.index_collection(collection_id, reindex=False)
    return {"collection_id": collection_id, "stats": stats}


@app.post("/collections/upload")
def upload_collection(
    background_tasks: BackgroundTasks,
    name: str = Query(""),
    collection_id: str = Query(""),
    append: bool = Query(False),
    file: UploadFile = File(...),
):
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Please upload a .zip file")
    ensure_data_dirs()
    if append and not collection_id:
        raise HTTPException(status_code=400, detail="collection_id is required for append")
    if append:
        collection = collections.get_collection_by_id(collection_id)
        if not collection:
            raise HTTPException(status_code=404, detail="Collection not found")
    else:
        if not name:
            raise HTTPException(status_code=400, detail="name is required")
        collection = collections.create_collection(name, "")
    upload_id = os.urandom(6).hex()
    zip_path = str(DATA_DIR / f"upload_{upload_id}.zip")
    with open(zip_path, "wb") as f:
        f.write(file.file.read())
    with UPLOAD_LOCK:
        UPLOAD_STATUS[upload_id] = {"status": "queued", "collection_id": collection["id"]}
    background_tasks.add_task(
        _index_collection_zip_background, zip_path, upload_id, collection["id"], append
    )
    return {"upload_id": upload_id, "collection_id": collection["id"]}


@app.get("/upload/status")
def upload_status(upload_id: str):
    with UPLOAD_LOCK:
        status = UPLOAD_STATUS.get(upload_id)
    if not status:
        raise HTTPException(status_code=404, detail="Upload not found")
    return {"upload_id": upload_id, **status}


def _index_collection_zip_background(
    zip_path: str, upload_id: str, collection_id: str, append: bool
) -> None:
    with UPLOAD_LOCK:
        UPLOAD_STATUS[upload_id] = {"status": "processing", "collection_id": collection_id}
    try:
        stats = indexer.index_collection_from_zip(collection_id, zip_path, append=append)
        with UPLOAD_LOCK:
            UPLOAD_STATUS[upload_id] = {
                "status": "ready",
                "stats": stats,
                "collection_id": collection_id,
            }
    except Exception as exc:
        with UPLOAD_LOCK:
            UPLOAD_STATUS[upload_id] = {
                "status": "error",
                "error": str(exc),
                "collection_id": collection_id,
            }
    finally:
        indexer.cleanup_path(zip_path)


@app.post("/retrieve")
def retrieve_endpoint(request: RetrieveRequest, debug: bool = Query(False)):
    logger.info(
        "retrieve:start mode=%s top_k=%s collection_id=%s debug=%s",
        request.mode,
        request.top_k,
        request.collection_id,
        debug,
    )
    if safety.contains_disallowed_query(request.query):
        return {
            "query": request.query,
            "results": [],
            "refusal": "Request blocked by safety policy.",
        }
    collection = collections.get_collection_by_id(request.collection_id)
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")
    conn = db.get_connection(collections.get_collection_db_path(request.collection_id))
    db.init_db(conn)
    db.init_db(conn)
    index = embeddings.load_index(collections.get_collection_faiss_path(request.collection_id))
    pool_size = max(1, request.top_k) + max(0, request.offset)
    results, debug_info = retrieval.retrieve(
        conn,
        request.query,
        request.mode,
        pool_size,
        index,
        request.redact,
        request.embeddings_device,
        request.hf_token or None,
        request.embeddings_engine,
        request.embeddings_worker,
        request.anchor_llm_enabled,
        request.llm_provider or None,
        request.llm_model or None,
        request.llm_base_url or None,
        request.llm_api_key or None,
        debug=debug,
    )
    results = results[: request.top_k]
    low_evidence = len(results) < 2
    logger.info("retrieve:done results=%s low_evidence=%s", len(results), low_evidence)
    conn.close()
    payload = {
        "query": request.query,
        "results": results,
        "collection_id": request.collection_id,
        "low_evidence": low_evidence,
    }
    if debug:
        payload["debug_info"] = debug_info
    return payload


@app.post("/ask")
def ask_endpoint(request: AskRequest, debug: bool = Query(False)):
    logger.info(
        "ask:start mode=%s top_k=%s answer_mode=%s collection_id=%s",
        request.mode,
        request.top_k,
        request.answer_mode,
        request.collection_id,
    )
    if safety.contains_disallowed_query(request.query):
        return {
            "query": request.query,
            "answer_markdown": "Request blocked by safety policy.",
            "sources": [],
            "citations": [],
        }
    collection = collections.get_collection_by_id(request.collection_id)
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")
    conn = db.get_connection(collections.get_collection_db_path(request.collection_id))
    index = embeddings.load_index(collections.get_collection_faiss_path(request.collection_id))
    start = max(0, request.offset)
    end = start + max(1, request.top_k)
    cached = _load_cache(request.session_id, request.query)
    if len(cached) < end:
        pool_size = max(1, request.top_k) + len(cached)
        results, debug_info = retrieval.retrieve(
            conn,
            request.query,
            request.mode,
            pool_size,
            index,
            request.redact,
            request.embeddings_device,
            request.hf_token or None,
            request.embeddings_engine,
            request.embeddings_worker,
            request.anchor_llm_enabled,
            request.llm_provider or None,
            request.llm_model or None,
            request.llm_base_url or None,
            request.llm_api_key or None,
            debug=debug,
        )
        for item in results:
            item["collection_id"] = request.collection_id
        _append_cache(request.session_id, request.query, results)
        cached = _load_cache(request.session_id, request.query)
    conn.close()
    results = cached[start:end]
    low_evidence = len(results) < 2
    logger.info("ask:retrieved sources=%s low_evidence=%s", len(results), low_evidence)
    citations = [
        {
            "doc_id": item["doc_id"],
            "filename": item["filename"],
            "rel_path": item["rel_path"],
            "page_num": item["page_num"],
            "quote": item["chunk_text"],
        }
        for item in results
    ]
    memory = _load_memory(request.session_id)
    if request.answer_mode == "evidence_view":
        answer = {"answer_markdown": "", "provider": "none"}
    else:
        answer = rag.generate_answer(
            request.query,
            results,
            request.answer_mode,
            {
                "provider": request.llm_provider,
                "model": request.llm_model,
                "base_url": request.llm_base_url,
                "api_key": request.llm_api_key,
                "low_evidence": low_evidence,
                "memory": memory,
            },
        )
    _remember_turn(request.session_id, request.query, answer.get("answer_markdown", ""), results)
    logger.info("ask:llm provider=%s", answer.get("provider", "none"))
    payload = {
        "query": request.query,
        "answer_markdown": answer.get("answer_markdown", ""),
        "sources": results,
        "citations": citations,
        "provider": answer.get("provider", "none"),
        "low_evidence": low_evidence,
        "offset": request.offset,
        "next_offset": end,
    }
    if debug:
        payload["retrieve_debug"] = debug_info
    return payload


@app.post("/query")
def query_endpoint(request: QueryRequest, debug: bool = Query(False)):
    logger.info("query:start mode=%s top_k=%s collection_id=%s", request.mode, request.top_k, request.collection_id)
    if safety.contains_disallowed_query(request.query):
        return {
            "query": request.query,
            "mode_used": "blocked",
            "answer_markdown": "Request blocked by safety policy.",
            "sources": [],
            "stats": {},
            "low_evidence": True,
        }
    collection = collections.get_collection_by_id(request.collection_id)
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")
    route = _route_query(request.query, request.mode)
    mode_used = route["mode"]
    conn = db.get_connection(collections.get_collection_db_path(request.collection_id))

    if mode_used == "count":
        term = request.query
        aliases = list(request.aliases or [])
        if request.anchor_llm_enabled:
            anchor_info = anchor_llm.extract_subjects(
                request.query,
                provider=request.llm_provider or None,
                model=request.llm_model or None,
                base_url=request.llm_base_url or None,
                api_key=request.llm_api_key or None,
            )
            subjects = anchor_info.get("subject_anchors", []) or []
            if subjects:
                term = subjects[0]
                aliases = subjects[1:] + aliases
        stats = count.count_mentions(
            conn,
            term,
            aliases=aliases,
            redact=request.redact,
            limit=request.top_k,
            debug=debug,
        )
        conn.close()
        payload = {
            "query": request.query,
            "mode_used": "count",
            "answer_markdown": "",
            "sources": stats.get("contexts", []),
            "stats": {
                "total_hits": stats.get("total_hits", 0),
                "unique_pages": stats.get("unique_pages", 0),
                "unique_docs": stats.get("unique_docs", 0),
            },
            "low_evidence": stats.get("total_hits", 0) == 0,
        }
        if debug:
            payload["debug_info"] = stats.get("debug", {})
            payload["debug_info"]["count_term"] = term
            payload["debug_info"]["count_aliases"] = aliases
        return payload

    if mode_used == "corpus_summary":
        index = embeddings.load_index(collections.get_collection_faiss_path(request.collection_id))
        summary_payload = summary.corpus_summary(
            query=request.query,
            conn=conn,
            index=index,
            top_k=request.top_k,
            embeddings_device=request.embeddings_device,
            hf_token=request.hf_token or None,
            embeddings_engine=request.embeddings_engine,
            embeddings_worker=request.embeddings_worker,
            llm_provider=request.llm_provider,
            llm_model=request.llm_model,
            llm_base_url=request.llm_base_url,
            llm_api_key=request.llm_api_key,
        )
        conn.close()
        return {
            "query": request.query,
            "mode_used": "corpus_summary",
            "answer_markdown": summary_payload.get("answer_markdown", ""),
            "themes": summary_payload.get("themes", []),
            "stats": summary_payload.get("stats", {}),
            "low_evidence": not summary_payload.get("themes"),
        }

    if mode_used == "search":
        search_query = request.query
        if request.anchor_llm_enabled:
            anchor_info = anchor_llm.extract_subjects(
                request.query,
                provider=request.llm_provider or None,
                model=request.llm_model or None,
                base_url=request.llm_base_url or None,
                api_key=request.llm_api_key or None,
            )
            subjects = anchor_info.get("subject_anchors", []) or []
            if subjects:
                search_query = subjects[0]
        results = search.fts_search(
            conn,
            search_query,
            limit=request.top_k,
            case_sensitive=False,
            redact=request.redact,
        )
        conn.close()
        return {
            "query": request.query,
            "mode_used": "search",
            "answer_markdown": "",
            "sources": results,
            "stats": {"result_count": len(results), "search_query": search_query},
            "low_evidence": len(results) < 2,
        }

    if mode_used == "email_filter":
        sender = request.sender.strip().lower()
        recipient = request.recipient.strip().lower()
        inferred_subjects = []
        anchor_info = None
        if not sender or not recipient:
            try:
                anchor_info = anchor_llm.extract_subjects(
                    request.query,
                    provider=request.llm_provider or None,
                    model=request.llm_model or None,
                    base_url=request.llm_base_url or None,
                    api_key=request.llm_api_key or None,
                )
                inferred_subjects = anchor_info.get("subject_anchors", []) or []
                logger.info(
                    "email_filter:anchor raw_len=%s subject_anchors=%s inferred=%s",
                    len(anchor_info.get("raw") or ""),
                    anchor_info.get("subject_anchors"),
                    inferred_subjects,
                )
            except Exception as e:
                logger.info("email_filter:anchor extract error %s", e)
                inferred_subjects = []
        inferred_topic: str | None = None
        query_topic = _extract_topic_from_query(request.query or "")
        if not sender and inferred_subjects:
            sender = inferred_subjects[0]
        if not recipient and len(inferred_subjects) > 1:
            # Prefer topic extracted from query ("regarding X", "about X") so we don't use a wrong second person from the LLM
            if query_topic:
                inferred_topic = query_topic
                recipient = None
            else:
                second_anchor = (inferred_subjects[1] or "").strip().lower()
                is_second_person = (
                    "@" in second_anchor
                    or bool(db.get_alias_emails_for_term(conn, second_anchor))
                )
                if is_second_person:
                    recipient = second_anchor
                else:
                    inferred_topic = second_anchor or None
                    recipient = None
        if query_topic and inferred_topic is None:
            inferred_topic = query_topic
        # Never use a topic (e.g. "food") as recipient; treat as topic-only so we match all to/from the person
        if recipient and inferred_topic and recipient == inferred_topic:
            recipient = None
        # With one inferred entity (or topic as second): match both FROM and TO/CC unless user said only "from" or only "to"
        if not recipient and sender:
            q = (request.query or "").lower()
            from_only = "emails from " in q and " to " not in q
            to_only = "emails to " in q and " from " not in q
            if not from_only and not to_only:
                recipient = sender
        logger.info(
            "email_filter:resolved sender=%s recipient=%s inferred_topic=%s",
            sender or "",
            recipient or "",
            inferred_topic or "",
        )
        sender_terms = []
        recipient_terms = []
        if sender:
            sender_terms.append(sender)
            sender_terms.extend(db.get_alias_emails_for_term(conn, sender))
        if recipient:
            recipient_terms.append(recipient)
            recipient_terms.extend(db.get_alias_emails_for_term(conn, recipient))
        strict_direction = bool(sender and recipient)
        # Use user-provided subject filter only; don't force topic into subject (e.g. "food" would exclude subject "Jerky")
        subject_contains = request.subject_contains.strip().lower() or None
        date_from = request.date_from.strip() or None
        date_to = request.date_to.strip() or None
        header_count = conn.execute("SELECT COUNT(*) FROM email_headers;").fetchone()[0]
        logger.info(
            "email_filter:sender_terms=%s recipient_terms=%s email_headers_rows=%s",
            sender_terms,
            recipient_terms,
            header_count,
        )
        primary_rows = []
        broad_rows = []
        if sender and recipient:
            primary_rows = db.query_email_headers(
                conn,
                sender_terms=sender_terms,
                recipient_terms=recipient_terms,
                strict_direction=True,
                subject_contains=subject_contains,
                date_from=date_from,
                date_to=date_to,
            )
            sender_rows = db.query_email_headers(
                conn,
                sender_terms=sender_terms,
                recipient_terms=[],
                strict_direction=False,
                subject_contains=subject_contains,
                date_from=date_from,
                date_to=date_to,
            )
            recipient_rows = db.query_email_headers(
                conn,
                sender_terms=[],
                recipient_terms=recipient_terms,
                strict_direction=False,
                subject_contains=subject_contains,
                date_from=date_from,
                date_to=date_to,
            )
            broad_rows = sender_rows + recipient_rows
        else:
            broad_rows = db.query_email_headers(
                conn,
                sender_terms=sender_terms,
                recipient_terms=recipient_terms,
                strict_direction=strict_direction,
                subject_contains=subject_contains,
                date_from=date_from,
                date_to=date_to,
            )
        email_rows = list({row["id"]: row for row in (primary_rows + broad_rows)}.values())
        primary_pages = {(row["doc_id"], row["page_num"]) for row in primary_rows}
        allowed_pages = list({(row["doc_id"], row["page_num"]) for row in email_rows})
        logger.info(
            "email_filter:matched emails=%s pages=%s sender=%s recipient=%s",
            len(email_rows),
            len(allowed_pages),
            request.sender,
            request.recipient,
        )
        resolved_emails = sorted(
            {
                email
                for row in email_rows
                for field in (row["from_addr"], row["to_addr"], row["cc_addr"])
                for email in (field or "").split(";")
                if email
            }
        )
        if debug:
            logger.info("email_filter:resolved_emails %s", ",".join(resolved_emails))
        email_mode = request.email_summary_mode or "auto"
        if email_mode == "auto":
            intent = anchor_llm.classify_email_intent(
                request.query,
                provider=request.llm_provider or None,
                model=request.llm_model or None,
                base_url=request.llm_base_url or None,
                api_key=request.llm_api_key or None,
            )
            use_summary = intent == "summary_thematic"
        else:
            use_summary = email_mode == "summary"
        if use_summary:
            summary_terms = [t for t in [sender, recipient] if t]
            if inferred_topic and inferred_topic not in summary_terms:
                summary_terms.append(inferred_topic)
            summary_payload = summary.email_filtered_summary(
                query=request.query,
                conn=conn,
                allowed_pages=allowed_pages,
                priority_pages=list(primary_pages),
                secondary_terms=summary_terms,
                embeddings_device=request.embeddings_device,
                hf_token=request.hf_token or None,
                embeddings_engine=request.embeddings_engine,
                embeddings_worker=request.embeddings_worker,
                llm_provider=request.llm_provider,
                llm_model=request.llm_model,
                llm_base_url=request.llm_base_url,
                llm_api_key=request.llm_api_key,
            )
            logger.info(
                "email_filter:summary type=%s themes=%s answer_len=%s pool_pages=%s pool_chunks=%s",
                summary_payload.get("summary_type"),
                len(summary_payload.get("themes") or []),
                len(summary_payload.get("answer_markdown") or ""),
                (summary_payload.get("stats") or {}).get("page_pool"),
                (summary_payload.get("stats") or {}).get("chunk_pool"),
            )
            conn.close()
            payload = {
                "query": request.query,
                "mode_used": "email_filter",
                "email_mode_used": "summary",
                "summary_type": summary_payload.get("summary_type"),
                "answer_markdown": summary_payload.get("answer_markdown", ""),
                "themes": summary_payload.get("themes", []),
                "sources": summary_payload.get("sources", []),
                "stats": {
                    "matched_emails": len(email_rows),
                    "matched_pages": len(allowed_pages),
                    "resolved_emails": resolved_emails,
                    "sender": request.sender,
                    "recipient": request.recipient,
                    **(summary_payload.get("stats") or {}),
                },
                "low_evidence": not summary_payload.get("answer_markdown") and not summary_payload.get("themes"),
            }
            if debug:
                raw_preview = (anchor_info or {}).get("raw") or ""
                payload["debug_info"] = {
                    "email_mode_used": "summary",
                    "summary_type": summary_payload.get("summary_type"),
                    "page_pool": (summary_payload.get("stats") or {}).get("page_pool"),
                    "chunk_pool": (summary_payload.get("stats") or {}).get("chunk_pool"),
                    "answer_len": len(summary_payload.get("answer_markdown") or ""),
                    "theme_count": len(summary_payload.get("themes") or []),
                    "matched_emails": len(email_rows),
                    "matched_pages": len(allowed_pages),
                    "inferred_subjects": inferred_subjects,
                    "subject_anchors_raw_preview": raw_preview[:500] if raw_preview else None,
                    "sender": sender,
                    "recipient": recipient,
                }
            return payload

        results = search.fts_search_scoped(
            conn,
            request.query,
            allowed_pages=allowed_pages,
            limit=request.top_k,
            redact=request.redact,
        )
        if sender and recipient and len(results) < request.top_k:
            secondary_pages = [page for page in allowed_pages if page not in primary_pages]
            fallback = search.scoped_term_search(
                conn,
                allowed_pages=secondary_pages,
                terms=[sender, recipient],
                limit=request.top_k - len(results),
                redact=request.redact,
            )
            results.extend(fallback)
        answer_markdown = ""
        if request.answer_mode not in ("sources_only", "evidence_view") and request.llm_provider != "none":
            answer_markdown = rag.generate_answer(
                request.query,
                results,
                request.answer_mode,
                {
                    "provider": request.llm_provider,
                    "model": request.llm_model,
                    "base_url": request.llm_base_url,
                    "api_key": request.llm_api_key,
                },
            ).get("answer_markdown", "")
        conn.close()
        payload = {
            "query": request.query,
            "mode_used": "email_filter",
            "email_mode_used": "search",
            "answer_markdown": answer_markdown,
            "sources": results,
            "stats": {
                "matched_emails": len(email_rows),
                "matched_pages": len(allowed_pages),
                "resolved_emails": resolved_emails,
                "sender": request.sender,
                "recipient": request.recipient,
            },
            "low_evidence": len(results) < 2,
        }
        if debug and anchor_info is not None:
            raw_preview = (anchor_info or {}).get("raw") or ""
            payload["debug_info"] = {
                "email_mode_used": "search",
                "inferred_subjects": inferred_subjects,
                "subject_anchors_raw_preview": raw_preview[:500] if raw_preview else None,
                "sender": sender,
                "recipient": recipient,
            }
        return payload

    index = embeddings.load_index(collections.get_collection_faiss_path(request.collection_id))
    start = max(0, request.offset)
    end = start + max(1, request.top_k)
    cached = _load_cache(request.session_id, request.query)
    if len(cached) < end:
        pool_size = max(1, request.top_k) + len(cached)
        results, debug_info = retrieval.retrieve(
            conn,
            request.query,
            request.mode if request.mode != "auto" else "hybrid",
            pool_size,
            index,
            request.redact,
            request.embeddings_device,
            request.hf_token or None,
            request.embeddings_engine,
            request.embeddings_worker,
            request.anchor_llm_enabled,
            request.llm_provider or None,
            request.llm_model or None,
            request.llm_base_url or None,
            request.llm_api_key or None,
            debug=debug,
        )
        for item in results:
            item["collection_id"] = request.collection_id
        _append_cache(request.session_id, request.query, results)
        cached = _load_cache(request.session_id, request.query)
    conn.close()
    results = cached[start:end]
    low_evidence = len(results) < 2
    citations = [
        {
            "doc_id": item["doc_id"],
            "filename": item["filename"],
            "rel_path": item["rel_path"],
            "page_num": item["page_num"],
            "quote": item["chunk_text"],
        }
        for item in results
    ]
    memory = _load_memory(request.session_id)
    answer = rag.generate_answer(
        request.query,
        results,
        request.answer_mode,
        {
            "provider": request.llm_provider,
            "model": request.llm_model,
            "base_url": request.llm_base_url,
            "api_key": request.llm_api_key,
            "low_evidence": low_evidence,
            "memory": memory,
        },
    )
    _remember_turn(request.session_id, request.query, answer.get("answer_markdown", ""), results)
    payload = {
        "query": request.query,
        "mode_used": "ask",
        "answer_markdown": answer.get("answer_markdown", ""),
        "sources": results,
        "citations": citations,
        "provider": answer.get("provider", "none"),
        "low_evidence": low_evidence,
        "offset": request.offset,
        "next_offset": end,
        "stats": {},
    }
    if debug:
        payload["retrieve_debug"] = debug_info
    return payload


@app.get("/email/known")
def get_known_emails(collection_id: str = Query(..., description="Collection ID")):
    if not collection_id:
        raise HTTPException(status_code=400, detail="collection_id is required")
    db_path = collections.get_collection_db_path(collection_id)
    if not db_path or not db_path.exists():
        return {"contacts": [], "aliases": []}
    conn = db.get_connection(db_path)
    db.init_db(conn)
    contacts = db.get_known_email_contacts(conn)
    aliases = db.get_alias_names_and_emails(conn)
    conn.close()
    canonical_names = list(dict.fromkeys(a["name"] for a in aliases))
    raw_contacts = [c for c in contacts if c["name"] not in set(canonical_names)]
    if raw_contacts and canonical_names:
        raw_names = [c["name"] for c in raw_contacts]
        mapping = contact_resolution.resolve_raw_to_canonical(raw_names, canonical_names)
        if mapping:
            for c in contacts:
                c["name"] = mapping.get(c["name"], c["name"])
            seen: set[tuple[str, str]] = set()
            deduped = []
            for c in contacts:
                key = (c["name"], c["email"])
                if key not in seen:
                    seen.add(key)
                    deduped.append(c)
            deduped.sort(key=lambda x: (x["name"].lower(), x["email"].lower()))
            contacts = deduped
    return {"contacts": contacts, "aliases": aliases}


@app.post("/email/aliases")
def add_email_aliases(request: EmailAliasRequest):
    if not request.collection_id or not request.name or not request.emails:
        raise HTTPException(status_code=400, detail="collection_id, name, and emails are required")
    conn = db.get_connection(collections.get_collection_db_path(request.collection_id))
    db.init_db(conn)
    added = db.add_email_aliases(conn, request.name, request.emails)
    conn.commit()
    conn.close()
    return {"name": request.name, "added": added}