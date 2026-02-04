import os
import threading
import logging
import re
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from . import anchor_llm, collections, count, db, embeddings, indexer, rag, retrieval, search, safety
from .utils import DATA_DIR, ensure_data_dirs


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


def _route_query(query: str, mode: str) -> dict:
    normalized = query.strip().lower()
    if mode and mode != "auto":
        return {"mode": mode}
    if not normalized:
        return {"mode": "ask"}
    if re.search(r"\b(how many|count|number of|frequency)\b", normalized):
        return {"mode": "count"}
    if re.search(r"\b(where is|show all|list all|find all|every instance)\b", normalized):
        return {"mode": "search"}
    return {"mode": "ask"}


@app.get("/collections")
def list_collections():
    return {"collections": collections.list_collections()}


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
