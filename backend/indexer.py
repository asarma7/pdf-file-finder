import os
import shutil
import subprocess
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import fitz

from . import chunking, collections, db, embeddings
from .utils import ensure_data_dirs, has_tool, sha256_file, stable_ocr_name


MIN_TEXT_LEN = 20


def find_pdfs(root: str) -> list[str]:
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(".pdf"):
                paths.append(os.path.join(dirpath, name))
    return sorted(paths)


def extract_page_text_with_pdftotext(path: str, page_num: int) -> str:
    try:
        result = subprocess.run(
            [
                "pdftotext",
                "-f",
                str(page_num + 1),
                "-l",
                str(page_num + 1),
                "-layout",
                path,
                "-",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def extract_text_from_pdf(path: str) -> list[str]:
    doc = fitz.open(path)
    pages = []
    for page in doc:
        pages.append(page.get_text().strip())
    return pages


def ocr_pdf(input_path: str, ocr_path: str) -> bool:
    try:
        subprocess.run(
            ["ocrmypdf", "--skip-text", "--quiet", input_path, ocr_path],
            check=True,
            capture_output=True,
            text=True,
        )
        return True
    except Exception:
        return False


def process_pdf(
    path: str, mtime: float, existing_sha256: str | None, ocr_cache_dir: str
) -> dict:
    try:
        sha256 = sha256_file(path)
        if existing_sha256 and sha256 == existing_sha256:
            return {"status": "unchanged", "path": path, "sha256": sha256, "mtime": mtime}

        pages = extract_text_from_pdf(path)
        needs_ocr = any(len(text) < MIN_TEXT_LEN for text in pages)
        if needs_ocr and has_tool("ocrmypdf"):
            ocr_name = stable_ocr_name(path, mtime)
            ocr_path = os.path.join(ocr_cache_dir, ocr_name)
            if not os.path.exists(ocr_path):
                if not ocr_pdf(path, ocr_path):
                    ocr_path = ""
            if ocr_path and os.path.exists(ocr_path):
                pages = extract_text_from_pdf(ocr_path)

        if not has_tool("ocrmypdf") and has_tool("pdftotext"):
            for idx, text in enumerate(pages):
                if len(text) < MIN_TEXT_LEN:
                    pages[idx] = extract_page_text_with_pdftotext(path, idx)

        return {
            "status": "indexed",
            "path": path,
            "sha256": sha256,
            "mtime": mtime,
            "pages": pages,
        }
    except Exception as exc:
        return {"status": "error", "path": path, "error": str(exc)}


def index_collection(collection_id: str, *, reindex: bool = False) -> dict:
    ensure_data_dirs()
    collection = collections.get_collection_by_id(collection_id)
    if not collection:
        return {"indexed": 0, "skipped": 0, "errors": 1}

    root = collection["root_path"]
    db_path = collections.get_collection_db_path(collection_id)
    ocr_cache_dir = str(collections.get_collection_ocr_cache_dir(collection_id))
    faiss_path = collections.get_collection_faiss_path(collection_id)

    conn = db.get_connection(db_path)
    db.init_db(conn)
    if reindex:
        db.reset_db(conn)
        if faiss_path.exists():
            faiss_path.unlink()

    pdfs = find_pdfs(root)
    existing_rows = conn.execute("SELECT * FROM docs;").fetchall()
    existing = {row["rel_path"]: row for row in existing_rows}

    to_process = []
    for path in pdfs:
        try:
            mtime = os.path.getmtime(path)
            size = os.path.getsize(path)
        except OSError:
            continue
        rel_path = os.path.relpath(path, root)
        if (
            not reindex
            and rel_path in existing
            and existing[rel_path]["mtime"] == mtime
            and existing[rel_path]["size"] == size
        ):
            continue
        existing_row = existing.get(rel_path)
        existing_sha = existing_row["sha256"] if existing_row else None
        to_process.append((path, rel_path, mtime, size, existing_sha))

    indexed = 0
    skipped = len(pdfs) - len(to_process)
    errors = 0

    workers = max(1, (os.cpu_count() or 2) // 2)
    index = embeddings.load_index(faiss_path)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_pdf, path, mtime, sha256, ocr_cache_dir): (path, rel_path, mtime, size)
            for path, rel_path, mtime, size, sha256 in to_process
        }
        for future in as_completed(futures):
            result = future.result()
            status = result.get("status")
            if status == "unchanged":
                skipped += 1
                continue
            if status == "error":
                errors += 1
                continue
            if status == "indexed":
                path = result["path"]
                rel_path = os.path.relpath(path, root)
                filename = os.path.basename(path)
                mtime = result["mtime"]
                size = os.path.getsize(path)
                existing_row = existing.get(rel_path)
                if existing_row:
                    old_doc_id = existing_row["id"]
                    old_chunk_ids = db.get_chunk_ids_for_doc(conn, old_doc_id)
                    db.delete_doc_data(conn, old_doc_id)
                    if index is not None:
                        embeddings.remove_vectors(index, old_chunk_ids)
                doc_id = db.upsert_doc(
                    conn,
                    rel_path=rel_path,
                    filename=filename,
                    sha256=result["sha256"],
                    mtime=mtime,
                    size=size,
                    pages_count=len(result["pages"]),
                )
                db.insert_pages(conn, doc_id, result["pages"])
                chunks_for_insert = []
                for page_num, page_text in enumerate(result["pages"]):
                    for chunk_index, (start, end, chunk_text) in enumerate(
                        chunking.chunk_text(page_text)
                    ):
                        chunks_for_insert.append(
                            (doc_id, page_num, chunk_index, chunk_text, start, end)
                        )
                inserted = db.insert_chunks(conn, doc_id, chunks_for_insert)
                db.insert_chunks_fts(conn, doc_id)
                conn.commit()
                if inserted:
                    texts = [item[1] for item in inserted]
                    ids = [item[0] for item in inserted]
                    vectors = embeddings.embed_texts(texts, use_worker=False)
                    if index is None:
                        index = embeddings.build_index(vectors.shape[1])
                    embeddings.add_vectors(index, vectors, ids)
                    embeddings.save_index(index, faiss_path)
                    conn.executemany(
                        "INSERT OR REPLACE INTO chunk_vectors (chunk_id, faiss_row_id) VALUES (?, ?);",
                        [(cid, cid) for cid in ids],
                    )
                    conn.commit()
                indexed += 1

    return {"indexed": indexed, "skipped": skipped, "errors": errors}


def index_collection_from_zip(
    collection_id: str,
    zip_path: str,
    *,
    reindex: bool = False,
    append: bool = False,
) -> dict:
    collection = collections.get_collection_by_id(collection_id)
    files_dir = collections.get_collection_files_dir(collection_id)
    target_root = files_dir
    if append and collection and collection.get("root_path"):
        target_root = Path(collection["root_path"])
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for member in zip_ref.infolist():
            extracted_path = Path(target_root, member.filename).resolve()
            if not str(extracted_path).startswith(str(target_root.resolve())):
                continue
            zip_ref.extract(member, target_root)
    collections.update_collection_root(collection_id, str(target_root))
    return index_collection(collection_id, reindex=reindex)


def cleanup_path(path: str) -> None:
    try:
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
    except OSError:
        return
