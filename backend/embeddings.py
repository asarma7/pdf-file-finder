from pathlib import Path
import numpy as np
import faiss
import os
import logging
from concurrent.futures import ProcessPoolExecutor

from .utils import MODELS_DIR


MODEL_NAME = "BAAI/bge-small-en-v1.5"
logger = logging.getLogger("docqa")
_ENGINE = None
_ENGINE_NAME = None
_POOL = None


def _normalize_engine(name: str) -> str:
    name = name.lower().strip()
    if name in ("fastembed", "fast"):
        return "fastembed"
    if name in ("sentence_transformers", "st", "torch"):
        return "sentence_transformers"
    if name in ("onnx", "onnx_sentence_transformers"):
        return "onnx"
    return "fastembed"


def _normalize_device(device: str | None) -> str:
    if not device or device == "auto":
        return "cpu"
    return device


class FastEmbedEngine:
    def __init__(self, model_name: str):
        from fastembed import TextEmbedding

        self.model_name = model_name
        self._model = TextEmbedding(model_name, cache_dir=str(MODELS_DIR))

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = list(self._model.embed(texts))
        arr = np.array(vectors, dtype="float32")
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms


class SentenceTransformersEngine:
    def __init__(self, model_name: str, device: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            model_name, cache_folder=str(MODELS_DIR), device=device
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.array(vectors, dtype="float32")


def _get_engine(engine_name: str, device: str, hf_token: str | None):
    global _ENGINE, _ENGINE_NAME
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    if _ENGINE is None or _ENGINE_NAME != engine_name:
        logger.info("embeddings:engine=%s device=%s", engine_name, device)
        if engine_name == "fastembed":
            _ENGINE = FastEmbedEngine(MODEL_NAME)
        elif engine_name == "sentence_transformers":
            _ENGINE = SentenceTransformersEngine(MODEL_NAME, device)
        elif engine_name == "onnx":
            _ENGINE = FastEmbedEngine(MODEL_NAME)
        else:
            _ENGINE = FastEmbedEngine(MODEL_NAME)
        _ENGINE_NAME = engine_name
    return _ENGINE


def _encode_in_process(
    texts: list[str], engine_name: str, device: str, hf_token: str | None
) -> np.ndarray:
    engine = _get_engine(engine_name, device, hf_token)
    return engine.encode(texts)


def _get_pool() -> ProcessPoolExecutor:
    global _POOL
    if _POOL is None:
        _POOL = ProcessPoolExecutor(max_workers=1)
    return _POOL


def embed_texts(
    texts: list[str],
    device: str | None = None,
    hf_token: str | None = None,
    engine: str | None = None,
    use_worker: bool = False,
) -> np.ndarray:
    engine_name = _normalize_engine(engine or os.getenv("EMBEDDINGS_ENGINE", "fastembed"))
    desired = _normalize_device(device)
    logger.info("embeddings:encode count=%s engine=%s device=%s", len(texts), engine_name, desired)
    if use_worker:
        pool = _get_pool()
        return pool.submit(_encode_in_process, texts, engine_name, desired, hf_token).result()
    model = _get_engine(engine_name, desired, hf_token)
    return model.encode(texts)


def build_index(dim: int) -> faiss.IndexIDMap2:
    base = faiss.IndexFlatIP(dim)
    return faiss.IndexIDMap2(base)


def save_index(index: faiss.IndexIDMap2, path: Path) -> None:
    faiss.write_index(index, str(path))


def load_index(path: Path) -> faiss.IndexIDMap2 | None:
    if not path.exists():
        return None
    return faiss.read_index(str(path))


def add_vectors(index: faiss.IndexIDMap2, vectors: np.ndarray, ids: list[int]) -> None:
    ids_array = np.array(ids, dtype="int64")
    index.add_with_ids(vectors, ids_array)


def remove_vectors(index: faiss.IndexIDMap2, ids: list[int]) -> None:
    if not ids:
        return
    ids_array = np.array(ids, dtype="int64")
    index.remove_ids(ids_array)
