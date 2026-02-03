# pdf-file-finder

Local document search + Q&A with FastAPI, SQLite FTS5, PyMuPDF, optional OCR, and hybrid retrieval.

## Quick Start (macOS)
1) Install dependencies:
```
brew install python ocrmypdf tesseract poppler
```
2) Create a virtualenv and install Python deps:
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
3) Unzip your PDFs (optional example dataset):
```
unzip "DataSet 8.zip" -d "DataSet 8"
```
Note: the example dataset ZIP is not included in the repo. If you do not have it, skip this step and use your own PDFs in step 4.
You can also download PDFs directly from DOJ press releases (see "Downloading DOJ documents" below).
4) Create a collection:
```
python scripts/create_collection.py --name "MyDocs" --root "/absolute/path/to/pdfs"
```
5) Index the collection:
```
python scripts/index_collection.py --collection "MyDocs"
```
6) Run the API server:
```
uvicorn backend.main:app --reload
```
7) Open:
```
http://127.0.0.1:8000
```

## No-code Upload Flow
1) Start the server:
```
uvicorn backend.main:app --reload
```
2) Open the web UI:
```
http://127.0.0.1:8000
```
3) Go to Collections tab.
4) Upload a `.zip` with PDFs and give it a name.
5) To append another ZIP to the same collection, check "Append to existing" and choose the collection.
6) Wait for the status to show "Ready", then search or ask.

## Downloading DOJ documents
1) Open the DOJ Office of Public Affairs press releases page:
```
https://www.justice.gov/opa/press-releases
```
2) Open a press release related to your topic.
3) Download any attached PDFs (often listed as "Attachments" or "Documents").
4) Put the PDFs into a local folder (for example, `~/Documents/doj_pdfs`).
5) Use that folder path as the `--root` when creating a collection.

## Capabilities
- Multi-collection support (each corpus is isolated).
- OCR fallback for scanned pages using `ocrmypdf` (if installed).
- Chunked indexing for precise citations and retrieval.
- Hybrid retrieval (keyword + semantic) with citations.
- Q&A modes: sources-only, strict Q&A, summary.
- Redaction of emails, phones, SSNs, and addresses in outputs.
- Safety filter for doxxing-style queries.

## How It Works (High Level)
1) PDFs are extracted page-by-page.
2) Pages are chunked into ~1000 character chunks.
3) Chunks are indexed in SQLite FTS5 and (optionally) embedded into FAISS.
4) Ask mode retrieves chunks, then the LLM synthesizes an answer with citations.

## Search Modes (Search + Ask)
- Keyword: SQLite FTS5 over chunks (exact/lexical).
  - Best for: names, direct phrases, exact terms.
- Semantic: FAISS vector search over embeddings.
  - Best for: paraphrases, vague questions, conceptual queries.
- Hybrid: merges keyword + semantic (RRF merge).
  - Best for: most natural language questions.

Examples:
- Keyword: `Spencer Kuvin` or `non-prosecution agreement`
- Semantic: `what did the investigation focus on?`
- Hybrid: `Which people are mentioned frequently in the case updates?`

## Answer Modes (Ask)
- Summary (default): short synthesis with citations.
  - Use for: high-level answers.
- Strict Q&A: direct answer with citations per claim.
  - Use for: precise, factual questions.
- Sources only: no generation, just excerpts.
  - Use for: manual review or when LLM is off.

## Ask Options (UI)
- Collection: which corpus to search.
- Mode: keyword / semantic / hybrid.
- Answer: summary / strict / sources only.
- Top K: number of chunks used for evidence.
- Redact outputs: mask emails/phones/SSNs/addresses in displayed text.
- CPU embeddings: force embeddings to run on CPU.
- Embeddings engine:
  - FastEmbed (default, most stable)
  - Sentence Transformers (Torch, heavier)
  - ONNX (mapped to FastEmbed currently)
- Worker process: runs query embeddings in a separate process to avoid crashes.
- LLM settings:
  - Provider: llama.cpp / Ollama / OpenAI-compatible / none
  - Model, Base URL, API key
- HF token: used to download embedding models if needed.

## LLM Providers
Set `LLM_PROVIDER` to one of:
- `none`
- `llama_cpp`
- `ollama`
- `openai_compat`

Environment variables:
- `LLM_BASE_URL` and `LLM_API_KEY` for OpenAI-compatible or llama.cpp servers
- `LLM_MODEL` for OpenAI-compatible or llama.cpp
- `OLLAMA_URL` and `OLLAMA_MODEL` for Ollama

## Embeddings Settings
To force CPU-only embeddings (avoids GPU OOM on some Macs):
```
export EMBEDDINGS_DEVICE=cpu
```

Embeddings engine (default fastembed):
```
export EMBEDDINGS_ENGINE=fastembed
```

Optional worker process for query embeddings:
```
export EMBEDDINGS_WORKER=1
```

## Model cache location (Hugging Face / embeddings)
Embedding models are downloaded and cached under `data/models/` by default. The folder is created automatically on first run.
If you want to pre-download models from Hugging Face, place them under `data/models/` (the app will read from that cache).

## Recommended Setup by Hardware
### Low RAM (<= 8 GB)
- Embeddings: CPU + FastEmbed + Worker
  ```
  export EMBEDDINGS_DEVICE=cpu
  export EMBEDDINGS_ENGINE=fastembed
  export EMBEDDINGS_WORKER=1
  ```
- LLM: none or external (avoid local models)

### Apple Silicon (M1/M2/M3) with 8–16 GB RAM
- Embeddings: CPU + FastEmbed + Worker
  ```
  export EMBEDDINGS_DEVICE=cpu
  export EMBEDDINGS_ENGINE=fastembed
  export EMBEDDINGS_WORKER=1
  ```
- LLM: llama.cpp or Ollama
  - llama.cpp: 7B–8B instruct, Q4 quant
  - Ollama: `llama3.1:8b` or `mistral:7b`

### Apple Silicon with 32–64 GB RAM
- Embeddings: CPU or MPS (GPU) if stable, FastEmbed or Sentence Transformers
  ```
  export EMBEDDINGS_ENGINE=fastembed
  ```
- LLM: llama.cpp 8B–13B Q4/Q5 (or Ollama)

### Intel Mac (8–16 GB RAM)
- Embeddings: CPU + FastEmbed + Worker
  ```
  export EMBEDDINGS_DEVICE=cpu
  export EMBEDDINGS_ENGINE=fastembed
  export EMBEDDINGS_WORKER=1
  ```
- LLM: Ollama with smaller models (7B–8B)

### External LLM (any hardware)
- Embeddings: keep local (FastEmbed CPU)
- LLM: OpenAI-compatible endpoint
  ```
  export LLM_PROVIDER=openai_compat
  export LLM_BASE_URL=https://api.openai.com
  export LLM_API_KEY=...
  ```

## PDF Viewer
If `pdfjs-dist` is present at `backend/static/pdfjs/web/viewer.html`, the app uses PDF.js with a pre-filled search term. Otherwise it falls back to the browser PDF viewer.

## CLI
```
python scripts/create_collection.py --name "MyDocs" --root "/absolute/path/to/pdfs"
python scripts/index_collection.py --collection "MyDocs"
python scripts/index_collection.py --collection "MyDocs" --reindex
python scripts/smoke_test.py --collection "MyDocs" --query "example question"
```

## Notes
- Indexing is incremental by default. Files with the same mtime and size are skipped.
- OCR output is cached under each collection folder.
- Each collection stores its own SQLite DB and FAISS index under `data/collections/<collection_id>/`.
- The `data/` indexes and any local `models/` files are generated locally and are not expected to be in the repo.


