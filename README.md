# pdf-file-finder

Local document search + Q&A with FastAPI, SQLite FTS5, PyMuPDF, optional OCR, and hybrid retrieval. Built for RAG over the Epstein files.

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

## Working with the Recently Released Epstein Files

1) Download the recently released Jeffrey Epstein court documents (PDFs). These files are often available as a ZIP archive on reputable news or court websites, or may be provided by specific information repositories.

2) Unzip the PDF files into a folder on your computer, for example: `~/Documents/epstein_pdfs`.

3) Use the path to this folder as the `--root` argument when creating a collection. For example:
```
https://www.justice.gov/opa/press-releases
```

4) Continue with the indexing and search steps as described above to interactively search, review, and analyze the Epstein documents.

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
  - Hard subject gate: results must contain detected subject anchors (e.g., person/entity).
  - Descriptor anchors (verbs/attributes) are used for soft scoring, not gating.

Examples:
- Keyword: `Spencer Kuvin` or `non-prosecution agreement`
- Semantic: `what did the investigation focus on?`
- Hybrid: `Which people are mentioned frequently in the case updates?`

## Anchor LLM (Ask)
The Anchor LLM extracts:
- **Subject anchors** (main person/org/place/entity). Results without these are dropped.
- **Descriptor anchors** (topical modifiers). These are used to gently boost ranking.

How it works (for normal users):
1) Turn on **Anchor LLM** in the Ask tab.
2) Ask your question normally.
3) The system uses the same LLM you selected in the UI to extract anchors.
4) Results that don’t mention the subject are filtered out.

Example:
- Question: `Is Bill Cosby mentioned in the Epstein files?`
- Subject anchors: `Bill Cosby`
- Descriptor anchors: `mentioned`, `files` (often ignored as generic)
- Returned results will only include chunks that mention **Bill Cosby**.

Tips:
- If you get no results, lower your Top K or rephrase with a clearer subject.
- If the Anchor LLM is off, the system falls back to rule-based anchors.

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
- Anchor LLM: use your selected LLM to detect subject/descriptor anchors.
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
  - Conversation memory: last 3 Q/A turns are used for follow-up context (in-memory, resets on restart).
  - Batch continue: use “Continue (next 10)” to fetch the next batch of sources for the same question.

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
Hugging Face models are stored under `/pdf-file-finder/models`, while embedding models used for user queries are downloaded and cached under `data/models/` by default. Both folders are created automatically on first run.  
To pre-download Hugging Face models, place them in `/pdf-file-finder/models`.  
To pre-download embedding models, place them in `data/models/` (the app will read from these cache locations).

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
- If a collection was created in a different directory and file paths changed, set `DOCQA_ROOT_OVERRIDE` to the new root. The server will use it when serving files and update the stored root path.


