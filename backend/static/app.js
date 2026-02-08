const queryInput = document.getElementById("query");
const searchBtn = document.getElementById("searchBtn");
const phraseToggle = document.getElementById("phrase");
const regexToggle = document.getElementById("regex");
const caseToggle = document.getElementById("case");
const redactToggle = document.getElementById("redact");
const resultsList = document.getElementById("resultsList");
const statusEl = document.getElementById("status");
const viewerFrame = document.getElementById("viewerFrame");
const zipInput = document.getElementById("zipInput");
const uploadBtn = document.getElementById("uploadBtn");
const uploadStatus = document.getElementById("uploadStatus");
const collectionSelect = document.getElementById("collectionSelect");
const collectionSelectAsk = document.getElementById("collectionSelectAsk");
const collectionsList = document.getElementById("collectionsList");
const collectionName = document.getElementById("collectionName");
const collectionRoot = document.getElementById("collectionRoot");
const createCollectionBtn = document.getElementById("createCollectionBtn");
const refreshCollectionsBtn = document.getElementById("refreshCollectionsBtn");
const zipCollectionName = document.getElementById("zipCollectionName");
const appendToggle = document.getElementById("appendToggle");
const appendCollectionSelect = document.getElementById("appendCollectionSelect");
const askQuery = document.getElementById("askQuery");
const askBtn = document.getElementById("askBtn");
const askMode = document.getElementById("askMode");
const answerMode = document.getElementById("answerMode");
const askTopK = document.getElementById("askTopK");
const askRedact = document.getElementById("askRedact");
const askDebug = document.getElementById("askDebug");
const anchorLLM = document.getElementById("anchorLLM");
const evidenceView = document.getElementById("evidenceView");
const askStatus = document.getElementById("askStatus");
const answerBox = document.getElementById("answerBox");
const sourcesList = document.getElementById("sourcesList");
const sourcesTitle = document.getElementById("sourcesTitle");
const askDebugPanel = document.getElementById("askDebugPanel");
const askDebugInfo = document.getElementById("askDebugInfo");
const cpuEmbeddings = document.getElementById("cpuEmbeddings");
const embeddingsEngine = document.getElementById("embeddingsEngine");
const embeddingsWorker = document.getElementById("embeddingsWorker");
const llmProvider = document.getElementById("llmProvider");
const llmModelSelect = document.getElementById("llmModelSelect");
const llmModel = document.getElementById("llmModel");
const llmBaseUrl = document.getElementById("llmBaseUrl");
const llmApiKey = document.getElementById("llmApiKey");
const hfToken = document.getElementById("hfToken");
const tabs = document.querySelectorAll(".tab");
const panels = document.querySelectorAll(".tab-panel");

function normalizeQuery(query) {
  if (regexToggle.checked) {
    return query;
  }
  if (phraseToggle.checked && !/^".*"$/.test(query)) {
    return `"${query}"`;
  }
  return query;
}

function getSelectedCollectionId() {
  return localStorage.getItem("collection_id") || "";
}

function setSelectedCollectionId(value) {
  localStorage.setItem("collection_id", value);
  if (collectionSelect) {
    collectionSelect.value = value;
  }
  if (collectionSelectAsk) {
    collectionSelectAsk.value = value;
  }
}

async function loadCollections() {
  const response = await fetch("/collections");
  const data = await response.json();
  const items = data.collections || [];
  collectionSelect.innerHTML = "";
  collectionSelectAsk.innerHTML = "";
  appendCollectionSelect.innerHTML = "";
  collectionsList.innerHTML = "";
  items.forEach((item) => {
    const option = document.createElement("option");
    option.value = item.id;
    option.textContent = item.name;
    collectionSelect.appendChild(option);
    const optionAsk = option.cloneNode(true);
    collectionSelectAsk.appendChild(optionAsk);
    const optionAppend = option.cloneNode(true);
    appendCollectionSelect.appendChild(optionAppend);
    const li = document.createElement("li");
    li.className = "result-item";
    li.innerHTML = `
      <div class="result-title">
        <span>${item.name}</span>
        <span class="page">${item.id}</span>
      </div>
      <div class="snippet">${item.root_path}</div>
      <div class="result-actions">
        <button class="selectBtn">Select</button>
        <button class="indexBtn">Index</button>
        <button class="reindexBtn">Reindex</button>
      </div>
    `;
    li.querySelector(".selectBtn").addEventListener("click", () => {
      setSelectedCollectionId(item.id);
    });
    li.querySelector(".indexBtn").addEventListener("click", async () => {
      await fetch(`/collections/${item.id}/index`, { method: "POST" });
    });
    li.querySelector(".reindexBtn").addEventListener("click", async () => {
      await fetch(`/collections/${item.id}/reindex`, { method: "POST" });
    });
    collectionsList.appendChild(li);
  });
  const selected = getSelectedCollectionId();
  if (selected && items.some((item) => item.id === selected)) {
    setSelectedCollectionId(selected);
  } else if (items.length > 0) {
    setSelectedCollectionId(items[0].id);
  }
}

async function runSearch() {
  const rawQuery = queryInput.value.trim();
  if (!rawQuery) {
    statusEl.textContent = "Enter a query to search.";
    resultsList.innerHTML = "";
    return;
  }

  const collectionId = getSelectedCollectionId();
  if (!collectionId) {
    statusEl.textContent = "Select a collection first.";
    return;
  }

  const query = normalizeQuery(rawQuery);
  const mode = regexToggle.checked ? "regex" : "fts";
  const params = new URLSearchParams({
    q: query,
    limit: "50",
    case_sensitive: caseToggle.checked ? "true" : "false",
    mode,
    redact: redactToggle.checked ? "true" : "false",
    collection_id: collectionId,
  });

  statusEl.textContent = "Searching...";
  resultsList.innerHTML = "";

  try {
    const response = await fetch(`/search?${params.toString()}`);
    const data = await response.json();
    const results = data.results || [];
    statusEl.textContent = `${results.length} result(s)`;
    if (results.length === 0) {
      return;
    }
    results.forEach((result) => {
      const li = document.createElement("li");
      li.className = "result-item";
      li.innerHTML = `
        <div class="result-title">
          <span>${result.filename}</span>
          <span class="page">Page ${result.page_num}</span>
        </div>
        <div class="snippet">${result.snippet}</div>
        <div class="result-actions">
          <button class="openBtn">Open</button>
          <a href="/view?collection_id=${collectionId}&doc_id=${result.doc_id}&page=${result.page_num}&term=${encodeURIComponent(rawQuery)}" target="_blank">Open in new tab</a>
          <a href="/file/${collectionId}/${result.doc_id}#page=${result.page_num}" target="_blank">Open PDF file</a>
        </div>
      `;
      li.querySelector(".openBtn").addEventListener("click", () => {
        viewerFrame.src = `/view?collection_id=${collectionId}&doc_id=${result.doc_id}&page=${result.page_num}&term=${encodeURIComponent(rawQuery)}`;
      });
      resultsList.appendChild(li);
    });
  } catch (err) {
    statusEl.textContent = "Search failed.";
  }
}

let sessionId = `${Date.now()}-${Math.random().toString(16).slice(2)}`;
let lastAskQuery = "";
let lastAskOffset = 0;
let lastCollectionId = "";

async function runAsk(offset = 0, append = false) {
  const question = askQuery.value.trim();
  if (!question) {
    askStatus.textContent = "Enter a question.";
    return;
  }
  const collectionId = getSelectedCollectionId();
  if (!collectionId) {
    askStatus.textContent = "Select a collection first.";
    return;
  }
  const queryChanged = question !== lastAskQuery;
  const collectionChanged = collectionId !== lastCollectionId;
  if (queryChanged || collectionChanged) {
    lastAskOffset = 0;
  }
  askStatus.textContent = "Thinking...";
  if (!append) {
    answerBox.textContent = "";
    sourcesList.innerHTML = "";
  }
  const payload = {
    query: question,
    collection_id: collectionId,
    mode: askMode.value,
    top_k: parseInt(askTopK.value, 10) || 10,
    offset: queryChanged || collectionChanged ? 0 : offset,
    answer_mode: evidenceView && evidenceView.checked ? "evidence_view" : answerMode.value,
    redact: askRedact.checked,
    embeddings_device: cpuEmbeddings.checked ? "cpu" : "auto",
    hf_token: hfToken.value.trim(),
    embeddings_engine: embeddingsEngine.value,
    embeddings_worker: embeddingsWorker.checked,
    anchor_llm_enabled: anchorLLM ? anchorLLM.checked : false,
    llm_provider: llmProvider.value,
    llm_model: llmModel.value.trim() || llmModelSelect.value,
    llm_base_url: llmBaseUrl.value.trim(),
    llm_api_key: llmApiKey.value.trim(),
    session_id: sessionId,
  };
  try {
    const debug = askDebug.checked;
    if (askDebugPanel) {
      askDebugPanel.open = false;
      askDebugPanel.style.display = "none";
    }
    if (askDebugInfo) {
      askDebugInfo.textContent = "";
    }
    const response = await fetch(debug ? "/query?debug=true" : "/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    const modeUsed = data.mode_used || "ask";
    const isEvidenceView = evidenceView && evidenceView.checked && modeUsed === "ask";
    if (modeUsed === "count") {
      const stats = data.stats || {};
      answerBox.textContent = `Total hits: ${stats.total_hits || 0}\nUnique pages: ${stats.unique_pages || 0}\nUnique docs: ${stats.unique_docs || 0}`;
      askStatus.textContent = stats.total_hits ? "Done" : "No matches found.";
    } else if (modeUsed === "search") {
      const stats = data.stats || {};
      answerBox.textContent = `Keyword results: ${stats.result_count || 0}`;
      askStatus.textContent = data.low_evidence ? "Low evidence found. Try keyword mode." : "Done";
    } else if (modeUsed === "corpus_summary") {
      answerBox.textContent = data.answer_markdown || "Corpus summary";
      askStatus.textContent = data.low_evidence ? "Low evidence found." : "Done";
    } else {
      if (isEvidenceView) {
        answerBox.textContent = "";
        askStatus.textContent = data.low_evidence ? "Low evidence found. Try keyword mode." : "Evidence view";
      } else {
        if (append && data.answer_markdown) {
          answerBox.textContent = `${answerBox.textContent}\n\n${data.answer_markdown}`;
        } else {
          answerBox.textContent = data.answer_markdown || "";
        }
        if (data.low_evidence) {
          askStatus.textContent = "Low evidence found. Try keyword mode.";
        } else {
          askStatus.textContent = "Done";
        }
      }
    }
    lastAskQuery = question;
    lastCollectionId = collectionId;
    lastAskOffset = data.next_offset || (offset + (parseInt(askTopK.value, 10) || 10));
    const sources = data.sources || [];
    if (askDebugPanel && askDebugInfo && data.retrieve_debug) {
      askDebugInfo.textContent = JSON.stringify(data.retrieve_debug, null, 2);
      askDebugPanel.style.display = "block";
    }
    if (modeUsed === "corpus_summary") {
      const themes = data.themes || [];
      sourcesTitle.textContent = "Themes";
      sourcesTitle.style.display = themes.length ? "block" : "none";
      sourcesList.style.display = themes.length ? "block" : "none";
      sourcesList.innerHTML = "";
      themes.forEach((theme) => {
        const li = document.createElement("li");
        li.className = "result-item";
        const citations = theme.citations || [];
        const snippets = citations
          .map((item) => {
            const link = `/view?collection_id=${collectionId}&doc_id=${item.doc_id}&page=${item.page_num}`;
            return `<li><a href="${link}" target="_blank">${item.filename} p.${item.page_num}</a>: ${item.snippet || ""}</li>`;
          })
          .join("");
        li.innerHTML = `
          <div class="result-title">
            <span>${theme.title || "Theme"}</span>
          </div>
          <div class="snippet">${theme.summary || ""}</div>
          <ul>${snippets}</ul>
        `;
        sourcesList.appendChild(li);
      });
    } else {
      sourcesTitle.textContent = "Sources";
      sourcesTitle.style.display = sources.length ? "block" : "none";
      sourcesList.style.display = sources.length ? "block" : "none";
      sourcesList.innerHTML = "";
      sources.forEach((source) => {
      const li = document.createElement("li");
      li.className = "result-item";
      li.innerHTML = `
        <div class="result-title">
          <span>${source.filename}</span>
          <span class="page">Page ${source.page_num}</span>
        </div>
        <div class="snippet">${source.snippet}</div>
        <div class="result-actions">
          <button class="openBtn">Open</button>
          <a href="/view?collection_id=${collectionId}&doc_id=${source.doc_id}&page=${source.page_num}&term=${encodeURIComponent(question)}" target="_blank">Open in new tab</a>
        </div>
      `;
      li.querySelector(".openBtn").addEventListener("click", () => {
        viewerFrame.src = `/view?collection_id=${collectionId}&doc_id=${source.doc_id}&page=${source.page_num}&term=${encodeURIComponent(question)}`;
      });
      sourcesList.appendChild(li);
      });
    }
    continueBtn.disabled = modeUsed !== "ask" || sources.length === 0;
  } catch (err) {
    askStatus.textContent = "Ask failed.";
  }
}

async function pollUploadStatus(uploadId) {
  let keepPolling = true;
  while (keepPolling) {
    try {
      const response = await fetch(`/upload/status?upload_id=${uploadId}`);
      const data = await response.json();
      if (data.status === "processing" || data.status === "queued") {
        uploadStatus.textContent = `Indexing... (${data.status})`;
        await new Promise((resolve) => setTimeout(resolve, 1500));
      } else if (data.status === "ready") {
        const stats = data.stats || {};
        uploadStatus.textContent = `Ready. Indexed ${stats.indexed || 0}, skipped ${stats.skipped || 0}, errors ${stats.errors || 0}.`;
        await loadCollections();
        keepPolling = false;
      } else {
        uploadStatus.textContent = `Error: ${data.error || "Unknown error"}`;
        keepPolling = false;
      }
    } catch (err) {
      uploadStatus.textContent = "Error checking upload status.";
      keepPolling = false;
    }
  }
}

async function uploadZip() {
  const file = zipInput.files[0];
  if (!file) {
    uploadStatus.textContent = "Choose a zip file first.";
    return;
  }
  const name = zipCollectionName.value.trim();
  const append = appendToggle.checked;
  const appendId = appendCollectionSelect.value;
  if (!append && !name) {
    uploadStatus.textContent = "Enter a collection name for the zip.";
    return;
  }
  if (append && !appendId) {
    uploadStatus.textContent = "Select a collection to append to.";
    return;
  }
  const formData = new FormData();
  formData.append("file", file);
  uploadStatus.textContent = "Uploading...";
  try {
    const query = new URLSearchParams();
    if (!append) {
      query.set("name", name);
    }
    if (append) {
      query.set("append", "true");
      query.set("collection_id", appendId);
    }
    const response = await fetch(`/collections/upload?${query.toString()}`, {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    if (!response.ok) {
      uploadStatus.textContent = data.detail || "Upload failed.";
      return;
    }
    uploadStatus.textContent = "Upload complete. Indexing...";
    if (data.collection_id) {
      setSelectedCollectionId(data.collection_id);
    }
    pollUploadStatus(data.upload_id);
  } catch (err) {
    uploadStatus.textContent = "Upload failed.";
  }
}

async function createCollection() {
  const name = collectionName.value.trim();
  const root = collectionRoot.value.trim();
  if (!name || !root) {
    return;
  }
  await fetch("/collections", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, root_path: root }),
  });
  collectionName.value = "";
  collectionRoot.value = "";
  await loadCollections();
}

tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    tabs.forEach((t) => t.classList.remove("active"));
    panels.forEach((p) => p.classList.remove("active"));
    tab.classList.add("active");
    const target = document.getElementById(`tab-${tab.dataset.tab}`);
    if (target) {
      target.classList.add("active");
    }
  });
});

searchBtn.addEventListener("click", runSearch);
queryInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    runSearch();
  }
});
askBtn.addEventListener("click", () => runAsk(0, false));
continueBtn.addEventListener("click", () => {
  askQuery.value = lastAskQuery;
  runAsk(lastAskOffset, true);
});
askQuery.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    runAsk();
  }
});
uploadBtn.addEventListener("click", uploadZip);
createCollectionBtn.addEventListener("click", createCollection);
refreshCollectionsBtn.addEventListener("click", loadCollections);
collectionSelect.addEventListener("change", (event) => {
  setSelectedCollectionId(event.target.value);
});
collectionSelectAsk.addEventListener("change", (event) => {
  setSelectedCollectionId(event.target.value);
});

loadCollections();

const modelOptions = {
  none: [{ value: "", label: "None" }],
  llama_cpp: [
    { value: "llama-3.1-8b-instruct", label: "Llama 3.1 8B Instruct" },
    { value: "llama-3.1-70b-instruct", label: "Llama 3.1 70B Instruct" },
    { value: "mistral-7b-instruct", label: "Mistral 7B Instruct" },
    { value: "qwen2-7b-instruct", label: "Qwen2 7B Instruct" },
  ],
  ollama: [
    { value: "llama3:8b", label: "Llama 3 8B" },
    { value: "llama3.1:8b", label: "Llama 3.1 8B" },
    { value: "mistral:7b", label: "Mistral 7B" },
    { value: "qwen2:7b", label: "Qwen2 7B" },
  ],
  openai_compat: [
    { value: "gpt-4o-mini", label: "gpt-4o-mini" },
    { value: "gpt-4o", label: "gpt-4o" },
    { value: "gpt-4.1-mini", label: "gpt-4.1-mini" },
  ],
};

function setModelOptions(provider) {
  const options = modelOptions[provider] || [];
  llmModelSelect.innerHTML = "";
  options.forEach((opt) => {
    const option = document.createElement("option");
    option.value = opt.value;
    option.textContent = opt.label;
    llmModelSelect.appendChild(option);
  });
}

function applyProviderDefaults(provider) {
  if (provider === "llama_cpp") {
    llmBaseUrl.value = llmBaseUrl.value || "http://127.0.0.1:8080";
  } else if (provider === "ollama") {
    llmBaseUrl.value = llmBaseUrl.value || "http://127.0.0.1:11434";
  } else if (provider === "openai_compat") {
    llmBaseUrl.value = llmBaseUrl.value || "https://api.openai.com";
  }
}

function onProviderChange() {
  const provider = llmProvider.value;
  setModelOptions(provider);
  applyProviderDefaults(provider);
}

llmProvider.addEventListener("change", onProviderChange);
onProviderChange();
