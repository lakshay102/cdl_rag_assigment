# Gender Budget RAG

A retrieval-augmented question answering web app over the Indian gender budget
corpus (`corpus/`). Questions are answered using a **local vector database**
(ChromaDB) for retrieval and an **LLM for grounded answer generation**. The LLM
can be a **local server** (llama.cpp, LM Studio, Ollama) or a **hosted API —
OpenAI or Google Gemini** — selected with environment variables. If no LLM is
configured, the app falls back to extractive answering.

## Layout

```
corpus/               source documents (PDF, CSV, XLS) - not modified
ingest.py             parse corpus -> chunk -> embed -> ChromaDB
app.py                FastAPI backend: hybrid retrieval, rerank, answering
llm.py                grounded generation + Hindi->English query translation
config.py             shared settings
static/index.html     the question / answer web page
run_eval.py           run a questions file and write answers.jsonl
requirements.txt      pinned dependencies
chroma_db/            persistent vector store (created by ingest.py)
answers.jsonl         system output for questions_eval.jsonl
SCOPE.md, NOTES.md    submission write-up
```

## Setup (venv only)

```powershell
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Optional: use a `.env` file

`config.py` loads `.env` from the project root automatically. Copy the example
and fill in the values you need:

```powershell
Copy-Item .env.example .env
```

Every variable the code reads is listed and documented in `.env.example`.
Variables already set in the real OS environment win over `.env` values, so the
PowerShell examples below can still be used instead.

## 1. Build the index

```powershell
.\venv\Scripts\python.exe ingest.py
```

Parses every file in `corpus/`, embeds the chunks locally, and writes a
persistent ChromaDB store in `chroma_db/`. Re-running rebuilds it. The first run
downloads the small `all-MiniLM-L6-v2` embedding model once (needs internet);
afterwards it is fully offline.

## 2. (Optional) choose an LLM

Every provider is called through its OpenAI-compatible API, so only the provider
name, model name and API key change. Pick one of the three options below.
Without any LLM the app still works (extractive mode).

### Option A — local server (default, no API key)

With llama.cpp:

```powershell
llama-server -m .\Llama-3.2-3B-Instruct-Q4_K_M.gguf --port 5555
```

The default provider is `local`, the base URL is `http://localhost:5555/v1`,
and the model name is auto-detected.

### Option B — OpenAI API key

```powershell
$env:RAG_LLM_PROVIDER = "openai"
$env:OPENAI_API_KEY   = "sk-..."          # your OpenAI key
$env:RAG_LLM_MODEL    = "gpt-4o-mini"     # optional; this is the default
.\venv\Scripts\python.exe app.py
```

Set `RAG_LLM_MODEL` to any chat model you have access to (e.g.
`gpt-4o`, `gpt-4.1-mini`). Omit it to use `gpt-4o-mini`.

### Option C — Google Gemini API key

Create a key in Google AI Studio, then:

```powershell
$env:RAG_LLM_PROVIDER = "gemini"
$env:GEMINI_API_KEY   = "AIza..."         # your Gemini / AI Studio key
$env:RAG_LLM_MODEL    = "gemini-3.1-flash-lite"  # optional; this is the default
.\venv\Scripts\python.exe app.py
```

`GOOGLE_API_KEY` also works instead of `GEMINI_API_KEY`. Set `RAG_LLM_MODEL` to
any Gemini model the key can use (e.g. `gemini-2.5-pro`, `gemini-2.0-flash`).

### Precedence and overrides

- Provider: `RAG_LLM_PROVIDER` → `local` if unset.
- Model: request `model` → `RAG_LLM_MODEL` → provider default.
- API key: `RAG_LLM_API_KEY` → `OPENAI_API_KEY` / `GEMINI_API_KEY` /
  `GOOGLE_API_KEY` → provider default.
- Endpoint: `RAG_LLM_BASE_URL` → provider default.

Keys stay on the server; the browser only sends provider/model names. The web
page has **LLM provider** and **Model** controls, and `POST /api/ask` accepts
optional `provider` and `model` fields, so you can switch models per request
without restarting the server. Provider defaults live in `config.py` under
`LLM_PROVIDERS`.

## 3. Start the web app

```powershell
.\venv\Scripts\python.exe app.py
```

Open <http://127.0.0.1:8000>, type a question, press **Ask** (or `Ctrl+Enter`).
The page shows the answer, its sources, the engine used, and the retrieved
context.

## How answering works

1. **Ingestion** turns PDF pages into chunks and CSV/XLS rows into one document
   per row. Each document is prefixed with `[source: file, page/row ...]` so
   years and jurisdictions are part of the embedded text. XLS header rows are
   auto-detected; free-text narrative rows are kept too.
2. A Hindi (Devanagari) question is translated to English so it can match the
   mostly-English / legacy-font index.
3. **Hybrid retrieval**: ChromaDB vector search is merged with an in-memory
   BM25-style lexical search, so distinctive tokens (scheme names, years,
   `<PART A Total>`) are never lost.
4. **Reranking** blends vector similarity, lexical score, exact
   capitalised/quoted scheme-name matches, year agreement, and jurisdiction
   agreement.
5. If the best cosine distance exceeds `RAG_MAX_DISTANCE` the app reports
   `answered: false`. Otherwise the configured LLM answers strictly from the
   retrieved sources (returning `NOT_IN_CORPUS` if they don't contain it), or
   the extractive fallback selects the best-matching passage.

## Configuration (environment variables)

These can be set as real environment variables or placed in a `.env` file
(see `.env.example`).

| Variable            | Default | Meaning                                    |
|---------------------|---------|--------------------------------------------|
| `RAG_CORPUS_DIR`    | `corpus`| folder with source documents               |
| `RAG_DB_DIR`        | `chroma_db` | vector store location                  |
| `RAG_TOP_K`         | `5`     | default number of chunks retrieved         |
| `RAG_RERANK_POOL`   | `60`    | candidates pulled before reranking         |
| `RAG_ALL_MAX_RESULTS` | `20`  | chunks kept when `k=0` ("all" in the UI)   |
| `RAG_MAX_DISTANCE`  | `0.65`  | cosine-distance cut-off for "answered"     |
| `RAG_PDF_CHUNK_CHARS` | `800` | PDF chunk size in characters              |
| `RAG_PDF_CHUNK_OVERLAP` | `150` | PDF chunk overlap in characters         |
| `RAG_PDF_CHUNK_MIN_CHARS` | `200` | merge shorter page tails into previous chunk |
| `RAG_PDF_HEADER_CONTEXT` | `1` | repeat unit/column headers on later chunks |
| `RAG_PDF_HEADER_CHARS` | `240` | max characters of repeated page header  |
| `RAG_EMBED_MAX_TOKENS` | `256` | embedding limit used in dry-run warnings |
| `RAG_USE_LLM`       | `1`     | set `0` to force extractive mode           |
| `RAG_LLM_PROVIDER`  | `local` | `local`, `openai`, or `gemini`             |
| `RAG_LLM_BASE_URL`  | provider preset | override the OpenAI-compatible endpoint |
| `RAG_LLM_MODEL`     | provider preset (`auto` for local) | model name |
| `RAG_LLM_API_KEY`   | provider env / `sk-local` for local | explicit key override |
| `OPENAI_API_KEY`    | —       | key used when `RAG_LLM_PROVIDER=openai`    |
| `GEMINI_API_KEY`    | —       | key used when `RAG_LLM_PROVIDER=gemini`    |
| `GOOGLE_API_KEY`    | —       | alternative Gemini key variable            |
| `RAG_LLM_MAX_TOKENS`| `512`   | max generated tokens                       |
| `RAG_LLM_TEMPERATURE`| `0.0`  | sampling temperature                       |
| `RAG_LLM_TIMEOUT`   | `120`   | request timeout in seconds                 |

### PDF parsing and chunking

`ingest.py` already uses **pdfplumber** (`extract_text(layout=True)`, with pypdf
as a fallback) for every PDF. `extract_tables()` is deliberately not used for
the Delhi statements: those tables have no ruling lines, so pdfplumber returns
zero tables even though `layout=True` preserves the rows. Adding table
extraction would therefore add noise, not structure, for those files.

Two things matter more than the parser:

1. **Embedding truncation.** The bundled `all-MiniLM-L6-v2` model truncates at
   256 tokens. Tabular text tokenizes densely: 800 characters is roughly 320
   tokens, so most 800-character chunks lose their tail before embedding.
2. **Missing context.** Unit and column headers only appear at the top of a
   page. A chunk that starts halfway down loses "(IN THOUSANDS)" and the
   column names, which is how a figure gets reported in the wrong unit.

The current chunker therefore:

- splits on line boundaries so a scheme name and its figures stay together;
- merges a tiny page tail into the previous chunk; and
- repeats a compact `[context: ...]` header (unit + column headings) on later
  chunks of the same page. Disable it with `RAG_PDF_HEADER_CONTEXT=0`.

Measure the distribution before changing the size:

```powershell
.\venv\Scripts\python.exe ingest.py --dry-run
.\venv\Scripts\python.exe ingest.py --dry-run --chunk-chars 600 --chunk-overlap 110
```

`--dry-run` prints chunks per file and token p50/p90/p95/max, and warns how many
chunks exceed the embedding limit.

#### Choosing a chunk size

Change size by a **factor of about 1.5**, not by a few characters:

```text
400 -> 600 -> 800 -> 1200 -> 1800
```

A sweep builds each size into its own database, so `chroma_db` is untouched:

```powershell
# Retrieval-only, no LLM calls: source recall and token stats
.\venv\Scripts\python.exe tune_chunks.py --no-llm --sizes 400,600,800,1200

# Answer quality with the configured provider (costs API calls)
.\venv\Scripts\python.exe tune_chunks.py --sizes 500,700,900 --limit 6
```

Each run writes `answers_c<size>.report.jsonl` and a combined `summary.json`
under `.sweeps/`. Read the table like this:

- **Source recall flat, numeric/keyword recall low** → the right chunks are
  already being retrieved; fix generation (prompt/unit handling), not chunking.
- **Source recall rises as chunks get smaller** → chunks were too coarse or
  truncated; keep the smallest size where recall plateaus.
- **Token p95 above 256** → the embedding model is truncating chunks; shrink
  until p95 is at or below the limit.
- **Source recall falls as chunks get smaller** → chunks lost the surrounding
  row/header context; move back up.

Pick the **knee**: the smallest size whose source hits and numeric/keyword
recall are within one question of the best. Larger chunks after that only add
truncation and context noise. Changing `RAG_PDF_CHUNK_CHARS`,
`RAG_PDF_CHUNK_OVERLAP` or `RAG_PDF_HEADER_CONTEXT` requires re-running
`ingest.py` and restarting the app.

## HTTP API

- `GET  /api/health` — index readiness, chunk count, LLM availability and the
  resolved provider/model (`llm_info`, `llm_providers`)
- `POST /api/ask` — body `{"question": "...", "k": 5}` →
  `{answered, answer, sources, engine, context, ...}`
  `k: 0` means "search every chunk in the index" (the web page's **all**
  option). It is slower, and only the best `RAG_ALL_MAX_RESULTS` chunks are
  passed to the LLM / returned in `context`.
  Optional fields `provider` (`local|openai|gemini`) and `model`
  (`"gpt-4o-mini"`, `"gemini-3.1-flash-lite"`, ...) override the environment for
  that one request. `llm_error` explains any provider/key failure.

## Evaluation / batch answering

Run the system over a questions file and write the submission format:

```powershell
.\venv\Scripts\python.exe run_eval.py questions_eval.jsonl answers.jsonl
```

Each line is `{"question_id", "answered", "answer", "sources"}`; `answered` is
false (empty answer) when the system decides the corpus lacks the information.
If the output path is omitted, `questions_dev.jsonl` writes `answers_dev.jsonl`
so the submission `answers.jsonl` is not overwritten.

### Dev-set scoring and the `tests` field

`questions_dev.jsonl` also has gold fields: `answer`, `source_files`,
`source_location`, `unit` and `tests`. When `run_eval.py` sees them it:

1. Sends the `tests` note to the LLM as a **focus note** — guidance about what
   the question is testing (unit conversion, year ambiguity, Hindi query, ...).
   It is explicitly not a source and cannot be cited. Disable it with
   `--no-hints` to measure the system without hints.
2. Scores each output against the gold answer: source-file recall, numeric
   recall (0.5% tolerance), keyword recall and whether the expected unit is
   stated.
3. Writes a per-question diagnostics file (default
   `answers_dev.report.jsonl`) and prints a summary.

```powershell
.\venv\Scripts\python.exe run_eval.py questions_dev.jsonl answers_dev.jsonl
.\venv\Scripts\python.exe run_eval.py questions_dev.jsonl answers_dev.jsonl --no-hints
.\venv\Scripts\python.exe run_eval.py questions_dev.jsonl answers_dev.jsonl --report my_report.jsonl
```

Gold answers/source files are **never** passed to the LLM; only the `tests`
description is used, and only for dev questions that contain it. The eval file
(`questions_eval.jsonl`) has no gold fields, so its output is unchanged.

