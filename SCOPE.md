# SCOPE

## What this system includes

- **Ingestion** of all eleven corpus files: PDFs (per-page text via `pdfplumber`,
  `pypdf` fallback), the Union CSV (one document per row, year columns detected),
  and the `.xls` (one document per row, header row auto-detected; free-text
  narrative rows kept separately).
- A **persistent ChromaDB** vector store with local, offline embeddings
  (`all-MiniLM-L6-v2` ONNX; cosine space).
- **Hybrid retrieval**: dense vector search merged with an in-memory BM25-style
  lexical search, so exact tokens (scheme names, years, "PART A Total") are not
  lost to embedding noise.
- A **reranker** blending vector similarity, lexical score, exact quoted /
  capitalised scheme-name matches, year agreement and jurisdiction agreement.
- **Grounded answering** with a local LLM (llama.cpp, Llama-3.2-3B) restricted to
  the retrieved sources, plus **Hindi→English query translation**. A strict
  "answer only from sources, else `NOT_IN_CORPUS`" rule drives the
  `answered: false` behaviour.
- A **FastAPI web page** and a **CLI (`run_eval.py`)** that emits `answers.jsonl`.

## What is excluded / handled imperfectly

- **Bilingual legacy-font PDFs.** The Hindi half extracts as garbage
  (`foÙk foHkkx`); only the English half is usable. Hindi queries therefore rely
  on machine translation, which is imperfect on a 3B model.
- **Borderless tables.** Column headings are not attached to rows, so the
  year→column mapping is guessed (the model assumes the last number is the most
  recent year). This causes wrong-year or wrong-unit answers.
- **Unit conversion and arithmetic** (thousands→crore, sums, percentages) are not
  performed; the model reports raw figures and sometimes mislabels the unit.
- **Spreadsheet quirks** (merged cells, blank padding) are partly handled by the
  header heuristic; some numeric rows remain noisy.
- **Aggregation rows** such as "PART A Total" are hard to rank and are sometimes
  missed.

## Why

The two-hour budget and the corpus's real-world messiness. Retrieval quality was
prioritised over perfect generation; remaining errors are documented rather than
hidden.
