# NOTES

## How it works

`ingest.py` reads every file in `corpus/` and turns it into documents: PDF pages
become ~800-character chunks (page number kept), CSV rows become one document
per row, and `.xls` rows become documents after detecting the real header row
(the stat20 header sits several rows down). Every document is prefixed with
`[source: file, page/row …]` at embedding time, which lets year and jurisdiction
tokens influence retrieval.

At query time (`app.py`) a question is optionally translated to English, then a
**hybrid search** runs: ChromaDB vector search is merged with a BM25-style
lexical index over all documents. The merged candidates are **reranked** using
vector similarity, lexical score, exact scheme-name ("Ladli Yojna") matches,
year agreement and jurisdiction agreement. If the best cosine distance exceeds a
threshold the system returns `answered: false` rather than guessing. Otherwise an
LLM answers strictly from the top sources, returning `NOT_IN_CORPUS` when they
lack the answer; it can be a local OpenAI-compatible server (`localhost:5555`)
or a hosted provider (OpenAI or Gemini) selected via environment variables and
overridable per request. If the LLM is unavailable the app falls back to
extractive answering.
`run_eval.py` runs the eval questions and writes `answers.jsonl`.

## What I'd do next

1. **Reconstruct tables** properly (e.g. `camelot`/`pdfplumber.extract_tables`)
   and attach column headers, fixing the year/unit ambiguity.
2. Add a **unit-conversion / arithmetic post-processor** so "10,00,000 thousand"
   becomes "Rs 100 crore" and totals are summed.
3. Use a **multilingual embedding model** and/or a stronger LLM; the 3B model
   sometimes refuses or mislabels units.
4. **Metadata filtering** on year/jurisdiction parsed from the question.
5. A scored **dev harness** (source-file and answer overlap) to tune ranking.

## An AI tool gave a misleading answer

While adding a "jurisdiction" bonus to the reranker, I let it match against the
document *filename*. This looked sensible but silently broke question D09:
"Union" in `MRF_13_Union_Budget.csv` matched every Union row, so a stat20
narrative sentence ("30 Ministries/Departments and 5 Union territories…") was
pushed out of the context and the model cited the wrong file. I only caught it by
running all 12 dev questions and comparing retrieved sources against the gold
`source_files` — the answer still looked plausible, which is exactly why source
checking mattered. I fixed it by matching jurisdiction only in document *text*,
with word boundaries. A similar over-eager phrase bonus helped one question but
regressed two others until the dev set exposed it.
