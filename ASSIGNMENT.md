# Assignment: Building a RAG System over Government Budget Documents

**Role:** Junior AI Developer  
**Time:** 2 hours. Please don't spend more.

---

## What you're building

A retrieval-augmented generation system that answers natural language questions over a corpus of Indian gender budget documents.

The corpus is real government data. It has not been cleaned for you.

> **You are not expected to parse every file perfectly. Document what you skipped or handled imperfectly, and why.**

## The corpus

Eleven files in `data/`: gender budget statements from the Government of India and from three state governments, spanning several years, as PDFs, a CSV, and an XLS. Some documents are English-only; others are bilingual English and Hindi.

## The questions

Two files ship with the corpus.

**`questions_dev.jsonl`** — 12 questions **with gold answers**, including source file and page. Use these to develop and sanity-check.

**`questions_eval.jsonl`** — 18 questions, **no answers provided**. Run your system over all of them and submit its raw output.

> A note on scope: a useful system should be able to recognise when the corpus does not contain the information needed to answer a question, and say so rather than guess.

---

## What to submit

A GitHub repository containing:

**1. Working code** — ingestion, indexing, and querying. It should run from a clean clone by following your README. Pin your dependencies.

**2. `SCOPE.md`** (~300 words) — what you included, what you excluded, what you handled imperfectly, and why. 

**3. `answers.jsonl`** — your system's output for all 18 evaluation questions, one JSON object per line:

```json
{"question_id": "E01", "answered": true, "answer": "Rs 23.3 crore", "sources": [{"file": "gender_budget_2023-24.pdf", "page": 1}]}
```

Set `"answered": false` when your system determines it cannot answer. Leave `answer` as an empty string in that case. Include `sources` wherever you can.

**4. `NOTES.md`** (~300 words) — how your system works and what you'd do next with more time. If possible, include one instance during this assignment where an AI tool gave you a wrong or misleading answer and how you caught it.

## How to submit

1. Create your **own public GitHub repository**.
2. Reply to the assignment email with the repository link.

---

## A suggested two hours

| | |
|---|---|
| Set up, look at the data | 20 min |
| Get an end-to-end pipeline running | 35 min |
| Deal with what you find | 30 min |
| Run the eval questions, capture output | 15 min |
| `SCOPE.md` and `NOTES.md` | 20 min |

If you run out of time, submit what you have with a note on where you stopped and what you'd have done next. An incomplete submission with clear reasoning beats a complete one without it.

## Ground rules

- **AI tools are expected and permitted.** Copilot, Claude, ChatGPT, Cursor, agents — all fine. We use them too. The reflection in `NOTES.md` is the only place we ask you to think about it.
- **Use any models you like** — hosted or local, open or closed. If you can't run a generation model at all, say so in `NOTES.md` and submit retrieval output instead.
- All source documents are **public government data**. Nothing you build here will be used by CivicDataLab. This assignment exists solely to assess your work.

Feel free to email us for any questions about the assignment.
