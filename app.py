"""FastAPI backend for the Gender Budget RAG web page.

Pipeline:
    question -> (optional Hindi->English translation)
             -> hybrid retrieval: ChromaDB vector search + lexical (BM25) search
             -> metadata/lexical rerank
             -> grounded answer from a local OpenAI-compatible LLM
                (extractive fallback if the LLM is unavailable)

Run:
    python app.py
Then open http://127.0.0.1:8000
"""
import math
import os
import re
from collections import Counter
from typing import Optional

import chromadb
from chromadb.config import Settings
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from config import (
    ALL_MAX_RESULTS,
    COLLECTION_NAME,
    DB_DIR,
    LLM_PROVIDERS,
    MAX_DISTANCE,
    RERANK_POOL,
    TOP_K,
    USE_LLM,
    get_embedding_function,
)

try:
    from llm import (
        NOT_IN_CORPUS,
        generate,
        get_llm_info,
        llm_ready,
        translate_to_english,
    )
except Exception:  # noqa: BLE001
    NOT_IN_CORPUS = "NOT_IN_CORPUS"

    def llm_ready(provider=None, model=None):
        return False

    def generate(question, hits, provider=None, model=None, test_hint=None):
        raise RuntimeError("LLM module unavailable")

    def translate_to_english(text, provider=None, model=None):
        raise RuntimeError("LLM module unavailable")

    def get_llm_info(provider=None, model=None):
        return {}


HERE = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="Gender Budget RAG", version="3.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STOPWORDS = set(
    """a an the of in on for to and or is are was were be been being what which
    how much many who whom whose when where did does do with under from at as by
    that this these those it its his her their there budget allocation allocated
    for is was""".split()
)

# State/UT names, matched against document text only (never filenames, which
# would let "Union" in "Union_Budget.csv" leak in for every question).
JURISDICTIONS = ["delhi", "odisha", "bihar", "chhattisgarh", "chandigarh"]


def tokenize(text):
    return [
        t
        for t in re.findall(r"[a-z0-9]+", (text or "").lower())
        if t not in STOPWORDS and len(t) > 1
    ]


def normalize_years(text):
    years = set()
    for m in re.finditer(r"((?:19|20)\d{2})\s*[-–/]\s*(\d{2,4})", text or ""):
        years.add(f"{m.group(1)}-{m.group(2)[-2:]}")
    return years


def mentions_jurisdiction(text, jurisdictions):
    low = (text or "").lower()
    return any(re.search(rf"\b{re.escape(j)}\b", low) for j in jurisdictions)


def raw_tokens(text):
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def key_phrases(question):
    """Extract distinctive multi-word phrases: quoted terms and capitalised
    scheme / ministry names (e.g. 'Ladli Yojna')."""
    phrases = []
    for m in re.findall(r"[\"']([^\"']{3,})[\"']", question):
        phrases.append(m)
    for m in re.finditer(
        r"([A-Z][A-Za-z0-9&.]+(?:\s+(?:of|and|for|the|in)?\s*[A-Z][A-Za-z0-9&.]+)+)",
        question,
    ):
        phrases.append(m.group(1))

    out = []
    for p in phrases:
        tokens = raw_tokens(p)
        if len(tokens) >= 2:
            out.append(tokens)
    return out


def phrase_bonus(q_phrases, doc_tokens):
    bonus = 0.0
    for phrase in q_phrases:
        n = len(phrase)
        for i in range(len(doc_tokens) - n + 1):
            if doc_tokens[i : i + n] == phrase:
                bonus += 0.25 * min(n, 5)
                break
    return bonus


# --------------------------------------------------------------------------- #
# Vector store + in-memory lexical index
# --------------------------------------------------------------------------- #
class LexicalIndex:
    """Tiny BM25-lite index over all documents, used to guarantee recall for
    distinctive tokens (scheme names, years, aggregate rows) that dense
    embeddings often miss."""

    def __init__(self, ids, documents, metadatas):
        self.ids = list(ids)
        self.documents = list(documents)
        self.metadatas = list(metadatas)
        self.doc_tokens = [set(tokenize(d)) for d in self.documents]
        self.df = Counter()
        for tokens in self.doc_tokens:
            for t in tokens:
                self.df[t] += 1
        self.n = max(len(self.doc_tokens), 1)
        self.id_to_pos = {doc_id: i for i, doc_id in enumerate(self.ids)}

    def score(self, query):
        q_tokens = set(tokenize(query))
        scored = []
        for i, tokens in enumerate(self.doc_tokens):
            s = 0.0
            for t in q_tokens:
                if t in tokens:
                    s += math.log(1.0 + self.n / (1.0 + self.df[t]))
            if s > 0:
                scored.append((s, i))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    def search(self, query, top_n):
        scored = self.score(query)
        if not scored:
            return [], {}
        max_score = scored[0][0] or 1.0
        results = []
        for s, i in scored[:top_n]:
            results.append(
                {
                    "id": self.ids[i],
                    "text": self.documents[i],
                    "metadata": self.metadatas[i] or {},
                    "distance": None,
                    "lex": round(s / max_score, 4),
                }
            )
        # normalised lookups so vector hits can get their lexical score too
        lex_by_id = {r["id"]: r["lex"] for r in results}
        return results, lex_by_id


_state = {"collection": None, "lexical": None}


def get_state():
    if _state["collection"] is None:
        if not os.path.isdir(DB_DIR):
            raise HTTPException(
                status_code=503,
                detail="Vector DB not found. Run `python ingest.py` first.",
            )
        client = chromadb.PersistentClient(
            path=DB_DIR, settings=Settings(anonymized_telemetry=False)
        )
        collection = client.get_collection(
            COLLECTION_NAME, embedding_function=get_embedding_function()
        )
        data = collection.get(include=["documents", "metadatas"])
        _state["collection"] = collection
        _state["lexical"] = LexicalIndex(data["ids"], data["documents"], data["metadatas"])
    return _state["collection"], _state["lexical"]


def retrieve(retrieval_query, pool):
    collection, lexical = get_state()
    result = collection.query(
        query_texts=[retrieval_query],
        n_results=pool,
        include=["documents", "metadatas", "distances"],
    )
    ids = (result.get("ids") or [[]])[0]
    docs = (result.get("documents") or [[]])[0]
    metas = (result.get("metadatas") or [[]])[0]
    dists = (result.get("distances") or [[]])[0]

    candidates = {}
    for cid, doc, meta, dist in zip(ids, docs, metas, dists):
        candidates[cid] = {
            "id": cid,
            "text": doc,
            "metadata": meta or {},
            "distance": float(dist),
            "lex": 0.0,
        }

    lex_hits, _ = lexical.search(retrieval_query, pool * 2)
    for hit in lex_hits:
        if hit["id"] in candidates:
            candidates[hit["id"]]["lex"] = hit["lex"]
        else:
            candidates[hit["id"]] = hit

    return list(candidates.values())


def rerank(question, candidates, k):
    """Blend vector similarity with lexical score and metadata signals."""
    q_tokens = set(tokenize(question))
    q_phrases = key_phrases(question)
    q_years = normalize_years(question)
    q_juris = {j for j in JURISDICTIONS if j in question.lower()}

    scored = []
    for cand in candidates:
        meta = cand["metadata"]
        doc = cand["text"]
        d_tokens = set(tokenize(doc))
        overlap = len(q_tokens & d_tokens) / max(len(q_tokens), 1)

        lexical = cand.get("lex", 0.0)
        if cand["distance"] is not None:
            similarity = 1.0 - cand["distance"]
        else:
            # Lexical-only hits come from the in-memory index and have no
            # vector distance. Give only a strong match on *informative* query
            # tokens a conservative pseudo-similarity. A shared year alone
            # (e.g. a Devanagari query whose only ASCII tokens are "2024 25")
            # must not promote unrelated pages over actual vector hits.
            informative = {t for t in q_tokens if not t.isdigit()}
            covered = len(informative & d_tokens) / max(len(informative), 1)
            strong = lexical >= 0.60 and covered >= 0.25
            similarity = 0.65 * lexical if strong else 0.0

        p_bonus = phrase_bonus(q_phrases, raw_tokens(doc))

        year_bonus = 0.0
        if q_years:
            d_years = normalize_years(doc) | normalize_years(meta.get("file", ""))
            year_bonus = 0.30 * len(q_years & d_years) / len(q_years)

        juris_bonus = 0.0
        if q_juris and mentions_jurisdiction(doc, q_juris):
            juris_bonus = 0.12

        # Aggregate rows are easy to outrank with scheme-level rows that share
        # the words "Part A" and "Total". When the question asks for the total
        # of a part, prefer the row that is that part's total.
        q_low = question.lower()
        doc_low = doc.lower()
        aggregate_bonus = 0.0
        if "total" in q_low:
            if re.search(r"\bpart a\b", q_low) and "part a total" in doc_low:
                aggregate_bonus = 0.75
            elif re.search(r"\bpart b\b", q_low) and "part b total" in doc_low:
                aggregate_bonus = 0.75
            elif "grand total" in q_low and "grand total" in doc_low:
                aggregate_bonus = 0.75

        score = (
            0.55 * similarity
            + 0.45 * lexical
            + p_bonus
            + year_bonus
            + juris_bonus
            + aggregate_bonus
        )
        enriched = dict(cand)
        enriched["score"] = round(score, 4)
        enriched["overlap"] = round(overlap, 3)
        enriched["phrase"] = p_bonus
        enriched["aggregate"] = aggregate_bonus
        scored.append(enriched)

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:k]


# --------------------------------------------------------------------------- #
# Extractive fallback
# --------------------------------------------------------------------------- #
def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+|\n+", text or "")
    return [p.strip() for p in parts if p.strip()]


def build_extractive_answer(question, hits, max_sentences=3):
    q_tokens = set(tokenize(question))
    candidates = []
    for rank, hit in enumerate(hits):
        rank_bonus = 1.0 / (1.0 + rank)
        for sentence in split_sentences(hit["text"]):
            s_tokens = set(tokenize(sentence))
            if not s_tokens:
                continue
            overlap = len(q_tokens & s_tokens)
            coverage = overlap / max(len(q_tokens), 1)
            digit_bonus = 0.15 if re.search(r"\d", sentence) else 0.0
            score = coverage + 0.08 * overlap + digit_bonus + 0.05 * rank_bonus
            candidates.append((score, overlap, sentence, hit))

    candidates.sort(key=lambda x: x[0], reverse=True)

    chosen, seen = [], set()
    for score, overlap, sentence, hit in candidates:
        if overlap == 0 and chosen:
            continue
        key = sentence.lower()
        if key in seen:
            continue
        seen.add(key)
        chosen.append((sentence, hit))
        if len(chosen) >= max_sentences:
            break

    if not chosen:
        return "", []
    return " ".join(s for s, _ in chosen), [h for _, h in chosen]


def summarize_source(meta):
    src = {"file": meta.get("file")}
    for key in ("page", "sheet", "row"):
        if meta.get(key) is not None:
            src[key] = meta[key]
    return src


def unique_sources(hits):
    sources, seen = [], set()
    for hit in hits:
        src = summarize_source(hit["metadata"])
        key = (src.get("file"), src.get("page"), src.get("sheet"), src.get("row"))
        if key in seen:
            continue
        seen.add(key)
        sources.append(src)
    return sources


def _cited_hits(answer, ranked):
    nums = [int(n) for n in re.findall(r"\[S(\d+)\]", answer)]
    chosen = [ranked[n - 1] for n in nums if 1 <= n <= len(ranked)]
    return chosen or ranked[:3]


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    # 0 means "every chunk in the index" (the UI's "all" option); it is slower
    # because the whole corpus is ranked. Positive values are capped at 50.
    k: int = Field(default=TOP_K, ge=0, le=50)
    # Optional per-request overrides; values from the environment are used when
    # these are omitted. Keys are never accepted from the client, only the
    # provider/model names.
    provider: Optional[str] = Field(
        default=None, description="local | openai | gemini"
    )
    model: Optional[str] = Field(
        default=None, description="Model name, e.g. gpt-4o-mini or gemini-3.1-flash-lite"
    )
    # Dev-set evaluation note (the `tests` field). Describes what the question
    # is testing (unit conversion, year ambiguity, ...). Passed to the LLM as
    # guidance only; it is never treated as a source of facts.
    test_hint: Optional[str] = Field(
        default=None,
        description="Optional dev-set `tests` note used as prompt guidance",
    )


@app.get("/")
def index_page():
    return FileResponse(os.path.join(HERE, "static", "index.html"))


@app.get("/api/health")
def health():
    ready = os.path.isdir(DB_DIR)
    count = None
    if ready:
        try:
            collection, _ = get_state()
            count = collection.count()
        except Exception:  # noqa: BLE001
            ready = False
    try:
        llm_info = get_llm_info()
    except Exception as exc:  # noqa: BLE001 - e.g. bad RAG_LLM_PROVIDER
        llm_info = {"error": str(exc)}

    return {
        "ready": ready,
        "chunks": count,
        "llm": llm_ready() if USE_LLM else False,
        "llm_info": llm_info,
        "llm_providers": sorted(LLM_PROVIDERS),
        "db_dir": DB_DIR,
    }


@app.post("/api/ask")
def ask(req: AskRequest):
    # Translate non-Latin (e.g. Hindi) questions so they can match the
    # mostly-English / legacy-font index.
    retrieval_query = req.question
    if USE_LLM and re.search(r"[\u0900-\u097F]", req.question):
        try:
            translated = translate_to_english(
                req.question, provider=req.provider, model=req.model
            )
            if translated:
                retrieval_query = translated
        except Exception:  # noqa: BLE001
            pass

    collection, _ = get_state()
    total = collection.count()

    if req.k == 0:
        # "all": score every chunk, but keep the answer/response bounded.
        pool = total
        keep = min(total, ALL_MAX_RESULTS)
    else:
        pool = max(RERANK_POOL, req.k)
        keep = req.k

    candidates = retrieve(retrieval_query, pool)
    if not candidates:
        return _response(req, False, "", [], [], None, "none", candidates=0)

    vector_distances = [
        c["distance"] for c in candidates if c["distance"] is not None
    ]
    best_distance = min(vector_distances) if vector_distances else None

    ranked = rerank(retrieval_query, candidates, keep)
    answered = best_distance is not None and best_distance <= MAX_DISTANCE
    if not answered:
        return _response(
            req, False, "", [], ranked, best_distance, "none", candidates=len(candidates)
        )

    engine = "extractive"
    answer, used = "", []
    llm_error = None

    if USE_LLM:
        try:
            text, model = generate(
                req.question,
                ranked,
                provider=req.provider,
                model=req.model,
                test_hint=req.test_hint,
            )
            if not text or NOT_IN_CORPUS in text:
                return _response(
                    req,
                    False,
                    "",
                    [],
                    ranked,
                    best_distance,
                    f"llm:{model}",
                    candidates=len(candidates),
                )
            answer = text
            engine = f"llm:{model}"
            used = _cited_hits(text, ranked)
        except Exception as exc:  # noqa: BLE001 - fall back to extractive
            llm_error = str(exc)
            answer, used = build_extractive_answer(req.question, ranked)
            engine = "extractive (llm unavailable)"
    else:
        answer, used = build_extractive_answer(req.question, ranked)

    return _response(
        req,
        True,
        answer,
        unique_sources(used),
        ranked,
        best_distance,
        engine,
        llm_error,
        candidates=len(candidates),
    )


def _response(
    req,
    answered,
    answer,
    sources,
    ranked,
    best_distance,
    engine,
    llm_error=None,
    candidates=None,
):
    return {
        "question": req.question,
        "answered": answered,
        "answer": answer,
        "sources": sources,
        "engine": engine,
        "llm_error": llm_error,
        "k_requested": req.k,
        "k_used": len(ranked),
        "candidates": candidates,
        "best_distance": round(best_distance, 4) if best_distance is not None else None,
        "threshold": MAX_DISTANCE,
        "context": [
            {
                "text": h["text"],
                "file": h["metadata"].get("file"),
                "page": h["metadata"].get("page"),
                "sheet": h["metadata"].get("sheet"),
                "row": h["metadata"].get("row"),
                "distance": round(h["distance"], 4) if h["distance"] is not None else None,
                "lex": h.get("lex"),
                "score": h.get("score"),
                "overlap": h.get("overlap"),
            }
            for h in ranked
        ],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
