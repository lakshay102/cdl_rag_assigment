"""Ingest the corpus into a persistent ChromaDB vector store.

Usage:
    python ingest.py            # build (or rebuild) the index

Sources handled:
    *.pdf          -> text extracted per page, chunked, page number kept
    *.csv          -> one document per row (year columns detected)
    *.xls / *.xlsx -> one document per row, per sheet; header row auto-detected,
                      free-text header/narrative rows kept as documents too
"""
import argparse
import os
import re

import pandas as pd
import pdfplumber
import chromadb
from chromadb.config import Settings
from pypdf import PdfReader

from config import (
    CORPUS_DIR,
    DB_DIR,
    COLLECTION_NAME,
    EMBED_MAX_TOKENS,
    PDF_CHUNK_CHARS,
    PDF_CHUNK_MIN_CHARS,
    PDF_CHUNK_OVERLAP,
    PDF_HEADER_CHARS,
    PDF_HEADER_CONTEXT,
    get_embedding_function,
)

BATCH_SIZE = 256
YEAR_RE = re.compile(r"(?:19|20)\d{2}\s*[-–/]\s*\d{2,4}")
_NULLS = {"", "nan", "none", "nat"}

# A line in these PDFs that tells us the unit or names the table columns.
_UNIT_RE = re.compile(
    r"\(\s*in\s+(?:thousand|lakh|crore)s?\s*\)"
    r"|in\s+(?:thousand|lakh|crore)s?\b"
    r"|rs\.?\s*in\s+(?:thousand|lakh|crore)s?",
    re.IGNORECASE,
)
_HEADER_HINTS = (
    "scheme",
    "s. no",
    "sl. no",
    "budget",
    "estimate",
    "actual",
    "revised",
    "expenditure",
    "outlay",
    "allocation",
    "ministry",
    "department",
    "category",
    "total",
)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def is_empty(value):
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() in _NULLS


def clean_text(text):
    text = (text or "").replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _mostly_latin(line):
    """True for English/ASCII lines, false for legacy-font Hindi garbage."""
    letters = [c for c in line if c.isalpha()]
    if not letters:
        return False
    latin = sum(1 for c in letters if ord(c) < 128)
    return latin / len(letters) >= 0.75


def _looks_like_header(line):
    low = line.lower()
    if len(line) < 4 or not _mostly_latin(line):
        return False
    # Column headings are short. Long lines are narrative sentences that
    # happen to contain words like "total" or "department".
    words = line.split()
    if len(words) > 14:
        return False
    if len(words) > 6 and line.rstrip().endswith((".", "।")):
        return False
    if _UNIT_RE.search(line):
        return True
    if len(words) <= 12 and any(hint in low for hint in _HEADER_HINTS):
        return True
    letters = [c for c in line if c.isalpha()]
    # All-caps title lines (e.g. "GENDER BUDGETING IN THE ANNUAL BUDGET").
    return (
        len(line) >= 8
        and bool(letters)
        and sum(1 for c in letters if c.isupper()) / len(letters) >= 0.8
    )


def extract_header_context(page_text, max_chars=PDF_HEADER_CHARS):
    """Return a compact unit + column-header line for a page.

    Delhi pages repeat "(IN THOUSANDS)" and the column headings on every page,
    but only at the top. Chunks after the first one otherwise lose the unit,
    which makes an answer like "3,00,000" impossible to interpret.
    """
    lines = [
        re.sub(r"\s+", " ", line).strip()
        for line in (page_text or "").splitlines()
    ]
    top = [line for line in lines if line][:20]
    # The useful part is the unit. Pages without one (narrative pages) get no
    # repeated context, which avoids copying random sentences into chunks.
    unit_line = next((line for line in top if _UNIT_RE.search(line)), None)
    if not unit_line:
        return ""

    picked = [unit_line]
    for line in top:
        if line == unit_line:
            continue
        if _looks_like_header(line):
            picked.append(line)
            if len(picked) >= 4:
                break
    # dict.fromkeys preserves order while dropping duplicates.
    return " | ".join(dict.fromkeys(picked))[:max_chars].rstrip(" |")


_TOKENIZER = None
_TOKENIZER_LOADED = False


def _get_tokenizer():
    """Load the same WordPiece tokenizer the embedding model uses, if present."""
    global _TOKENIZER, _TOKENIZER_LOADED
    if _TOKENIZER_LOADED:
        return _TOKENIZER
    _TOKENIZER_LOADED = True
    try:
        from tokenizers import Tokenizer

        path = os.path.join(
            os.path.expanduser("~"),
            ".cache",
            "chroma",
            "onnx_models",
            "all-MiniLM-L6-v2",
            "onnx",
            "tokenizer.json",
        )
        if os.path.exists(path):
            _TOKENIZER = Tokenizer.from_file(path)
            # The saved tokenizer.json carries a 128-token limit/padding, but
            # chroma's embedding function overrides it to 256 at runtime.
            _TOKENIZER.no_truncation()
            _TOKENIZER.no_padding()
    except Exception:  # noqa: BLE001 - stats are best-effort
        _TOKENIZER = None
    return _TOKENIZER


def count_tokens(text):
    tokenizer = _get_tokenizer()
    if tokenizer is not None:
        return len(tokenizer.encode(text or "", add_special_tokens=False).ids)
    # Conservative fallback: words plus punctuation.
    return len(re.findall(r"\w+|[^\w\s]", text or ""))


def token_stats(texts):
    counts = sorted(count_tokens(text) for text in texts)
    if not counts:
        return {"count": 0, "p50": 0, "p90": 0, "p95": 0, "max": 0, "over": 0}

    def pct(fraction):
        index = min(len(counts) - 1, round(fraction * (len(counts) - 1)))
        return counts[index]

    return {
        "count": len(counts),
        "p50": pct(0.50),
        "p90": pct(0.90),
        "p95": pct(0.95),
        "max": counts[-1],
        "over": sum(1 for c in counts if c > EMBED_MAX_TOKENS),
    }


def chunk_text(
    text,
    size=None,
    overlap=None,
    min_chars=None,
):
    """Split text into chunks, preferring line boundaries (table rows).

    `size` and `overlap` are character counts. Line snapping keeps a scheme
    name and its figures on the same row instead of splitting a row in half.
    """
    size = PDF_CHUNK_CHARS if size is None else max(int(size), 1)
    overlap = PDF_CHUNK_OVERLAP if overlap is None else max(int(overlap), 0)
    min_chars = PDF_CHUNK_MIN_CHARS if min_chars is None else max(int(min_chars), 0)
    overlap = min(overlap, size // 2)

    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            newline = text.rfind("\n", start + max(size // 2, 1), end)
            if newline != -1:
                end = newline
        # Fold a short page tail into this chunk instead of creating a tiny one.
        if len(text) - end <= max(min_chars // 2, 40):
            end = len(text)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        next_start = max(end - overlap, start + 1)
        # Back up to the beginning of the line so overlap starts on a full row.
        previous_newline = text.rfind("\n", 0, next_start)
        if previous_newline != -1 and previous_newline + 1 > start:
            next_start = previous_newline + 1
        start = next_start

    if len(chunks) > 1 and len(chunks[-1]) < min_chars:
        tail = chunks.pop()
        chunks[-1] = (chunks[-1] + "\n" + tail).strip()
    return chunks


def part_of(category):
    c = str(category).lower()
    if c.startswith("part a"):
        return "A"
    if c.startswith("part b"):
        return "B"
    if "grand" in c:
        return "A+B"
    return ""


def source_tag(rel, page=None, sheet=None, row=None):
    tag = f"[source: {rel}"
    if page is not None:
        tag += f", page {page}"
    if sheet is not None:
        tag += f", sheet {sheet}"
    if row is not None:
        tag += f", row {row}"
    return tag + "]"


def pairs_to_text(pairs):
    return "; ".join(f"{k}: {v}" for k, v in pairs if not is_empty(v))


# --------------------------------------------------------------------------- #
# Readers -> list of {"id", "text", "metadata"}
# --------------------------------------------------------------------------- #
def read_pdf(
    path,
    rel,
    chunk_chars=None,
    chunk_overlap=None,
    header_context=None,
    header_chars=None,
):
    chunk_chars = PDF_CHUNK_CHARS if chunk_chars is None else chunk_chars
    chunk_overlap = (
        PDF_CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap
    )
    header_context = (
        PDF_HEADER_CONTEXT if header_context is None else header_context
    )
    header_chars = PDF_HEADER_CHARS if header_chars is None else header_chars

    docs = []
    try:
        with pdfplumber.open(path) as pdf:
            pages = []
            for page in pdf.pages:
                try:
                    # layout=True keeps table columns on their original rows.
                    # extract_tables() cannot find these borderless tables
                    # (it returns 0 tables for the Delhi PDFs), so layout text
                    # is the reliable representation here.
                    txt = page.extract_text(layout=True)
                except Exception:
                    txt = page.extract_text()
                pages.append(clean_text(txt))
    except Exception as exc:  # noqa: BLE001 - fall back to pypdf
        print(f"    pdfplumber failed ({exc}); falling back to pypdf")
        reader = PdfReader(path)
        pages = [clean_text(p.extract_text()) for p in reader.pages]

    for page_no, page_text in enumerate(pages, start=1):
        header = source_tag(rel, page=page_no)
        context = (
            extract_header_context(page_text, header_chars)
            if header_context
            else ""
        )
        for ci, chunk in enumerate(
            chunk_text(page_text, size=chunk_chars, overlap=chunk_overlap)
        ):
            # The first chunk already begins with the page's own header.
            text = chunk
            if ci > 0 and context:
                text = f"[context: {context}] {chunk}"
            docs.append(
                {
                    "id": f"{rel}:p{page_no}:c{ci}",
                    "text": f"{header} {text}",
                    "metadata": {
                        "file": rel,
                        "page": page_no,
                        "chunk": ci,
                        "type": "pdf",
                    },
                }
            )
    return docs


def read_csv(path, rel):
    df = None
    last_err = None
    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding=encoding)
            break
        except UnicodeDecodeError as exc:
            last_err = exc
    if df is None:
        raise last_err

    columns = [str(c) for c in df.columns]
    year_cols = [c for c in columns if YEAR_RE.search(c)]
    id_like = ("category", "sub-category", "ministry", "scheme")

    docs = []
    for i, row in df.iterrows():
        values = row.tolist()
        pairs = list(zip(columns, values))
        text = f"{source_tag(rel, row=int(i))} {pairs_to_text(pairs)}"
        meta = {
            "file": rel,
            "row": int(i),
            "type": "csv",
            "part": part_of(row.get("Category", "")),
            "ministry": str(row.get("Ministry/Department", "")).strip(),
            "scheme": str(row.get("Scheme", "")).strip(),
            "years": ", ".join(str(row.get(c, "")) for c in year_cols),
        }
        docs.append({"id": f"{rel}:row{i}", "text": text, "metadata": meta})
    return docs


def _find_header_row(df):
    for i in range(min(len(df), 40)):
        joined = " ".join(
            str(v).lower() for v in df.iloc[i].tolist() if not is_empty(v)
        )
        if ("ministry" in joined or "department" in joined) and "total" in joined:
            return i
    return None


def _build_names(df, hdr):
    upper = df.iloc[hdr - 1].tolist() if hdr > 0 else [None] * df.shape[1]
    lower = df.iloc[hdr].tolist()
    names = []
    for j in range(df.shape[1]):
        lo = "" if is_empty(lower[j]) else str(lower[j]).strip()
        up = "" if is_empty(upper[j]) else str(upper[j]).strip()
        name = f"{up} {lo}".strip() if up else lo
        names.append(name or f"col{j}")
    return names


def read_excel(path, rel):
    docs = []
    sheets = pd.read_excel(path, sheet_name=None, header=None, dtype=str)
    for sheet_name, df in sheets.items():
        if df is None or df.empty:
            continue
        hdr = _find_header_row(df)
        names = _build_names(df, hdr) if hdr is not None else None

        for i in range(len(df)):
            values = df.iloc[i].tolist()
            if all(is_empty(v) for v in values):
                continue
            if names is not None and i > hdr:
                text_body = pairs_to_text(
                    [(names[j], values[j]) for j in range(len(values))]
                )
            else:
                text_body = " | ".join(
                    str(v).strip() for v in values if not is_empty(v)
                )
            if not text_body.strip():
                continue
            docs.append(
                {
                    "id": f"{rel}:{sheet_name}:row{i}",
                    "text": f"{source_tag(rel, sheet=sheet_name, row=i)} {text_body}",
                    "metadata": {
                        "file": rel,
                        "sheet": str(sheet_name),
                        "row": int(i),
                        "type": "xls",
                    },
                }
            )
    return docs


def collect_documents(
    corpus_dir=None,
    chunk_chars=None,
    chunk_overlap=None,
    header_context=None,
    header_chars=None,
):
    docs = []
    corpus_dir = CORPUS_DIR if corpus_dir is None else corpus_dir
    if not os.path.isdir(corpus_dir):
        raise SystemExit(f"Corpus directory not found: {corpus_dir}")

    for name in sorted(os.listdir(corpus_dir)):
        path = os.path.join(corpus_dir, name)
        if not os.path.isfile(path):
            continue
        ext = name.rsplit(".", 1)[-1].lower()
        print(f"Reading {name} ...")
        if ext == "pdf":
            docs.extend(
                read_pdf(
                    path,
                    name,
                    chunk_chars=chunk_chars,
                    chunk_overlap=chunk_overlap,
                    header_context=header_context,
                    header_chars=header_chars,
                )
            )
        elif ext == "csv":
            docs.extend(read_csv(path, name))
        elif ext in ("xls", "xlsx"):
            docs.extend(read_excel(path, name))
        else:
            print("    skipped (unsupported extension)")
    return docs


# --------------------------------------------------------------------------- #
# Indexing
# --------------------------------------------------------------------------- #
def build_index(docs, db_dir=None, collection_name=None):
    db_dir = DB_DIR if db_dir is None else db_dir
    collection_name = (
        COLLECTION_NAME if collection_name is None else collection_name
    )
    client = chromadb.PersistentClient(
        path=db_dir, settings=Settings(anonymized_telemetry=False)
    )
    try:
        client.delete_collection(collection_name)
        print(f"Dropped existing collection '{collection_name}'.")
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )

    total = len(docs)
    for start in range(0, total, BATCH_SIZE):
        batch = docs[start : start + BATCH_SIZE]
        collection.add(
            ids=[d["id"] for d in batch],
            documents=[d["text"] for d in batch],
            metadatas=[d["metadata"] for d in batch],
        )
        print(f"  indexed {min(start + BATCH_SIZE, total)}/{total}")
    return total


def print_stats(docs):
    """Print chunk/character/token distribution for a dry run."""
    print("\nChunk statistics (the embedding model truncates at "
          f"{EMBED_MAX_TOKENS} tokens):")
    by_file = {}
    for doc in docs:
        by_file.setdefault(doc["metadata"].get("file", "?"), []).append(doc["text"])

    for name in sorted(by_file):
        texts = by_file[name]
        lengths = sorted(len(text) for text in texts)
        stats = token_stats(texts)
        print(
            f"  {name:32s} chunks={len(texts):4d} "
            f"chars p50={lengths[len(lengths) // 2]:4d} "
            f"max={lengths[-1]:4d} | "
            f"tokens p50={stats['p50']:3d} p95={stats['p95']:3d} "
            f"max={stats['max']:3d} over={stats['over']:4d}"
        )

    pdf_texts = [
        doc["text"] for doc in docs if doc["metadata"].get("type") == "pdf"
    ]
    if pdf_texts:
        stats = token_stats(pdf_texts)
        over = stats["over"]
        print(
            f"\nPDF totals: {len(pdf_texts)} chunks | "
            f"tokens p50={stats['p50']} p90={stats['p90']} "
            f"p95={stats['p95']} max={stats['max']}"
        )
        if over:
            print(
                f"WARNING: {over} PDF chunk(s) exceed {EMBED_MAX_TOKENS} "
                "tokens and will be truncated by the embedding model. "
                "Reduce RAG_PDF_CHUNK_CHARS."
            )


def main():
    parser = argparse.ArgumentParser(description="Build the RAG vector index.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Rebuild the index (the collection is always recreated).",
    )
    parser.add_argument(
        "--corpus", default=None, help=f"Corpus folder (default: {CORPUS_DIR})"
    )
    parser.add_argument(
        "--db-dir", default=None, help=f"Vector store folder (default: {DB_DIR})"
    )
    parser.add_argument(
        "--collection",
        default=None,
        help=f"Collection name (default: {COLLECTION_NAME})",
    )
    parser.add_argument(
        "--chunk-chars",
        type=int,
        default=None,
        help=f"PDF chunk size in characters (default: {PDF_CHUNK_CHARS})",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help=f"PDF chunk overlap in characters (default: {PDF_CHUNK_OVERLAP})",
    )
    parser.add_argument(
        "--header-chars",
        type=int,
        default=None,
        help=f"Max characters of repeated page header (default: {PDF_HEADER_CHARS})",
    )
    parser.add_argument(
        "--no-header-context",
        action="store_true",
        help="Do not repeat unit/column headers on later chunks of a page.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and report chunk statistics without writing the index.",
    )
    args = parser.parse_args()

    chunk_chars = PDF_CHUNK_CHARS if args.chunk_chars is None else args.chunk_chars
    chunk_overlap = (
        PDF_CHUNK_OVERLAP if args.chunk_overlap is None else args.chunk_overlap
    )
    if chunk_chars < 100:
        raise SystemExit("--chunk-chars must be at least 100")
    if chunk_overlap < 0 or chunk_overlap >= chunk_chars:
        raise SystemExit("--chunk-overlap must be >= 0 and smaller than chunk size")

    docs = collect_documents(
        corpus_dir=args.corpus,
        chunk_chars=chunk_chars,
        chunk_overlap=chunk_overlap,
        header_context=not args.no_header_context,
        header_chars=args.header_chars,
    )
    if not docs:
        raise SystemExit("No documents found to index.")

    if args.dry_run:
        print_stats(docs)
        print(
            f"\nDry run: would index {len(docs)} chunks with "
            f"chunk={chunk_chars} overlap={chunk_overlap} chars, "
            f"header_context={'off' if args.no_header_context else 'on'}."
        )
        return

    pdf_texts = [
        doc["text"] for doc in docs if doc["metadata"].get("type") == "pdf"
    ]
    if pdf_texts:
        stats = token_stats(pdf_texts)
        print(
            f"PDF chunks: {len(pdf_texts)} | tokens p50={stats['p50']} "
            f"p95={stats['p95']} max={stats['max']}"
        )
        if stats["over"]:
            print(
                f"WARNING: {stats['over']} PDF chunk(s) exceed "
                f"{EMBED_MAX_TOKENS} tokens and will be truncated by the "
                "embedding model. Run with --dry-run to inspect, then reduce "
                "RAG_PDF_CHUNK_CHARS."
            )

    count = build_index(
        docs, db_dir=args.db_dir, collection_name=args.collection
    )
    db_dir = DB_DIR if args.db_dir is None else args.db_dir
    collection_name = (
        COLLECTION_NAME if args.collection is None else args.collection
    )
    print(
        f"\nDone. Indexed {count} chunks into '{db_dir}' "
        f"(collection: {collection_name})."
    )
    print(f"Chunk size: {chunk_chars} chars, overlap: {chunk_overlap} chars.")
    print("Start the web app with:  python app.py")


if __name__ == "__main__":
    main()
