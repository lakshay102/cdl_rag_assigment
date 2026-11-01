"""Run the RAG system over a questions file and write answers.jsonl.

Dev questions (questions_dev.jsonl) carry extra gold fields:
    answer, source_files, source_location, unit, tests

When those fields are present this script:
  * sends the `tests` note to the LLM as a per-question focus note (it explains
    what is being tested, e.g. "unit conversion", "year ambiguity"); disable
    that with --no-hints;
  * scores the answer against the gold answer, source files and unit;
  * writes a per-question report (default: <dst>.report.jsonl).

The submitted `answers.jsonl` format is unchanged: one
{"question_id", "answered", "answer", "sources"} object per line.

Usage:
    python run_eval.py questions_eval.jsonl answers.jsonl
    python run_eval.py questions_dev.jsonl          # -> answers_dev.jsonl
    python run_eval.py questions_dev.jsonl answers_dev.jsonl
    python run_eval.py questions_dev.jsonl answers_dev.jsonl --no-hints
    python run_eval.py questions_dev.jsonl answers_dev.jsonl --report dev.report.jsonl
    python run_eval.py questions_dev.jsonl --limit 6
"""
import argparse
import json
import os
import re

from app import AskRequest, ask, tokenize
from config import USE_LLM

NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")

# The first unit mentioned in the gold `unit` field is the one being tested.
UNIT_ALIASES = {
    "crore": ("crore", "cr"),
    "lakh": ("lakh", "lac"),
    "thousand": ("thousand",),
    "rupee": ("rupee", "rupees", "rs", "inr", "\u20b9"),
    "percent": ("percent", "percentage", "%"),
}


def load_rows(path):
    """Read one JSON object per line and fail loudly on malformed rows."""
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not row.get("question_id") or not row.get("question"):
                raise SystemExit(
                    f"{path}:{line_no}: each row needs question_id and question"
                )
            rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Scoring helpers for the dev set
# --------------------------------------------------------------------------- #
def extract_numbers(text):
    values = []
    for token in NUM_RE.findall(text or ""):
        try:
            values.append(float(token.replace(",", "")))
        except ValueError:
            continue
    # Preserve order but drop duplicates.
    return list(dict.fromkeys(values))


def _same_number(a, b):
    if a == b:
        return True
    if not a or not b:
        return False
    return abs(a - b) / max(abs(a), abs(b)) <= 0.005


def number_recall(predicted, gold):
    """Fraction of gold numbers that appear in the predicted answer."""
    gold_numbers = extract_numbers(gold)
    if not gold_numbers:
        return None, [], []
    predicted_numbers = extract_numbers(predicted)
    matched, missing = [], []
    for gold_value in gold_numbers:
        if any(_same_number(gold_value, value) for value in predicted_numbers):
            matched.append(gold_value)
        else:
            missing.append(gold_value)
    return len(matched) / len(gold_numbers), matched, missing


def _basename(path):
    return os.path.basename(str(path).replace("\\", "/")).lower()


def source_recall(sources, gold_files):
    """Fraction of the gold source files returned by the retriever."""
    gold = list(dict.fromkeys(_basename(f) for f in (gold_files or []) if f))
    if not gold:
        return None, [], []
    actual = {_basename(s.get("file")) for s in (sources or []) if s.get("file")}
    matched = [name for name in gold if name in actual]
    missing = [name for name in gold if name not in actual]
    return len(matched) / len(gold), matched, missing


def keyword_recall(predicted, gold):
    """Fraction of meaningful gold answer tokens present in the prediction."""
    gold_tokens = set(tokenize(gold))
    if not gold_tokens:
        return None
    predicted_tokens = set(tokenize(predicted))
    return len(gold_tokens & predicted_tokens) / len(gold_tokens)


def _contains_alias(text, aliases):
    for alias in aliases:
        if alias in ("%", "\u20b9"):
            if alias in text:
                return True
        elif re.search(rf"\b{re.escape(alias)}\b", text):
            return True
    return False


def unit_hit(predicted, unit):
    """True when the answer states the unit the gold answer is reported in."""
    if not unit:
        return None
    gold_unit = unit.lower()
    expected = next(
        (
            name
            for name, aliases in UNIT_ALIASES.items()
            if _contains_alias(gold_unit, aliases)
        ),
        None,
    )
    if expected is None:
        return None
    return _contains_alias((predicted or "").lower(), UNIT_ALIASES[expected])


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _fmt(value):
    return "n/a" if value is None else f"{value:.0%}"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Run the RAG system over a questions file."
    )
    parser.add_argument("src", nargs="?", default="questions_eval.jsonl")
    parser.add_argument(
        "dst",
        nargs="?",
        default=None,
        help=(
            "Output file. Default: answers.jsonl, or answers_dev.jsonl when "
            "the questions file name contains 'dev'."
        ),
    )
    parser.add_argument(
        "--report",
        default=None,
        help=(
            "Where to write per-question diagnostics. Default: "
            "<dst>.report.jsonl when the file contains gold fields."
        ),
    )
    parser.add_argument(
        "--no-hints",
        action="store_true",
        help="Do not send the dev-set `tests` note to the LLM.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only answer the first N questions (useful for a cheap trial).",
    )
    args = parser.parse_args()

    dst = args.dst
    if dst is None:
        stem = os.path.splitext(os.path.basename(args.src))[0]
        dst = "answers_dev.jsonl" if "dev" in stem.lower() else "answers.jsonl"

    rows = load_rows(args.src)
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be at least 1")
        rows = rows[: args.limit]
    has_gold = any(
        row.get("answer") or row.get("source_files") or row.get("unit")
        for row in rows
    )
    report_path = args.report
    if report_path is None and has_gold:
        report_path = os.path.splitext(dst)[0] + ".report.jsonl"

    totals = {
        "answered": 0,
        "source": [],
        "numbers": [],
        "keywords": [],
        "units": [],
    }

    out = open(dst, "w", encoding="utf-8")
    report_out = (
        open(report_path, "w", encoding="utf-8") if report_path else None
    )
    try:
        for i, q in enumerate(rows, start=1):
            hint = None if args.no_hints else q.get("tests")
            result = None
            try:
                result = ask(AskRequest(question=q["question"], test_hint=hint))
            except Exception as exc:  # noqa: BLE001
                print(f"    error: {exc}")

            metrics = {}
            if result is None:
                record = {
                    "question_id": q["question_id"],
                    "answered": False,
                    "answer": "",
                    "sources": [],
                }
                report = {
                    "question_id": q["question_id"],
                    "answered": False,
                    "answer": "",
                    "sources": [],
                    "engine": None,
                    "llm_error": None,
                    "test_hint": hint,
                    "gold_answer": q.get("answer"),
                    "gold_sources": q.get("source_files"),
                    "metrics": metrics,
                    "error": "ask() failed",
                }
                print(f"[{i}/{len(rows)}] {q['question_id']} answered=False")
            else:
                answered = result["answered"]
                record = {
                    "question_id": q["question_id"],
                    "answered": answered,
                    "answer": result["answer"] if answered else "",
                    "sources": result["sources"],
                }

                src_value, src_matched, src_missing = source_recall(
                    result.get("sources"), q.get("source_files")
                )
                if src_value is not None:
                    metrics["source_recall"] = round(src_value, 4)
                    metrics["source_hit"] = src_value == 1.0
                    metrics["sources_matched"] = src_matched
                    metrics["sources_missing"] = src_missing

                num_value, num_matched, num_missing = number_recall(
                    result.get("answer"), q.get("answer")
                )
                if num_value is not None:
                    metrics["number_recall"] = round(num_value, 4)
                    metrics["numbers_matched"] = num_matched
                    metrics["numbers_missing"] = num_missing

                kw_value = keyword_recall(result.get("answer"), q.get("answer"))
                if kw_value is not None:
                    metrics["keyword_recall"] = round(kw_value, 4)

                u_hit = unit_hit(result.get("answer"), q.get("unit"))
                if u_hit is not None:
                    metrics["unit_hit"] = u_hit

                report = {
                    "question_id": q["question_id"],
                    "answered": answered,
                    "answer": result["answer"],
                    "sources": result["sources"],
                    "engine": result.get("engine"),
                    "llm_error": result.get("llm_error"),
                    "test_hint": hint,
                    "gold_answer": q.get("answer"),
                    "gold_sources": q.get("source_files"),
                    "metrics": metrics,
                }

                totals["answered"] += int(answered)
                if src_value is not None:
                    totals["source"].append(src_value)
                if num_value is not None:
                    totals["numbers"].append(num_value)
                if kw_value is not None:
                    totals["keywords"].append(kw_value)
                if u_hit is not None:
                    totals["units"].append(1.0 if u_hit else 0.0)

                flags = []
                if src_value is not None:
                    flags.append(
                        "src=" + ("hit" if src_value == 1.0 else f"{src_value:.0%}")
                    )
                if num_value is not None:
                    flags.append(f"nums={num_value:.0%}")
                if u_hit is not None:
                    flags.append("unit=" + ("ok" if u_hit else "MISS"))
                print(
                    f"[{i}/{len(rows)}] {q['question_id']} "
                    f"answered={answered} " + " ".join(flags)
                )

            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            if report_out is not None:
                report_out.write(json.dumps(report, ensure_ascii=False) + "\n")
                report_out.flush()
    finally:
        out.close()
        if report_out is not None:
            report_out.close()

    print(f"wrote {dst}")
    if report_path:
        print(f"wrote {report_path}")
        print(
            "summary: "
            f"answered {totals['answered']}/{len(rows)}"
            f" | source recall {_fmt(_mean(totals['source']))}"
            f" | number recall {_fmt(_mean(totals['numbers']))}"
            f" | keyword recall {_fmt(_mean(totals['keywords']))}"
            f" | unit hit {_fmt(_mean(totals['units']))}"
        )
        if USE_LLM and not args.no_hints and any(row.get("tests") for row in rows):
            print(
                "hints: dev `tests` notes were sent to the LLM "
                "(disable with --no-hints)"
            )


if __name__ == "__main__":
    main()
