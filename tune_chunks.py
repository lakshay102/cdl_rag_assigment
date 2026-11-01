"""Sweep PDF chunk sizes and score each index on the dev questions.

Each size is built into its own Chroma directory/collection, so the production
`chroma_db` is never touched. The chunk/overlap values are passed through the
same environment variables the app uses:

    RAG_PDF_CHUNK_CHARS, RAG_PDF_CHUNK_OVERLAP, RAG_DB_DIR, RAG_COLLECTION

Examples
--------
    # Retrieval-only sweep (no LLM calls, fast, free)
    python tune_chunks.py --no-llm --sizes 400,600,800,1200

    # Full answer-quality sweep with the configured provider (costs API calls)
    python tune_chunks.py --sizes 500,700,900 --limit 6

    # Compare with/without the repeated page header
    python tune_chunks.py --no-llm --no-header --sizes 600,800

The script prints a table and writes `summary.json` in the work directory.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(cmd, env):
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        print(output)
        raise SystemExit(f"command failed ({proc.returncode}): {' '.join(cmd)}")
    return output


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def pct(value):
    return "n/a" if value is None else f"{value:.0%}"


def load_report(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def score(report_path):
    rows = load_report(report_path)
    metrics = [row.get("metrics") or {} for row in rows]
    return {
        "questions": len(rows),
        "answered": sum(1 for row in rows if row.get("answered")),
        "source_hit": mean([m.get("source_hit") for m in metrics]),
        "source_recall": mean([m.get("source_recall") for m in metrics]),
        "number_recall": mean([m.get("number_recall") for m in metrics]),
        "keyword_recall": mean([m.get("keyword_recall") for m in metrics]),
        "unit_hit": mean(
            [None if m.get("unit_hit") is None else float(bool(m["unit_hit"])) for m in metrics]
        ),
    }


def parse_token_stats(output):
    match = re.search(
        r"PDF totals:\s*(\d+) chunks\s*\|\s*tokens p50=(\d+) p90=(\d+) "
        r"p95=(\d+) max=(\d+)",
        output,
    )
    if not match:
        return {}
    over = re.search(r"WARNING:\s*(\d+) PDF chunk", output)
    return {
        "pdf_chunks": int(match.group(1)),
        "token_p50": int(match.group(2)),
        "token_p90": int(match.group(3)),
        "token_p95": int(match.group(4)),
        "token_max": int(match.group(5)),
        "over_limit": int(over.group(1)) if over else 0,
    }


def chunks_from_ingest(output):
    match = re.search(r"Done\. Indexed (\d+) chunks", output)
    return int(match.group(1)) if match else None


def main():
    parser = argparse.ArgumentParser(
        description="Build several chunk sizes and score them on dev questions."
    )
    parser.add_argument(
        "--sizes",
        default="400,600,800,1200",
        help="Comma-separated PDF chunk sizes in characters.",
    )
    parser.add_argument(
        "--overlap-ratio",
        type=float,
        default=0.18,
        help="Overlap as a fraction of chunk size (default: 0.18).",
    )
    parser.add_argument(
        "--src",
        default="questions_dev.jsonl",
        help="Questions file with gold fields (default: questions_dev.jsonl).",
    )
    parser.add_argument(
        "--workdir",
        default=".sweeps",
        help="Where to put the per-size DBs, answers and reports (default: .sweeps).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run the first N questions (useful for a cheap trial).",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Set RAG_USE_LLM=0 (retrieval/extractive metrics only, no API cost).",
    )
    parser.add_argument(
        "--hints",
        action="store_true",
        help="Send the dev-set `tests` notes to the LLM (default: off for a fair comparison).",
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Disable the repeated page unit/column-header context.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete the work directory before starting.",
    )
    args = parser.parse_args()

    sizes = [int(part.strip()) for part in args.sizes.split(",") if part.strip()]
    if not sizes:
        raise SystemExit("--sizes must contain at least one number")
    workdir = Path(args.workdir)
    if not workdir.is_absolute():
        workdir = ROOT / workdir
    if args.clean and workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    summary = []
    for size in sizes:
        overlap = max(40, int(round(size * args.overlap_ratio)))
        overlap = min(overlap, size // 2)
        db_dir = workdir / f"chroma_c{size}"
        collection = f"gender_budget_c{size}"
        env = os.environ.copy()
        env.update(
            {
                "RAG_DB_DIR": str(db_dir),
                "RAG_COLLECTION": collection,
                "RAG_PDF_CHUNK_CHARS": str(size),
                "RAG_PDF_CHUNK_OVERLAP": str(overlap),
                "RAG_PDF_HEADER_CONTEXT": "0" if args.no_header else "1",
            }
        )
        if args.no_llm:
            env["RAG_USE_LLM"] = "0"

        print(f"\n=== chunk={size} overlap={overlap} ===")
        dry_output = run(
            [
                sys.executable,
                "ingest.py",
                "--dry-run",
                "--chunk-chars",
                str(size),
                "--chunk-overlap",
                str(overlap),
                *(["--no-header-context"] if args.no_header else []),
            ],
            env,
        )
        stats = parse_token_stats(dry_output)
        print(
            f"tokens p50={stats.get('token_p50')} p95={stats.get('token_p95')} "
            f"max={stats.get('token_max')} over-limit={stats.get('over_limit')}"
        )

        build_output = run(
            [
                sys.executable,
                "ingest.py",
                "--chunk-chars",
                str(size),
                "--chunk-overlap",
                str(overlap),
                *(["--no-header-context"] if args.no_header else []),
            ],
            env,
        )
        chunks = chunks_from_ingest(build_output)
        print(f"indexed {chunks} chunks into {db_dir}")

        dst = workdir / f"answers_c{size}.jsonl"
        report = workdir / f"answers_c{size}.report.jsonl"
        eval_cmd = [
            sys.executable,
            "run_eval.py",
            args.src,
            str(dst),
            "--report",
            str(report),
        ]
        if not args.hints:
            eval_cmd.append("--no-hints")
        if args.limit:
            eval_cmd.extend(["--limit", str(args.limit)])
        run(eval_cmd, env)

        result = {"chunk_chars": size, "overlap_chars": overlap, "chunks": chunks}
        result.update(stats)
        result.update(score(report))
        summary.append(result)
        print(
            f"answered {result['answered']}/{result['questions']} | "
            f"source {pct(result['source_hit'])} | "
            f"numbers {pct(result['number_recall'])} | "
            f"keywords {pct(result['keyword_recall'])} | "
            f"unit {pct(result['unit_hit'])}"
        )

    summary_path = workdir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n" + "=" * 88)
    print(
        f"{'size':>6} {'chunks':>7} {'tok p95':>8} {'over':>6} "
        f"{'ans':>6} {'source':>7} {'nums':>6} {'kw':>6} {'unit':>6}"
    )
    for row in summary:
        print(
            f"{row['chunk_chars']:>6} {str(row.get('chunks')):>7} "
            f"{str(row.get('token_p95')):>8} {str(row.get('over_limit')):>6} "
            f"{row['answered']:>3}/{row['questions']:<2} "
            f"{pct(row['source_hit']):>7} {pct(row['number_recall']):>6} "
            f"{pct(row['keyword_recall']):>6} {pct(row['unit_hit']):>6}"
        )
    print(f"\nsummary written to {summary_path}")
    print(
        "Pick the smallest size where source hits and numeric/keyword recall "
        "have plateaued (within one question of the best), and where token p95 "
        "is at or below the embedding limit. Larger chunks only add truncation "
        "and context noise after that point."
    )


if __name__ == "__main__":
    main()
