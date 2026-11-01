"""Measure whether hybrid retrieval + reranking actually improve over raw vector
search, using questions_dev.jsonl (which ships gold source_files).

Usage:
    python eval_dev.py [k]

Reports source-file hit@1 and hit@k for:
  - vector-only  (raw ChromaDB distance order)
  - hybrid+rerank (what the app actually uses)
"""
import json
import sys

from app import retrieve, rerank

K = int(sys.argv[1]) if len(sys.argv) > 1 else 5


def first_gold_rank(hits, gold):
    for rank, h in enumerate(hits, start=1):
        if h["metadata"].get("file") in gold:
            return rank
    return None


def main():
    rows = [json.loads(l) for l in open("questions_dev.jsonl", encoding="utf-8") if l.strip()]

    results = {"vector": [], "rerank": []}
    for q in rows:
        gold = set(q.get("source_files") or [])
        cands = retrieve(q["question"], 60)

        vector_only = sorted(
            [c for c in cands if c.get("distance") is not None],
            key=lambda c: c["distance"],
        )[:K]
        reranked = rerank(q["question"], cands, K)

        results["vector"].append(first_gold_rank(vector_only, gold))
        results["rerank"].append(first_gold_rank(reranked, gold))

    print(f"Dev questions: {len(rows)}   k={K}\n")
    header = f"{'method':<16}{'hit@1':>8}{'hit@k':>8}{'miss':>8}"
    print(header)
    print("-" * len(header))
    for name, ranks in results.items():
        hit1 = sum(1 for r in ranks if r == 1)
        hitk = sum(1 for r in ranks if r is not None)
        miss = sum(1 for r in ranks if r is None)
        print(f"{name:<16}{hit1:>8}{hitk:>8}{miss:>8}")

    print("\nPer-question first gold rank (vector -> rerank):")
    for q, v, r in zip(rows, results["vector"], results["rerank"]):
        flag = "" if (r or 999) <= (v or 999) else "   <-- regressed"
        print(f"  {q['question_id']}: {v} -> {r}{flag}")


if __name__ == "__main__":
    main()
