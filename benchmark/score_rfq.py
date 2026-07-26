#!/usr/bin/env python3
"""Score every questionnaire-corpus run file and emit one markdown table.

Reads whatever run files exist under results/, so it is safe to call while
arms are still being added. Output is the table the write-up quotes, so the
numbers in RESULTS.md always trace back to a command anyone can re-run.

  python3 score_rfq.py
  python3 score_rfq.py --k 10 > /tmp/table.md
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent

# (arm label, run-file stem, results subdir)
ARMS = [
    ("BM25 (lexical)", "bm25", "exp8"),
    ("semantic (embed the answer)", "semantic", "exp8"),
    ("hybrid — blueprint weights", "hybrid", "exp8"),
    ("RRF fusion (untrained)", "rrf", "exp8"),
    ("q2q-semantic — the library model", "q2q-semantic", "exp8"),
    ("q2q-RRF", "q2q-rrf", "exp8"),
    ("ColBERT late interaction", "colbert", "exp11"),
    ("cross-encoder over semantic", "ce-semantic", "exp9"),
    ("cross-encoder over hybrid", "ce-hybrid", "exp9"),
    ("cross-encoder over library recall", "ce-q2q", "exp9"),
    ("distilled student (served natively)", "student", "exp13"),
    ("semantic — int8 binary vectors", "semantic-binary", "gap3"),
    ("semantic — float32 vectors", "semantic-float", "gap3"),
]

SETS = {
    "para": ("queries-rfq-para.jsonl", "qrels-rfq-para.tsv"),
    "verbatim": ("queries-rfq.jsonl", "qrels-rfq.tsv"),
}


def score(run, queries, qrels, k):
    """Return (recall, mrr, ndcg, leaks) or None if the run file is absent."""
    if not run.exists() or run.stat().st_size < 10:
        return None
    out = subprocess.run(
        [sys.executable, str(HERE / "evaluate.py"), str(run), "--k", str(k),
         "--queries", str(HERE / queries), "--qrels", str(HERE / qrels),
         "--docs", str(HERE / "docs-rfq.jsonl")],
        capture_output=True, text=True).stdout
    vals, leaks = None, None
    for line in out.splitlines():
        if line.startswith("ALL"):
            p = line.split()
            vals = (float(p[2]), float(p[3]), float(p[4]))
        if "leakage" in line:
            leaks = line.split()[-1]
    return (*vals, leaks) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()
    res = HERE / "results"

    print(f"### Tenant-filtered, nDCG@{args.k}\n")
    print("| arm | paraphrase (47q) | verbatim (232q) |")
    print("|---|---|---|")
    for label, stem, sub in ARMS:
        row = []
        for s in ("para", "verbatim"):
            q, qr = SETS[s]
            r = score(res / sub / f"{s}-{stem}-filtered.txt", q, qr, args.k)
            row.append(f"{r[2]:.3f}" if r else "—")
        print(f"| {label} | {row[0]} | {row[1]} |")

    print(f"\n### What the tenant filter is worth (paraphrase set, nDCG@{args.k})\n")
    print("| arm | filtered | unfiltered | delta | cross-tenant leaks |")
    print("|---|---|---|---|---|")
    q, qr = SETS["para"]
    for label, stem, sub in ARMS:
        f = score(res / sub / f"para-{stem}-filtered.txt", q, qr, args.k)
        u = score(res / sub / f"para-{stem}.txt", q, qr, args.k)
        if not f or not u:
            continue
        print(f"| {label} | {f[2]:.3f} | {u[2]:.3f} | +{f[2]-u[2]:.3f} | {u[3]} |")

    print(f"\n### Full metrics, tenant-filtered (recall@{args.k} / MRR@{args.k} / nDCG@{args.k})\n")
    print("| arm | paraphrase | verbatim |")
    print("|---|---|---|")
    for label, stem, sub in ARMS:
        row = []
        for s in ("para", "verbatim"):
            q, qr = SETS[s]
            r = score(res / sub / f"{s}-{stem}-filtered.txt", q, qr, args.k)
            row.append(f"{r[0]:.3f} / {r[1]:.3f} / {r[2]:.3f}" if r else "—")
        print(f"| {label} | {row[0]} | {row[1]} |")


if __name__ == "__main__":
    main()
