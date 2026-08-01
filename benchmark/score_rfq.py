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

# (arm label, run-file stem); run files are located by scanning RESULT_DIRS
# for {set}-{stem}-filtered.txt, so an arm re-run on a new query set (e.g. the
# scaled paraphrase set in exp15) is picked up without touching this table.
ARMS = [
    ("BM25 (lexical)", "bm25"),
    ("semantic (embed the answer)", "semantic"),
    ("hybrid — blueprint weights", "hybrid"),
    ("RRF fusion (untrained)", "rrf"),
    ("q2q-semantic — the library model", "q2q-semantic"),
    ("q2q-RRF", "q2q-rrf"),
    ("ColBERT late interaction", "colbert"),
    ("cross-encoder over semantic", "ce-semantic"),
    ("cross-encoder over hybrid", "ce-hybrid"),
    ("cross-encoder over library recall", "ce-q2q"),
    ("distilled student (served natively)", "student"),
    ("semantic — int8 binary vectors", "semantic-binary"),
    ("semantic — float32 vectors", "semantic-float"),
    # Exps 14-15: HyDE query transformation — paraphrase sets only (~2 s LLM
    # call per query at generation time; see hyde_generate.py)
    ("HyDE hypothetical answer", "hydeans-semantic"),
    ("HyDE answer + query", "hydeanscat-semantic"),
    ("HyDE hypothetical question", "hydeq-q2q-semantic"),
    ("HyDE question + query", "hydeqcat-q2q-semantic"),
]

RESULT_DIRS = ["exp8", "exp9", "exp11", "exp13", "exp14", "exp15", "gap3"]

SETS = {
    "para": ("queries-rfq-para.jsonl", "qrels-rfq-para.tsv"),
    "para-all": ("queries-rfq-para-all.jsonl", "qrels-rfq-para-all.tsv"),
    "verbatim": ("queries-rfq.jsonl", "qrels-rfq.tsv"),
}


def find_run(setname, stem, filtered=True):
    name = f"{setname}-{stem}{'-filtered' if filtered else ''}.txt"
    for sub in RESULT_DIRS:
        p = HERE / "results" / sub / name
        if p.exists():
            return p
    return HERE / "results" / "absent" / name


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
    print("| arm | paraphrase (47q) | scaled paraphrase (232q) | verbatim (232q) |")
    print("|---|---|---|---|")
    for label, stem in ARMS:
        row = []
        for s in ("para", "para-all", "verbatim"):
            q, qr = SETS[s]
            r = score(find_run(s, stem), q, qr, args.k)
            row.append(f"{r[2]:.3f}" if r else "—")
        print(f"| {label} | {row[0]} | {row[1]} | {row[2]} |")

    print(f"\n### What the tenant filter is worth (paraphrase set, nDCG@{args.k})\n")
    print("| arm | filtered | unfiltered | delta | cross-tenant leaks |")
    print("|---|---|---|---|---|")
    q, qr = SETS["para"]
    for label, stem in ARMS:
        f = score(find_run("para", stem), q, qr, args.k)
        u = score(find_run("para", stem, filtered=False), q, qr, args.k)
        if not f or not u:
            continue
        print(f"| {label} | {f[2]:.3f} | {u[2]:.3f} | +{f[2]-u[2]:.3f} | {u[3]} |")

    print(f"\n### Full metrics, tenant-filtered (recall@{args.k} / MRR@{args.k} / nDCG@{args.k})\n")
    print("| arm | paraphrase | scaled paraphrase | verbatim |")
    print("|---|---|---|---|")
    for label, stem in ARMS:
        row = []
        for s in ("para", "para-all", "verbatim"):
            q, qr = SETS[s]
            r = score(find_run(s, stem), q, qr, args.k)
            row.append(f"{r[0]:.3f} / {r[1]:.3f} / {r[2]:.3f}" if r else "—")
        print(f"| {label} | {row[0]} | {row[1]} | {row[2]} |")


if __name__ == "__main__":
    main()
