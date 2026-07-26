#!/usr/bin/env python3
"""Score a retrieval run against the RFP benchmark.

Usage:
    python evaluate.py run.txt [--k 10]

Run file format (TREC): query_id  Q0  doc_id  rank  score  tag
Reports Recall@k, MRR@k, nDCG@k overall and per cluster, plus:
  - tenant leakage: any returned doc belonging to a different tenant
  - unanswerable handling: docs returned for queries with no relevant docs
"""
import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_qrels(path):
    qrels = defaultdict(dict)
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        qid, _, docid, grade = line.split()
        qrels[qid][docid] = int(grade)
    return qrels


def load_run(path):
    run = defaultdict(list)
    for line in open(path):
        if not line.strip():
            continue
        parts = line.split()
        qid, docid = parts[0], parts[2]
        run[qid].append(docid)
    return run


def ndcg(ranked, rels, k):
    dcg = sum(
        (2 ** rels.get(d, 0) - 1) / math.log2(i + 2)
        for i, d in enumerate(ranked[:k])
    )
    ideal = sorted(rels.values(), reverse=True)
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal[:k]))
    return dcg / idcg if idcg > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--queries", default=str(HERE / "queries.jsonl"))
    ap.add_argument("--qrels", default=str(HERE / "qrels.tsv"))
    ap.add_argument("--docs", default=str(HERE / "docs.jsonl"))
    args = ap.parse_args()

    docs = {d["id"]: d for d in load_jsonl(args.docs)}
    queries = {q["query_id"]: q for q in load_jsonl(args.queries)}
    qrels = load_qrels(args.qrels)
    run = load_run(args.run)

    per_cluster = defaultdict(list)
    leaks, unanswerable_hits = [], []

    for qid, q in sorted(queries.items()):
        ranked = run.get(qid, [])
        for d in ranked[: args.k]:
            if d in docs and docs[d]["tenant"] != q["tenant"]:
                leaks.append((qid, d))

        relevant = {d: g for d, g in qrels.get(qid, {}).items() if g > 0}
        if not relevant:  # unanswerable query
            if ranked:
                unanswerable_hits.append((qid, len(ranked[: args.k])))
            continue

        hits = [d for d in ranked[: args.k] if d in relevant]
        recall = len(set(hits)) / len(relevant)
        rr = 0.0
        for i, d in enumerate(ranked[: args.k]):
            if d in relevant:
                rr = 1 / (i + 1)
                break
        n = ndcg(ranked, qrels[qid], args.k)
        per_cluster[q["cluster"]].append((qid, recall, rr, n))

    all_rows = [r for rows in per_cluster.values() for r in rows]
    if not all_rows:
        sys.exit("no scored queries — is the run file empty?")

    def avg(rows, i):
        return sum(r[i] for r in rows) / len(rows)

    print(f"{'cluster':<18}{'n':>3}{'recall@k':>10}{'mrr@k':>8}{'ndcg@k':>8}")
    for cluster, rows in sorted(per_cluster.items()):
        print(f"{cluster:<18}{len(rows):>3}{avg(rows,1):>10.3f}{avg(rows,2):>8.3f}{avg(rows,3):>8.3f}")
    print("-" * 47)
    print(f"{'ALL':<18}{len(all_rows):>3}{avg(all_rows,1):>10.3f}{avg(all_rows,2):>8.3f}{avg(all_rows,3):>8.3f}")

    print(f"\ntenant leakage violations: {len(leaks)}")
    for qid, d in leaks:
        print(f"  {qid}: returned {d} (tenant {docs[d]['tenant']}, query tenant {queries[qid]['tenant']})")
    for qid, n in unanswerable_hits:
        print(f"unanswerable {qid}: {n} docs returned in top-{args.k} "
              f"(fine for retrieval; the generator must decline — check answer-stage behaviour)")


if __name__ == "__main__":
    main()
