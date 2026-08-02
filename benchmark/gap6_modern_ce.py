#!/usr/bin/env python3
"""Gap 6: is "retire the cross-encoder" true of cross-encoders, or of one
2020-era model? Findings 24/29 rest entirely on ms-marco MiniLM-L-6. This
reranks the same candidate pools with a current reranker
(BAAI/bge-reranker-v2-m3, 568M, offline on MPS) and compares.

Pools: top-50 tenant-filtered candidates from the library model (q2q) and
hybrid first phases (results/gap4/, --hits 50). Reranker reads the stored
question + answer text, like the deployed ce_tokens input. Default eval is the
original 47 paraphrases so numbers sit next to Exp 9/10's 0.578/0.561;
--subset all scores the full scaled set next to ce-q2q's 0.683.

  python3 gap6_modern_ce.py
  python3 gap6_modern_ce.py --subset all
"""
import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

from sentence_transformers import CrossEncoder

from evaluate import load_jsonl, load_qrels, load_run, ndcg

HERE = Path(__file__).parent
MODEL = "BAAI/bge-reranker-v2-m3"
K = 10

POOLS = {
    "q2q":    "results/gap4/para-all50-q2q-semantic-filtered.txt",
    "hybrid": "results/gap4/para-all50-hybrid-filtered.txt",
}
BASELINES = {  # per-query comparators, restricted to whichever subset runs
    "q2q-first-phase": "results/exp15/para-all-q2q-semantic-filtered.txt",
    "old-ce-q2q":      "results/exp15/para-all-ce-q2q-filtered.txt",
}


def boot(diffs, rng, n_boot=10000):
    n = len(diffs)
    delta = sum(diffs) / n
    hits = sum(1 for _ in range(n_boot)
               if ((s := sum(diffs[rng.randrange(n)] for _ in range(n)) / n) <= 0) == (delta > 0))
    return delta, min(1.0, 2 * hits / n_boot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=["orig", "all"], default="orig")
    ap.add_argument("--depth", type=int, default=50)
    args = ap.parse_args()

    docs = {d["id"]: f'{d["title"]} {d["text"]}' for d in load_jsonl(HERE / "docs-rfq.jsonl")}
    qrels = load_qrels(HERE / "qrels-rfq-para-all.tsv")
    queries = [q for q in load_jsonl(HERE / "queries-rfq-para-all.jsonl")
               if any(g > 0 for g in qrels.get(q["query_id"], {}).values())]
    if args.subset == "orig":
        queries = [q for q in queries if q["cluster"] == "questionnaire-para"]
    qids = [q["query_id"] for q in queries]
    qtext = {q["query_id"]: q["query"] for q in queries}

    model = CrossEncoder(MODEL, device="mps", max_length=512)
    print(f"{len(qids)} queries, depth {args.depth}, {MODEL}")

    per = {}
    for pool_name, runfile in POOLS.items():
        pool = load_run(HERE / runfile)
        pairs, index = [], []
        for qid in qids:
            for d in pool.get(qid, [])[:args.depth]:
                pairs.append((qtext[qid], docs[d]))
                index.append((qid, d))
        t0 = time.time()
        scores = model.predict(pairs, batch_size=16, show_progress_bar=False)
        secs = time.time() - t0
        ranked = defaultdict(list)
        for (qid, d), s in zip(index, scores):
            ranked[qid].append((float(s), d))
        per[pool_name] = {qid: ndcg([d for _, d in sorted(ranked[qid], reverse=True)],
                                    qrels[qid], K) for qid in qids}
        mean = sum(per[pool_name].values()) / len(qids)
        print(f"bge over {pool_name:<7} nDCG@10 = {mean:.3f}   "
              f"({len(pairs)} pairs, {secs:.0f}s, {secs/len(qids):.2f}s/query)")

    rng = random.Random(6)
    base = {}
    for name, runfile in BASELINES.items():
        run = load_run(HERE / runfile)
        base[name] = {qid: ndcg(run.get(qid, []), qrels[qid], K) for qid in qids}
        print(f"{name:<18} nDCG@10 = {sum(base[name].values())/len(qids):.3f}")
    for pool_name in POOLS:
        for bname in BASELINES:
            d, p = boot([per[pool_name][q] - base[bname][q] for q in qids], rng)
            print(f"bge-{pool_name} vs {bname}: {d:+.3f} p≈{p:.4f}")

    out = HERE / "results" / "gap6"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"bge-{args.subset}.json").write_text(json.dumps(
        {p: {q: round(v, 6) for q, v in per[p].items()} for p in per}, indent=1))


if __name__ == "__main__":
    main()
