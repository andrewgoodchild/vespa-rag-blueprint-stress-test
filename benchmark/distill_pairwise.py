#!/usr/bin/env python3
"""Exp 13 (pairwise variant): distill the cross-encoder into a pairwise-linear
student. Trains a logistic model on within-query feature differences ordered by
teacher score (RankNet-style), yielding a linear scorer w·features. Evaluates on
the 47 held-out paraphrases against teacher and ColBERT on the same top-50 pool.

Pointwise regression (distill_train.py) underperformed both teacher and the best
single feature; the pairwise objective recovers to teacher-beating quality — the
same lesson as Exps 2 vs 4.
"""
import json
import math
import time
import urllib.request
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = __import__("pathlib").Path(__file__).parent
FEATS = ["bm25(title)", "bm25(chunks)", "max_chunk_sim_scores",
         "avg_top_3_chunk_sim_scores", "title_sim", "colbert_max_sim"]
WHERE = ('userInput(@query) or ({label:"t",targetHits:200}nearestNeighbor(title_embedding,embedding)) '
         'or ({label:"c",targetHits:200}nearestNeighbor(chunk_embeddings,embedding))')


def search(p, retries=4):
    for _ in range(retries):
        try:
            req = urllib.request.Request("http://127.0.0.1:8080/search/",
                data=json.dumps(p).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.load(r)
        except Exception:
            time.sleep(3)
    return {"root": {}}


def ndcg(ranked, rels, k=10):
    dcg = sum((2 ** rels.get(d, 0) - 1) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted(rels.values(), reverse=True)
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal[:k]))
    return dcg / idcg if idcg else 0.0


def main():
    rows = [l.split(",") for l in open(HERE / "distill_train.csv").read().splitlines()[1:]]
    by_q = defaultdict(list)
    for r in rows:
        by_q[r[0]].append(([float(r[4 + i]) for i in range(6)], float(r[-1])))
    Xd, yd = [], []
    for items in by_q.values():
        for fa, ta in items:
            for fb, tb in items:
                if ta - tb > 0.5:
                    d = np.array(fa) - np.array(fb)
                    Xd.append(d); yd.append(1)
                    Xd.append(-d); yd.append(0)
    Xd, yd = np.array(Xd), np.array(yd)
    sc = StandardScaler(with_mean=False).fit(Xd)
    m = LogisticRegression(fit_intercept=False, max_iter=2000).fit(sc.transform(Xd), yd)
    w = m.coef_[0] / sc.scale_
    print(f"pairwise student: {len(Xd)} pairs; weights",
          {FEATS[i]: round(float(w[i]), 3) for i in np.argsort(-np.abs(w))})

    qrels = defaultdict(dict)
    for line in open(HERE / "qrels-rfq-para.tsv"):
        if line.strip():
            a, _, dd, g = line.split(); qrels[a][dd] = int(g)
    queries = [json.loads(l) for l in open(HERE / "queries-rfq-para.jsonl")]
    stu, tea, col = [], [], []
    for q in queries:
        p = {"query": q["query"], "timeout": "60s", "hits": 50, "presentation.summary": "no-chunks",
             "yql": f"select * from rfq where tenant contains '{q['tenant']}' and ({WHERE})",
             "ranking.profile": "collect-distill", "ranking.globalPhase.rerankCount": 50,
             "ranking.features.query(embedding)": "embed(nomicmb, @query)",
             "ranking.features.query(float_embedding)": "embed(nomicmb, @query)",
             "input.query(q_tokens)": "embed(tokenizer, @query)",
             "input.query(qt)": "embed(colbert, @query)"}
        cands = []
        for h in search(p)["root"].get("children", []):
            f = h.get("fields", {}); mf = f.get("matchfeatures")
            if mf and "id" in f:
                cands.append((f["id"], [float(mf.get(k, 0)) for k in FEATS], h.get("relevance", 0.0)))
        if not cands:
            continue
        rels = qrels[q["query_id"]]
        sco = np.array([c[1] for c in cands]) @ w
        stu.append(ndcg([c[0] for c, _ in sorted(zip(cands, sco), key=lambda z: -z[1])], rels))
        tea.append(ndcg([c[0] for c in sorted(cands, key=lambda c: -c[2])], rels))
        col.append(ndcg([c[0] for c in sorted(cands, key=lambda c: -c[1][5])], rels))
    n = len(stu)
    print(f"\n=== {n} held-out paraphrases (nDCG@10, top-50 pool) ===")
    print(f"  teacher (cross-encoder)          {sum(tea)/n:.3f}")
    print(f"  colbert_max_sim alone            {sum(col)/n:.3f}")
    print(f"  STUDENT (pairwise-linear)        {sum(stu)/n:.3f}")

    # Persist the learned scorer so the deployed rank profile can be regenerated
    # rather than hand-copied: student_expr.json is pasted into the
    # second-phase of rfq/student-pairwise.profile.
    weights = {k: float(v) for k, v in zip(FEATS, w)}
    (HERE / "student_weights.json").write_text(json.dumps(weights, indent=1) + "\n")
    expr = " + ".join(f"{v:.6f}*{k}" for k, v in weights.items())
    (HERE / "student_expr.json").write_text(json.dumps({"expr": expr}) + "\n")
    print(f"\nwrote student_weights.json and student_expr.json")
    print("  paste into vespa-app/schemas/rfq/student-pairwise.profile second-phase:")
    print(f"  {expr}")


if __name__ == "__main__":
    main()
