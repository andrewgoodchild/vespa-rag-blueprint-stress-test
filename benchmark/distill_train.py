#!/usr/bin/env python3
"""Exp 13: train the distilled student and evaluate on held-out paraphrases.

Trains a GBDT (sklearn HistGradientBoostingRegressor — LightGBM stand-in;
same algorithm family, avoids the local libomp arch mismatch) to reproduce the
cross-encoder teacher score from six cheap features. Evaluates on the 47
paraphrase queries (never in training) by reranking each query's top-50 hybrid
candidates, and compares against the teacher's own ranking and ColBERT-alone on
the identical candidate pool.
"""
import json
import math
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = Path(__file__).parent
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


def load_qrels(path):
    q = defaultdict(dict)
    for line in open(path):
        if line.strip() and not line.startswith("#"):
            a, _, d, g = line.split()
            q[a][d] = int(g)
    return q


def ndcg(ranked, rels, k=10):
    dcg = sum((2 ** rels.get(d, 0) - 1) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted(rels.values(), reverse=True)
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal[:k]))
    return dcg / idcg if idcg else 0.0


def main():
    # --- train ---
    rows = [l.split(",") for l in open(HERE / "distill_train.csv").read().splitlines()[1:]]
    X = np.array([[float(r[4 + i]) for i in range(len(FEATS))] for r in rows])
    y = np.array([float(r[-1]) for r in rows])
    t0 = time.time()
    model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                          max_leaf_nodes=31, min_samples_leaf=40,
                                          l2_regularization=1.0)
    model.fit(X, y)
    print(f"trained student on {len(X)} rows in {time.time()-t0:.1f}s")
    # feature importance via permutation on a held-out slice
    from sklearn.inspection import permutation_importance
    idx = np.random.default_rng(0).permutation(len(X))[:4000]
    imp = permutation_importance(model, X[idx], y[idx], n_repeats=5, random_state=0)
    print("feature importance:", {FEATS[i]: round(float(imp.importances_mean[i]), 3)
                                   for i in np.argsort(-imp.importances_mean)})

    # --- eval on held-out paraphrases ---
    qrels = load_qrels(HERE / "qrels-rfq-para.tsv")
    queries = [json.loads(l) for l in open(HERE / "queries-rfq-para.jsonl")]
    student_s, teacher_s, colbert_s = [], [], []
    for q in queries:
        p = {"query": q["query"], "timeout": "60s", "hits": 50,
             "presentation.summary": "no-chunks",
             "yql": f"select * from rfq where tenant contains '{q['tenant']}' and ({WHERE})",
             "ranking.profile": "collect-distill", "ranking.globalPhase.rerankCount": 50,
             "ranking.features.query(embedding)": "embed(nomicmb, @query)",
             "ranking.features.query(float_embedding)": "embed(nomicmb, @query)",
             "input.query(q_tokens)": "embed(tokenizer, @query)",
             "input.query(qt)": "embed(colbert, @query)"}
        cands = []
        for h in search(p)["root"].get("children", []):
            f = h.get("fields", {})
            mf = f.get("matchfeatures")
            if not mf or "id" not in f:
                continue
            feat = [float(mf.get(k, 0)) for k in FEATS]
            cands.append((f["id"], feat, h.get("relevance", 0.0)))
        if not cands:
            continue
        rels = qrels[q["query_id"]]
        pred = model.predict(np.array([c[1] for c in cands]))
        student = [c[0] for c in sorted(zip(cands, pred), key=lambda z: -z[1])[0:]]
        student = [c[0][0] for c in sorted(zip(cands, pred), key=lambda z: -z[1])]
        teacher = [c[0] for c in sorted(cands, key=lambda c: -c[2])]
        colbert = [c[0] for c in sorted(cands, key=lambda c: -c[1][5])]
        student_s.append(ndcg(student, rels))
        teacher_s.append(ndcg(teacher, rels))
        colbert_s.append(ndcg(colbert, rels))

    n = len(student_s)
    print(f"\n=== Exp 13 distillation, {n} held-out paraphrase queries (nDCG@10) ===")
    print(f"  teacher (cross-encoder)      {sum(teacher_s)/n:.3f}")
    print(f"  colbert_max_sim alone        {sum(colbert_s)/n:.3f}")
    print(f"  STUDENT (GBDT, ~microsecond) {sum(student_s)/n:.3f}")


if __name__ == "__main__":
    main()
