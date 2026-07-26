#!/usr/bin/env python3
"""Experiment 4: pairwise (RankNet-style) training of the linear first phase.

Exp 2 showed pointwise logistic regression cannot learn the freshness feature
(stale docs vanish among ~720 negatives). Here the training unit is a
within-query preference pair: for docs a, b in the same query with
grade(a) > grade(b), the model is trained on the feature difference
f(a) - f(b). Graded qrels supply the crucial stale-vs-current contrasts
directly (e.g. soc2-2025 grade 2 vs soc2-2023 grade 1).

Same grouped 5-fold CV and query-parameter scoring as train_linear.py
(intercept fixed at 0 — rank-invariant). Reads features from exp2's
features.csv. Outputs pairwise-6f.txt / pairwise-7f.txt + coefficients.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from train_linear import FEATURES6, FEATURES7, load_qrels, score_query

HERE = Path(__file__).parent


def load_features(path):
    rows = []
    for r in csv.DictReader(open(path)):
        rows.append(
            {"query_id": r["query_id"], "doc_id": r["doc_id"], **{f: float(r[f]) for f in FEATURES7}}
        )
    return rows


def make_pairs(rows, qrels, qids, features):
    """Feature diffs for every within-query pair with unequal grades."""
    by_q = {}
    for r in rows:
        by_q.setdefault(r["query_id"], []).append(r)
    X, y = [], []
    for qid in qids:
        grades = qrels.get(qid, {})
        docs = by_q.get(qid, [])
        for a in docs:
            ga = grades.get(a["doc_id"], 0)
            if ga == 0:
                continue
            va = np.array([a[f] for f in features])
            for b in docs:
                gb = grades.get(b["doc_id"], 0)
                if gb >= ga:
                    continue
                vb = np.array([b[f] for f in features])
                X.append(va - vb)
                y.append(1)
                X.append(vb - va)
                y.append(0)
    return np.array(X), np.array(y)


def fit_pairwise(X, y):
    scaler = StandardScaler(with_mean=False).fit(X)  # diffs are symmetric; no centering
    model = LogisticRegression(fit_intercept=False, max_iter=2000)
    model.fit(scaler.transform(X), y)
    return model.coef_[0] / scaler.scale_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:8080")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--features-csv", default=str(HERE / "results/exp2/features.csv"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--queries", default=str(HERE / "queries.jsonl"))
    ap.add_argument("--qrels", default=str(HERE / "qrels.tsv"))
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    queries = [json.loads(l) for l in open(args.queries)]
    qrels = load_qrels(args.qrels)
    rows = load_features(args.features_csv)

    rng = np.random.default_rng(args.seed)
    trainable = sorted(q["query_id"] for q in queries if any(g > 0 for g in qrels.get(q["query_id"], {}).values()))
    order = list(rng.permutation(trainable))
    folds = [set(order[i :: args.folds]) for i in range(args.folds)]

    all_coefs = {}
    for arm, profile, features in [
        ("pairwise-6f", "learned-linear", FEATURES6),
        ("pairwise-7f", "learned-linear-v2", FEATURES7),
    ]:
        lines = []
        for i, held_out in enumerate(folds):
            train_qids = set(trainable) - held_out
            X, y = make_pairs(rows, qrels, train_qids, features)
            coef = fit_pairwise(X, y)
            coefs = dict(zip(features, coef.tolist()))
            all_coefs[f"{arm}-fold{i}"] = coefs
            targets = [q for q in queries if q["query_id"] in held_out] + (
                [q for q in queries if q["query_id"] not in trainable] if i == 0 else []
            )
            for q in targets:
                for rank, (docid, score) in enumerate(
                    score_query(args.endpoint, q, profile, coefs, 0.0), 1
                ):
                    lines.append(f"{q['query_id']} Q0 {docid} {rank} {score:.6f} {arm}")
        (outdir / f"{arm}.txt").write_text("\n".join(lines) + "\n")
        print(f"{arm}: wrote {len(lines)} result lines, last fold pairs={len(X)}")

    (outdir / "coefficients.json").write_text(json.dumps(all_coefs, indent=1))


if __name__ == "__main__":
    main()
