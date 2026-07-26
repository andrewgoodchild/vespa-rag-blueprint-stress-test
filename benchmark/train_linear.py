#!/usr/bin/env python3
"""Experiment 2: retrain the learned-linear first phase on this corpus.

Collects the blueprint's 6 ranking features (+ freshness) per (query, doc)
via the collect-v2 rank profile, trains logistic regression with 5-fold
cross-validation grouped by query, and scores each held-out query by passing
that fold's coefficients as query parameters to the learned-linear /
learned-linear-v2 profiles (no redeploy needed).

Outputs TREC run files: retrained-6f.txt and retrained-7f.txt in --outdir,
plus features.csv and per-fold coefficients in coefficients.json.
"""
import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).parent

FEATURES6 = [
    "avg_top_3_chunk_sim_scores",
    "avg_top_3_chunk_text_scores",
    "bm25(title)",
    "bm25(chunks)",
    "max_chunk_sim_scores",
    "max_chunk_text_scores",
]
FEATURES7 = FEATURES6 + ["modified_freshness"]

PARAM_NAME = {
    "avg_top_3_chunk_sim_scores": "avg_top_3_chunk_sim_scores_param",
    "avg_top_3_chunk_text_scores": "avg_top_3_chunk_text_scores_param",
    "bm25(title)": "bm25_title_param",
    "bm25(chunks)": "bm25_chunks_param",
    "max_chunk_sim_scores": "max_chunk_sim_scores_param",
    "max_chunk_text_scores": "max_chunk_text_scores_param",
    "modified_freshness": "modified_freshness_param",
}

HYBRID_WHERE = (
    "userInput(@query) or "
    '({label:"title_label", targetHits:100}nearestNeighbor(title_embedding, embedding)) or '
    '({label:"chunks_label", targetHits:100}nearestNeighbor(chunk_embeddings, embedding))'
)


def search(endpoint, params):
    req = urllib.request.Request(
        f"{endpoint}/search/",
        data=json.dumps(params).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def collect_features(endpoint, queries):
    """One row per (query, candidate doc) with the 7 raw feature values."""
    rows = []
    for q in queries:
        resp = search(
            endpoint,
            {
                "yql": f"select * from doc where {HYBRID_WHERE}",
                "query": q["query"],
                "ranking.profile": "collect-v2",
                "ranking.features.query(embedding)": "embed(nomicmb, @query)",
                "ranking.features.query(float_embedding)": "embed(nomicmb, @query)",
                "hits": 100,
                "presentation.summary": "no-chunks",
                "timeout": "20s",
            },
        )
        for h in resp["root"].get("children", []):
            mf = h["fields"].get("matchfeatures")
            if not mf:
                continue
            rows.append(
                {
                    "query_id": q["query_id"],
                    "doc_id": h["fields"]["id"],
                    **{f: float(mf[f]) for f in FEATURES7},
                }
            )
    return rows


def fit_fold(rows, qrels, train_qids, features):
    """Logistic regression on standardized features; return raw-space coefs."""
    X, y = [], []
    for r in rows:
        if r["query_id"] not in train_qids:
            continue
        X.append([r[f] for f in features])
        y.append(1 if qrels.get(r["query_id"], {}).get(r["doc_id"], 0) > 0 else 0)
    X, y = np.array(X), np.array(y)
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(class_weight="balanced", max_iter=2000)
    model.fit(scaler.transform(X), y)
    raw_coef = model.coef_[0] / scaler.scale_
    raw_intercept = model.intercept_[0] - float(np.sum(model.coef_[0] * scaler.mean_ / scaler.scale_))
    return dict(zip(features, raw_coef.tolist())), raw_intercept


def score_query(endpoint, q, profile, coefs, intercept, tenant_filter=False):
    where = HYBRID_WHERE
    if tenant_filter:
        where = f"tenant contains '{q['tenant']}' and ({HYBRID_WHERE})"
    params = {
        "yql": f"select * from doc where {where}",
        "query": q["query"],
        "ranking.profile": profile,
        "ranking.features.query(embedding)": "embed(nomicmb, @query)",
        "ranking.features.query(float_embedding)": "embed(nomicmb, @query)",
        "ranking.features.query(intercept)": intercept,
        "hits": 10,
        "presentation.summary": "no-chunks",
        "timeout": "20s",
    }
    for f, c in coefs.items():
        params[f"ranking.features.query({PARAM_NAME[f]})"] = c
    resp = search(endpoint, params)
    return [
        (h["fields"]["id"], h.get("relevance", 0.0))
        for h in resp["root"].get("children", [])
        if "fields" in h and "id" in h["fields"]
    ]


def load_qrels(path):
    qrels = {}
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        qid, _, docid, grade = line.split()
        qrels.setdefault(qid, {})[docid] = int(grade)
    return qrels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:8080")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--queries", default=str(HERE / "queries.jsonl"))
    ap.add_argument("--qrels", default=str(HERE / "qrels.tsv"))
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    queries = [json.loads(l) for l in open(args.queries)]
    qrels = load_qrels(args.qrels)

    rows = collect_features(args.endpoint, queries)
    with open(outdir / "features.csv", "w") as f:
        f.write("query_id,doc_id," + ",".join(FEATURES7) + ",label\n")
        for r in rows:
            label = 1 if qrels.get(r["query_id"], {}).get(r["doc_id"], 0) > 0 else 0
            f.write(
                f"{r['query_id']},{r['doc_id']},"
                + ",".join(f"{r[f]:.6f}" for f in FEATURES7)
                + f",{label}\n"
            )
    print(f"collected {len(rows)} feature rows")

    # Folds grouped by query. Only queries with at least one positive can train.
    rng = np.random.default_rng(args.seed)
    trainable = sorted(q["query_id"] for q in queries if any(g > 0 for g in qrels.get(q["query_id"], {}).values()))
    order = list(rng.permutation(trainable))
    folds = [set(order[i :: args.folds]) for i in range(args.folds)]

    all_coefs = {}
    for arm, profile, features in [
        ("retrained-6f", "learned-linear", FEATURES6),
        ("retrained-7f", "learned-linear-v2", FEATURES7),
    ]:
        lines = []
        for i, held_out in enumerate(folds):
            train_qids = set(trainable) - held_out
            coefs, intercept = fit_fold(rows, qrels, train_qids, features)
            all_coefs[f"{arm}-fold{i}"] = {"intercept": intercept, **coefs}
            # Unanswerable queries (no positives anywhere) are scored with fold 0.
            targets = [q for q in queries if q["query_id"] in held_out] + (
                [q for q in queries if q["query_id"] not in trainable] if i == 0 else []
            )
            for q in targets:
                for rank, (docid, score) in enumerate(
                    score_query(args.endpoint, q, profile, coefs, intercept), 1
                ):
                    lines.append(f"{q['query_id']} Q0 {docid} {rank} {score:.6f} {arm}")
        (outdir / f"{arm}.txt").write_text("\n".join(lines) + "\n")
        print(f"{arm}: wrote {len(lines)} result lines")

    (outdir / "coefficients.json").write_text(json.dumps(all_coefs, indent=1))


if __name__ == "__main__":
    main()
