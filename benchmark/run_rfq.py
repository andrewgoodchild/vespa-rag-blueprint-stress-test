#!/usr/bin/env python3
"""Run the questionnaire-corpus arms against the local Vespa rfq schema.

One TREC run file per (query set, arm). Every arm is tenant-filtered: the hard
filter is a compliance boundary in this scenario, not a ranking preference, so
the filtered configuration is the one worth comparing. Pass --unfiltered to
also produce the leaky baseline that quantifies what the filter is worth.

  python3 run_rfq.py --outdir results/exp8
  python3 run_rfq.py --arms bm25,semantic --queries queries-rfq-para.jsonl
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
ENDPOINT = "http://127.0.0.1:8080/search/"

NN_CHUNKS = ('({label:"c", targetHits:100}nearestNeighbor(chunk_embeddings, embedding))')
NN_TITLE = ('({label:"t", targetHits:100}nearestNeighbor(title_embedding, embedding))')

# where-clause fragment per arm, the rank profile it scores with, and whether
# the query embedding is needed (bm25 is purely lexical, so it is not).
ARMS = {
    "bm25":         ("userInput(@query)", "bm25-only", False),
    "semantic":     (f"({NN_TITLE} or {NN_CHUNKS})", "semantic-only", True),
    "hybrid":       (f"(userInput(@query) or {NN_TITLE} or {NN_CHUNKS})", "learned-linear", True),
    "rrf":          (f"(userInput(@query) or {NN_TITLE} or {NN_CHUNKS})", "rrf-hybrid", True),
    "q2q-semantic": (NN_TITLE, "q2q-semantic", True),
    "q2q-rrf":      (f"(userInput(@query) or {NN_TITLE})", "q2q-rrf", True),
    # Gap 3: identical candidate pool (every doc in the tenant), scored off the
    # int8-binary embedding vs the float32 one. Only the stored representation
    # differs, so the delta is the cost of binarisation and nothing else.
    "semantic-binary": ("true", "semantic-only", True),
    "semantic-float":  ("true", "semantic-float", True),
    # cross-encoder rerank of a shortlist — accurate and ~3s/query
    "ce-semantic":  (f"({NN_TITLE} or {NN_CHUNKS})", "ce-semantic", True),
    "ce-hybrid":    (f"(userInput(@query) or {NN_TITLE} or {NN_CHUNKS})", "ce-hybrid", True),
    "ce-q2q":       (NN_TITLE, "ce-q2q", True),
    # the distilled student, served natively — no cross-encoder in the path
    "student":      (f"(userInput(@query) or {NN_TITLE} or {NN_CHUNKS})", "student-pairwise", True),
}

# The blueprint's shipped first-phase coefficients. The `hybrid` arm exists to
# measure them as-shipped, so they are passed explicitly rather than inherited
# from the `hybrid` query profile (which is bound to the synthetic `doc` schema
# and carries no tenant filter).
LINEAR_PARAMS = {
    "ranking.features.query(intercept)": -7.798639,
    "ranking.features.query(avg_top_3_chunk_sim_scores_param)": 13.383840,
    "ranking.features.query(avg_top_3_chunk_text_scores_param)": 0.203145,
    "ranking.features.query(bm25_chunks_param)": 0.159914,
    "ranking.features.query(bm25_title_param)": 0.191867,
    "ranking.features.query(max_chunk_sim_scores_param)": 10.067169,
    "ranking.features.query(max_chunk_text_scores_param)": 0.153392,
}


def search(params, retries=4):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                ENDPOINT, data=json.dumps(params).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except Exception as e:
            if attempt == retries - 1:
                return {"root": {}, "_error": str(e)}
            time.sleep(2 * (attempt + 1))
    return {"root": {}}


def run_arm(arm, queries, filtered, hits):
    where, profile, needs_embedding = ARMS[arm]
    lines, errors = [], 0
    t0 = time.time()
    for q in queries:
        clause = f"tenant contains '{q['tenant']}' and {where}" if filtered else where
        params = {
            "query": q["query"],
            "yql": f"select * from rfq where {clause}",
            "ranking.profile": profile,
            "hits": hits,
            "timeout": "30s",
            "presentation.summary": "no-chunks",
        }
        if needs_embedding:
            # the embedder id is required: the app configures three of them
            params["ranking.features.query(embedding)"] = "embed(nomicmb, @query)"
            params["ranking.features.query(float_embedding)"] = "embed(nomicmb, @query)"
        if profile == "learned-linear":
            params.update(LINEAR_PARAMS)
        if profile.startswith("ce-"):
            params["input.query(q_tokens)"] = "embed(tokenizer, @query)"
            params["timeout"] = "60s"
        if profile == "student-pairwise":
            params["input.query(qt)"] = "embed(colbert, @query)"
        resp = search(params)
        if "_error" in resp:
            errors += 1
            continue
        if resp["root"].get("errors"):
            errors += 1
            continue
        for rank, h in enumerate(resp["root"].get("children", []), start=1):
            f = h.get("fields", {})
            if "id" not in f:
                continue
            lines.append(f'{q["query_id"]} Q0 {f["id"]} {rank} {h.get("relevance", 0):.6f} {arm}')
    return lines, time.time() - t0, errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="queries-rfq.jsonl")
    ap.add_argument("--outdir", default=str(HERE / "results" / "exp8"))
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--hits", type=int, default=10)
    ap.add_argument("--tag", default=None, help="filename prefix; default from query file")
    ap.add_argument("--unfiltered", action="store_true", help="also run without the tenant filter")
    args = ap.parse_args()

    queries = [json.loads(l) for l in open(HERE / args.queries)]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = args.tag if args.tag is not None else ("para" if "para" in args.queries else "verbatim")

    modes = [True, False] if args.unfiltered else [True]
    print(f"{len(queries)} queries from {args.queries} -> {outdir}")
    for arm in args.arms.split(","):
        arm = arm.strip()
        if arm not in ARMS:
            print(f"  ! unknown arm {arm}, skipping")
            continue
        for filtered in modes:
            lines, secs, errors = run_arm(arm, queries, filtered, args.hits)
            name = f"{tag}-{arm}{'-filtered' if filtered else ''}.txt"
            (outdir / name).write_text("\n".join(lines) + "\n")
            note = f"  ({errors} query errors)" if errors else ""
            print(f"  {name:<40} {len(lines):>5} rows  {secs:>6.0f}s{note}")


if __name__ == "__main__":
    main()
