#!/usr/bin/env python3
"""Run the benchmark queries against a local RAG-blueprint Vespa instance.

Produces one TREC run file per arm in --outdir, plus raw.json with full
responses (including selected top-3 chunks) for diagnostics.

Arms:
  bm25            weakAnd lexical retrieval, bm25(title)+bm25(chunks) ranking
  semantic        nearestNeighbor retrieval, max chunk cosine ranking
  hybrid          blueprint 'hybrid' query profile (learned-linear first phase)
  gbdt            blueprint 'hybrid-with-gbdt' (LightGBM second phase)
  hybrid-filtered hybrid + hard tenant filter
"""
import argparse
import json
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent

HYBRID_WHERE = (
    "userInput(@query) or "
    '({label:"title_label", targetHits:100}nearestNeighbor(title_embedding, embedding)) or '
    '({label:"chunks_label", targetHits:100}nearestNeighbor(chunk_embeddings, embedding))'
)


def arm_params(arm, q):
    embeds = {
        "ranking.features.query(embedding)": "embed(nomicmb, @query)",
        "ranking.features.query(float_embedding)": "embed(nomicmb, @query)",
    }
    if arm == "bm25":
        return {
            "yql": "select * from doc where userInput(@query)",
            "ranking.profile": "bm25-only",
            "hits": 10,
            "presentation.summary": "no-chunks",
            **embeds,
        }
    if arm == "semantic":
        return {
            "yql": (
                "select * from doc where "
                '({label:"title_label", targetHits:100}nearestNeighbor(title_embedding, embedding)) or '
                '({label:"chunks_label", targetHits:100}nearestNeighbor(chunk_embeddings, embedding))'
            ),
            "ranking.profile": "semantic-only",
            "hits": 10,
            "presentation.summary": "no-chunks",
            **embeds,
        }
    if arm == "hybrid":
        return {"queryProfile": "hybrid", "hits": 10}
    if arm == "gbdt":
        return {"queryProfile": "hybrid-with-gbdt", "hits": 10}
    if arm == "hybrid-filtered":
        return {
            "queryProfile": "hybrid",
            "hits": 10,
            "yql": f"select * from doc where tenant contains '{q['tenant']}' and ({HYBRID_WHERE})",
        }
    raise ValueError(arm)


def search(endpoint, params):
    body = json.dumps(params).encode()
    req = urllib.request.Request(
        f"{endpoint}/search/", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:8080")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--arms", default="bm25,semantic,hybrid,gbdt,hybrid-filtered")
    ap.add_argument("--queries", default=str(HERE / "queries.jsonl"))
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    queries = [json.loads(l) for l in open(args.queries)]
    raw = {}

    for arm in args.arms.split(","):
        lines = []
        for q in queries:
            params = {"query": q["query"], "timeout": "15s", **arm_params(arm, q)}
            resp = search(args.endpoint, params)
            hits = resp.get("root", {}).get("children", []) or []
            raw.setdefault(arm, {})[q["query_id"]] = [
                {
                    "id": h["fields"].get("id"),
                    "relevance": h.get("relevance"),
                    "chunks_top3": h["fields"].get("chunks_top3"),
                }
                for h in hits
                if "fields" in h
            ]
            rank = 0
            for h in hits:
                if "fields" not in h or "id" not in h["fields"]:
                    continue
                rank += 1
                lines.append(
                    f"{q['query_id']} Q0 {h['fields']['id']} {rank} {h.get('relevance', 0):.6f} {arm}"
                )
        (outdir / f"{arm}.txt").write_text("\n".join(lines) + "\n")
        print(f"{arm}: wrote {len(lines)} result lines")

    (outdir / "raw.json").write_text(json.dumps(raw, indent=1))


if __name__ == "__main__":
    main()
