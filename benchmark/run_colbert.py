#!/usr/bin/env python3
"""Experiment 11: ColBERT late-interaction retrieval on the questionnaire corpus.

Ranks all tenant docs by MaxSim between the query's ColBERT token embeddings
and each doc's stored token embeddings. Corpus is small enough (~235 docs per
tenant) to score every candidate in first-phase, so no ANN needed.
"""
import json
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent


def search(p, retries=4):
    for _ in range(retries):
        try:
            req = urllib.request.Request(
                "http://127.0.0.1:8080/search/",
                data=json.dumps(p).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.load(r)
        except Exception:
            time.sleep(3)
    return {"root": {}}


def run(qfile, tag, outdir):
    queries = [json.loads(l) for l in open(HERE / qfile)]
    lines = []
    t0 = time.time()
    for q in queries:
        p = {
            "query": q["query"],
            "timeout": "30s",
            "hits": 10,
            "presentation.summary": "no-chunks",
            "yql": f"select * from rfq where tenant contains '{q['tenant']}'",
            "ranking.profile": "colbert-maxsim",
            "input.query(qt)": "embed(colbert, @query)",
        }
        resp = search(p)
        hits = [h for h in resp["root"].get("children", []) if "fields" in h and "id" in h["fields"]]
        for rank, h in enumerate(hits, 1):
            lines.append(f'{q["query_id"]} Q0 {h["fields"]["id"]} {rank} {h.get("relevance",0):.6f} colbert')
    (outdir / f"{tag}-colbert-filtered.txt").write_text("\n".join(lines) + "\n")
    print(f"{tag} colbert: {len(queries)} queries in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    outdir = HERE / "results" / "exp11"
    outdir.mkdir(parents=True, exist_ok=True)
    run("queries-rfq-para.jsonl", "para", outdir)
    run("queries-rfq.jsonl", "verbatim", outdir)
