#!/usr/bin/env python3
"""Exp 13 data-gen: label (query, doc) pairs with the cross-encoder teacher.

Uses each of the 940 canonical questionnaire questions as a query, retrieves top-50
hybrid candidates within its own tenant, and records the cross-encoder teacher
score plus the cheap student features (via the collect-distill profile's
match-features). Output: distill_train.csv. The 47 paraphrase queries are NOT
used here — they stay held out for evaluation.
"""
import json
import time
import urllib.request
from pathlib import Path

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


def main():
    docs = [json.loads(l) for l in open(HERE / "docs-rfq.jsonl")]
    out = open(HERE / "distill_train.csv", "w")
    out.write("qsrc,tenant,doc_id,is_gold," + ",".join(f"f{i}" for i in range(len(FEATS))) + ",teacher\n")
    t0, n = time.time(), 0
    for i, src in enumerate(docs):
        p = {"query": src["title"], "timeout": "60s", "hits": 50,
             "presentation.summary": "no-chunks",
             "yql": f"select * from rfq where tenant contains '{src['tenant']}' and ({WHERE})",
             "ranking.profile": "collect-distill", "ranking.globalPhase.rerankCount": 50,
             "ranking.features.query(embedding)": "embed(nomicmb, @query)",
             "ranking.features.query(float_embedding)": "embed(nomicmb, @query)",
             "input.query(q_tokens)": "embed(tokenizer, @query)",
             "input.query(qt)": "embed(colbert, @query)"}
        resp = search(p)
        for h in resp["root"].get("children", []):
            f = h.get("fields", {})
            mf = f.get("matchfeatures")
            if not mf or "id" not in f:
                continue
            row = [f"{mf.get(k, 0):.6f}" for k in FEATS]
            is_gold = 1 if f["id"] == src["id"] else 0
            out.write(f"{src['id']},{src['tenant']},{f['id']},{is_gold},"
                      + ",".join(row) + f",{h.get('relevance', 0):.6f}\n")
            n += 1
        if (i + 1) % 100 == 0:
            print(f"{i+1}/940 queries, {n} rows, {time.time()-t0:.0f}s", flush=True)
    out.close()
    print(f"DISTILL COLLECT DONE: {n} rows from {len(docs)} queries in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
