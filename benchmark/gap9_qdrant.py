#!/usr/bin/env python3
"""Gap 9: engine bake-off — the winning recipe served by Qdrant vs Vespa.

The recipe that won the study (library model, symmetric prefixes, float
titles, hard tenant filter) is pure dense retrieval plus a filter — the one
configuration every vector database can also express. This deploys it on
Qdrant (REST, stdlib only; cosine over the same nomic embeddings, tenant as
an indexed keyword payload) and compares against the committed Vespa served
runs (results/gap7/*-q2q-sym-filtered.txt).

With model and config held fixed this measures serving fidelity, not model
quality — the expected result is a tie. Three deltas decompose the stack:

  offline numpy vs qdrant-exact   feed + scoring fidelity (should be ~0)
  qdrant-exact  vs qdrant-hnsw    ANN-under-filter loss (toy at 940 docs)
  qdrant        vs vespa-served   whole-system delta, including the
                                  embedding path (client PyTorch vs
                                  in-engine ONNX — part of each system)

Arms are scored per-query (nDCG@10) on the scaled paraphrase set and
verbatim; TREC run files -> results/gap9/, per-query scores ->
results/gap9/bakeoff.json.

  docker run -d --name qdrant-bakeoff -p 127.0.0.1:6333:6333 qdrant/qdrant
  python3 gap9_qdrant.py
"""
import json
import random
import time
import urllib.request
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

from evaluate import load_jsonl, load_qrels, load_run, ndcg

HERE = Path(__file__).parent
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
QDRANT = "http://127.0.0.1:6333"
COLL = "rfq_q2q_sym"
PREFIX = "search_query: "  # symmetric: query prefix on BOTH sides (Gap 7)

SETS = {
    "para-all": ("queries-rfq-para-all.jsonl", "qrels-rfq-para-all.tsv"),
    "verbatim": ("queries-rfq.jsonl", "qrels-rfq.tsv"),
}
VESPA_RUNS = {
    "para-all": "results/gap7/para-all-q2q-sym-filtered.txt",
    "verbatim": "results/gap7/verbatim-q2q-sym-filtered.txt",
}


def req(method, path, body=None):
    r = urllib.request.Request(
        QDRANT + path, method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r) as resp:
        return json.loads(resp.read())


def main():
    docs = load_jsonl(HERE / "docs-rfq.jsonl")
    model = SentenceTransformer("nomic-ai/modernbert-embed-base", device=DEVICE)
    D = model.encode([PREFIX + d["title"] for d in docs],
                     normalize_embeddings=True, batch_size=16, show_progress_bar=False)

    try:
        req("DELETE", f"/collections/{COLL}")
    except urllib.error.HTTPError:
        pass
    req("PUT", f"/collections/{COLL}",
        {"vectors": {"size": D.shape[1], "distance": "Cosine"}})
    req("PUT", f"/collections/{COLL}/index?wait=true",
        {"field_name": "tenant", "field_schema": "keyword"})
    for i in range(0, len(docs), 100):
        req("PUT", f"/collections/{COLL}/points?wait=true", {"points": [
            {"id": j, "vector": D[j].tolist(),
             "payload": {"doc_id": docs[j]["id"], "tenant": docs[j]["tenant"]}}
            for j in range(i, min(i + 100, len(docs)))]})
    n = req("GET", f"/collections/{COLL}")["result"]["points_count"]
    print(f"fed {n}/{len(docs)} points, dim {D.shape[1]}, device {DEVICE}")

    out = HERE / "results" / "gap9"
    out.mkdir(parents=True, exist_ok=True)
    per = {}      # set -> arm -> qid -> ndcg
    lat = []      # qdrant hnsw search ms
    for sname, (qfile, qrfile) in SETS.items():
        qrels = load_qrels(HERE / qrfile)
        queries = [q for q in load_jsonl(HERE / qfile)
                   if any(g > 0 for g in qrels.get(q["query_id"], {}).values())]
        Q = model.encode([PREFIX + q["query"] for q in queries],
                         normalize_embeddings=True, batch_size=16, show_progress_bar=False)
        per[sname] = {}

        for arm, exact in (("qdrant-hnsw", False), ("qdrant-exact", True)):
            scores, lines = {}, []
            for qi, q in enumerate(queries):
                t0 = time.perf_counter()
                hits = req("POST", f"/collections/{COLL}/points/search", {
                    "vector": Q[qi].tolist(), "limit": 10, "with_payload": ["doc_id"],
                    "filter": {"must": [{"key": "tenant", "match": {"value": q["tenant"]}}]},
                    "params": {"exact": exact}})["result"]
                if not exact:
                    lat.append((time.perf_counter() - t0) * 1000)
                ranked = [h["payload"]["doc_id"] for h in hits]
                scores[q["query_id"]] = ndcg(ranked, qrels[q["query_id"]], 10)
                lines += [f"{q['query_id']} Q0 {d} {r+1} {h['score']:.6f} {arm}"
                          for r, (d, h) in enumerate(zip(ranked, hits))]
            per[sname][arm] = scores
            (out / f"{sname}-{arm}-filtered.txt").write_text("\n".join(lines) + "\n")

        vrun = load_run(HERE / VESPA_RUNS[sname])
        per[sname]["vespa-served"] = {
            q["query_id"]: ndcg(vrun[q["query_id"]][:10], qrels[q["query_id"]], 10)
            for q in queries}
        print(f"{sname}: {len(queries)} queries done")

    # offline numpy control from Gap 8 (identical embeddings + exact scan)
    g8 = json.loads((HERE / "results" / "gap8" / "splade.json").read_text())
    for sname in SETS:
        per[sname]["offline-numpy"] = g8[sname]["nomic-q2q-sym"]

    para_q = [q for q in load_jsonl(HERE / SETS["para-all"][0])
              if q["query_id"] in per["para-all"]["qdrant-hnsw"]]
    splits = {
        "para-all": [q["query_id"] for q in para_q],
        "oos-185": [q["query_id"] for q in para_q
                    if q["cluster"] == "questionnaire-para-new"],
        "verbatim": list(per["verbatim"]["qdrant-hnsw"]),
    }

    def mean(sname, a, qids):
        return sum(per[sname][a][q] for q in qids) / len(qids)

    rng = random.Random(7)
    def boot(sname, a, b, qids):
        d = [per[sname][a][q] - per[sname][b][q] for q in qids]
        n = len(d)
        delta = sum(d) / n
        if all(abs(x) < 1e-12 for x in d):
            return delta, 1.0
        hits = sum(1 for _ in range(10000)
                   if ((s := sum(d[rng.randrange(n)] for _ in range(n)) / n) <= 0) == (delta > 0))
        return delta, min(1.0, 2 * hits / 10000)

    arms = ["offline-numpy", "qdrant-exact", "qdrant-hnsw", "vespa-served"]
    print(f"\n{'arm':<14} {'full-232':>9} {'oos-185':>8} {'verbatim':>9}")
    for a in arms:
        print(f"{a:<14} {mean('para-all', a, splits['para-all']):>9.3f}"
              f" {mean('para-all', a, splits['oos-185']):>8.3f}"
              f" {mean('verbatim', a, splits['verbatim']):>9.3f}")

    print("\npaired bootstrap (delta, p) —")
    for a, b, note in [("qdrant-exact", "offline-numpy", "feed fidelity"),
                       ("qdrant-hnsw", "qdrant-exact", "ANN-under-filter loss"),
                       ("qdrant-hnsw", "vespa-served", "whole-system delta")]:
        for sname, key in (("para-all", "para-all"), ("verbatim", "verbatim")):
            d, p = boot(sname, a, b, splits[key])
            print(f"  {a} vs {b} [{sname}]: {d:+.4f} p={p:.4f}   ({note})")

    lat.sort()
    print(f"\nqdrant hnsw+filter search (excl. embed): "
          f"p50 {lat[len(lat)//2]:.1f} ms, p95 {lat[int(len(lat)*.95)]:.1f} ms")

    (out / "bakeoff.json").write_text(json.dumps(
        {s: {a: {q: round(v, 6) for q, v in per[s][a].items()} for a in per[s]}
         for s in per}, indent=1))


if __name__ == "__main__":
    main()
