#!/usr/bin/env python3
"""Gap 7b: embedder bake-off on the library model (q2q), offline.

Every ranking number in the study is conditional on one embedder (nomic
modernbert-embed-base), and Gap 5 showed embedder choice dominates embedder
fine-tuning. This compares current-generation embedders on the q2q task —
embed the incoming paraphrase, rank the stored questions within tenant by
cosine — on the scaled 232-query paraphrase set:

  nomic-asym   modernbert-embed-base, standard prefixes (the study baseline)
  nomic-sym    same model, query prefix on BOTH sides (symmetric task,
               symmetric conditioning — offline twin of the served Gap 7a)
  bge-m3       BAAI/bge-m3 dense, no prefixes
  qwen3-raw    Qwen3-Embedding-0.6B, no instruction
  qwen3-inst   Qwen3-Embedding-0.6B with a task instruction on the query side
               — instruction conditioning aimed at the HyDE finding: normalise
               the paraphrase toward question space inside the forward pass,
               at zero extra query cost

Offline mirrors gap2_finetune.evaluate (exact within-tenant cosine); Gap 5
measured serving costs ~0.02 vs offline, so compare arms within this table,
not against served numbers. Per-query scores -> results/gap7/bakeoff.json.

  python3 gap7_embed_bakeoff.py
"""
import gc
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from evaluate import load_jsonl, load_qrels, ndcg

HERE = Path(__file__).parent
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

INSTRUCTION = ("Instruct: Given a buyer's informally worded security question, "
               "retrieve the standard security-questionnaire question that asks "
               "the same thing.\nQuery: ")

ARMS = {
    "nomic-asym": ("nomic-ai/modernbert-embed-base", "search_query: ", "search_document: "),
    "nomic-sym":  ("nomic-ai/modernbert-embed-base", "search_query: ", "search_query: "),
    "bge-m3":     ("BAAI/bge-m3", "", ""),
    "qwen3-raw":  ("Qwen/Qwen3-Embedding-0.6B", "", ""),
    "qwen3-inst": ("Qwen/Qwen3-Embedding-0.6B", INSTRUCTION, ""),
}


def main():
    docs = load_jsonl(HERE / "docs-rfq.jsonl")
    qrels = load_qrels(HERE / "qrels-rfq-para-all.tsv")
    queries = [q for q in load_jsonl(HERE / "queries-rfq-para-all.jsonl")
               if any(g > 0 for g in qrels.get(q["query_id"], {}).values())]
    doc_ids = [d["id"] for d in docs]
    doc_tenant = {d["id"]: d["tenant"] for d in docs}
    print(f"{len(docs)} docs, {len(queries)} queries, device {DEVICE}")

    per = {}
    loaded = {}
    for arm, (model_name, qprefix, dprefix) in ARMS.items():
        if model_name not in loaded:
            loaded.clear()
            gc.collect()
            if DEVICE == "mps":
                torch.mps.empty_cache()
            loaded[model_name] = SentenceTransformer(model_name, device=DEVICE)
        model = loaded[model_name]
        D = model.encode([dprefix + d["title"] for d in docs],
                         normalize_embeddings=True, batch_size=16, show_progress_bar=False)
        Q = model.encode([qprefix + q["query"] for q in queries],
                         normalize_embeddings=True, batch_size=16, show_progress_bar=False)
        scores = {}
        for qi, q in enumerate(queries):
            sims = D @ Q[qi]
            order = np.argsort(-sims)
            ranked = [doc_ids[j] for j in order if doc_tenant[doc_ids[j]] == q["tenant"]][:10]
            scores[q["query_id"]] = ndcg(ranked, qrels[q["query_id"]], 10)
        per[arm] = scores
        print(f"{arm:<11} done")

    orig = [q["query_id"] for q in queries if q["cluster"] == "questionnaire-para"]
    new = [q["query_id"] for q in queries if q["cluster"] == "questionnaire-para-new"]
    full = orig + new

    def mean(a, qids):
        return sum(per[a][q] for q in qids) / len(qids)

    rng = random.Random(7)
    def boot(a, b, qids):
        d = [per[a][q] - per[b][q] for q in qids]
        n = len(d)
        delta = sum(d) / n
        hits = sum(1 for _ in range(10000)
                   if ((s := sum(d[rng.randrange(n)] for _ in range(n)) / n) <= 0) == (delta > 0))
        return delta, min(1.0, 2 * hits / 10000)

    print(f"\n{'arm':<11} {'orig-47':>8} {'new-185':>8} {'full-232':>9}   vs nomic-asym (full | oos)")
    for arm in ARMS:
        line = f"{arm:<11} {mean(arm, orig):>8.3f} {mean(arm, new):>8.3f} {mean(arm, full):>9.3f}"
        if arm != "nomic-asym":
            df, pf = boot(arm, "nomic-asym", full)
            dn, pn = boot(arm, "nomic-asym", new)
            line += f"   {df:+.3f} p={pf:.4f} | {dn:+.3f} p={pn:.4f}"
        print(line)

    out = HERE / "results" / "gap7"
    out.mkdir(parents=True, exist_ok=True)
    (out / "bakeoff.json").write_text(json.dumps(
        {a: {q: round(v, 6) for q, v in per[a].items()} for a in per}, indent=1))


if __name__ == "__main__":
    main()
