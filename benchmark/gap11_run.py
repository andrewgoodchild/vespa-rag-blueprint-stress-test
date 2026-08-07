#!/usr/bin/env python3
"""Gap 11b: the non-graph arms on the multi-hop query sets.

  library  the winning recipe unchanged (symmetric q2q, tenant-filtered,
           exact scan) — how far does single-vector retrieval already get
           on questions it was never built for?
  decomp   LLM query decomposition (google/gemini-2.5-flash, temp 0,
           cached to gap11_decomp.jsonl so re-runs need no key), each
           sub-question retrieved with the library model, results merged
           rank-interleaved. One cheap LLM call — the null hypothesis
           every graph arm must beat.

Metrics per cluster: compositional — nDCG@10, parts-recall@10 (mean
fraction of grade-2 constituents in the top 10) and all-parts@10 (the one
that matters: every constituent present, or the drafted answer is
confidently wrong); global — nDCG@10 and control-coverage@10 (distinct
controls of the domain reached in the top 10 / min(10, controls in
domain)). Paired bootstrap decomp vs library. Runs -> results/gap11/.

  python3 gap11_run.py
"""
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

from evaluate import load_jsonl, load_qrels, ndcg
from gap11_gen_queries import llm

HERE = Path(__file__).parent
OUT = HERE / "results" / "gap11"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
PREFIX = "search_query: "

DECOMP_PROMPT = """Split this buyer question to a software vendor into the minimal \
set of self-contained sub-questions (1 to 4). Each sub-question must stand alone. \
Return one sub-question per line, nothing else.

{query}"""


def decompositions(queries):
    cache_path = HERE / "gap11_decomp.jsonl"
    cache = {}
    if cache_path.exists():
        cache = {json.loads(l)["query_id"]: json.loads(l)["subs"]
                 for l in open(cache_path) if l.strip()}
    for q in queries:
        if q["query_id"] not in cache:
            out = llm(DECOMP_PROMPT.format(query=q["query"]), temperature=0.0)
            subs = [s.strip("-• ").strip() for s in out.splitlines() if s.strip()]
            cache[q["query_id"]] = subs[:4] or [q["query"]]
            cache_path.write_text("\n".join(
                json.dumps({"query_id": k, "subs": v})
                for k, v in cache.items()) + "\n")
    return cache


def main():
    docs = load_jsonl(HERE / "docs-rfq.jsonl")
    control = {d["id"]: d["control_id"] for d in docs}
    ids = [d["id"] for d in docs]
    tenant = {d["id"]: d["tenant"] for d in docs}

    model = SentenceTransformer("nomic-ai/modernbert-embed-base", device=DEVICE)
    D = model.encode([PREFIX + d["title"] for d in docs],
                     normalize_embeddings=True, batch_size=16,
                     show_progress_bar=False)

    def retrieve(text, ten, k=10):
        v = model.encode([PREFIX + text], normalize_embeddings=True)[0]
        sims = D @ v
        order = sims.argsort()[::-1]
        return [ids[j] for j in order if tenant[ids[j]] == ten][:k]

    sets = {
        "multi": (load_jsonl(HERE / "queries-rfq-multi.jsonl"),
                  load_qrels(HERE / "qrels-rfq-multi.tsv")),
        "global": (load_jsonl(HERE / "queries-rfq-global.jsonl"),
                   load_qrels(HERE / "qrels-rfq-global.tsv")),
    }
    decomp = decompositions(sets["multi"][0] + sets["global"][0])
    n_subs = [len(v) for v in decomp.values()]
    print(f"decompositions: mean {sum(n_subs)/len(n_subs):.1f} subs/query")

    OUT.mkdir(parents=True, exist_ok=True)
    per = defaultdict(lambda: defaultdict(dict))  # arm -> metric -> qid -> val
    runs = defaultdict(list)
    for sname, (queries, qrels) in sets.items():
        for q in queries:
            qid, ten = q["query_id"], q["tenant"]
            ranked = {"library": retrieve(q["query"], ten)}
            merged, seen = [], set()
            sub_lists = [retrieve(s, ten) for s in decomp[qid]]
            for rank in range(10):
                for sl in sub_lists:
                    if rank < len(sl) and sl[rank] not in seen:
                        seen.add(sl[rank])
                        merged.append(sl[rank])
            ranked["decomp"] = merged[:10]

            rels = qrels[qid]
            parts = [d for d, g in rels.items() if g == 2]
            n_ctl = min(10, len({control[d] for d in rels}))
            for arm, rk in ranked.items():
                runs[arm] += [f"{qid} Q0 {d} {i+1} {1.0/(i+1):.4f} {arm}"
                              for i, d in enumerate(rk)]
                per[arm]["ndcg-" + sname][qid] = ndcg(rk, rels, 10)
                if sname == "multi":
                    found = sum(1 for p in parts if p in rk)
                    per[arm]["parts-recall"][qid] = found / len(parts)
                    per[arm]["all-parts"][qid] = float(found == len(parts))
                else:
                    hit = {control[d] for d in rk if d in rels}
                    per[arm]["coverage"][qid] = len(hit) / n_ctl
    for arm, lines in runs.items():
        (OUT / f"gap11-{arm}.txt").write_text("\n".join(lines) + "\n")

    rng = random.Random(7)
    def boot(metric, a, b):
        qids = list(per[a][metric])
        d = [per[a][metric][q] - per[b][metric][q] for q in qids]
        n = len(d)
        delta = sum(d) / n
        if all(abs(x) < 1e-12 for x in d):
            return delta, 1.0
        hits = sum(1 for _ in range(10000)
                   if ((s := sum(d[rng.randrange(n)] for _ in range(n)) / n) <= 0) == (delta > 0))
        return delta, min(1.0, 2 * hits / 10000)

    metrics = ["ndcg-multi", "parts-recall", "all-parts", "ndcg-global", "coverage"]
    print(f"\n{'metric':<14} {'library':>8} {'decomp':>8}   delta (p)")
    for m in metrics:
        la = sum(per["library"][m].values()) / len(per["library"][m])
        de = sum(per["decomp"][m].values()) / len(per["decomp"][m])
        d, p = boot(m, "decomp", "library")
        print(f"{m:<14} {la:>8.3f} {de:>8.3f}   {d:+.3f} (p={p:.4f})")

    (OUT / "arms.json").write_text(json.dumps(
        {a: {m: {q: round(v, 6) for q, v in per[a][m].items()}
             for m in per[a]} for a in per}, indent=1))


if __name__ == "__main__":
    main()
