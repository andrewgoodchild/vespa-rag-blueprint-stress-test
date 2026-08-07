#!/usr/bin/env python3
"""Gap 11c: LightRAG graph retrieval — per-tenant graphs, and the shared
graph nobody should build but everybody does.

Two graph configurations over the same 940 answers ("Q: <question>\nA:
<answer>" per document, one doc per unit):

  tenant   four separate graphs, one per vendor — tenant isolation done
           right, graph edition
  shared   one graph over all four vendors — the naive multi-tenant
           deployment. Near-duplicate corpora merge entities by design
           ("SOC 2", "AES-256", "quarterly access review" are one node
           each), so this configuration exists to *measure* cross-tenant
           leakage, not to win.

LLM: google/gemini-2.5-flash via OpenRouter (temp 0); embeddings: local
nomic (the study embedder). Queries run in local and hybrid modes with
only_need_context=True — we score retrieval, not generation. Retrieved
context is mapped back to doc ids by normalized title containment, in
order of appearance, top-10. Builds are resumable (LightRAG dedupes by
content hash) — re-run after an interruption and it continues.

  python3 gap11_lightrag.py --config tenant --step build
  python3 gap11_lightrag.py --config shared --step build
  python3 gap11_lightrag.py --config tenant --step query
  python3 gap11_lightrag.py --config shared --step query
  python3 gap11_lightrag.py --step score
"""
import argparse
import asyncio
import json
import random
from collections import defaultdict
from pathlib import Path

from evaluate import load_jsonl, load_qrels, ndcg

HERE = Path(__file__).parent
OUT = HERE / "results" / "gap11"
MODEL = "google/gemini-2.5-flash"
TENANTS = ["tanager_geo", "kestrel_cloud", "orrery_suite", "pellucid_data"]


def norm(s):
    return " ".join(
        "".join(c.lower() if c.isalnum() or c.isspace() else " " for c in s).split())


# --------------------------------------------------------------- lightrag glue
def make_llm_func():
    import aiohttp
    import certifi
    import ssl
    from gap11_gen_queries import api_key
    key = api_key()
    ctx = ssl.create_default_context(cafile=certifi.where())
    sem = asyncio.Semaphore(16)

    async def llm_func(prompt, system_prompt=None, history_messages=[], **kw):
        msgs = ([{"role": "system", "content": system_prompt}] if system_prompt else [])
        msgs += list(history_messages) + [{"role": "user", "content": prompt}]
        body = {"model": MODEL, "temperature": 0.0, "max_tokens": 4096,
                "messages": msgs}
        async with sem:
            for attempt in range(5):
                try:
                    async with aiohttp.ClientSession(
                            connector=aiohttp.TCPConnector(ssl=ctx)) as s:
                        async with s.post(
                                "https://openrouter.ai/api/v1/chat/completions",
                                json=body, timeout=aiohttp.ClientTimeout(total=180),
                                headers={"Authorization": f"Bearer {key}"}) as r:
                            out = await r.json()
                    if "choices" not in out:
                        err = out.get("error", {})
                        if err.get("code") == 402:  # out of credits: permanent
                            raise SystemExit(f"OpenRouter 402: {err.get('message')}")
                        raise RuntimeError(f"OpenRouter error: {str(err)[:200]}")
                    return out["choices"][0]["message"]["content"]
                except SystemExit:
                    raise
                except Exception:
                    if attempt == 4:
                        raise
                    await asyncio.sleep(3 * 2 ** attempt)
    return llm_func


def make_embed_func():
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    from lightrag.utils import EmbeddingFunc
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer("nomic-ai/modernbert-embed-base", device=device)

    async def embed(texts):
        return np.asarray(model.encode(
            ["search_query: " + t for t in texts],
            normalize_embeddings=True, batch_size=16, show_progress_bar=False))
    return EmbeddingFunc(embedding_dim=768, max_token_size=8192, func=embed)


async def open_rag(wdir):
    from lightrag import LightRAG
    from lightrag.kg.shared_storage import initialize_pipeline_status
    rag = LightRAG(working_dir=str(wdir), llm_model_func=make_llm_func(),
                   embedding_func=make_embed_func(), llm_model_max_async=16)
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag


def doc_text(d):
    return f"Q: {d['title']}\nA: {d['text']}"


async def build(config, only=None):
    # one graph per PROCESS: LightRAG's pipeline status is process-global,
    # and building several graphs in one interpreter cross-contaminates
    # their doc queues (observed: tenant stores accumulating other
    # tenants' docs). Drive each graph as its own invocation via --only.
    docs = load_jsonl(HERE / "docs-rfq.jsonl")
    groups = ({t: [d for d in docs if d["tenant"] == t] for t in TENANTS}
              if config == "tenant" else {"shared": docs})
    if only:
        groups = {only: groups[only]}
    for name, group in groups.items():
        wdir = OUT / f"lr-{name}"
        wdir.mkdir(parents=True, exist_ok=True)
        rag = await open_rag(wdir)
        texts = [doc_text(d) for d in group]
        print(f"[{name}] inserting {len(texts)} docs "
              f"(resumable — already-processed docs are skipped)")
        await rag.ainsert(texts)
        await rag.finalize_storages()
        print(f"[{name}] build complete")


async def run_queries(config):
    docs = load_jsonl(HERE / "docs-rfq.jsonl")
    # map retrieved context back to docs by ANSWER text: question titles are
    # identical across tenants by construction, answers are tenant-specific
    # (verified: 940/940 unique at 120 normalized chars)
    keys = [(d["id"], norm(d["text"])[:120]) for d in docs]
    queries = (load_jsonl(HERE / "queries-rfq-multi.jsonl")
               + load_jsonl(HERE / "queries-rfq-global.jsonl"))
    from lightrag import QueryParam

    rags = {}
    for q in queries:
        gname = q["tenant"] if config == "tenant" else "shared"
        if gname not in rags:
            rags[gname] = await open_rag(OUT / f"lr-{gname}")
    results = {}
    for mode in ("local", "hybrid"):
        out_path = OUT / f"lightrag-{config}-{mode}.jsonl"
        done = {}
        if out_path.exists():
            done = {json.loads(l)["query_id"]: json.loads(l)
                    for l in open(out_path) if l.strip()}
        rows = []
        for q in queries:
            if q["query_id"] in done:
                rows.append(done[q["query_id"]])
                continue
            gname = q["tenant"] if config == "tenant" else "shared"
            ctx = await rags[gname].aquery(
                q["query"], param=QueryParam(mode=mode, only_need_context=True))
            nctx = norm(ctx or "")
            hits = sorted((nctx.find(k), did) for did, k in keys
                          if k in nctx)
            ranked = [did for _, did in hits][:10]
            rows.append({"query_id": q["query_id"], "tenant": q["tenant"],
                         "ranked": ranked, "n_ctx_chars": len(ctx or ""),
                         "ctx": ctx or ""})  # kept so re-mapping is offline
            out_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            print(f"{mode} {q['query_id']} -> {len(ranked)} docs")
        out_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        results[mode] = rows
    for gname, rag in rags.items():
        await rag.finalize_storages()


def score():
    docs = load_jsonl(HERE / "docs-rfq.jsonl")
    control = {d["id"]: d["control_id"] for d in docs}
    tenant_of = {d["id"]: d["tenant"] for d in docs}
    qrels = load_qrels(HERE / "qrels-rfq-multi.tsv")
    qrels.update(load_qrels(HERE / "qrels-rfq-global.tsv"))
    arms = json.loads((OUT / "arms.json").read_text())

    per = {a: arms[a] for a in arms}
    leak = {}
    for config in ("tenant", "shared"):
        for mode in ("local", "hybrid"):
            p = OUT / f"lightrag-{config}-{mode}.jsonl"
            if not p.exists():
                continue
            name = f"lr-{config}-{mode}"
            per[name] = defaultdict(dict)
            leak[name] = 0
            for l in open(p):
                r = json.loads(l)
                qid, ranked = r["query_id"], r["ranked"]
                leak[name] += sum(1 for d in ranked
                                  if tenant_of[d] != r["tenant"])
                rels = qrels[qid]
                parts = [d for d, g in rels.items() if g == 2]
                if qid.startswith("mq"):
                    per[name]["ndcg-multi"][qid] = ndcg(ranked, rels, 10)
                    found = sum(1 for x in parts if x in ranked)
                    per[name]["parts-recall"][qid] = found / len(parts)
                    per[name]["all-parts"][qid] = float(found == len(parts))
                else:
                    per[name]["ndcg-global"][qid] = ndcg(ranked, rels, 10)
                    n_ctl = min(10, len({control[d] for d in rels}))
                    hit = {control[d] for d in ranked if d in rels}
                    per[name]["coverage"][qid] = len(hit) / n_ctl

    rng = random.Random(7)
    def boot(m, a, b):
        qids = [q for q in per[a].get(m, {}) if q in per[b].get(m, {})]
        if not qids:
            return 0.0, 1.0
        d = [per[a][m][q] - per[b][m][q] for q in qids]
        n = len(d)
        delta = sum(d) / n
        if all(abs(x) < 1e-12 for x in d):
            return delta, 1.0
        hits = sum(1 for _ in range(10000)
                   if ((s := sum(d[rng.randrange(n)] for _ in range(n)) / n) <= 0) == (delta > 0))
        return delta, min(1.0, 2 * hits / 10000)

    metrics = ["ndcg-multi", "parts-recall", "all-parts", "ndcg-global", "coverage"]
    names = [a for a in per if per[a]]
    print(f"{'arm':<18}" + "".join(f"{m:>14}" for m in metrics) + f"{'leaked':>8}")
    for a in names:
        row = "".join(
            f"{(sum(per[a][m].values())/len(per[a][m])) if per[a].get(m) else float('nan'):>14.3f}"
            for m in metrics)
        print(f"{a:<18}{row}{leak.get(a, 0):>8}")
    print("\npaired bootstrap vs decomp —")
    for a in names:
        if a.startswith("lr-"):
            for m in metrics:
                d, p = boot(m, a, "decomp")
                print(f"  {a} {m}: {d:+.3f} p={p:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["tenant", "shared"], default="tenant")
    ap.add_argument("--step", choices=["build", "query", "score"], required=True)
    ap.add_argument("--only", help="build a single graph (one graph per process)")
    args = ap.parse_args()
    if args.step == "build":
        asyncio.run(build(args.config, args.only))
    elif args.step == "query":
        asyncio.run(run_queries(args.config))
    else:
        score()


if __name__ == "__main__":
    main()
