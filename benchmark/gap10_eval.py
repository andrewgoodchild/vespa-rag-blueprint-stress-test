#!/usr/bin/env python3
"""Gap 10c: score the winning retrieval recipe over each parsed corpus.

The question the whole gap exists for: how much of the clean-text retrieval
quality survives the document layer? Each parsed corpus (results/gap10/
parsed-{arm}.jsonl) is indexed exactly like the clean control — library
model, symmetric prefixes, float title vectors, exact within-tenant scan —
and scored with the same qrels. Missing answers, mangled questions and junk
units are all charged to the arm that produced them, the way production
would charge them. Per-tenant splits are per-flavour splits by
construction (every query targets one tenant).

  python3 gap10_eval.py
"""
import json
import random
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

from evaluate import load_jsonl, load_qrels, ndcg

HERE = Path(__file__).parent
OUT = HERE / "results" / "gap10"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
PREFIX = "search_query: "

SETS = {
    "para-all": ("queries-rfq-para-all.jsonl", "qrels-rfq-para-all.tsv"),
    "verbatim": ("queries-rfq.jsonl", "qrels-rfq.tsv"),
}
TENANTS = ["tanager_geo", "kestrel_cloud", "orrery_suite", "pellucid_data"]
FLAVOUR = {"tanager_geo": "pdf-tables", "kestrel_cloud": "docx-headings",
           "orrery_suite": "xlsx", "pellucid_data": "pdf-prose"}


def main():
    corpora = {"clean": load_jsonl(HERE / "docs-rfq.jsonl")}
    for arm in ("naive", "structured", "docling"):
        p = OUT / f"parsed-{arm}.jsonl"
        if p.exists():
            corpora[arm] = load_jsonl(p)
        else:
            print(f"({arm}: no parsed corpus yet, skipping)")

    model = SentenceTransformer("nomic-ai/modernbert-embed-base", device=DEVICE)
    sets = {}
    for sname, (qfile, qrfile) in SETS.items():
        qrels = load_qrels(HERE / qrfile)
        queries = [q for q in load_jsonl(HERE / qfile)
                   if any(g > 0 for g in qrels.get(q["query_id"], {}).values())]
        Q = model.encode([PREFIX + q["query"] for q in queries],
                         normalize_embeddings=True, batch_size=16,
                         show_progress_bar=False)
        sets[sname] = (queries, qrels, Q)

    per = {}  # corpus -> set -> qid -> ndcg
    for cname, docs in corpora.items():
        D = model.encode([PREFIX + d["title"] for d in docs],
                         normalize_embeddings=True, batch_size=16,
                         show_progress_bar=False)
        ids = [d["id"] for d in docs]
        tenant = {d["id"]: d["tenant"] for d in docs}
        per[cname] = {}
        for sname, (queries, qrels, Q) in sets.items():
            scores = {}
            for qi, q in enumerate(queries):
                sims = D @ Q[qi]
                order = sims.argsort()[::-1]
                ranked = [ids[j] for j in order
                          if tenant[ids[j]] == q["tenant"]][:10]
                scores[q["query_id"]] = ndcg(ranked, qrels[q["query_id"]], 10)
            per[cname][sname] = scores
        print(f"{cname}: indexed {len(docs)} units, scored")

    rng = random.Random(7)
    def boot(a, b, sname, qids):
        d = [per[a][sname][q] - per[b][sname][q] for q in qids]
        n = len(d)
        delta = sum(d) / n
        if all(abs(x) < 1e-12 for x in d):
            return delta, 1.0
        hits = sum(1 for _ in range(10000)
                   if ((s := sum(d[rng.randrange(n)] for _ in range(n)) / n) <= 0) == (delta > 0))
        return delta, min(1.0, 2 * hits / 10000)

    para_q = sets["para-all"][0]
    by_tenant = {t: [q["query_id"] for q in para_q if q["tenant"] == t]
                 for t in TENANTS}
    all_para = [q["query_id"] for q in para_q]
    all_verb = [q["query_id"] for q in sets["verbatim"][0]]

    hdr = "  ".join(f"{FLAVOUR[t]:>13}" for t in TENANTS)
    print(f"\nnDCG@10, paraphrase-232 by flavour       {hdr}   "
          f"{'para-all':>8} {'verbatim':>8}")
    for cname in per:
        row = "  ".join(
            f"{sum(per[cname]['para-all'][q] for q in by_tenant[t]) / len(by_tenant[t]):>13.3f}"
            for t in TENANTS)
        pa = sum(per[cname]["para-all"][q] for q in all_para) / len(all_para)
        vb = sum(per[cname]["verbatim"][q] for q in all_verb) / len(all_verb)
        print(f"{cname:<40} {row}   {pa:>8.3f} {vb:>8.3f}")

    print("\npaired bootstrap vs clean control —")
    for cname in per:
        if cname == "clean":
            continue
        for sname, qids in (("para-all", all_para), ("verbatim", all_verb)):
            d, p = boot(cname, "clean", sname, qids)
            print(f"  {cname} [{sname}]: {d:+.4f} p={p:.4f}")
        for t in TENANTS:
            d, p = boot(cname, "clean", "para-all", by_tenant[t])
            print(f"      {FLAVOUR[t]:>13} para: {d:+.4f} p={p:.4f}")

    (OUT / "eval.json").write_text(json.dumps(
        {c: {s: {q: round(v, 6) for q, v in per[c][s].items()}
             for s in per[c]} for c in per}, indent=1))


if __name__ == "__main__":
    main()
