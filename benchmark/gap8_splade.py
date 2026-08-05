#!/usr/bin/env python3
"""Gap 8: learned sparse retrieval (SPLADE), offline.

Every retrieval family in the study has been measured except one: learned
sparse. SPLADE expands query and document into weighted vocabulary terms via
an MLM head, promising exactly the blend this workload wants — BM25's exact-
match ceiling on verbatim re-runs plus learned expansion to survive
paraphrase, where plain BM25 collapses (0.279 on the 47q set). Two questions:
does the expansion actually close the paraphrase gap, and does SPLADE obey
the study's central rule (match the stored *question*, not the answer)?

Arms (model x field), scored on the scaled paraphrase set (orig-47 /
new-185 / full-232) and the 232-query verbatim set:

  nomic-answer   dense, answer text, standard prefixes — offline twin of the
                 served `semantic` arm
  nomic-q2q      dense, stored question, standard prefixes (study baseline)
  nomic-q2q-sym  dense, stored question, query prefix both sides (Gap 7 winner)
  splade-answer  SPLADE++ (naver/splade-cocondenser-ensembledistil), answer
  splade-q2q     SPLADE++, stored question

Same exact within-tenant scan as the Gap 7 bake-off; compare arms within
this table, not against served numbers (serving costs ~0.02, Gap 5).
Per-query scores -> results/gap8/splade.json.

  python3 gap8_splade.py
"""
import gc
import json
import random
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForMaskedLM, AutoTokenizer

from evaluate import load_jsonl, load_qrels, ndcg

HERE = Path(__file__).parent
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SPLADE_MODEL = "naver/splade-cocondenser-ensembledistil"

SETS = {
    "para-all": ("queries-rfq-para-all.jsonl", "qrels-rfq-para-all.tsv"),
    "verbatim": ("queries-rfq.jsonl", "qrels-rfq.tsv"),
}

# arm -> (encoder key, doc field, query prefix, doc prefix)
ARMS = {
    "nomic-answer":  ("nomic", "text", "search_query: ", "search_document: "),
    "nomic-q2q":     ("nomic", "title", "search_query: ", "search_document: "),
    "nomic-q2q-sym": ("nomic", "title", "search_query: ", "search_query: "),
    "splade-answer": ("splade", "text", "", ""),
    "splade-q2q":    ("splade", "title", "", ""),
}


def splade_encode(model, tok, texts, batch_size=16):
    reps = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = tok(texts[i:i + batch_size], padding=True, truncation=True,
                        max_length=512, return_tensors="pt").to(DEVICE)
            logits = model(**batch).logits
            w = torch.log1p(torch.relu(logits))
            w = (w * batch["attention_mask"].unsqueeze(-1)).max(dim=1).values
            reps.append(w.cpu().float().numpy())
    return np.concatenate(reps)


def main():
    docs = load_jsonl(HERE / "docs-rfq.jsonl")
    doc_ids = [d["id"] for d in docs]
    doc_tenant = {d["id"]: d["tenant"] for d in docs}

    sets = {}
    for name, (qfile, qrfile) in SETS.items():
        qrels = load_qrels(HERE / qrfile)
        queries = [q for q in load_jsonl(HERE / qfile)
                   if any(g > 0 for g in qrels.get(q["query_id"], {}).values())]
        sets[name] = (queries, qrels)
        print(f"{name}: {len(queries)} queries")
    print(f"{len(docs)} docs, device {DEVICE}")

    per = {name: {} for name in SETS}  # set -> arm -> qid -> ndcg
    encoders = {}

    def get_encoder(key):
        if key not in encoders:
            encoders.clear()
            gc.collect()
            if DEVICE == "mps":
                torch.mps.empty_cache()
            if key == "nomic":
                encoders[key] = SentenceTransformer(
                    "nomic-ai/modernbert-embed-base", device=DEVICE)
            else:
                tok = AutoTokenizer.from_pretrained(SPLADE_MODEL)
                mdl = AutoModelForMaskedLM.from_pretrained(SPLADE_MODEL).to(DEVICE).eval()
                encoders[key] = (mdl, tok)
        return encoders[key]

    for arm, (enc_key, field, qprefix, dprefix) in ARMS.items():
        enc = get_encoder(enc_key)
        if enc_key == "nomic":
            D = enc.encode([dprefix + d[field] for d in docs],
                           normalize_embeddings=True, batch_size=16,
                           show_progress_bar=False)
        else:
            D = splade_encode(*enc, [d[field] for d in docs])
        for name, (queries, qrels) in sets.items():
            if enc_key == "nomic":
                Q = enc.encode([qprefix + q["query"] for q in queries],
                               normalize_embeddings=True, batch_size=16,
                               show_progress_bar=False)
            else:
                Q = splade_encode(*enc, [q["query"] for q in queries])
            for qi, q in enumerate(queries):
                sims = D @ Q[qi]
                order = np.argsort(-sims)
                ranked = [doc_ids[j] for j in order
                          if doc_tenant[doc_ids[j]] == q["tenant"]][:10]
                per[name][arm] = per[name].get(arm, {})
                per[name][arm][q["query_id"]] = ndcg(ranked, qrels[q["query_id"]], 10)
        print(f"{arm:<14} done")

    para_q, _ = sets["para-all"]
    orig = [q["query_id"] for q in para_q if q["cluster"] == "questionnaire-para"]
    new = [q["query_id"] for q in para_q if q["cluster"] == "questionnaire-para-new"]

    def mean(name, a, qids):
        return sum(per[name][a][q] for q in qids) / len(qids)

    rng = random.Random(7)
    def boot(name, a, b, qids):
        d = [per[name][a][q] - per[name][b][q] for q in qids]
        n = len(d)
        delta = sum(d) / n
        hits = sum(1 for _ in range(10000)
                   if ((s := sum(d[rng.randrange(n)] for _ in range(n)) / n) <= 0) == (delta > 0))
        return delta, min(1.0, 2 * hits / 10000)

    full = orig + new
    verb = [q["query_id"] for q in sets["verbatim"][0]]
    print(f"\n{'arm':<14} {'orig-47':>8} {'new-185':>8} {'full-232':>9} {'verbatim':>9}")
    for arm in ARMS:
        print(f"{arm:<14} {mean('para-all', arm, orig):>8.3f}"
              f" {mean('para-all', arm, new):>8.3f}"
              f" {mean('para-all', arm, full):>9.3f}"
              f" {mean('verbatim', arm, verb):>9.3f}")

    print("\npaired bootstrap (delta, p) —")
    pairs = [
        ("splade-q2q", "splade-answer", "para-all", full, "splade obeys the library rule?"),
        ("splade-answer", "nomic-answer", "para-all", full, "sparse vs dense on answers"),
        ("splade-q2q", "nomic-q2q", "para-all", full, "sparse vs dense baseline q2q"),
        ("splade-q2q", "nomic-q2q-sym", "para-all", full, "sparse vs Gap 7 winner"),
        ("splade-q2q", "nomic-q2q-sym", "para-all", new, "  ... out-of-sample only"),
        ("splade-q2q", "nomic-q2q-sym", "verbatim", verb, "sparse vs Gap 7 winner, verbatim"),
    ]
    for a, b, name, qids, note in pairs:
        d, p = boot(name, a, b, qids)
        print(f"  {a} vs {b} [{name}]: {d:+.3f} p={p:.4f}   ({note})")

    out = HERE / "results" / "gap8"
    out.mkdir(parents=True, exist_ok=True)
    (out / "splade.json").write_text(json.dumps(
        {name: {a: {q: round(v, 6) for q, v in arms[a].items()} for a in arms}
         for name, arms in per.items()}, indent=1))


if __name__ == "__main__":
    main()
