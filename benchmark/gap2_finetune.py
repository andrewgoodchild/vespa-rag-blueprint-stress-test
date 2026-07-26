#!/usr/bin/env python3
"""Gap 2 step 2: contrastive fine-tune a small embedder on the paraphrase pairs,
evaluate base vs fine-tuned on the HELD-OUT 47 paraphrases — all offline, no Vespa.

Retrieval is question->question (the library model): embed the incoming
paraphrase and each doc's stored question (title), rank within tenant by cosine,
score nDCG@10 against the paraphrase qrels. Baseline = the base MiniLM; then
fine-tune with MultipleNegativesRankingLoss on (reworded, canonical) pairs and
re-score. The delta isolates what contrastive fine-tuning buys on this task.
"""
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

HERE = Path(__file__).parent
BASE = "sentence-transformers/all-MiniLM-L6-v2"
DEVICE = "mps"


def load_eval():
    docs = [json.loads(l) for l in open(HERE / "docs-rfq.jsonl")]
    queries = [json.loads(l) for l in open(HERE / "queries-rfq-para.jsonl")]
    qrels = defaultdict(dict)
    for line in open(HERE / "qrels-rfq-para.tsv"):
        if line.strip():
            q, _, d, g = line.split()
            qrels[q][d] = int(g)
    return docs, queries, qrels


def ndcg(ranked, rels, k=10):
    dcg = sum((2 ** rels.get(d, 0) - 1) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted(rels.values(), reverse=True)
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal[:k]))
    return dcg / idcg if idcg else 0.0


def evaluate(model, docs, queries, qrels):
    # embed all doc questions (titles) and eval paraphrases
    doc_ids = [d["id"] for d in docs]
    doc_tenant = {d["id"]: d["tenant"] for d in docs}
    D = model.encode([d["title"] for d in docs], normalize_embeddings=True,
                     batch_size=64, show_progress_bar=False, device=DEVICE)
    Q = model.encode([q["query"] for q in queries], normalize_embeddings=True,
                     batch_size=64, show_progress_bar=False, device=DEVICE)
    scores = []
    for qi, q in enumerate(queries):
        sims = D @ Q[qi]
        order = np.argsort(-sims)
        ranked = [doc_ids[j] for j in order if doc_tenant[doc_ids[j]] == q["tenant"]][:10]
        scores.append(ndcg(ranked, qrels[q["query_id"]]))
    return sum(scores) / len(scores)


def main():
    docs, queries, qrels = load_eval()
    pairs = [json.loads(l) for l in open(HERE / "gap2_pairs.jsonl")]
    print(f"{len(docs)} docs, {len(queries)} eval paraphrases, {len(pairs)} training pairs")

    base = SentenceTransformer(BASE, device=DEVICE)
    base_ndcg = evaluate(base, docs, queries, qrels)
    print(f"BASE  ({BASE}) nDCG@10 = {base_ndcg:.3f}")

    model = SentenceTransformer(BASE, device=DEVICE)
    examples = [InputExample(texts=[p["anchor"], p["positive"]]) for p in pairs]
    loader = DataLoader(examples, batch_size=32, shuffle=True)
    loss = losses.MultipleNegativesRankingLoss(model)
    model.fit(train_objectives=[(loader, loss)], epochs=3,
              warmup_steps=int(0.1 * len(loader) * 3), show_progress_bar=False)
    tuned_ndcg = evaluate(model, docs, queries, qrels)
    print(f"TUNED (contrastive, 3 epochs) nDCG@10 = {tuned_ndcg:.3f}")
    print(f"\ndelta from fine-tuning: {tuned_ndcg - base_ndcg:+.3f}")
    (HERE / "results" / "gap2").mkdir(parents=True, exist_ok=True)
    (HERE / "results" / "gap2" / "result.json").write_text(
        json.dumps({"base": base_ndcg, "tuned": tuned_ndcg, "pairs": len(pairs)}, indent=1))


if __name__ == "__main__":
    main()
