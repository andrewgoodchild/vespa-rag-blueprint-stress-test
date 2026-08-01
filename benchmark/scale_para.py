#!/usr/bin/env python3
"""Scale the paraphrase set 47 -> 232: one paraphrase twin per verbatim query.

Exp 6's lesson operationalised: finding 35 (HyDE normalise-and-keep, +0.084,
p~0.064 at n=47) needs a bigger query set before it can graduate. This script
extends the paraphrase set to cover every verbatim question_id, reusing
build_synthetic_rfq_corpus.gen_paraphrase verbatim — same model, same prompt,
same temperature — because the paraphrase prompt is the corpus's difficulty
dial (see LABBOOK 2026-07-27) and a new prompt would silently move it.

The original 47 queries are copied through byte-identical (they are the set
finding 35 was formed on); the 185 new ones get cluster
"questionnaire-para-new" so evaluate.py reports the out-of-sample subset
separately. New generations are cached into .synthetic-cache.json under the
same "paraphrases" key the build script uses, so a full corpus rebuild picks
them up for free.

Writes queries-rfq-para-all.jsonl + qrels-rfq-para-all.tsv (structural grades,
same rule as the build script: same question grade 2, covered sibling
sub-question grade 1).

  python3 scale_para.py
"""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from build_synthetic_rfq_corpus import gen_paraphrase, load_cache, save_cache

HERE = Path(__file__).parent


def main():
    verbatim = [json.loads(l) for l in open(HERE / "queries-rfq.jsonl")]
    existing = [json.loads(l) for l in open(HERE / "queries-rfq-para.jsonl")]
    have = {q["orig"] for q in existing}
    covered = {}
    for d in (json.loads(l) for l in open(HERE / "docs-rfq.jsonl")):
        covered.setdefault(d["tenant"], set()).add(d["question_id"])

    cache_path = HERE / ".synthetic-cache.json"
    cache = load_cache(cache_path)
    cache.setdefault("paraphrases", {})

    todo = [v for v in verbatim
            if v["query_id"] not in have and v["question_id"] not in cache["paraphrases"]]
    print(f"{len(verbatim)} verbatim queries, {len(existing)} already paraphrased, "
          f"{len(todo)} to generate")
    for attempt in range(3):
        if not todo:
            break
        with ThreadPoolExecutor(max_workers=8) as ex:
            res = list(ex.map(lambda v: (v, gen_paraphrase(v["query"])), todo))
        for v, p in res:
            if p:
                cache["paraphrases"][v["question_id"]] = p
        save_cache(cache_path, cache)
        todo = [v for v, p in res if not p]
        if todo:
            print(f"  retry {attempt + 1}: {len(todo)} failed the quality gate")
    if todo:
        raise SystemExit(f"{len(todo)} paraphrases still missing after retries")

    by_orig = {q["orig"]: q for q in existing}
    n_new = 0
    with open(HERE / "queries-rfq-para-all.jsonl", "w") as fq, \
         open(HERE / "qrels-rfq-para-all.tsv", "w") as fr:
        for v in verbatim:
            if v["query_id"] in by_orig:
                row = by_orig[v["query_id"]]
            else:
                n_new += 1
                row = {"query_id": "p" + v["query_id"], "orig": v["query_id"],
                       "tenant": v["tenant"], "cluster": "questionnaire-para-new",
                       "query": cache["paraphrases"][v["question_id"]]}
            fq.write(json.dumps(row) + "\n")
            qid, tenant = v["question_id"], v["tenant"]
            control = qid.split(".")[0]
            for qid2 in sorted(covered[tenant]):
                doc_id = f"{tenant}-{qid2.lower()}"
                if qid2 == qid:
                    fr.write(f'{row["query_id"]}\t0\t{doc_id}\t2\n')
                elif qid2.split(".")[0] == control:
                    fr.write(f'{row["query_id"]}\t0\t{doc_id}\t1\n')
    print(f"wrote queries-rfq-para-all.jsonl: {len(verbatim)} queries "
          f"({len(existing)} original + {n_new} new) and qrels-rfq-para-all.tsv")

    old = set((HERE / "qrels-rfq-para.tsv").read_text().splitlines())
    new = set((HERE / "qrels-rfq-para-all.tsv").read_text().splitlines())
    missing = old - new
    if missing:
        raise SystemExit(f"qrels regression: {len(missing)} original lines missing")
    print("original 47 queries' qrels reproduced exactly")


if __name__ == "__main__":
    main()
