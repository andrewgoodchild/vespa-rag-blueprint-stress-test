#!/usr/bin/env python3
"""Experiment 5: staleness as a dedup tie-break (post-retrieval prototype).

Exp 2/4 showed a global freshness feature is unlearnable and conceptually
wrong: freshness only discriminates between near-duplicates on the same
topic. This reranker applies exactly that: within a query's result list,
cluster near-duplicate docs (content-word Jaccard >= --tau, same tenant),
then reorder each cluster's members across the positions they occupy so the
newest doc comes first. Docs outside clusters are untouched.

In production this would live in a Vespa global-phase expression or a
custom searcher; a run-file transform is the cheapest honest prototype.

Usage: rerank_fresh.py in_run.txt out_run.txt [--tau 0.25]
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
STOP = set("the a an and or of to in for is are with by on at as from we our your you all any not no".split())


def content_words(text):
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_run")
    ap.add_argument("out_run")
    ap.add_argument("--tau", type=float, default=0.25)
    args = ap.parse_args()

    docs = {}
    for line in open(HERE / "docs.jsonl"):
        d = json.loads(line)
        docs[d["id"]] = {"tenant": d["tenant"], "created": d["created"], "words": content_words(d["text"])}

    run = defaultdict(list)
    for line in open(args.in_run):
        p = line.split()
        run[p[0]].append((p[2], float(p[4])))

    out = []
    swaps = 0
    for qid, ranked in run.items():
        ids = [d for d, _ in ranked]
        # union-find over near-duplicate pairs
        parent = {d: d for d in ids}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                da, db = docs.get(a), docs.get(b)
                if not da or not db or da["tenant"] != db["tenant"]:
                    continue
                inter = da["words"] & db["words"]
                union = da["words"] | db["words"]
                if union and len(inter) / len(union) >= args.tau:
                    parent[find(a)] = find(b)

        clusters = defaultdict(list)
        for d in ids:
            clusters[find(d)].append(d)

        new_order = list(ids)
        for members in clusters.values():
            if len(members) < 2:
                continue
            positions = sorted(new_order.index(d) for d in members)
            by_fresh = sorted(members, key=lambda d: (docs[d]["created"], -ids.index(d)), reverse=True)
            if [new_order[p] for p in positions] != by_fresh:
                swaps += 1
            for pos, d in zip(positions, by_fresh):
                new_order[pos] = d

        for rank, d in enumerate(new_order, 1):
            # descending pseudo-scores keep the file well-formed for TREC tools
            out.append(f"{qid} Q0 {d} {rank} {float(len(new_order) - rank):.1f} fresh-rerank")

    Path(args.out_run).write_text("\n".join(out) + "\n")
    print(f"reranked {len(run)} queries, clusters reordered in {swaps} query-cluster cases")


if __name__ == "__main__":
    main()
