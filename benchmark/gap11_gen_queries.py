#!/usr/bin/env python3
"""Gap 11a: synthesize the multi-hop query sets graph retrieval is for.

The existing workload is single-hop by construction — one query, one stored
answer. Graph RAG's pitch is exactly the queries that isn't: buyers who
bundle several controls into one question, and buyers who ask for a
domain-level overview. Two new clusters:

  compositional  60 queries, each built from 2-3 specific questionnaire
                 questions (45 same-domain, 15 cross-domain). The LLM
                 phrases one natural buyer question asking all parts at
                 once; qrels grade every constituent 2 and its control-
                 siblings 1. A system that fetches two of three parts
                 produces a confidently wrong answer — so the metric that
                 matters is all-parts@10, not nDCG alone.
  global         30 queries asking for an overview of one domain
                 ("summarize your access-control posture"); qrels grade
                 the tenant's whole domain 1, scored on control coverage.

Part selection is seeded (11) and deterministic; only the phrasing is
LLM-generated (google/gemini-2.5-flash via OpenRouter — the same
different-family-from-the-corpus-generator convention as HyDE). Generated
files are committed, so downstream steps need no API key. Resume-safe:
already-phrased queries are skipped.

  python3 gap11_gen_queries.py
"""
import json
import os
import random
import ssl
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import certifi

from build_synthetic_rfq_corpus import TAXONOMY

HERE = Path(__file__).parent
MODEL = "google/gemini-2.5-flash"
SSL_CTX = ssl.create_default_context(cafile=certifi.where())
DOMAIN = {code: (name, desc) for code, name, _, desc in TAXONOMY}


def api_key():
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    env = HERE.parent / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            k, _, v = line.strip().partition("=")
            if k == "OPENROUTER_API_KEY" and v:
                return v
    raise SystemExit("OPENROUTER_API_KEY not set (env or repo .env)")


def llm(prompt, temperature=0.7, retries=4):
    body = {"model": MODEL, "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}]}
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                "https://openrouter.ai/api/v1/chat/completions",
                data=json.dumps(body).encode(),
                headers={"Authorization": f"Bearer {api_key()}",
                         "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120, context=SSL_CTX) as r:
                out = json.loads(r.read())
            return out["choices"][0]["message"]["content"].strip().strip('"')
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


COMPO_PROMPT = """You write questions that a buyer's procurement team sends to a \
software vendor during a security review. Combine the following questionnaire \
questions into ONE natural question the buyer would ask in their own words. It \
must ask for all of them together, informally, without copying the questionnaire \
wording. One or two sentences. Return only the question.

{parts}"""

GLOBAL_PROMPT = """You write questions that a buyer's procurement team sends to a \
software vendor during a security review. Write ONE informal question asking the \
vendor to give an overview of their {name} practices — the area covering {desc}. \
The buyer wants a summary of the whole area, in their own words, not a specific \
detail. One sentence. Return only the question."""


def build_plan(docs):
    rng = random.Random(11)
    tenants = sorted({d["tenant"] for d in docs})
    ctl = defaultdict(lambda: defaultdict(list))  # tenant -> control -> [qid]
    for d in docs:
        ctl[d["tenant"]][d["control_id"]].append(d["question_id"])
    dom = {t: defaultdict(list) for t in tenants}  # tenant -> domain -> [ctl]
    for t in tenants:
        for c in ctl[t]:
            dom[t][c.split("-")[0]].append(c)

    compo, seen = [], set()
    i = 0
    while len(compo) < 60:
        t = tenants[i % 4]
        i += 1
        cross = len(compo) >= 45
        if cross:
            d1, d2 = rng.sample([d for d in dom[t] if dom[t][d]], 2)
            controls = [rng.choice(dom[t][d1]), rng.choice(dom[t][d2])]
        else:
            cands = [d for d in dom[t] if len(dom[t][d]) >= 3]
            d1 = rng.choice(cands)
            controls = rng.sample(dom[t][d1], rng.choice([2, 2, 3]))
        parts = tuple(sorted(rng.choice(ctl[t][c]) for c in controls))
        if (t, parts) in seen:
            continue
        seen.add((t, parts))
        compo.append({"query_id": f"mq-{len(compo)+1:03d}", "tenant": t,
                      "cluster": "rfq-multi",
                      "kind": "cross-domain" if cross else "same-domain",
                      "parts": list(parts)})

    eligible = [(t, d) for t in tenants for d in sorted(dom[t])
                if len(dom[t][d]) >= 4]
    glob = [{"query_id": f"gq-{j+1:03d}", "tenant": t, "cluster": "rfq-global",
             "domain": d}
            for j, (t, d) in enumerate(rng.sample(eligible, 30))]
    return compo, glob


def main():
    docs = [json.loads(l) for l in open(HERE / "docs-rfq.jsonl")]
    title = {d["question_id"]: d["title"] for d in docs}
    compo, glob = build_plan(docs)

    # qrels first — they depend only on the deterministic plan
    with open(HERE / "qrels-rfq-multi.tsv", "w") as f:
        for q in compo:
            grades = {}
            for part in q["parts"]:
                grades[f"{q['tenant']}-{part.lower()}"] = 2
                control = part.rsplit(".", 1)[0]
                for d in docs:
                    if d["tenant"] == q["tenant"] and d["control_id"] == control:
                        grades.setdefault(d["id"], 1)
            for did, g in sorted(grades.items()):
                f.write(f"{q['query_id']}\t0\t{did}\t{g}\n")
    with open(HERE / "qrels-rfq-global.tsv", "w") as f:
        for q in glob:
            for d in docs:
                if d["tenant"] == q["tenant"] and \
                        d["control_id"].startswith(q["domain"] + "-"):
                    f.write(f"{q['query_id']}\t0\t{d['id']}\t1\n")

    for path, plan in ((HERE / "queries-rfq-multi.jsonl", compo),
                       (HERE / "queries-rfq-global.jsonl", glob)):
        done = {}
        if path.exists():
            done = {json.loads(l)["query_id"]: json.loads(l)
                    for l in open(path) if l.strip()}
        out = []
        for q in plan:
            if q["query_id"] in done and done[q["query_id"]].get("query"):
                out.append(done[q["query_id"]])
                continue
            if "parts" in q:
                parts = "\n".join(f"- {title[p]}" for p in q["parts"])
                q["query"] = llm(COMPO_PROMPT.format(parts=parts))
            else:
                name, desc = DOMAIN[q["domain"]]
                q["query"] = llm(GLOBAL_PROMPT.format(name=name, desc=desc))
            out.append(q)
            print(f"{q['query_id']}  {q['query'][:90]}")
            path.write_text("\n".join(json.dumps(x) for x in out) + "\n")
        path.write_text("\n".join(json.dumps(x) for x in out) + "\n")
    print(f"\n{len(compo)} compositional + {len(glob)} global queries ready")


if __name__ == "__main__":
    main()
