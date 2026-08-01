#!/usr/bin/env python3
"""Generate the HyDE query files for Exp 14 from the paraphrase set.

For each query in the input set, one LLM call produces two hypothetical
texts: the *answer* a vendor's security team would write (classic HyDE — maps
the query into answer space) and the *standard questionnaire wording* of the
question (the symmetric variant — maps it into question space, where Exp 8
found the signal lives). Four query files come out, named after the input and
keyed by the original query_id so the matching qrels apply unchanged:

  <input>-hyde-ans.jsonl      query = hypothetical answer
  <input>-hyde-ans-cat.jsonl  query = paraphrase + hypothetical answer
  <input>-hyde-q.jsonl        query = hypothetical standard question
  <input>-hyde-q-cat.jsonl    query = paraphrase + hypothetical question

The generator is deliberately a different model family from the gpt-4o-mini
that built the corpus: both the stored answers and the hypothetical ones are
LLM text, and same-family register match would flatter HyDE (RESULTS.md
limitation 1). Raw generations are cached in hyde_generations.jsonl, so
re-runs only call the API for missing query_ids.

  python3 hyde_generate.py
  python3 hyde_generate.py --model anthropic/claude-3.5-haiku
"""
import argparse
import json
import ssl
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import certifi

HERE = Path(__file__).parent
MODEL = "google/gemini-2.5-flash"
CTX = ssl.create_default_context(cafile=certifi.where())

PROMPT = """A buyer evaluating a B2B software vendor asked this security question in their own words:

"{query}"

Return a JSON object with exactly two string fields:

"question": the formal wording this item would have in a standard security questionnaire (SIG/CAIQ-style control language). One question, no preamble.

"answer": the answer a vendor's security team would keep in their approved answer library for it. 80-120 words, first person plural, concrete about practices but with no company or product names."""


def api_key():
    env = HERE.parent / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("OPENROUTER_API_KEY"):
                return line.split("=", 1)[1].strip()
    raise SystemExit(f"No OPENROUTER_API_KEY in {env}")


def llm(model, query, temperature=0.7):
    body = {"model": model,
            "messages": [{"role": "user", "content": PROMPT.format(query=query)}],
            "max_tokens": 500, "temperature": temperature,
            "response_format": {"type": "json_object"}}
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {api_key()}",
                 "Content-Type": "application/json"})
    for attempt in range(5):
        try:
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=120, context=CTX) as r:
                text = json.load(r)["choices"][0]["message"]["content"]
            text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
            out = json.loads(text)
            if not (out.get("question") and out.get("answer")):
                raise ValueError(f"missing field in {out.keys()}")
            out["secs"] = round(time.time() - t0, 2)
            return out
        except Exception as e:
            if attempt == 4:
                print(f"  ! giving up on: {query[:60]}... ({e})", file=sys.stderr)
                return None
            time.sleep(2 * (attempt + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--queries", default="queries-rfq-para.jsonl")
    args = ap.parse_args()

    queries = [json.loads(l) for l in open(HERE / args.queries)]
    cache_path = HERE / "hyde_generations.jsonl"
    cache = {}
    if cache_path.exists():
        for l in open(cache_path):
            g = json.loads(l)
            cache[g["query_id"]] = g

    missing = [q for q in queries if q["query_id"] not in cache]
    print(f"{len(queries)} queries, {len(missing)} to generate with {args.model}")
    if missing:
        with ThreadPoolExecutor(max_workers=8) as ex:
            res = list(ex.map(lambda q: (q, llm(args.model, q["query"])), missing))
        with open(cache_path, "a") as f:
            for q, out in res:
                if out is None:
                    continue
                g = {"query_id": q["query_id"], "orig_query": q["query"],
                     "question": out["question"], "answer": out["answer"],
                     "model": args.model, "secs": out["secs"]}
                cache[q["query_id"]] = g
                f.write(json.dumps(g) + "\n")
        print(f"  {sum(1 for _, o in res if o)} generated")

    done = [q for q in queries if q["query_id"] in cache]
    if len(done) < len(queries):
        sys.exit(f"only {len(done)}/{len(queries)} generated — re-run to fill the gaps")

    stem = args.queries.removesuffix(".jsonl")
    variants = {
        f"{stem}-hyde-ans.jsonl":     lambda q, g: g["answer"],
        f"{stem}-hyde-ans-cat.jsonl": lambda q, g: f'{q["query"]} {g["answer"]}',
        f"{stem}-hyde-q.jsonl":       lambda q, g: g["question"],
        f"{stem}-hyde-q-cat.jsonl":   lambda q, g: f'{q["query"]} {g["question"]}',
    }
    for name, text in variants.items():
        with open(HERE / name, "w") as f:
            for q in done:
                g = cache[q["query_id"]]
                f.write(json.dumps({**q, "query": text(q, g), "orig_query": q["query"]}) + "\n")
        print(f"wrote {name}")

    secs = [cache[q["query_id"]]["secs"] for q in done]
    print(f"generation latency: mean {sum(secs)/len(secs):.2f}s, "
          f"max {max(secs):.2f}s per query")


if __name__ == "__main__":
    main()
