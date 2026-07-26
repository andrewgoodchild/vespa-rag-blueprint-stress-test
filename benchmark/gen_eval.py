#!/usr/bin/env python3
"""Gap 1: the generation stage (RAG answer) + faithfulness/abstention eval.

For each question: retrieve top-k past Q&A from Vespa, prompt an LLM to draft a
questionnaire answer grounded ONLY in that context, then judge the result.
- Answerable questions (real paraphrases): score faithfulness (grounded in
  context) and correctness (matches the vendor's gold answer).
- Unanswerable questions (crafted, absent from the questionnaire): score abstention (did it
  decline instead of hallucinating).

LLM calls go through OpenRouter (key in .env). Generator and judge both use a
cheap model; the generator≠judge separation is weak (same model) — noted as a
limitation. Retrieval uses the deployed student-pairwise profile.
"""
import json
import ssl
import time
import urllib.request
from pathlib import Path

import certifi

HERE = Path(__file__).parent
CTX = ssl.create_default_context(cafile=certifi.where())
GEN_MODEL = "openai/gpt-4o-mini"
JUDGE_MODEL = "openai/gpt-4o-mini"


def api_key():
    """OPENROUTER_API_KEY from .env at the repo root."""
    env = HERE.parent / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("OPENROUTER_API_KEY"):
                return line.split("=", 1)[1].strip()
    raise SystemExit(
        f"No OPENROUTER_API_KEY found. Create {env} containing:\n"
        "  OPENROUTER_API_KEY=<your key>\n"
        "Only the generation scripts need it; the committed corpus and results "
        "reproduce every number in RESULTS.md with no LLM calls.")


WHERE = ('userInput(@query) or ({label:"t",targetHits:100}nearestNeighbor(title_embedding,embedding)) '
         'or ({label:"c",targetHits:100}nearestNeighbor(chunk_embeddings,embedding))')

# Unanswerable but plausible questions — genuinely absent from a security
# questionnaire (commercial / ESG / HR / trivia). Abstention is the correct answer.
UNANSWERABLE = [
    ("kestrel_cloud", "What is your annual subscription pricing for enterprise customers?"),
    ("kestrel_cloud", "What is your company's net-zero carbon target date?"),
    ("tanager_geo", "Do you offer a free trial, and how long does it last?"),
    ("orrery_suite", "How many patents does your company currently hold?"),
    ("pellucid_data", "What is your customer refund and cancellation policy?"),
    ("tanager_geo", "What programming languages is your product built in?"),
    ("pellucid_data", "Who is your current Chief Executive Officer?"),
    ("orrery_suite", "What is your average employee Glassdoor rating?"),
]


def llm(model, messages, max_tokens=400, temperature=0.0):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json"})
    for _ in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60, context=CTX) as r:
                return json.load(r)["choices"][0]["message"]["content"].strip()
        except Exception:
            time.sleep(3)
    return ""


def retrieve(query, tenant, k=3):
    p = {"query": query, "timeout": "20s", "hits": k,
         "yql": f"select * from rfq where tenant contains '{tenant}' and ({WHERE})",
         "ranking.profile": "student-pairwise",
         "presentation.summary": "no-chunks",
         "ranking.features.query(embedding)": "embed(nomicmb, @query)",
         "ranking.features.query(float_embedding)": "embed(nomicmb, @query)",
         "input.query(qt)": "embed(colbert, @query)"}
    req = urllib.request.Request("http://127.0.0.1:8080/search/",
        data=json.dumps(p).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        resp = json.load(r)
    out = []
    for h in resp["root"].get("children", []):
        f = h.get("fields", {})
        if "title" in f and "chunks" in f:
            out.append((f["title"], " ".join(f["chunks"])))
    return out


GEN_SYS = ("You are drafting an answer to a vendor security questionnaire. Use ONLY the "
           "provided past Q&A context. If the context does not contain the information needed "
           "to answer, reply with exactly: NOT COVERED IN OUR DOCUMENTATION. Do not use outside "
           "knowledge. Be concise (2-4 sentences).")


def generate(query, ctx):
    ctxstr = "\n\n".join(f"[Past Q] {q}\n[Past A] {a}" for q, a in ctx) or "(no context retrieved)"
    return llm(GEN_MODEL, [
        {"role": "system", "content": GEN_SYS},
        {"role": "user", "content": f"CONTEXT:\n{ctxstr}\n\nQUESTION: {query}\n\nANSWER:"}])


def judge_answerable(query, gold, answer, ctx):
    ctxstr = "\n\n".join(f"{a}" for _, a in ctx)
    prompt = (f"QUESTION: {query}\n\nRETRIEVED CONTEXT:\n{ctxstr}\n\nGOLD ANSWER (vendor's real answer): "
              f"{gold}\n\nCANDIDATE ANSWER: {answer}\n\n"
              "Score the candidate answer. Return ONLY compact JSON: "
              '{"faithful": 0 or 1, "correct": 0 or 1}. '
              "faithful=1 iff every claim in the candidate is supported by the retrieved context "
              "(no outside facts, no invention). correct=1 iff the candidate conveys the same key "
              "fact as the gold answer. If the candidate says NOT COVERED, faithful=1 and correct=0.")
    r = llm(JUDGE_MODEL, [{"role": "user", "content": prompt}], max_tokens=40)
    try:
        j = json.loads(r[r.find("{"):r.rfind("}") + 1])
        return int(j.get("faithful", 0)), int(j.get("correct", 0))
    except Exception:
        return None, None


def abstained(answer):
    a = answer.lower()
    return "not covered" in a or "does not contain" in a or "no information" in a or "cannot" in a[:40]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-answerable", type=int, default=20)
    args = ap.parse_args()

    qrels = {}
    for line in open(HERE / "qrels-rfq-para.tsv"):
        if line.strip():
            q, _, d, g = line.split()
            if int(g) == 2:
                qrels[q] = d
    docs = {json.loads(l)["id"]: json.loads(l) for l in open(HERE / "docs-rfq.jsonl")}
    queries = [json.loads(l) for l in open(HERE / "queries-rfq-para.jsonl")][: args.n_answerable]

    results = {"answerable": [], "unanswerable": []}
    print("=== answerable ===")
    for q in queries:
        gold = docs.get(qrels.get(q["query_id"], ""), {}).get("text", "")
        ctx = retrieve(q["query"], q["tenant"])
        ans = generate(q["query"], ctx)
        faith, corr = judge_answerable(q["query"], gold, ans, ctx)
        results["answerable"].append({"q": q["query"], "faith": faith, "corr": corr,
                                      "abstained": abstained(ans)})
        print(f"  {q['query_id']}: faithful={faith} correct={corr}{' [ABSTAINED]' if abstained(ans) else ''}")

    print("=== unanswerable (abstention test) ===")
    for tenant, query in UNANSWERABLE:
        ctx = retrieve(query, tenant)
        ans = generate(query, ctx)
        ab = abstained(ans)
        results["unanswerable"].append({"q": query, "abstained": ab})
        print(f"  [{tenant}] {'ABSTAINED ✓' if ab else 'ANSWERED ✗ → ' + ans[:70]}  | {query[:50]}")

    A = [r for r in results["answerable"] if r["faith"] is not None]
    n = len(A)
    faith_rate = sum(r["faith"] for r in A) / n
    corr_rate = sum(r["corr"] for r in A) / n
    answered = [r for r in A if not r["abstained"]]
    hallucination = sum(1 for r in answered if r["faith"] == 0) / max(len(answered), 1)
    U = results["unanswerable"]
    abstain_rate = sum(r["abstained"] for r in U) / len(U)

    print(f"\n=== Gap 1 generation-stage results ===")
    print(f"answerable ({n}): faithfulness {faith_rate:.2f} · correctness {corr_rate:.2f}")
    print(f"  hallucination rate (unfaithful among answered): {hallucination:.2f}")
    print(f"unanswerable ({len(U)}): abstention rate {abstain_rate:.2f}")
    (HERE / "results" / "gap1").mkdir(parents=True, exist_ok=True)
    (HERE / "results" / "gap1" / "gen_eval.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
