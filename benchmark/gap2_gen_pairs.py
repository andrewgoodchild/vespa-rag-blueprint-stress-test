#!/usr/bin/env python3
"""Gap 2 step 1: generate (paraphrase -> canonical question) training pairs.

The corpus has no natural paraphrase supervision (questionnaire questions are
standardized, so cross-vendor question text is identical, not reworded). So we
synthesize it: ask an LLM to reword each canonical question the way a real
buyer's questionnaire would. Each (reworded, canonical) becomes a positive pair
for contrastive fine-tuning. Question_ids used in the held-out eval set are
EXCLUDED so the fine-tune never sees the eval targets.

Output: gap2_pairs.jsonl  {anchor: reworded, positive: canonical}
"""
import json
import ssl
import time
import urllib.request
from pathlib import Path

import certifi

HERE = Path(__file__).parent
CTX = ssl.create_default_context(cafile=certifi.where())
MODEL = "openai/gpt-4o-mini"


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
        "Only the generation scripts need it; the committed pairs in "
        "gap2_pairs.jsonl reproduce Exp 12 with no LLM calls.")


def llm(messages, max_tokens=800):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0.7}
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


def main():
    # held-out eval question_ids (from the paraphrase qrels' gold docs) -> exclude
    eval_qids = set()
    docs = {json.loads(l)["id"]: json.loads(l) for l in open(HERE / "docs-rfq.jsonl")}
    for line in open(HERE / "qrels-rfq-para.tsv"):
        if line.strip():
            _, _, d, g = line.split()
            if d in docs:
                eval_qids.add(docs[d]["question_id"])

    # one canonical question per question_id (dedup across vendors)
    canon = {}
    for d in docs.values():
        canon.setdefault(d["question_id"], d["title"])
    train_items = [(qid, q) for qid, q in sorted(canon.items()) if qid not in eval_qids]
    print(f"{len(canon)} unique questions, {len(eval_qids)} held out for eval, {len(train_items)} for training")

    out = open(HERE / "gap2_pairs.jsonl", "w")
    n = 0
    BATCH = 6
    t0 = time.time()
    for i in range(0, len(train_items), BATCH):
        chunk = train_items[i:i + BATCH]
        numbered = "\n".join(f"{j+1}. {q}" for j, (_, q) in enumerate(chunk))
        prompt = ("Reword each security-questionnaire question below the way a real customer's "
                  "questionnaire might phrase it — same meaning, natural/informal wording, different "
                  "vocabulary and sentence shape. Return ONLY a JSON array of strings, one reworded "
                  f"question per input, in order.\n\n{numbered}")
        r = llm([{"role": "user", "content": prompt}])
        try:
            arr = json.loads(r[r.find("["):r.rfind("]") + 1])
        except Exception:
            continue
        for (qid, canonical), reworded in zip(chunk, arr):
            if isinstance(reworded, str) and len(reworded) > 10:
                out.write(json.dumps({"anchor": reworded, "positive": canonical, "qid": qid}) + "\n")
                n += 1
        if (i // BATCH) % 10 == 0:
            print(f"  {i+len(chunk)}/{len(train_items)} questions, {n} pairs, {time.time()-t0:.0f}s", flush=True)
    out.close()
    print(f"GAP2 PAIRS DONE: {n} pairs in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
