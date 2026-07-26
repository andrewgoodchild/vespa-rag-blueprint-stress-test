#!/usr/bin/env python3
"""Build the clean-room RFQ corpus: our own questionnaire, our own vendors.

This replaces an earlier corpus parsed from a published industry questionnaire
completed by real vendors. Nothing here is derived from a third-party document: the
control taxonomy below is ours, the questions are written to it, and the four
vendors answering them are fictional. The corpus is therefore committed, and
the whole study clones and runs without downloading anything.

It is built to be a structural analogue of a real questionnaire corpus, because
the study's findings depend on those structural properties rather than on the
specific text:

  * one shared question set answered independently by every tenant, so the
    corpus is near-duplicate by construction (the central difficulty);
  * answers that name their own vendor and product, so cross-tenant hits are
    detectable and tenant isolation is a real compliance boundary;
  * uneven coverage — no tenant answers every question — so "no answer exists
    for this tenant" is a genuine case;
  * multi-part controls, so a sibling sub-question is a graded near-miss
    rather than a clean negative.

Question and answer text come from an LLM (OpenRouter, key in .env), prompted
per control and per question. Generation is cached in --cache so a re-run is
free and reproducible; the committed .jsonl files are the source of truth.

Usage:
  python3 build_synthetic_rfq_corpus.py                # build everything
  python3 build_synthetic_rfq_corpus.py --questions-only
"""
import argparse
import json
import random
import ssl
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import certifi

HERE = Path(__file__).parent
CTX = ssl.create_default_context(cafile=certifi.where())
MODEL = "openai/gpt-4o-mini"
SEED = 20260726

# ---------------------------------------------------------------- taxonomy --
# Our own control taxonomy. Seventeen domains covering the topics any security
# questionnaire covers — those topics are facts about the field, not anyone's
# property — but the grouping, the codes, the control counts and every question
# written against them are ours.
TAXONOMY = [
    ("GOV", "Governance, Risk & Compliance Program", 11,
     "the security program itself: ownership, risk register, board reporting, control framework"),
    ("AUD", "Audit, Assurance & Certification", 9,
     "internal and external audit, third-party attestations, evidence handling, findings tracking"),
    ("POL", "Policy Management & Exceptions", 8,
     "policy lifecycle, approval, communication, exception handling and expiry"),
    ("PER", "Personnel Security & Awareness", 14,
     "background screening, onboarding and offboarding, training, acceptable use, disciplinary process"),
    ("ACC", "Identity & Access Management", 16,
     "authentication, authorisation, privileged access, joiner-mover-leaver, access review, federation"),
    ("CRY", "Cryptography & Key Management", 15,
     "encryption at rest and in transit, algorithms, key generation, rotation, escrow, HSMs"),
    ("DAT", "Data Classification, Retention & Privacy", 18,
     "classification, residency, retention and deletion, subject rights, minimisation, secure disposal"),
    ("NET", "Network Security & Segmentation", 12,
     "perimeter and internal segmentation, firewalls, DDoS, remote access, wireless, egress control"),
    ("APP", "Application Security & Secure SDLC", 11,
     "secure design, code review, dependency and secrets management, application testing, API security"),
    ("CHG", "Change, Release & Configuration Management", 10,
     "change approval, release process, baselines, drift detection, rollback, emergency change"),
    ("VUL", "Vulnerability & Patch Management", 11,
     "scanning cadence, severity-based SLAs, penetration testing, remediation tracking, exceptions"),
    ("MON", "Logging, Monitoring & Telemetry", 13,
     "log sources and retention, integrity, time sync, alerting, SIEM, customer-visible telemetry"),
    ("INC", "Incident Detection & Response", 12,
     "detection, triage and severity, containment, customer notification timelines, forensics, post-incident review"),
    ("RES", "Business Continuity & Disaster Recovery", 13,
     "BIA, RTO and RPO, backup and restore testing, failover, pandemic and site-loss planning"),
    ("PHY", "Physical & Environmental Controls", 10,
     "data centre access, visitor handling, CCTV, power and cooling, environmental monitoring, media destruction"),
    ("SUP", "Supply Chain & Subprocessor Management", 9,
     "vendor due diligence, subprocessor disclosure, contractual flow-down, ongoing monitoring, exit"),
    ("END", "Endpoint & Device Management", 5,
     "corporate device hardening, disk encryption, MDM, BYOD, patching, lost-device response"),
]

# ------------------------------------------------------------------ tenants --
# Four fictional vendors. Distinct enough that an answer naming its own product
# is identifiable, similar enough that all four plausibly answer the same
# questionnaire — which is exactly what makes cross-tenant retrieval tempting.
TENANTS = [
    {"tenant": "tanager_geo", "company": "Tanager Geospatial",
     "product": "Tanager Atlas, a hosted geospatial mapping and analytics platform",
     "voice": "measured and engineering-led; cites its shared-responsibility model often",
     "coverage": 258},
    {"tenant": "kestrel_cloud", "company": "Kestrel Cloud Infrastructure",
     "product": "Kestrel Cloud, an IaaS and managed-container platform",
     "voice": "terse and control-reference heavy; leans on its own compliance programme names",
     "coverage": 214},
    {"tenant": "orrery_suite", "company": "Orrery Software",
     "product": "Orrery Business Suite, a multi-tenant SaaS finance and HR application",
     "voice": "formal and policy-quoting; describes process before technology",
     "coverage": 234},
    {"tenant": "pellucid_data", "company": "Pellucid Data Systems",
     "product": "Pellucid Lakehouse, a managed data warehouse and analytics service",
     "voice": "plain and specific; gives concrete intervals and named tooling",
     "coverage": 234},
]

# Sub-question shape, mirroring how real questionnaires split controls: mostly
# single-part, a long tail of multi-part ones. 197 controls -> 261 questions.
SUBQ_PROFILE = [(1, 144), (2, 46), (3, 5), (5, 2)]

# Size of the hard paraphrase set — the study's headline case.
N_PARA = 47


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
        "You only need this to rebuild the corpus from scratch — docs-rfq.jsonl "
        "and the query sets are committed, so every result reproduces without it.")


def llm(messages, max_tokens=1600, temperature=0.4, seed=None):
    key = api_key()
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature, "response_format": {"type": "json_object"}}
    if seed is not None:
        body["seed"] = seed
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=120, context=CTX) as r:
                return json.loads(json.load(r)["choices"][0]["message"]["content"])
        except Exception as e:
            if attempt == 4:
                print(f"  ! giving up: {e}", file=sys.stderr)
                return None
            time.sleep(2 * (attempt + 1))
    return None


def allocate_controls():
    """Deterministic control -> sub-question-count map across the taxonomy."""
    rng = random.Random(SEED)
    total = sum(n for _, _, n, _ in TAXONOMY)
    sizes = [s for s, count in SUBQ_PROFILE for _ in range(count)]
    assert len(sizes) == total, f"profile has {len(sizes)} controls, taxonomy wants {total}"
    rng.shuffle(sizes)
    out, i = [], 0
    for code, name, n, blurb in TAXONOMY:
        for c in range(1, n + 1):
            out.append({"control": f"{code}-{c:02d}", "domain": code, "domain_name": name,
                        "blurb": blurb, "nsub": sizes[i]})
            i += 1
    return out


def gen_domain_questions(domain):
    """Generate every control's questions for one domain in a single call.

    Per-control generation produced heavy duplication: given the same domain
    blurb sixteen times, the model returns the same question sixteen ways, and
    identical question text across control ids silently corrupts the qrels (a
    doc answering the twin question is graded 0 while being word-for-word as
    relevant). Generating the domain as a unit lets the model see what it has
    already asked and keep every control distinct.
    """
    code, name, _, blurb = domain["meta"]
    ctrls = domain["controls"]
    spec = "\n".join(f"- {c['control']}: {c['nsub']} question(s)" for c in ctrls)
    msg = [
        {"role": "system", "content":
         "You write vendor security questionnaires. You produce neutral, "
         "assessor-style questions. Reply with JSON only."},
        {"role": "user", "content":
         f"Domain: {name} — covering {blurb}.\n\n"
         f"Write questions for all {len(ctrls)} controls in this domain:\n{spec}\n\n"
         "Rules:\n"
         "- Each question asks a cloud service provider about ONE specific practice.\n"
         "- **The controls in a domain deliberately overlap in subject matter and "
         "should be easy to confuse with one another** — that is what a real "
         "questionnaire looks like, and it is the point of this exercise. Several "
         "controls may circle the same underlying topic.\n"
         "- **But no two questions may be paraphrases of each other.** Each must ask "
         "for something an expert could tell apart: a different object, scope, actor, "
         "artifact, trigger, or interval. If two answers to two of your questions "
         "could be identical and both correct, rewrite one.\n"
         "- Where a control has more than one question, those are parts of the SAME "
         "control differing in a specific way (does it exist / how often is it "
         "reviewed / who approves it) — closest of all, but still distinguishable.\n"
         "- 20 to 40 words each — real questionnaire items are long and compound, "
         "stacking several verbs or conditions into one sentence. End with a question "
         "mark. No numbering, no preamble.\n"
         "- Vary the openings across the domain.\n\n"
         'Reply as {"CODE-01": ["..."], "CODE-02": ["...", "..."], ...} using the exact '
         "control references above as keys."},
    ]
    r = llm(msg, max_tokens=4000, seed=SEED)
    if not r:
        return {}
    # Keep whatever came back complete. A domain where one control returned too
    # few questions should not discard the other sixteen — the caller re-runs
    # only what is still missing, so successive attempts converge.
    out = {}
    for c in ctrls:
        qs = r.get(c["control"])
        if not isinstance(qs, list):
            continue
        qs = [q.strip() for q in qs if isinstance(q, str) and q.strip().endswith("?")]
        if len(qs) >= c["nsub"]:
            out[c["control"]] = qs[:c["nsub"]]
    return out


def gen_answers(question, control):
    who = "\n".join(
        f"- {t['tenant']}: {t['company']}, which sells {t['product']}. Tone: {t['voice']}."
        for t in TENANTS)
    msg = [
        {"role": "system", "content":
         "You write the security-questionnaire answers a cloud vendor's compliance "
         "team files. Reply with JSON only."},
        {"role": "user", "content":
         f"Questionnaire control {control}.\nQuestion: {question}\n\n"
         f"Four different vendors each answer it:\n{who}\n\n"
         "Rules:\n"
         "- Answer as each vendor, in its own voice. Name the company or its product "
         "at least once in each answer, but vary where it falls — most answers should "
         "NOT open with it.\n"
         "- All four describe broadly similar industry practice — they are answering "
         "the same question honestly — but differ in specifics: intervals, tooling, "
         "role names, standards cited, and which parts are the customer's "
         "responsibility. They must NOT be paraphrases of one another.\n"
         "- Vary the lengths a lot, the way real filed answers do. Most should run 45 to "
         "90 words, but make at least one of the four noticeably terse (a single "
         "sentence under 25 words) and, where the question warrants detail, let one run "
         "long (150+ words). Do not give all four the same length.\n"
         "- Prose, no bullet points, no headings, no yes/no prefix.\n"
         "- Invent plausible specifics. Do not mention any real company.\n\n"
         'Reply as {"tanager_geo": "...", "kestrel_cloud": "...", '
         '"orrery_suite": "...", "pellucid_data": "..."}'},
    ]
    return llm(msg, max_tokens=1600, seed=SEED)


def gen_paraphrase(question):
    """Reword a question the way a buyer would — deliberately sharing as little
    vocabulary with the original as possible. That gap is the hard case."""
    msg = [
        {"role": "system", "content":
         "You rewrite security-questionnaire questions the way a real buyer would "
         "phrase them in their own words. Reply with JSON only."},
        {"role": "user", "content":
         f"Original question: {question}\n\n"
         "Rewrite it as the question a *procurement analyst* would type — someone who "
         "needs the same information but does not know your terminology:\n"
         "- share as few words as possible with the original; avoid its security "
         "jargon entirely and ask in plain business language;\n"
         "- be looser and less precise than the original, the way a real person is — "
         "drop the formal qualifiers, ask the underlying thing;\n"
         "- one sentence, 10 to 22 words, ending in a question mark;\n"
         "- do not add requirements the original did not ask for, and do not name the "
         "control or standard.\n\n"
         'Reply as {"paraphrase": "..."}'},
    ]
    r = llm(msg, max_tokens=200, temperature=0.7, seed=SEED)
    if not r or not isinstance(r.get("paraphrase"), str):
        return None
    p = r["paraphrase"].strip()
    return p if p.endswith("?") else None


def load_cache(path):
    return json.loads(path.read_text()) if path.exists() else {}


def save_cache(path, data):
    path.write_text(json.dumps(data, indent=1, sort_keys=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(HERE))
    ap.add_argument("--cache", default=str(HERE / ".synthetic-cache.json"))
    ap.add_argument("--questions-only", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    outdir, cache_path = Path(args.outdir), Path(args.cache)
    cache = load_cache(cache_path)
    cache.setdefault("questions", {})
    cache.setdefault("answers", {})

    controls = allocate_controls()
    print(f"taxonomy: {len(TAXONOMY)} domains, {len(controls)} controls, "
          f"{sum(c['nsub'] for c in controls)} questions")

    # ---- phase A: questions, one call per domain ----------------------------
    domains = [{"meta": d, "controls": [c for c in controls if c["domain"] == d[0]]}
               for d in TAXONOMY]
    todo = [d for d in domains
            if any(c["control"] not in cache["questions"] for c in d["controls"])]
    print(f"phase A: {len(todo)} domains need questions ({len(cache['questions'])} controls cached)")
    if todo:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for dom, got in zip(todo, ex.map(gen_domain_questions, todo)):
                if got:
                    cache["questions"].update(got)
                else:
                    print(f"  ! {dom['meta'][0]} failed", file=sys.stderr)
        save_cache(cache_path, cache)
    missing = [c["control"] for c in controls if c["control"] not in cache["questions"]]
    if missing:
        sys.exit(f"failed to generate questions for {len(missing)} controls: {missing[:5]}")

    questions = {}  # question_id -> {text, control, domain}
    for c in controls:
        for i, q in enumerate(cache["questions"][c["control"]], start=1):
            questions[f"{c['control']}.{i}"] = {"text": q, "control": c["control"],
                                                "domain": c["domain"]}
    print(f"  {len(questions)} questions")

    # Distinct question text is a correctness requirement, not a nicety: two
    # control ids sharing wording means a doc answering the twin is graded 0
    # while being exactly as relevant, which silently punishes precisely the
    # question-matching arms this study is about.
    seen = {}
    for qid, q in sorted(questions.items()):
        seen.setdefault(q["text"].strip().lower(), []).append(qid)
    dups = {t: ids for t, ids in seen.items() if len(ids) > 1}
    if dups:
        print(f"\n!! {len(dups)} question texts are shared by "
              f"{sum(len(v) for v in dups.values())} control ids", file=sys.stderr)
        for t, ids in list(dups.items())[:5]:
            print(f"   {ids}: {t[:90]}", file=sys.stderr)
        sys.exit("refusing to build a corpus with duplicate questions — delete the "
                 "affected entries from the cache and re-run to regenerate them")
    print(f"  all {len(questions)} question texts distinct")
    if args.questions_only:
        return

    # ---- phase B: answers ---------------------------------------------------
    todo = [q for q in sorted(questions) if q not in cache["answers"]]
    print(f"phase B: {len(todo)} questions need answers ({len(cache['answers'])} cached)")
    if todo:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            results = list(ex.map(lambda q: (q, gen_answers(questions[q]["text"], q)), todo))
        for q, res in results:
            if res and all(t["tenant"] in res for t in TENANTS):
                cache["answers"][q] = {t["tenant"]: str(res[t["tenant"]]).strip() for t in TENANTS}
        save_cache(cache_path, cache)
    missing = [q for q in questions if q not in cache["answers"]]
    if missing:
        sys.exit(f"failed to generate answers for {len(missing)} questions: {missing[:5]}")

    # ---- coverage: no tenant answers everything -----------------------------
    rng = random.Random(SEED)
    qids = sorted(questions)
    covered = {}
    for t in TENANTS:
        drop = len(qids) - t["coverage"]
        covered[t["tenant"]] = set(qids) - set(rng.sample(qids, drop))

    # ---- write docs ---------------------------------------------------------
    docs_path = outdir / "docs-rfq.jsonl"
    n = 0
    with open(docs_path, "w") as f:
        for t in TENANTS:
            for qid in qids:
                if qid not in covered[t["tenant"]]:
                    continue
                f.write(json.dumps({
                    "id": f"{t['tenant']}-{qid.lower()}",
                    "tenant": t["tenant"], "doc_type": "questionnaire_answer",
                    "question_id": qid, "control_id": qid.split(".")[0],
                    "title": questions[qid]["text"],
                    "text": cache["answers"][qid][t["tenant"]],
                }) + "\n")
                n += 1
    print(f"wrote {docs_path.name}: {n} docs")

    # ---- write queries + qrels ---------------------------------------------
    tenants = [t["tenant"] for t in TENANTS]
    nq = 0
    with open(outdir / "queries-rfq.jsonl", "w") as fq, open(outdir / "qrels-rfq.tsv", "w") as fr:
        for i, qid in enumerate(qids):
            tenant = tenants[i % len(tenants)]
            if qid not in covered[tenant]:
                continue
            nq += 1
            query_id = f"r{qid.lower()}"
            fq.write(json.dumps({
                "query_id": query_id, "tenant": tenant, "cluster": "questionnaire",
                "query": questions[qid]["text"], "question_id": qid,
            }) + "\n")
            control = qid.split(".")[0]
            for qid2 in sorted(covered[tenant]):
                doc_id = f"{tenant}-{qid2.lower()}"
                if qid2 == qid:
                    fr.write(f"{query_id}\t0\t{doc_id}\t2\n")
                elif qid2.split(".")[0] == control:
                    fr.write(f"{query_id}\t0\t{doc_id}\t1\n")
    print(f"wrote queries-rfq.jsonl: {nq} queries, and qrels-rfq.tsv")

    # ---- phase C: the hard paraphrase set -----------------------------------
    cache.setdefault("paraphrases", {})
    verbatim = [json.loads(l) for l in open(outdir / "queries-rfq.jsonl")]
    picks = random.Random(SEED + 1).sample(sorted(v["question_id"] for v in verbatim), N_PARA)
    todo = [q for q in picks if q not in cache["paraphrases"]]
    print(f"phase C: {len(todo)} paraphrases needed ({len(cache['paraphrases'])} cached)")
    if todo:
        qtext = {q: questions[q]["text"] for q in todo}
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            res = list(ex.map(lambda q: (q, gen_paraphrase(qtext[q])), todo))
        for q, p in res:
            if p:
                cache["paraphrases"][q] = p
        save_cache(cache_path, cache)
    picks = [q for q in picks if q in cache["paraphrases"]]

    by_qid = {v["question_id"]: v for v in verbatim}
    with open(outdir / "queries-rfq-para.jsonl", "w") as fq, \
         open(outdir / "qrels-rfq-para.tsv", "w") as fr:
        for qid in picks:
            src = by_qid[qid]
            tenant, query_id = src["tenant"], "p" + src["query_id"]
            fq.write(json.dumps({
                "query_id": query_id, "orig": src["query_id"], "tenant": tenant,
                "cluster": "questionnaire-para", "query": cache["paraphrases"][qid],
            }) + "\n")
            control = qid.split(".")[0]
            for qid2 in sorted(covered[tenant]):
                doc_id = f"{tenant}-{qid2.lower()}"
                if qid2 == qid:
                    fr.write(f"{query_id}\t0\t{doc_id}\t2\n")
                elif qid2.split(".")[0] == control:
                    fr.write(f"{query_id}\t0\t{doc_id}\t1\n")
    print(f"wrote queries-rfq-para.jsonl: {len(picks)} paraphrases, and qrels-rfq-para.tsv")


if __name__ == "__main__":
    main()
