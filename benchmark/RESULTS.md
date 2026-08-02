# Results: RFP benchmark vs the Vespa RAG Blueprint (run 2026-07-12)

Setup: RAG Blueprint deployed unmodified to local Vespa (Docker, 6GB) except:
OpenAI/secrets stripped from `services.xml`, `tenant`/`doc_type` fields added,
`bm25-only` and `semantic-only` rank profiles added for stage isolation, and
`id`/`tenant` added to the `top_3_chunks` document summary. Embedder is the
blueprint's own ModernBERT (nomic modernbert-embed-base, 768d → 96-byte binary).
39 docs fed, 20 queries × 5 arms. Raw runs in `results/`, reproduce with
`run_queries.py` + `evaluate.py`.

## Headline numbers (k=10, 19 scored queries)

| arm | recall@10 | MRR@10 | nDCG@10 | leaked docs |
|---|---|---|---|---|
| bm25-only | 0.908 | 0.919 | 0.885 | 61 |
| semantic-only | 0.961 | **0.974** | **0.950** | 49 |
| hybrid (blueprint learned-linear) | **0.987** | 0.917 | 0.933 | 49 |
| hybrid + GBDT second phase | 0.934 | 0.834 | 0.846 | 56 |
| hybrid + hard tenant filter | **0.987** | 0.917 | 0.933 | **0** |

Corpus is small, so recall is near ceiling everywhere; the ordering metrics and
per-query diagnostics carry the signal.

## Findings

**1. The blueprint's shipped learned models do not transfer — the GBDT phase
actively degrades quality on a new corpus.** The LightGBM second phase (trained
on the blueprint's synthetic personal-docs dataset, with `open_count`,
`favorite`, `modified_freshness` features) is the *worst* arm: nDCG 0.846 vs
0.950 for plain semantic. Concrete damage: on the offboarding paraphrase (q03)
it demotes the correct doc to rank 3 below the onboarding doc; on insurance
(q11) it ranks the stale 2023 policy ($5M) above the current 2026 one ($10M) —
despite having a freshness feature; its scores saturate (0.84–0.92) so margins
between right and wrong docs collapse. The learned-linear first phase transfers
somewhat better but still loses MRR to pure semantic (0.917 vs 0.974). Lesson
for the target pipeline: phased ranking is only as good as per-corpus training
data; collect judgments from your own click/copy logs and retrain, never ship a
model across corpora. This is why the target architecture puts evaluation tooling ahead of
model work.

**2. Tenant isolation must be a filter; it is free.** Unfiltered, every arm
returns 49–61 cross-tenant docs in the top-10 across the run — including
`acme-sla` (99.5%) and `nimbus-sla` (99.9%) for granite_cap's SLA question
(committed number: 99.95%). With the blueprint's `rag` profile stuffing up to
50 hits into LLM context, wrong-company numbers would enter generated answers.
The hard filter (`tenant contains ...` AND'd into YQL) removed all violations
with *identical* relevance metrics. There is no tradeoff; it just has to be
plumbed in (the blueprint has no tenant concept out of the box).

**3. Nothing in the default pipeline handles staleness.** On the SOC 2 query
(q01), semantic-only ranks the 2023 attestation (narrower scope, noted
exceptions) *first* and the current 2025 report *third*, behind even the SOC 1
doc; GBDT also prefers 2023. BM25/hybrid got it right only because the word
"current" happens to appear in the 2025 doc. Same pattern for insurance (q11
under GBDT). Embeddings cannot see recency; the blueprint has the machinery
(`freshness(modified_timestamp)` rank feature) but it only enters via the
untransferable GBDT model. An RFP product needs freshness (or answer-lifecycle
state like approved/superseded) as an explicit first-phase term or filter.

**4. Hybrid earns its keep on paraphrase, BM25 alone is the weakest arm.**
BM25 misses the offboarding doc entirely in the top-4 for "what happens when
staff leave" (paraphrase cluster nDCG 0.552 vs 0.942 semantic) — the modern
768d embedder resolves offboarding/termination/deprovisioning easily. Notably
the embedder also handled acronyms (IRAP, SOC 2) better than expected —
lexical-exact nDCG 0.844 — so on this corpus semantic-only beat hybrid. Don't
assume BM25 is pulling weight; measure it (the operating philosophy's "diagnose before you
optimise" philosophy, vindicated).

**5. Hard negatives were NOT the failure mode expected.** At-rest vs
in-transit, retention vs backup, breach-history vs breach-policy: every arm
put the right doc at rank 1 (hard-negative cluster: 1.000 across the board).
With good doc-level embeddings over well-scoped documents, the cross-encoder
global phase has little to add at this scale. The expensive reranker should be
justified by measurement, not assumed — at production scale (thousands of past
responses, messier chunking) this deserves a re-test.

**6. Fixed 1024-char chunking splits answers mid-sentence.** The privileged-
password paragraph landed half in chunk 1 ("minimum 20-character…") and half
in chunk 2 ("FIDO2… rotation every 90 days"). Here top-3 selection rescued it
by taking both halves (small doc, 4 chunks), but on real policy docs partial
selection would feed the generator half a policy statement — a subtle-wrong-
answer factory. Also observed: the generic intro chunk ("This document
summarises…") won a top-3 context slot on q14 — summary-ish chunks are cosine
attractors and waste context budget. Semantic/paragraph-aware chunking is the
fix; `qrels-chunks.tsv` gives the harness to measure it.

**7. Near-duplicate boilerplate confuses every arm.** The four company-overview
docs (two stale) produced the worst cluster everywhere (best MRR 0.5); bm25 and
hybrid even rank the integrations doc above all four overviews. The blueprint
has no diversity/dedup mechanism. For a corpus made of hundreds of past
questionnaire responses — near-duplicates by construction — this is arguably
the most workload-relevant gap of all.

**8. Unanswerable queries retrieve confidently.** q19 (on-prem deployment,
answered nowhere) returned 10 docs in every arm with unremarkable score
distributions. Score calibration / abstention has to live above retrieval.

## Operational notes

- Deploy was genuinely painless: zip → REST deploy worked first try; embedder
  model auto-downloaded in-container; feed of 39 docs took 11s including
  document-side embedding. The blueprint's local story is solid.
- Vespa needed the Docker VM bumped from 2GB (Mac default) to 6GB.
- Blueprint quirks found: the `top_3_chunks` document summary omits `id` (any
  run-file tooling needs it added); `min-redundancy: 2` in a single-node local
  deploy; the query-profile YQL embeds the full `select`, so adding a filter
  means rebuilding the where-clause.
- Container: `docker stop rag-blueprint` to stop, `docker start rag-blueprint`
  to resume (state persists in the container).

---

# Experiment 2: per-corpus retraining of the linear first phase (run 2026-07-23)

Response to Finding 1 (learned models don't transfer): retrain the
learned-linear first phase on this corpus's own qrels, logistic regression on
the blueprint's 6 features (+ a freshness variant via new `learned-linear-v2`
profile), 5-fold cross-validation grouped by query, scored by passing fold
coefficients as query parameters — no redeploy per model. Code:
`train_linear.py`; artifacts: `results/exp2/`.

| arm | nDCG@10 |
|---|---|
| semantic-only (zero-shot) | **0.950** |
| hybrid, blueprint weights | 0.933 |
| retrained-6f (grouped CV) | 0.902 |
| retrained-7f +freshness (grouped CV) | 0.882 |
| retrained-6f (in-sample, optimistic) | 0.953 |
| retrained-7f (in-sample) | 0.932 |

**9. Retraining needs label volume: 19 queries lose to zero-shot.** Honest
CV retraining (0.902) underperforms plain semantic search (0.950); the
in-sample fit (0.953) overstates quality by ~5 nDCG points — the exact
"fool yourself" gap the evaluation-rigour requirement is about. The
model collapses onto `max_chunk_sim_scores` (fold coefficients swing 8.5→32)
and at best rediscovers "trust the embedder".

**10. Freshness is unlearnable pointwise at this scale.** The freshness
coefficient came out negative in 4/5 folds (sign-flipped in the fifth); the
freshness cluster stayed at 0.815 in every retrained arm. Staleness needs
pairwise stale-vs-current labels, a hand-set prior, or an answer-lifecycle
filter — not a pointwise classifier over mostly-irrelevant negatives.

Operational note: coefficient-as-query-parameter made the whole CV loop run
against one deployment — the `learned-linear` profile design is genuinely good
for fast ranking iteration.

---

# Experiments 3 & 4 (run 2026-07-23)

## Experiment 4: pairwise training — `train_pairwise.py`, `results/exp4/`

Same 19 queries, same 6/7 features, same deployment as Exp 2; only the
objective changed — logistic regression on within-query feature differences
for every unequal-grade pair (~1,944 pairs/fold), grouped 5-fold CV.

| arm | nDCG@10 (CV) |
|---|---|
| **pairwise-6f** | **0.956** — best arm of the study |
| semantic-only | 0.950 |
| hybrid, blueprint weights | 0.933 |
| pairwise-7f (+freshness) | 0.932 |
| pointwise retrained (Exp 2) | 0.902 |

**11. The objective was the bottleneck, not the label count.** Pointwise
0.902 → pairwise 0.956 with identical data. Graded pairs multiply ~45
positives into ~2,000 training examples that match the actual task
(ordering); fold coefficients stabilise (max_sim 22–26 vs pointwise's
8.5–32) and weight spreads across avg/max similarity + bm25(title). The
freshness *cluster* reached 1.000 without any freshness feature — though q01
still ranks the stale SOC 2 above the current one (grade-1 vs grade-2 miss).

**12. A global additive freshness term is the wrong model, full stop.** Even
pairwise, its coefficient is negative in 4/5 folds: relevant docs are not
systematically fresher across the corpus — freshness only discriminates
*between near-duplicates on the same topic*. Staleness wants
dedup-then-prefer-fresh or an interaction/lifecycle mechanism, not a term in
a linear scorer.

## Experiment 3: paragraph-aware chunking — `docp` schema, `results/exp3/`

New `docp` schema identical to `doc` except chunks are fed pre-split on
paragraph boundaries (`array<string>` + `input chunks | embed`) instead of
in-schema fixed-1024 chunking. Same 39 docs, same profiles; paragraph indices
align exactly with `qrels-chunks.tsv`.

| arm | fixed-1024 | paragraph |
|---|---|---|
| bm25 nDCG@10 | 0.885 | 0.885 |
| semantic nDCG@10 | 0.950 | 0.935 |
| hybrid (blueprint weights) nDCG@10 | 0.933 | 0.936 |

**13. Paragraph chunking is invisible at doc level and decisive at chunk
level.** Doc-level metrics barely move, but the top-3 chunk selection — the
part the LLM actually sees — flips from broken to right: q14 now selects the
grade-2 password paragraph *first and intact* (fixed-1024 split it across two
chunks and spent a slot on the generic intro); q15 selects the grade-2
pentest paragraph first, grade-1 customer-testing second, intro gone. The
intro-attractor was an artifact of multi-topic 1024-char blobs — "about
everything" text that cosine similarity loves. Topically-pure chunks make the
similarity signal discriminative. Only chunk-level evaluation detects any of
this; doc-level nDCG is blind to it.

## Study conclusion so far

Best pipeline found: **paragraph chunking + pairwise-trained linear first
phase + hard tenant filter** (with staleness-as-dedup-tie-break, abstention,
and query-set scale-up still open). Every improvement came from measurement
the blueprint doesn't ship: leakage accounting, grouped CV, chunk-level
qrels — the evaluation harness the target architecture calls for, doing the actual work.

---

# Experiments 5–7 (run 2026-07-23)

Exp 5: `rerank_fresh.py`, `results/exp5/`. Exp 6: 60 new queries
(`queries-v2.jsonl`, merged as `queries-all.jsonl`/`qrels-all.tsv`),
`results/exp6/`. Exp 7: abstention analysis over the Exp 6 semantic run.

## Final numbers (nDCG@10, 69 scored queries, grouped CV where trained)

| arm | nDCG@10 |
|---|---|
| semantic-only | 0.873 |
| hybrid, blueprint weights | 0.874 |
| pointwise retrained (CV) | 0.869 |
| pairwise retrained (CV) | 0.873 |
| hybrid + tenant filter | 0.909 |
| pairwise + tenant filter | 0.910 |
| **pairwise + filter + dedup-freshness rerank** | **0.921** |

Paired bootstrap (per-query nDCG deltas): tenant filter **+0.035, p<0.001**;
dedup-freshness rerank **+0.011, p≈0.011**; pairwise vs semantic **−0.001,
p≈0.98**; full pipeline vs blueprint-weights+filter +0.013, p≈0.39.

**14. Dedup-then-prefer-fresh fixes staleness cheaply.** Near-duplicate
clustering (content-word Jaccard ≥0.25, same tenant) + newest-first within
cluster fixed q01's stale SOC 2 with zero regressions on 20 queries
(0.956→0.970) and replicated significantly on 80 (+0.011, p≈0.011). Stale
pairs sit at 0.29–0.49 Jaccard vs 0.05–0.14 for legitimate hard negatives —
except surface-form rewrites (0.16), which need embedding-based dedup.

**15. Scaling the query set killed the pairwise headline — retraction of
finding 11's stronger claim.** At 69 queries, pairwise-CV vs zero-shot
semantic is a statistical tie (p≈0.98); the 19-query win was noise. What
survives: pairwise is the better *objective* (stabler coefficients, never
worse than pointwise, which trails at 0.869). Learned ranking has still not
beaten zero-shot at any label scale tested.

**16. The tenant filter is the single biggest relevance intervention in the
study: +0.035 nDCG, p<0.001** — on a realistic query mix, cross-tenant
near-matches (other tenants' SLAs, overviews, certifications) actively crowd
out correct same-tenant answers; 219 leaked docs in the unfiltered hybrid
run. Isolation is not just compliance, it is relevance.

**17. Score-threshold abstention is a first-pass signal only: AUC 0.884.**
Max chunk-cosine separates answerable (mean 0.297) from unanswerable (0.214),
but no threshold is clean — catching 9/11 unanswerables costs 16% false
abstention. The residual failures are topic-match-without-fact-match (BYOK →
at-rest encryption doc at 0.296), which only generation-stage answerability
verification can catch.

## Final study conclusion

Every durable improvement was structural — paragraph chunking, hard tenant
filter, dedup-freshness tie-breaking — and every seductive learned-model win
(shipped GBDT, small-sample pairwise) died under better measurement. The
evaluation harness (leakage accounting, grouped CV, paired bootstrap,
chunk-level qrels, a scaled query set) was the product. Final pipeline:
0.921 vs 0.874 unfiltered blueprint baseline.

---

# Experiment 8: the questionnaire corpus (re-measured 2026-07-27)

Corpus: 940 answers to a 261-question security questionnaire from four
fictional vendors — Tanager Geospatial, Kestrel Cloud, Orrery Software,
Pellucid Data. The 17-domain / 197-control taxonomy, the questions and the
answers are all generated by `build_synthetic_rfq_corpus.py` and committed.
Vendors = tenants; `rfq` schema; structural qrels. 232 verbatim queries + 47
paraphrase queries. Runs in `results/exp8/`, table via `score_rfq.py`.

| arm (tenant-filtered) | verbatim (232q) | paraphrase (47q) |
|---|---|---|
| bm25 | **0.957** | 0.279 |
| semantic | 0.915 | **0.505** |
| hybrid (blueprint weights) | 0.963 | 0.436 |
| RRF global-phase fusion (untrained) | 0.950 | 0.489 |

**18. The BM25/semantic verdict inverts completely between the two query
styles.** On verbatim re-runs BM25 is near-perfect (0.957): the query *is* the
stored question, so lexical overlap is total and the task is essentially a
lookup. On paraphrases BM25 collapses to 0.279 while semantic holds 0.505 — a
procurement analyst asking in plain business language shares almost no
vocabulary with the stored questionnaire wording, and lexical matching has
nothing to grip. No single retrieval mode survives both styles, which is why
hybrid retrieval is forced by the workload rather than chosen for elegance. The
small adversarial corpus was too easy to show this at all.

**19. Multi-tenant crowding dwarfs every ranking effect.** Unfiltered, ~335–360
cross-tenant docs pollute the top-10 per arm, and the tenant filter is worth
**+0.149 to +0.294 nDCG** depending on arm — an order of magnitude more than
the +0.035 it bought on the small corpus. It is worth most to the library model
(+0.294), which is also the best arm: the better your retrieval, the more of
its top-10 a competitor's identical answers would otherwise steal. Four vendors answering the same 261
questions is precisely the scenario's data shape; isolation-as-filter goes from
"significant improvement" to "non-negotiable". Note this is the *relevance*
argument; the compliance argument was never in question.

**20. Reciprocal rank fusion — one line of Vespa global-phase config, no
training — beats the blueprint's trained learned-linear on paraphrases**
(0.489 vs 0.436). Implementation note: RRF inputs must be exported as
match-features; chunk-tensor expressions cannot be evaluated container-side.
The structural-beats-learned pattern holds again: an untrained rank-fusion rule
outperforms coefficients fitted on someone else's corpus.

Caveats: paraphrases are generated, not collected from real buyers; qrels are
structural (same-question grade 2, sibling sub-question grade 1), not
human-judged.

---

# Experiments 9–12: attacking the real hard case (runs 2026-07-24/25)

All on the questionnaire corpus (`rfq` schema), paraphrase set = 47 reworded
questions, tenant-filtered, nDCG@10. Prior best was RRF 0.489.

| technique | paraphrase | verbatim (232q) | query latency |
|---|---|---|---|
| blueprint hybrid | 0.436 | 0.963 | fast |
| RRF fusion (Exp 8) | 0.489 | 0.950 | fast |
| semantic (embed the answer) | 0.505 | 0.915 | fast |
| ColBERT late interaction (Exp 11) | 0.517 | 0.958 | ~60 ms/q |
| cross-encoder over hybrid (Exp 9) | 0.561 | 0.960 | ~2.5 s/q |
| cross-encoder over semantic (Exp 9) | 0.576 | 0.961 | ~2.5 s/q |
| library recall + cross-encoder (Exp 10) | 0.578 | 0.961 | ~2.5 s/q |
| **q2q-semantic — the library model (Exp 10)** | **0.635** | 0.961 | **fast** |

**21. The cross-encoder is a real lever, but a mid-table one** (0.505 → 0.576).
It is the target architecture's "expensive cross-encoder" stage and it does lift the hard
case — but at ~2.5 s/query on CPU, and it is beaten outright by a change that
costs nothing (finding 22).

**22. The library model is the biggest structural insight: match questions,
not answers.** Embedding the incoming query against the stored *question*
instead of the *answer* lifts single-vector semantic 0.505 → **0.635** — same
embedder, same corpus, only the indexed field changed. It is the best arm
tested, and the cheapest. This is why real RFP tools curate a Q&A library
rather than doing document RAG. *(Gap 7 later found another +0.062 sitting
in this arm for free — embed both sides with the query prefix and keep float
title vectors — finding 41.)*

**23. Fusing lexical signal back into the library model hurts it** (q2q-RRF
0.496 vs q2q-semantic 0.635). Paraphrases share little vocabulary with the
stored wording, so BM25 contributes mostly noise to the fusion; adding a weak
signal to a strong one is not free.

**24. Stacking the cross-encoder on top of the library model makes it worse**
(0.578 vs 0.635), at ~300× the query cost. This is the study's sharpest
negative result. *(Gap 6 later narrowed it: the harm is a property of this
2020-era model — a modern reranker is at-or-above the library model, but
still never significantly above it. "Harmful" retracted, "doesn't earn its
cost" confirmed — finding 40.)* The cross-encoder rescores query-against-*answer*, which
reintroduces exactly the near-duplicate confusion that matching on questions
removes — four vendors' answers to one question are near-identical prose, and a
reranker reading that prose cannot tell them apart as well as a question-to-
question cosine can. Retrieve against the right field and the expensive stage
has nothing left to contribute.

**25. Every cross-encoder variant lands in much the same place** (ce-semantic
0.576, ce-hybrid 0.561, ce-q2q 0.578) despite very different first phases —
whatever the cheap stage hands it, the reranker converges on its own ordering,
so first-phase quality is largely wasted on it.

**25b. On verbatim queries the entire ranking stack is redundant.** BM25 alone
scores 0.957; hybrid 0.963, the library model 0.961, and all three cross-encoder
variants 0.960–0.961. When the query *is* the stored question, a term-overlap
score is already at ceiling and every model above it — learned linear, ColBERT,
a 2.5 s/query transformer — buys nothing measurable. Half of a real workload's
traffic may look like this, which is a strong argument for routing by query
style rather than sending everything down the expensive path. Whatever the first phase hands it, the reranker
converges on the same ordering — its own opinion dominates, so first-phase
quality is wasted on it. That is a strong hint the reranker, not recall, is the
binding constraint here.

**26. ColBERT late interaction lands between fusion and the cross-encoder**
(0.517 vs 0.576) at ~40× lower query cost. Document token embeddings are
precomputed at index time; query-time is a cheap MaxSim. It does not beat the
cross-encoder on this corpus — unlike on the earlier one, where it did — but it
gets within 0.02 for 60 ms instead of 2.5 s. Cost moves to indexing (a third
model resident; 181 s to feed 940 docs).

**27. First-phase recall is a real constraint on this query set.** Recall@10
runs 0.43–0.68 across arms on paraphrases — on a third or more of queries the
gold answer never reaches the top-10 at all, so no reranker can save it. That is an ordering problem,
not a retrieval one — and it is why the library model (which changes what
"similar" means) beats every reranker (which only reorders what it is given).

---

# Experiment 13: local cross-encoder distillation (re-measured 2026-07-27)

Teacher = deployed ms-marco cross-encoder. `collect-distill` profile exports
teacher score + 6 cheap features per hit; **47,000** teacher-labelled rows from
940 canonical questions × top-50 candidates (~39 min). Students evaluated on
the 47 held-out paraphrases (top-50 pool). Scripts: `distill_collect.py`,
`distill_train.py` (pointwise GBDT), `distill_pairwise.py` (pairwise linear).

| ranker | nDCG@10 | query cost |
|---|---|---|
| **pointwise-GBDT student** | **0.604** | ~µs |
| teacher (cross-encoder) | 0.561 | ~2.5 s |
| pairwise-linear student | 0.553 | ~µs |
| colbert_max_sim alone | 0.540 | ~60 ms |

**28. Distillation beats its own teacher — and this time the *pointwise*
student won, reversing a result confirmed three times.** Exps 2, 4 and the
earlier distillation runs all found pointwise losing and pairwise winning; here
pointwise-GBDT reaches 0.604 against pairwise-linear's 0.553, and beats the
0.561 teacher it was trained to imitate.

Read that reversal carefully, because it is partly an artifact of how the two
students are built. **The comparison confounds objective with model class**:
`distill_train.py` fits a non-linear gradient-boosted tree, `distill_pairwise.py`
fits a *linear* model on pairwise differences. On the earlier corpora the
pairwise objective was worth more than the non-linearity; on this one the
ordering flips. So the honest statement is not "pointwise beats pairwise" — it
is that the objective advantage is smaller than previously claimed, and can be
outweighed by model capacity. The original three-times-confirmed finding was
over-stated: it compared a linear pairwise model against a linear pointwise one
in Exps 2 and 4, but against a *tree* here, and only the like-for-like halves of
that comparison support the general claim.

A student beating its teacher is not paradoxical: the teacher supplies an
ordering signal over 47,000 pairs, and a model with the right inductive bias can
generalise that signal better than the teacher generalises its own scores —
especially when the teacher is itself mediocre on this task (0.561, below the
plain library model's 0.635).

**29. The query-time cross-encoder is retirable outright here.** Both students
and ColBERT land at or above the teacher's quality at a fraction of the cost,
and the best of them exceeds it by 0.043. Its role is as an offline *teacher*
that never runs at query time. (Local libomp/arch note: student uses sklearn
GBDT/logistic; a production Vespa deploy would export LightGBM for the native
`lightgbm()` second-phase — which is also what would be needed to serve the
better pointwise-GBDT student, since a tree cannot be written as the linear
`second-phase` expression the pairwise student compiles to.)

**30. Native serving: the deployed student reranks in ~8 ms vs the
cross-encoder's ~2.5 s.** Deployed as a Vespa `second-phase`
(`student-pairwise` profile), the pairwise-linear student reproduces its offline
0.553 on paraphrases and scores 0.964 on verbatim queries — the best verbatim
number in the study, marginally ahead of BM25's 0.957. The learned weights
concentrate on `title_sim` (6.71) with the chunk-similarity features next
(3.70) and `colbert_max_sim` last (0.25): the student independently rediscovers
the library model of finding 22, and a colbert-free variant would shed the
query-side ColBERT embedding cost at little quality loss.
`distill_pairwise.py` now writes `student_weights.json` / `student_expr.json`,
and the deployed profile is regenerated from them rather than hand-copied.

---

# Experiment 14: HyDE — query-side transformation, both directions (run 2026-08-01)

The one untested intervention class: transform the *query* instead of the
ranking or the index. `hyde_generate.py` makes one LLM call per paraphrase
query (gemini-2.5-flash — a different family from the gpt-4o-mini that wrote
the corpus, to blunt the same-generator confound) returning two hypothetical
texts: the vendor *answer* a security team would file (classic HyDE — into
answer space) and the *standard questionnaire wording* being paraphrased (the
symmetric variant — into question space, where finding 22 says the signal is).
Each is run alone and concatenated with the original query (query2doc-style);
answer variants score through the `semantic` arm, question variants through
`q2q-semantic`. Same query_ids, same qrels. Generation: mean 1.99 s/query, so
HyDE sits in the cross-encoder's latency class, not the fast class. Runs in
`results/exp14/`, generations cached in `hyde_generations.jsonl`.

| arm (tenant-filtered, 47 paraphrases) | nDCG@10 | vs baseline | p (paired bootstrap) |
|---|---|---|---|
| semantic (Exp 8 baseline) | 0.505 | — | — |
| HyDE hypothetical answer | 0.538 | +0.033 | 0.63 |
| HyDE answer + original query | 0.604 | +0.099 | 0.084 |
| q2q-semantic — library model (Exp 10 baseline) | 0.635 | — | — |
| HyDE hypothetical question | 0.640 | +0.005 | 0.93 |
| **HyDE question + original query** | **0.719** | **+0.084** | **0.064** |

**34. The target space matters more than the technique, again.** Answer-space
HyDE spends ~2 s of LLM generation per query and tops out at 0.604 with the
concat — still below the zero-cost, zero-LLM library model (0.635, p≈0.65).
The mechanism of finding 24 reappears on the query side: a hypothetical answer
is precisely the generic near-duplicate prose this corpus is saturated with,
so recall jumps (0.617 → 0.741) while MRR barely moves (0.475 → 0.493) — more
same-topic material retrieved, no better separation between four vendors'
near-identical answers. Finding 22 survives its strongest challenger: an LLM
call into the wrong space is worth less than a field change into the right one.

**35. Normalise-and-keep is the best retrieval number on the paraphrase set
(0.719) — suggestive, not confirmed (n=47, p≈0.064).** The reconstructed
question alone merely ties the library model (0.640): the LLM reliably recovers
the formal register but sometimes reconstructs the wrong control, and then the
query is thrown. Keeping the original paraphrase alongside it hedges both
failure modes — the reconstruction supplies vocabulary, the paraphrase keeps
intent. Recall@10 rises 0.684 → 0.777, attacking finding 27's binding
constraint (gold never reaching the top-10), which no reranker can touch — and
against the best reranker at the same latency (ce-q2q, 0.578) the margin is
+0.141 at p≈0.007, which *is* significant. Per Exp 6's lesson, the headline
number needs a scaled paraphrase set before it graduates from suggestive to
confirmed; the safe claims today are the significant ones: HyDE-into-question-
space beats every cross-encoder variant at equal cost, and answer-space HyDE
never catches the library model.

Caveats: all three texts (stored question, paraphrase, reconstruction) are LLM
prose — the affinity confound of limitation 1 applies with a family change as
the only mitigation; single sample per query at temperature 0.7 rather than
the HyDE paper's multi-sample embedding average; verbatim set not run (BM25 is
at 0.957 and a rewrite can only dilute exact overlap).

> **Verdict from Exp 15:** finding 35's headline did not survive the scaled
> query set — the +0.084 edge over the library model attenuated to **+0.023
> out-of-sample (p≈0.24)**. Findings 34 and the cross-encoder comparison
> survived, both significant at scale. See below.

---

# Experiment 15: scale the paraphrase set 47 → 232 (run 2026-08-01)

Finding 35 sat at p≈0.064 on 47 queries — the exact shape of result Exp 6
killed at scale — and the 47 queries had also formed the hypothesis.
`scale_para.py` gives every verbatim query a paraphrase twin: the original 47
copied through byte-identical, plus 185 new ones from
`build_synthetic_rfq_corpus.gen_paraphrase` unchanged (the paraphrase prompt is
the difficulty dial; a new prompt would silently move it). The new queries are
the out-of-sample test. All fast arms plus ce-q2q and the student re-run on
the 232 (`results/exp15/`); ColBERT and the other cross-encoder variants were
not. Trained-component evals (Gap 2) must keep to the original 47 — the scaled
set overlaps their training question_ids.

| arm (tenant-filtered, nDCG@10) | orig-47 | new-185 (oos) | full-232 |
|---|---|---|---|
| BM25 | 0.312 | 0.354 | 0.346 |
| semantic | 0.505 | 0.604 | 0.584 |
| **q2q-semantic — library model** | 0.635 | 0.766 | **0.740** |
| distilled student | 0.556 | 0.684 | 0.658 |
| ce-q2q (cross-encoder) | 0.578 | 0.710 | 0.683 |
| HyDE answer + query | 0.604 | 0.637 | 0.631 |
| HyDE question | 0.640 | 0.693 | 0.682 |
| HyDE question + query | 0.719 | 0.789 | 0.775 |

**36. Scale strikes again: the normalise-and-keep edge over the library model
is not confirmed.** Out-of-sample it is +0.023 at p≈0.24; the full-232 p≈0.046
is dragged under 0.05 by the 47 queries the claim was built on — in-sample
flattery, the study's oldest lesson (Exp 6, Exp 13). Demoted to an unconfirmed
~+0.02–0.03 tendency at ~2 s/query. What survived, all significant
out-of-sample: HyDE-into-question-space **beats the cross-encoder at the same
latency** (+0.079, p≈0.001); **pure rewriting is actively harmful** (−0.073
vs q2q, p≈0.003 — keeping the original query alongside is worth +0.096,
p<0.0001); **answer-space HyDE never competes** (−0.129 vs q2q, p<0.0001 —
finding 34 at scale). The practical recommendation is unchanged and
strengthened: the library model, alone, on the raw query. Spend the ~2 s LLM
call only if you were about to spend it on a cross-encoder — the rewrite is
strictly better than the rerank here.

**37. The library model's dominance replicates at scale — and the instrument
drifts.** q2q (0.740) significantly beats the cross-encoder stacked on it
(0.683, p≈0.001) and the distilled student (0.658, p<0.0001) on 232 queries:
findings 22/24 graduate to confirmed. Recorded alongside: every arm scores
higher on the new 185 than the original 47 (q2q 0.766 vs 0.635 — ~3σ against
sampling noise) despite an identical generation prompt five days apart, so the
difficulty dial of limitation 1 **drifts between generation runs**;
cross-set absolute numbers are unsafe, within-query paired comparisons
unaffected. And re-running the committed lexical arms shifts BM25/hybrid/RRF
±0.01–0.03 on identical queries and qrels (the live index's BM25 scores differ
from the state that produced the July run files; semantic arms reproduce
exactly) — all Exp 15 comparisons are within one index state.

---

# Gaps 4–6: the follow-ups the first fifteen experiments left open (run 2026-08-01)

**Gap 4 — mixed workload & fusion (analysis of committed runs only).** On a
464-query blend (232 verbatim + 232 paraphrase twins), the library model
alone scores **0.850**; an oracle per-style router reaches 0.852 — **+0.001
headroom**, because q2q is already within 0.003 of the verbatim ceiling. And
RRF-fusing the best arms (q2q, HyDE rewrite, student; top-50 pools in
`results/gap4/`) never beats the best parent (all p>0.3).

**38. No router, no fusion: serve the library model alone.** Finding 25b's
routing suggestion dissolves on measurement, and finding 23's weak-signal
lesson extends to fusing two strong arms — their rankings mostly agree, so
fusion can only blur the better one.

**Gap 5 — the Gap 2 fine-tune, finally served.** `gap5_export_minilm.py`
reproduces the training (base 0.591 = `results/gap2/result.json` exactly;
tuned 0.617 vs July's 0.622, within the seed band), saves the checkpoint,
exports base + tuned MiniLM to ONNX, and serves both as `q2q-mini` /
`q2q-ft` rank profiles over angular-NN question embeddings. Original 47
only (the scaled set overlaps the training question_ids); `results/gap5/`.
Served: nomic **0.635** > fine-tuned MiniLM 0.595 > base MiniLM 0.573; the
fine-tune's +0.023 (p≈0.21) does not close the −0.062 gap to nomic.

**39. Embedder choice dominates embedder fine-tuning.** Finding 11 is true
within a model family and misleading across them: pick the strongest base
embedder first, then fine-tune that. (Untested follow-up: the same 195-pair
contrastive fine-tune applied to nomic itself.)

**Gap 6 — a modern cross-encoder revisits finding 24.**
`gap6_modern_ce.py`: BAAI/bge-reranker-v2-m3 (568M, ~1.7 s/query on MPS)
reranks the same tenant-filtered top-50 library-model recall the old
cross-encoder saw. Old ce-q2q 0.578/0.683 (47/232); bge **0.678/0.765**;
free q2q first phase 0.635/0.740. bge vs old ce: **+0.082, p<0.0001** — the
modern model is categorically better. bge vs the free library model: +0.043
p≈0.33, +0.025 p≈0.11, +0.021 p≈0.21 (oos) — never significant. bge vs the
HyDE rewrite at equal cost: p≈0.59, a tie. Per-query scores in
`results/gap6/`.

**40. "Actively harmful" is retracted; "doesn't earn its cost" survives.**
Finding 24's harm was the shipped model, not reranking: a current reranker
is at-or-above the library model everywhere — and never significantly above
it. Both expensive techniques (modern rerank, LLM rewrite) now show the same
unconfirmed ~+0.02 at 232 queries; the free library model remains the
recommendation.

---

# Gap 7: prefix symmetry + an embedder bake-off (run 2026-08-02)

The library model is a **symmetric** task — question against question — but
the blueprint's nomic embedder conditions its sides asymmetrically
(`search_query:` / `search_document:` prepends). Gap 7a serves a second
component with the query prefix on *both* sides (`title_emb_sym`,
`q2q-sym` profile) against an identically-scored asymmetric control
(`q2q-float`), so the only difference is the document-side prefix. Gap 7b
(`gap7_embed_bakeoff.py`, offline) races nomic against bge-m3 and
Qwen3-Embedding-0.6B, the latter with and without a task instruction aimed
at capturing HyDE's normalise-toward-question-space gain inside the
embedder. Runs in `results/gap7/`.

| served arm (tenant-filtered) | orig-47 | oos-185 | full-232 | verbatim |
|---|---|---|---|---|
| q2q — binary, asym (Exp 15 baseline) | 0.635 | 0.766 | 0.740 | 0.961 |
| q2q — float, asym | 0.662 | 0.788 | 0.762 | 0.964 |
| **q2q — float, symmetric prefixes** | **0.703** | **0.827** | **0.802** | **0.965** |

**41. Match a symmetric task symmetrically — the biggest free win since the
library model itself.** The pure prefix effect is **+0.039, p<0.0001,
identical out-of-sample** — the first intervention since the tenant filter
to clear significance on the held-out subset. Float titles over binarized
add +0.023 (p≈0.017/0.045; 940 title vectors ≈ 3 KB each — binarization
buys nothing worth having). Combined vs the study's headline arm: **+0.062,
p<0.0001**; verbatim 0.965 and blend 0.883 are the best numbers in the
study, the oracle-router headroom closes to zero by construction, and the
free arm now sits at-or-above the ~2 s HyDE rewrite (+0.027, p≈0.11) —
strictly dominating it.

**42. Configuration beat model shopping.** Offline on the same harness:
same-model symmetric conditioning +0.042 (p<0.0001) vs bge-m3 +0.003 (tie)
vs Qwen3-0.6B +0.030 raw / +0.034 instructed (marginal, not significant
out-of-sample; the instruction itself adds only +0.004 over raw). The
instructed-embedder route to HyDE's gain mostly wasn't there at 0.6B.
Caveats: bake-off offline (serving ≈ −0.02, Gap 5), one seed, larger
Qwen variants and API embedders untested, prefix result shown on one model
family. Ops note: loading a third embedder pushed the 6 GB container past
Vespa's default feed-block limits (HTTP 507 mid-feed at 77% disk vs the
0.75 default); explicit `resource-limits` in services.xml fixed it.

---

# Evaluation methodology & limitations

## How things were measured

- **Metrics.** nDCG@10 as the primary ranking metric, with Recall@10 and MRR@10
  reported per cluster. Graded relevance (0/1/2) throughout; grade-0 rows are
  deliberate hard negatives, tracked not omitted.
- **Harness.** `evaluate.py` (stdlib only) reads TREC-format run files and
  graded qrels, and additionally reports **cross-tenant leakage** (a
  domain-specific correctness check) and **unanswerable-query behaviour**.
- **Statistical rigour.** Learned rankers were evaluated under **5-fold
  cross-validation grouped by query** (never score a query with weights trained
  on it). Key comparisons used a **paired bootstrap** over per-query nDCG deltas
  — this is what retracted the small-sample "pairwise beats zero-shot" claim
  (Exp 6, p≈0.98) and confirmed the tenant filter (+0.035, p<0.001) and
  freshness rerank (+0.011, p≈0.011).
- **Chunk-level judgments.** `qrels-chunks.tsv` grades individual paragraphs,
  which is the only instrument that detected the paragraph-chunking win (Exp 3);
  document-level metrics were blind to it.

## Limitations (read the numbers with these in mind)

1. **Both corpora are authored, and the questionnaire corpus is LLM-generated.
   This is the single biggest threat to validity and it got worse, not better.**
   Exps 1–7 use 39 LLM-written documents. Exps 8–15 use 940 answers generated
   by `gpt-4o-mini` against a taxonomy we wrote, answered by four vendors we
   invented, with paraphrases generated by the same model, judged by
   **structural qrels** (same-question = grade 2, sibling sub-question = grade
   1). No independent human judged relevance, and no sentence in the corpus was
   written by a real compliance team.

   Three specific hazards follow. **Register uniformity:** one model wrote all
   940 answers, so they are more homogeneous in length, vocabulary and structure
   than real filed answers — the measured spread is 264–905 characters where the
   real corpus ran 32–2749. That flattens exactly the lexical variation BM25
   feeds on. Question wording is the other half: BM25 scores 0.957 on verbatim
   queries here but only 0.279 on paraphrases, a far wider swing than the real
   corpus produced, because generated questions and generated paraphrases sit
   further apart in vocabulary than a real questionnaire and a real buyer do. **Generator-embedder affinity:** the
   text was produced by a language model and is being retrieved by language
   models; there is no way to rule out that generated prose is easier to embed
   coherently than human prose, which would inflate every semantic arm.
   **Paraphrase symmetry:** the paraphrases were generated from the questions by
   a model instructed to minimise vocabulary overlap, which is a cleaner and
   more consistent transformation than a real buyer rewording a question.

   Prior versions of these experiments ran on real published questionnaires.
   That corpus could not be redistributed and is not part of this repository;
   what it measured is preserved in the superseded entries of
   [../LABBOOK.md](../LABBOOK.md). Where the two disagree — BM25's verbatim
   dominance, ColBERT vs cross-encoder, whether the full stack beats the library
   model — **trust neither number, and treat the disagreement itself as the
   finding**: it marks precisely where the conclusion was corpus-dependent
   rather than architectural.
2. **Thin coverage in places.** Several synthetic clusters have 1–3 queries; the
   real paraphrase set is 47; most queries have a single grade-2 target. Small
   n widens the error bars the bootstrap already reports.
3. **Reranking comparisons are recall-bounded.** The distillation eval (Exp 13)
   reranks a fixed top-50 hybrid candidate pool, so it measures reranking
   quality given that recall ceiling, not end-to-end recall.
4. **Single-node, local.** All latency/quality figures are from one Docker node
   on a laptop. No multi-node cluster sizing, throughput-under-concurrency, or
   geo-distributed reliability was tested (the production-systems half of the
   target architecture). Latency figures are indicative, not SLA-grade.
5. **Tooling substitutions.** LightGBM was replaced by scikit-learn GBDT/logistic
   due to a local libomp/arch mismatch (equivalent algorithms; a production
   Vespa deploy would export LightGBM for the native `lightgbm()` operator).

6. **Narrow domain — the other major external-validity threat.** The
   questionnaire corpus is entirely security-compliance content, which is close
   to a best case for retrieval and *not* representative of a full RFP. It has a
   fixed 261-question taxonomy that every tenant answers, a homogeneous formal
   register, and short atomic answers. A real RFP also contains bespoke,
   non-standardized areas — pricing (tabular/conditional), product capabilities,
   implementation approach, legal terms, references — with wider vocabulary,
   longer and more structured answers, and less verbatim question recurrence.
   Consequences:
   - **Likely portable (structural/architectural):** tenant isolation as a hard
     filter; the library model *as an architecture*; the cost-quality ordering
     of phased ranking; pointwise-loses/pairwise-wins; contrastive fine-tuning
     helping (possibly *more* where zero-shot embedders are weaker); the
     binarisation direction.
   - **Corpus-inflated (magnitudes that will attenuate or reverse):** every
     absolute number, and specifically the size of the library model's
     0.505→0.635 jump, the tenant-crowding severity, and the finding that the
     cross-encoder *hurts* on top of the library model — that last one depends
     on how near-duplicate the answer text is, which this corpus maximises.
   - **Untested regimes:** tabular/numeric/conditional content (pricing, SLAs)
     and synthesis/essay questions ("describe your approach to X"), where the
     generation stage does real composition and "faithful abstention on a
     retrieval miss" (finding 31) is no longer a complete answer.
   - **Data-model mismatch:** our tenants are 4 *vendors answering an identical
     question set*, which maximises cross-tenant near-duplication. The
     scenario's real shape is *many customers, each with a bespoke library
     spanning all RFP areas* — likely overstating cross-tenant crowding and
     understating the within-library near-duplicate problem (one company's
     slightly-different answers to similar questions over time), which we barely
     touched.
   The honest resolution is the study's own recursive lesson — the small corpus
   was near-ceiling, so a harder one was necessary; this one is authored, so a
   *real* one is necessary next. Validate on a genuine RFP corpus spanning
   multiple question categories before trusting any magnitude here. Expected
   outcome: structural wins replicate; the library edge shrinks on bespoke
   areas; semantic looks *better* on high-diversity content; generation becomes
   the harder problem on essay/synthesis questions.

## The highest-value hardening step

Every real number rests on author-created judgments (limitation 1). The most
valuable next validation is **not** a better metric or a bigger model — it is
**independent, domain-expert relevance judgments** — subject-matter experts
rating results to build graded qrels. The
recommended before/after is **structural qrels vs expert-judged qrels** on the
top experiments — i.e. does the ranking of our arms survive when a human, not
the author, defines relevance? Our code-based harness stays the measurement
instrument (bootstrap, grouped CV, leakage accounting, chunk-level qrels); the
expert judgments harden the *ground truth* the instrument runs against.

---

# Gap 1: generation stage (RAG answer) — re-measured 2026-07-27

Retrieve top-3 past Q&A (student-pairwise) → LLM drafts an answer grounded only
in context, declining if uncovered → judge faithfulness + correctness. 20
answerable paraphrases + 8 crafted unanswerable questions. OpenRouter
gpt-4o-mini (generator = judge; self-judge caveat). `gen_eval.py`,
`results/gap1/`.

| metric | top-3 |
|---|---|
| faithfulness | 1.00 |
| hallucination (among answered) | 0.00 |
| correctness | 0.50 |
| abstention on unanswerable | 8/8 |

**31. The generator is exemplary; retrieval is the ceiling.** All 8 genuinely
unanswerable questions were declined — no invented prices, CEOs, patent counts
or carbon targets — and faithfulness is 1.00 with zero hallucination among
answered questions. Correctness 0.50 is set entirely by retrieval: the gold
answer reaches the top-3 context only about half the time on these hard
paraphrases, and the honest generator converts those misses into "NOT COVERED"
rather than confident wrong answers.

This is the safe failure mode for a compliance product, and it is the clearest
vindication of spending the effort upstream — "get retrieval wrong and every
answer downstream is wrong too." Note the retrieval here runs through the
`student-pairwise` profile at 0.553 on paraphrases, not the best available arm;
routing generation through the plain library model (0.635) would likely lift
correctness further, which is the cheapest available improvement to the
end-to-end system.

---

# Gap 3: binarisation / quantisation tradeoff — re-measured 2026-07-27

Parallel float32 embedding fields vs the int8-binary default, identical
candidate pool (every doc in the tenant, ranked by cosine), paraphrase set.
Only the stored representation differs. Runs in `results/gap3/`.

| doc representation | bytes/vec | nDCG@10 |
|---|---|---|
| int8 binary (pack_bits, hamming) | 96 | 0.505 |
| float32 | 3072 | 0.513 |

**32. Binarisation costs ~0.008 nDCG (~1.6% rel.) for 32× storage.** Full float
retrieves better on the hard case (0.513 vs 0.505), but only just — the binary
default leaves very little on the table here, and 32× smaller vectors is why it
is right at scale. The direction replicates across all three corpora measured;
the *size* of the cost does not (~12% rel. on the real corpus, ~6% on an earlier
build of this one, ~1.6% now), so treat "how much binarisation costs" as
corpus-dependent and worth measuring per deployment rather than assuming.
Resolution either way: binary ANN recall + float rescoring of the top-k recovers
most of the gap (the blueprint's binary-doc/float-query asymmetry already leans
this way).

---

# Gap 2: contrastive fine-tuning of the embedder — run 2026-07-26

Offline (PyTorch MPS, Vespa not involved). **195** LLM-generated (reworded,
canonical) question pairs, eval question_ids held out; fine-tune
all-MiniLM-L6-v2 with MultipleNegativesRankingLoss for 3 epochs; eval
question→question on the held-out 47 paraphrases. `gap2_gen_pairs.py`,
`gap2_finetune.py`, `results/gap2/`.

| model | nDCG@10 |
|---|---|
| base all-MiniLM-L6-v2 | 0.591 |
| + contrastive fine-tune | **0.622** |

**33. Contrastive fine-tuning gives the clearest lift of any training in the
study — the one trained component that beats its zero-shot baseline.** +0.031
from 195 generated pairs, closely replicating the +0.029 and +0.032 measured on
two earlier corpora — three independent replications of the same effect size. Unlike the learned *rankers* (which tied or lost, Exps 2/6/13), fine-
tuning the *representation* on paraphrase→canonical attacks the actual failure
mode — vocabulary mismatch — and works. It is the lever most likely to raise the
retrieval ceiling Gap 1 exposed, and unlike the structural wins it responds to
more data.

Caveat weaker than the earlier run: this is a single fine-tune, where the
earlier corpus was checked across 3 seeds and positive in all of them. The
effect size matches, but seed-robustness was not re-verified.
