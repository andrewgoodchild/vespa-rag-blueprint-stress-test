# Lab notebook — vespa-rag-blueprint-stress-test

Stress-testing the retrieval architecture specified in
`spec/target-architecture.md` (a fictional multi-tenant RFP/questionnaire
scenario). One entry per session; newest last. Detailed numbers live in
`benchmark/RESULTS.md` — this file records what was done, why, and what came
next.

---

## 2026-07-11 — Hypothesis and benchmark construction

**Question.** What stack does the target architecture imply, and where would it
break?

**Hypothesis.** The target architecture's requirements (tensor-based chunk selection, phased
ranking cheap→cross-encoder, hybrid BM25+semantic, linear regression and
LightGBM in Python, embedding binarisation/quantisation) map ~1:1 onto the
**Vespa RAG Blueprint** (`vespa-engine/sample-apps/rag-blueprint`). Confirmed
by reading the blueprint tutorial and repo — the mapping table is in
`benchmark/README.md`.

**Built.** `benchmark/`: 39 synthetic docs across three fictional tenants
spanning the scenario's customer archetypes (nimbus_sec SaaS vendor, acme_sat
satellite operator, granite_cap fund), 20 queries in 11 adversarial clusters
(lexical-exact, paraphrase, hard-negative, tenant-isolation, freshness,
negation-trap, chunk-selection, multi-doc, near-duplicates, domain-vocab,
unanswerable), graded TREC qrels incl. paragraph-level judgments for the
chunk-selection queries, and a stdlib scorer (`evaluate.py`) that also reports
cross-tenant leakage and unanswerable-query behaviour.

---

## 2026-07-12 — Experiment 1: run the blueprint against the benchmark

**Setup.** Blueprint deployed to local Vespa in Docker (Docker VM raised
2GB→6GB). App adaptations, all in `vespa-app/`: OpenAI/secrets stripped
(cloud-only), `tenant`+`doc_type` fields added, `bm25-only` and
`semantic-only` profiles added to isolate stages, `id`/`tenant` added to the
`top_3_chunks` summary. Embedder = blueprint's ModernBERT (768d → 96-byte
binary, in-engine). Fed 39 docs (11s), ran 20 queries × 5 arms via
`run_queries.py`.

**Results** (nDCG@10): bm25 0.885 · semantic 0.950 · hybrid learned-linear
0.933 · +GBDT second phase 0.846 · hybrid+tenant-filter 0.933 with **zero**
leakage (unfiltered arms leaked 49–61 cross-tenant docs).

**Findings** (full detail in RESULTS.md):
1. The blueprint's shipped LightGBM model **actively degrades** ranking on a
   foreign corpus — learned rankers don't transfer.
2. Tenant isolation as a YQL filter is free; ranking never provides it.
3. Staleness is unhandled: stale 2023 docs outrank current ones except by
   lexical luck. Freshness only enters via the untransferable GBDT.
4. Hard negatives did NOT fail — cross-encoder value unproven at this scale.
5. Fixed 1024-char chunking splits answers mid-sentence; generic intro chunks
   are cosine attractors that win top-3 context slots.
6. Near-duplicate boilerplate is the worst cluster in every arm; no
   dedup/diversity mechanism exists.
7. Unanswerable queries retrieve 10 confident docs; abstention must live
   above retrieval.

**Next planned:** retrain the learned phases on this corpus's own qrels
(attacks findings 1 and 3); later, paragraph-aware chunking vs
`qrels-chunks.tsv` (finding 5).

---

## 2026-07-23 — Experiment 2: per-corpus retraining of the linear first phase

**Question.** Finding 1 says the blueprint's learned weights don't transfer.
Can we recover (or beat) semantic-only's nDCG 0.950 by retraining the
learned-linear first phase on our own corpus — without touching the deployed
model? And does adding a freshness feature fix finding 3 (staleness)?

**Design.**
- Collect the 6 blueprint features per (query, doc) via the
  `collect-training-data` rank profile; add a `collect-v2` profile that also
  emits `freshness(modified_timestamp)` (3-year linear decay, as in the
  blueprint's GBDT profile).
- Train logistic regression (blueprint's own method), positives = qrels
  grade > 0, features standardized then coefficients mapped back to raw
  feature space.
- **5-fold cross-validation grouped by query** — never score a query with
  weights trained on it. 19 scored queries is tiny; CV keeps us honest
  (the operating philosophy's "don't fool yourself with noisy results").
- Score held-out queries by passing each fold's coefficients as *query
  parameters* to the existing `learned-linear` profile (they're query inputs —
  retraining needs no redeploy; this is the operational point of the design).
- Arms: `retrained-6f` (same features as blueprint hybrid) and
  `retrained-7f` (+ freshness, via new `learned-linear-v2` profile).
- Baselines to beat: hybrid-with-blueprint-weights 0.933, semantic-only 0.950.
- GBDT retrain skipped: no lightgbm locally, and 19 queries cannot support a
  GBDT honestly anyway.

**Run log.** Added `collect-v2` + `learned-linear-v2` profiles, redeployed
(session 4; container had been up 11 days, 39 docs still indexed). Collected
780 feature rows (20 queries × 39 docs — the hybrid recall base matches
everything at this corpus size). Trained per fold, scored held-out queries via
query-param coefficients (`train_linear.py`); artifacts in
`benchmark/results/exp2/` (features.csv, coefficients.json, run files).

**Results (nDCG@10, k=10):**

| arm | nDCG@10 | note |
|---|---|---|
| semantic-only (no learning) | **0.950** | zero-shot baseline |
| hybrid, blueprint weights | 0.933 | foreign-trained transfer |
| retrained-6f, grouped 5-fold CV | 0.902 | honest estimate |
| retrained-7f (+freshness), CV | 0.882 | freshness *hurt* |
| retrained-6f, in-sample | 0.953 | optimistic upper bound |
| retrained-7f, in-sample | 0.932 | |

**Conclusions.**
1. **19 labeled queries is not enough to beat a strong zero-shot baseline
   honestly.** CV-retrained linear (0.902) loses to plain semantic (0.950).
   The in-sample fit reaches 0.953 — a ~5-point optimism gap that is exactly
   the "fool yourself with noisy results" trap. Retraining
   *is* the right response to Exp 1's transfer failure, but only once you have
   label volume — hence the emphasis on instrumenting production to
   capture relevance signals.
2. **The learner collapses onto the embedding score.** Across folds the model
   puts nearly all weight on `max_chunk_sim_scores` (coefficient swings
   8.5→32 between folds — huge variance) and its in-sample best (0.953) merely
   matches semantic-only (0.950). With this little data, learning-to-rank
   rediscovers "trust the embedder" and adds nothing.
3. **Freshness cannot be learned pointwise from this data.** The freshness
   coefficient came out *negative* in 4/5 folds (and +7.8 in the fifth —
   pure sign instability): stale docs are a handful of the 720 negatives, so
   classification loss never sees the stale-vs-current contrast. The freshness
   cluster stayed at 0.815 in every retrained arm. Fixing staleness needs
   pairwise labels (stale vs current same-topic pairs), a hand-set prior, or a
   lifecycle filter (approved/superseded) — not a naive pointwise regressor.

**Next candidates:** paragraph-aware chunking vs `qrels-chunks.tsv` (Exp 1
finding 5); pairwise training for freshness; scale the corpus/query set to
find where retraining starts to win.

---

## 2026-07-23 — Experiment 4: pairwise training (freshness follow-up)

**Question.** Exp 2 showed pointwise logistic regression can't use 19 queries
(loses to zero-shot) and can't learn freshness. Does a pairwise (RankNet-style)
objective — training on within-query preference pairs from the *graded* qrels —
fix either problem with the same data, features, and deployment?

**Design.** For docs a,b in the same query with grade(a) > grade(b), train on
feature difference f(a)−f(b) (logistic, no intercept, symmetric pairs).
~1,944 pairs per fold from the same 19 queries. Same grouped 5-fold CV, same
query-parameter scoring. `train_pairwise.py`, artifacts `results/exp4/`.

**Results (nDCG@10, grouped CV):** pairwise-6f **0.956** — best arm of the
whole study (semantic 0.950, blueprint hybrid 0.933, pointwise CV 0.902).
pairwise-7f (+freshness) 0.932 — the freshness feature *still* hurts.

**Conclusions.**
1. **The objective was the bottleneck, not the label count.** Same 19 queries:
   pointwise 0.902 → pairwise 0.956. Graded pairs multiply the training signal
   (~45 positives → ~2,000 pairs) and match the actual task (ordering).
   Coefficients are far more stable across folds (max_sim 22–26 vs 8.5–32)
   and spread weight across avg+max similarity plus bm25(title), rather than
   collapsing onto one feature.
2. **Freshness cluster fixed without the freshness feature** (0.815 → 1.000):
   better-balanced similarity weights let the current docs win. But q01 still
   ranks the stale SOC 2 (grade 1) above the current one — staleness is
   reduced, not solved.
3. **A global freshness term is the wrong model even pairwise** (coefficient
   negative in 4/5 folds): across all pairs, relevant docs are not
   systematically fresher — freshness only matters *between near-duplicates on
   the same topic*. It's a tie-breaker, which points to dedup-then-prefer-fresh
   or an interaction feature, not an additive term.

---

## 2026-07-23 — Experiment 3: paragraph-aware chunking

**Question.** Exp 1 finding 5: fixed 1024-char chunking splits answers
mid-sentence and generic intro chunks win top-3 context slots. Does
paragraph-boundary chunking fix chunk selection, and does it cost anything at
doc level?

**Design.** New `docp` schema — identical to `doc` except `chunks` is fed
pre-split on paragraph boundaries (`array<string>` + `input chunks | embed`)
instead of in-schema `chunk fixed-length 1024`. Same embedder, features, and
profiles (copied). Fed the same 39 docs (paragraph indices now align exactly
with `qrels-chunks.tsv`). Ran bm25/semantic/hybrid-blueprint-weights against
both schemas. Artifacts: `results/exp3/`.

**Results.**
- **Doc-level: a wash.** bm25 identical (0.885); semantic 0.950→0.935; hybrid
  0.933→0.936. Chunking granularity barely moves document ranking on this
  corpus.
- **Chunk-level: decisive.** q14: selected paragraphs {6,5,3} with the grade-2
  password paragraph ranked *first*, intact (fixed-1024 had split it across
  two chunks and spent a slot on the generic intro). q15: selected {5,6,2}
  with the grade-2 pentest paragraph first and grade-1 customer-testing
  paragraph second; intro no longer selected in either query.

**Conclusions.** The intro-chunk attractor was an artifact of multi-topic
1024-char blobs — a chunk containing intro+governance+risk text is "about
everything" and cosine loves it. Topically-pure paragraph chunks make the
similarity signal discriminative, deliver answers unsplit, and cost nothing at
doc level. Since the generator only sees the selected chunks, this is a pure
win for answer quality that doc-level metrics are blind to — chunk-level
evaluation (qrels-chunks) is the only instrument that detects it.

**Study status.** Best pipeline found: paragraph chunking + pairwise-trained
linear first phase + hard tenant filter. Remaining open items: staleness as
dedup-tie-breaking (q01), abstention for unanswerables, scale-up of the query
set.

---

## 2026-07-23 — Experiment 5: staleness as a dedup tie-break

**Question.** Exps 2/4 showed freshness is unlearnable as a global feature.
Does the alternative — near-duplicate clustering + prefer-fresh within
cluster — fix the remaining staleness failures (q01) without regressions?

**Design.** Post-retrieval reranker (`rerank_fresh.py`): union-find clusters
over same-tenant docs in the result list with content-word Jaccard ≥ 0.25,
each cluster's members reordered newest-first across their occupied
positions. Threshold chosen by inspecting the pair-similarity distribution:
stale pairs 0.29–0.49, hard-negative pairs 0.05–0.14 — clean separation,
except the marketing-rewrite overview pair (0.16), which lexical similarity
cannot reach (embedding-based dedup would).

**Results.** pairwise-6f 0.956 → **0.970** nDCG; q01's current SOC 2 now
rank 1; recall/MRR unchanged; no regressions. On the 80-query set (Exp 6) the
gain replicates: **+0.011 nDCG, p≈0.011** (paired bootstrap). Freshness
belongs in dedup tie-breaking, exactly as Exp 4 predicted.

---

## 2026-07-23 — Experiment 6: scaling the query set 20 → 80

**Question.** Do the small-set conclusions survive 4× more queries? Wrote 60
new queries (`queries-v2.jsonl` + `qrels-v2.tsv`): 50 answerable across all
clusters incl. reverse-direction hard negatives (q29, q31, q41), 10 new
unanswerables for Exp 7. Reran all arms + grouped-CV retraining on the merged
80-query set (69 scored); paired bootstrap for significance.

**Results (nDCG@10, 69 queries):** semantic 0.873 · hybrid-blueprint 0.874 ·
pointwise-CV 0.869 · pairwise-CV 0.873 · hybrid+filter 0.909 ·
pairwise+filter 0.910 · **pairwise+filter+fresh 0.921**.

**Conclusions.**
1. **A finding died, honourably: Exp 4's "pairwise beats zero-shot" was
   noise.** At 69 queries pairwise-CV vs semantic is Δ=−0.001, p≈0.98. The
   19-query margin (0.956 vs 0.950) did not replicate. Pairwise remains the
   better *objective* (more stable, never worse than pointwise) but learned
   ranking still hasn't beaten zero-shot at this label scale.
2. **The tenant filter is a significant relevance win, not just compliance:
   +0.035 nDCG, p<0.001** — with a realistic query mix, cross-tenant docs
   actively crowd out correct answers (219 leaked docs in the unfiltered
   run). Biggest single intervention in the study.
3. **Fresh rerank replicates** (+0.011, p≈0.011). The durable improvements
   are all structural — filter, dedup-freshness, paragraph chunking — not
   the learned model.

---

## 2026-07-23 — Experiment 7: abstention on unanswerable queries

**Question.** 11 unanswerable queries now exist (q19 + 10 new). Can the
retrieval score alone tell "no answer exists"?

**Design.** Max chunk-cosine score per query from the semantic arm (cosine is
query-comparable, unlike learned scores); AUC + threshold sweep, answerable
(69) vs unanswerable (11).

**Results.** AUC **0.884**. Means separate (0.297 vs 0.214) but the overlap
zone is fatal for a hard threshold: τ=0.26 catches 9/11 while falsely
abstaining on 11/69 answerable (16%); catching all 11 costs half the
answerable set. The worst offender is exactly the predicted trap: the BYOK
query scores 0.296 against the at-rest encryption doc — right topic, missing
fact. A retrieval-score threshold is a usable *first-pass* signal, but
topic-match-without-fact-match failures require answerability verification at
the generation stage.

**Study status.** Complete story: measurement infrastructure (leakage
accounting, grouped CV, paired bootstrap, chunk-level qrels, scaled query
set) produced every real improvement and killed two seductive false ones
(GBDT transfer, small-sample pairwise win). Final pipeline: paragraph
chunking + tenant filter + dedup-freshness rerank + (either linear scorer),
0.921 vs 0.874 baseline on the 80-query set.

---

## 2026-07-24 — Experiment 8: real data (published questionnaire answers) + first new Vespa feature (RRF)

> **Superseded 2026-07-27.** The numbers in this entry were measured on a corpus
> parsed from a published industry security questionnaire, which could not be
> redistributed and was replaced. The entry is kept as the record of what was
> done at the time, with vendor names genericised; for numbers that
> describe the committed corpus see the 2026-07-28 entry and
> `benchmark/RESULTS.md`.


**Question.** Everything so far is synthetic. Do the findings survive contact
with real data from the actual problem domain?

**Data.** Four major cloud vendors publish their completed security
questionnaires (261 questions) specifically for prospect due-diligence:
questionnaires for prospect due diligence. Parsed 940 real
question+answer docs (`build_rfq_corpus.py` → `docs-rfq.jsonl`; PDFs fetched
at run time, not committed). Vendors = tenants; new `rfq` schema. Two query
sets: 234 *verbatim* questionnaire questions (structural qrels: same-tenant
same-question grade 2, sibling sub-questions grade 1) and 47 hand-written
*paraphrases* (`queries-rfq-para.jsonl`) as the style-control. Also tried the
first new Vespa feature from the exploration backlog: **reciprocal rank
fusion** via a `global-phase` expression (`rrf-hybrid.profile`) — RRF inputs
must be shipped as match-features; evaluating chunk-tensor expressions
container-side fails.

**Results (nDCG@10, tenant-filtered):**

| arm | verbatim (234) | paraphrase (47) |
|---|---|---|
| bm25 | **0.942** | 0.327 |
| semantic | 0.480 | 0.377 |
| hybrid (blueprint learned-linear) | 0.843 | 0.390 |
| RRF fusion (untrained) | 0.671 | **0.453** |

Unfiltered: bm25 0.625, semantic 0.268, hybrid 0.546 — **1,700+ leaked
cross-tenant docs (~74% of top-10 slots)**; the filter is worth +0.32 nDCG
on bm25.

**Conclusions.**
1. **Query style flips the winner completely.** Verbatim questionnaire
   re-runs: BM25 near-perfect (0.942), semantic collapses (0.480) — 940
   same-domain compliance answers are semantically indistinguishable at the
   sub-question level. Paraphrased questions: BM25 collapses (0.327),
   semantic overtakes, and *untrained RRF fusion wins* (0.453). This is the
   real justification for hybrid retrieval that the synthetic corpus was too
   easy to show — a real RFP workload is a mix of both styles.
2. **The real problem is much harder than the synthetic one.** Best
   paraphrase score is 0.453 (vs 0.92+ on synthetic). Headroom everywhere.
3. **Tenant crowding on real data is an order of magnitude worse than
   synthetic** (+0.32 nDCG from the filter vs +0.035): four vendors answering
   identical questions is the near-duplicate worst case — and it is exactly
   the scenario's data shape.
4. **RRF — one line of global-phase config, zero training — beats the
   blueprint's trained linear model on paraphrases.** Structural-beats-
   learned, again.

**Caveats.** Paraphrases are mine (47 of them); qrels are structural, not
human-judged; one grade-2 doc per query. Real *documents*, semi-real
*judgments*.

**Vespa feature backlog** (from docs exploration, in rough order of promise):
- **Cross-encoder in global-phase** (`onnx-model` + tokenInputIds; e.g.
  bge-reranker-base, rerank-count 25–100) — the target architecture's "expensive cross-encoder"
  stage, still untested; the 0.45 paraphrase ceiling is its natural target.
- **In-engine dedup/diversity**: `collapsefield`, grouping `max(1)`,
  match-phase/second-phase `diversity { attribute, min-groups }` — would
  replace the Python fresh-rerank with engine-native mechanisms.
- **ColBERT embedder** (token-level late interaction, `tensor(token{},x)`
  MaxSim) — for the paraphrase gap where single-vector cosine fails.
- **SPLADE embedder** (learned sparse) — BM25-like precision with learned
  term expansion; interesting exactly where bm25/semantic split.
- **Significance model** (global IDF) — corpus-level term statistics when
  tenant filtering shrinks the effective corpus per query.

---

## 2026-07-24/25 — Experiments 9–11: attacking the hard case (published-questionnaire paraphrase set)

> **Superseded 2026-07-27.** The numbers in this entry were measured on a corpus
> parsed from a published industry security questionnaire, which could not be
> redistributed and was replaced. The entry is kept as the record of what was
> done at the time, with vendor names genericised; for numbers that
> describe the committed corpus see the 2026-07-28 entry and
> `benchmark/RESULTS.md`.


Motivation: Exp 8 exposed a real ceiling — reworded questions against 940
same-domain answers topped out at nDCG 0.453 (untrained RRF). These
experiments test the techniques real RFP-automation vendors actually use.
All numbers are nDCG@10 on the 47-query paraphrase set, tenant-filtered,
against the `rfq` published-questionnaire corpus. Baselines: semantic (answer embed) 0.377,
blueprint hybrid 0.390, RRF 0.453.

### Exp 9 — cross-encoder reranking (`ce-semantic`, `ce-hybrid`)

ms-marco MiniLM-L-6 cross-encoder as a Vespa `global-phase` ONNX model
(`onnx-model` + tokenInputIds/tokenTypeIds/tokenAttentionMask,
rerank-count 50). New `ce_tokens` attribute stores the WordPiece tokens of
title+text; query tokenised via a `hugging-face-tokenizer` component.

| arm | paraphrase | verbatim (234q) |
|---|---|---|
| ce-semantic | 0.701 | 0.834 |
| ce-hybrid | **0.717** | **0.945** |

**Finding 21.** The cross-encoder is the single largest lever on the hard
case: 0.453 → 0.717 (+58%). This is the "expensive cross-encoder" stage the
brief specifies, and on real data it is unambiguously worth it — the one place a
learned model decisively beat structural methods. Cost: ~2.7 s/query on CPU
at rerank-count 50 (the reason it is a final-stage reranker, not first-phase).

### Exp 10 — the library model and structural routing

Real RFP tools retrieve against a curated library of past *questions*, not
answers. Tested question-to-question matching and control-ID routing.

| arm | paraphrase |
|---|---|
| q2q-bm25 (lexical, match stored question) | 0.251 |
| route-bm25 (lexical cross-tenant vote → tenant lookup) | 0.242 |
| route-sem (semantic vote → lookup) | 0.485 |
| q2q-rrf | 0.519 |
| q2q-semantic (match stored question, single-vector) | 0.648 |
| **ce-q2q (library recall + cross-encoder)** | **0.774** |

**Finding 22 — the biggest structural insight of the study.** Matching the
incoming question against the stored *question* text instead of the *answer*
nearly doubles single-vector semantic retrieval on paraphrases (0.377 →
0.648). Same embedder, same query; only the indexed field changed. This is
why every serious RFP product is built around a curated Q&A library rather
than raw-document RAG — question↔question paraphrase distance is far smaller
than question↔answer distance.

**Finding 23.** Lexical library matching *fails* (q2q-bm25 0.251): reworded
questions share almost no exact tokens with the formal questionnaire phrasing. The
library model only pays off with semantic matching.

**Finding 24.** Explicit routing/voting indirection underperforms direct
question matching (route-sem 0.485 < q2q-semantic 0.648): collapsing to a
control-ID vote discards ranking signal the direct scorer keeps. Structure
helps as a *retrieval target* (match questions), not as a *pipeline hop*
(classify then look up) — at least at this corpus size.

**Finding 25.** The full industry stack wins: library-model recall feeding
the cross-encoder (ce-q2q) reaches 0.774, beating both plain cross-encoder
(0.717) and library-alone (0.648). Recall stage and rerank stage compound.
On verbatim queries ce-q2q ties ce-hybrid (0.946 vs 0.945) — near-perfect,
as expected when the question is already in the library.

### Exp 11 — ColBERT late interaction (`colbert-maxsim`) — the cross-encoder killer

Vespa `colbert-embedder` (colbertv2.0, 436 MB ONNX, third model in-engine;
required Docker VM 6→8 GB). Per-doc token embeddings stored as
`tensor<int8>(dt{}, x[16])` over title+text; query-time MaxSim
(`sum over query tokens of max over doc tokens`) as first-phase. Corpus is
small enough to MaxSim every tenant candidate — no ANN needed.

| arm | paraphrase | verbatim | latency/query |
|---|---|---|---|
| cross-encoder (ce-hybrid) | 0.717 | 0.945 | ~2700 ms (CPU, rerank 50) |
| **ColBERT MaxSim** | **0.752** | **0.948** | **~60 ms** |
| ce-q2q (library + CE) | 0.774 | 0.946 | ~2700 ms |

**Finding 26 — late interaction beats the expensive cross-encoder here, at
~40× lower query cost.** ColBERT scores 0.752 on paraphrases (above the
standalone cross-encoder's 0.717, just under the full library+CE stack's
0.774) and is the single best arm on verbatim (0.948). Because document token
embeddings are precomputed at index time, query-time is a cheap MaxSim tensor
op — ~60 ms/query vs the cross-encoder's ~2.7 s. On this corpus late
interaction largely obsoletes the query-time cross-encoder: near-equal quality,
40× cheaper, no per-query transformer forward pass. The cost moves to index
time (heavier feed: 325 s to embed 940 docs, and a 3rd model resident in RAM).

**Answering "how do you go after the expensive cross-encoder": measured.**
The winning move on this data is not to optimise the cross-encoder but to
replace it with late interaction on the recall side. Remaining cross-encoder
cost-reduction options (untested, noted for future work): rerank-count sweep
(we rerank 50 of ~235; top-10 likely suffices), sequence 256→128 (answers
median ~80 tokens), smaller/quantized cross-encoder (MiniLM-L-2, int8),
cascade/early-exit on first-phase margin, and distillation (cross-encoder as
teacher for a fast student — the target architecture's LightGBM-training pattern).

## Final real-data paraphrase scoreboard (nDCG@10, 47q, tenant-filtered)

| technique | nDCG@10 | note |
|---|---|---|
| semantic (embed the answer) | 0.377 | blueprint-style RAG |
| blueprint hybrid | 0.390 | |
| RRF fusion | 0.453 | prior best (Exp 8) |
| q2q-semantic (embed the question) | 0.648 | library model |
| cross-encoder | 0.717 | expensive rerank |
| ColBERT late interaction | 0.752 | ~40× cheaper than CE |
| library recall + cross-encoder | **0.774** | full industry stack |

Every gain past 0.453 came from a technique real RFP vendors use: match
questions not answers, late interaction, cross-encoder rerank. The synthetic
corpus (Exps 1–7) could not have surfaced any of them — it was already near
ceiling. Real data was necessary to find the real levers.

### Exp 12a — cross-encoder rerank-count sweep (paraphrase set)

| rerankCount | nDCG@10 | ms/query |
|---|---|---|
| 50 | 0.717 | 2436 |
| 25 | 0.664 | 1266 |
| 15 | 0.580 | 790 |
| 10 | 0.518 | 561 |
| 5 | 0.500 | 327 |
| 3 | 0.483 | 233 |

**Finding 27.** Rerank-count reduction is NOT a free speedup on this corpus:
quality degrades almost linearly with the window (0.717→0.518 at rc=10). The
semantic first-phase ranks the gold answer deep, so the cross-encoder needs a
wide window to reach it. Corollary: the cost problem is a *first-phase recall*
problem. ColBERT (Exp 11) already ranks the gold answer near the top at 60
ms/query — so ColBERT-as-first-phase + a small cross-encoder window is the
efficient frontier to test, and plain ColBERT (0.752) already beats the
full-window cross-encoder (0.717) outright. Latency scales linearly with
rerank-count as expected (~48 ms per reranked pair, CPU).

---

## 2026-07-25 — Experiment 13: local cross-encoder distillation ("Student A")

> **Superseded 2026-07-27.** The numbers in this entry were measured on a corpus
> parsed from a published industry security questionnaire, which could not be
> redistributed and was replaced. The entry is kept as the record of what was
> done at the time, with vendor names genericised; for numbers that
> describe the committed corpus see the 2026-07-28 entry and
> `benchmark/RESULTS.md`.


**Question.** Can we get cross-encoder quality without the ~2.4 s/query cost by
distilling it into a cheap student — entirely on the 16 GB M1, no Modal?

**Setup.** Teacher = the deployed ms-marco cross-encoder. A `collect-distill`
rank profile exposes, in one pass, the teacher score (global-phase relevance)
plus six cheap student features via match-features: bm25(title), bm25(chunks),
max_chunk_sim, avg_top_3_sim, title_sim (q2q), colbert_max_sim. Labelling pass
over the 940 canonical questionnaire questions × top-50 hybrid candidates produced
37,926 teacher-labelled rows (stopped at 700/940 queries — already ample). The
47 paraphrases were held out. (LightGBM hit a local libomp arch mismatch —
Intel Homebrew vs arm64 Python — so the student is sklearn's GBDT /
logistic; a production deploy would export LightGBM for Vespa's `lightgbm()`.)

**Results (nDCG@10, 47 held-out paraphrases, top-50 candidate pool):**

| student objective | nDCG@10 | query cost |
|---|---|---|
| teacher (cross-encoder) | 0.717 | ~2.4 s |
| colbert_max_sim alone (a feature) | 0.744 | ~60 ms |
| **pointwise GBDT** (regress teacher score) | 0.678 | ~µs |
| **pairwise-linear** (RankNet on teacher order) | **0.747** | ~µs |

**Finding 28 — distillation works, but only with the right objective; and it
is the third confirmation of pointwise-loses/pairwise-wins.** The pointwise
GBDT student (0.678) underperformed both the teacher and its own best single
feature — regressing to teacher *magnitudes* distorted the *order*. Switching to
a pairwise objective (RankNet on teacher-ordered feature diffs) recovered to
0.747 — matching ColBERT and **beating the teacher** — at ~microsecond query
cost vs the teacher's 2.4 s. Same features, same data, only the loss changed.
Exactly the Exp 2 vs Exp 4 result, now a third time.

**Finding 29 — you can retire the query-time cross-encoder on this corpus.**
Two independent cheap paths reach teacher-or-better quality: ColBERT late
interaction (0.744–0.752, no training) and a pairwise-distilled linear scorer
(0.747, trained once offline). The pairwise student even leans on title_sim
(q2q, weight 6.8) over colbert — the library-model signal carries it. The
expensive reranker is a *teacher*, not a *serving* component.

**Answering "how do you go after the expensive cross-encoder", concluded.**
Measured menu: (1) don't shrink its rerank window — not free (Exp 12a); (2)
replace it at serving time with ColBERT late interaction (Exp 11); (3) distill
it offline into a pairwise-linear/GBDT student that serves in microseconds
(Exp 13). All three run locally; none need the GPU/Modal budget. The cross-
encoder's role collapses to an offline teacher.

### Exp 13b — native deployment + real serving latency

Deployed the pairwise-linear student as a native Vespa `second-phase`
(`student-pairwise.profile`): cheap hybrid first-phase (bm25+sim) selects the
window, the linear student reranks top-50; no cross-encoder in the serving path.
Weights hardcoded from `student_weights.json`.

| metric | value |
|---|---|
| nDCG@10 (47 paraphrases) | 0.747 — exactly matches the offline distill eval |
| server querytime (Vespa) | p50 **7 ms**, p90 9 ms, max 12 ms |
| client wall-clock (incl. query embedding) | p50 138 ms, p90 152 ms |
| cross-encoder it replaces | ~2400 ms |

**Finding 30 — the distilled student serves at ~7 ms of ranking vs the
cross-encoder's ~2.4 s (≈340× less ranking compute), at equal-or-better
quality.** Deployment fidelity was exact (0.747 → 0.747). The ~130 ms residual
in wall-clock is query-side embedding (nomic + colbert forward passes on the
query), shared by all neural arms and cacheable. Notable: the student's weights
lean overwhelmingly on `title_sim` (6.79) with `colbert_max_sim` near-negligible
(0.24) — a colbert-free student would drop the ~100 ms colbert query-embedding
and likely hold quality, an obvious next optimisation. The expensive cross-
encoder is now fully out of the query path, replaced by a linear second-phase.

---

## 2026-07-25 — Gap 1: the generation stage (RAG answer) via OpenRouter

> **Superseded 2026-07-27.** The numbers in this entry were measured on a corpus
> parsed from a published industry security questionnaire, which could not be
> redistributed and was replaced. The entry is kept as the record of what was
> done at the time, with vendor names genericised; for numbers that
> describe the committed corpus see the 2026-07-28 entry and
> `benchmark/RESULTS.md`.


**Question.** The whole study measured retrieval; the product actually *drafts
answers*. Does the generation stage produce correct, grounded answers — and
does it decline when it should? (OpenRouter key in `.env`, gpt-4o-mini for both
generation and judging — self-judge, a noted limitation. `gen_eval.py`.)

**Design.** For each question: retrieve top-k past Q&A from the deployed
student-pairwise profile → prompt the LLM to answer grounded ONLY in that
context, instructed to reply "NOT COVERED" if the context lacks the answer →
judge faithfulness (grounded, no invention) + correctness (matches the vendor's
gold answer). 20 real answerable paraphrases + 8 crafted unanswerable questions
(pricing, ESG, CEO, patents — genuinely absent from a security
questionnaire).

**Results.**
| metric | top-3 | top-5 |
|---|---|---|
| faithfulness (answered are grounded) | **1.00** | **1.00** |
| hallucination rate | **0.00** | 0.00 |
| correctness (conveys gold fact) | 0.50 | 0.55 |
| abstained on answerable | 10/20 | 8/20 |
| abstention on truly-unanswerable | **8/8 (1.00)** | — |

**Finding 31 — the generation stage is exemplary; retrieval is the ceiling.**
Zero hallucination: every answer the model gave was grounded in the retrieved
context, and it declined all 8 truly-unanswerable questions (never invented a
price, a CEO, or a carbon target). Correctness sits at ~0.50 not because the
generator is weak but because on the hard paraphrase set the gold answer only
reaches the top-k context about half the time — and the honest generator turns
those retrieval misses into "NOT COVERED" rather than confident wrong answers.
Correctness rising with more context (0.50→0.55 at k=5) confirms the bottleneck
is retrieval recall, not generation. This is the safe failure mode for a
compliance product, and it vindicates the study's core premise: get retrieval wrong and every
answer downstream is wrong too. The 13 retrieval experiments were the right
place to spend the effort — the answer is only as good as what feeds it.

**Caveats.** Generator = judge (both gpt-4o-mini) → self-judge bias; structural
qrels, not human-rated; a hand-crafted unanswerable set of 8.

---

## 2026-07-25 — Gap 3: binarisation / quantisation quality-cost tradeoff

> **Superseded 2026-07-27.** The numbers in this entry were measured on a corpus
> parsed from a published industry security questionnaire, which could not be
> redistributed and was replaced. The entry is kept as the record of what was
> done at the time, with vendor names genericised; for numbers that
> describe the committed corpus see the 2026-07-28 entry and
> `benchmark/RESULTS.md`.


**Question.** The blueprint (and the target architecture's binarisation/quantisation requirement) stores
int8 binary embeddings with hamming distance. What does binarisation cost in
quality, and save in space? Added parallel float32 embedding fields
(`title_emb_f`, `chunk_emb_f`) + a `semantic-float` profile; compared against
the binary `semantic-only` on the same candidate pool (all tenant docs ranked
by cosine — isolates representation, no ANN confound), real paraphrase set.

**Results (nDCG@10, paraphrase set).**
| doc representation | bytes/vector | nDCG@10 |
|---|---|---|
| int8 binary (pack_bits, hamming) | 96 | 0.379 |
| float32 | 3072 | 0.431 |

**Finding 32 — binarisation costs ~0.053 nDCG (~12% relative) for a 32× storage
saving.** On the hard paraphrase case, full-precision embeddings retrieve
noticeably better (0.431 vs 0.379), so the blueprint's binary default does leave
measurable quality on the table — a lift comparable in size to some of the
study's structural wins. But 32× smaller vectors is exactly why it's the right
default at the target architecture's stated scale (thousands of past responses across dozens of
customers). The practical resolution is the blueprint's own asymmetric design
(binary doc + float query) plus a float rescoring stage over the top-k: keep
binary for cheap ANN recall, recover most of the +0.053 by rescoring finalists
in float. Quantisation is a recall-tier decision, not an all-or-nothing one.

---

## 2026-07-26 — Gap 2: contrastive fine-tuning of the embedder

> **Superseded 2026-07-27.** The numbers in this entry were measured on a corpus
> parsed from a published industry security questionnaire, which could not be
> redistributed and was replaced. The entry is kept as the record of what was
> done at the time, with vendor names genericised; for numbers that
> describe the committed corpus see the 2026-07-28 entry and
> `benchmark/RESULTS.md`.


**Question.** Gap 1 showed the RAG answer is capped by retrieval recall. Can
contrastive fine-tuning — the target architecture's "sentence transformers, contrastive
fine-tuning" requirement — lift that ceiling? Done fully offline on the M1 (Vespa
stopped to free RAM; PyTorch MPS).

**Design.** No natural paraphrase supervision exists (questionnaire questions are
standardized), so synthesized it: LLM-reworded 190 canonical questions into
buyer-style phrasings → (reworded, canonical) positive pairs, with the 71
eval question_ids **held out**. Fine-tuned `all-MiniLM-L6-v2` with
MultipleNegativesRankingLoss (in-batch negatives, 3 epochs). Evaluated
question→question retrieval (embed paraphrase vs stored question, rank within
tenant) on the held-out 47 paraphrases, base vs tuned. `gap2_gen_pairs.py`,
`gap2_finetune.py`.

**Results (nDCG@10, held-out 47 paraphrases).**
| model | nDCG@10 |
|---|---|
| base all-MiniLM-L6-v2 (off-the-shelf) | 0.656 |
| + contrastive fine-tune (190 pairs) | 0.681 – 0.690 (3 seeds, mean ~0.685) |

**Finding 33 — contrastive fine-tuning gives a robust lift, and it's the first
trained component in the study that beats its zero-shot baseline.** +0.029 mean
(+0.025 to +0.034, positive in all 3 runs) from just 190 synthetic pairs. Two
things stand out: (1) the base MiniLM (0.656) already edges the larger deployed
nomic embedder on q2q (0.648) — a small off-the-shelf model is competitive in
this narrow domain; (2) unlike the learned *rankers* (which repeatedly tied
zero-shot — Exps 2/6/13), fine-tuning the *representation* directly on the
paraphrase→canonical task works, because the objective matches the actual
failure mode (vocabulary mismatch). This is the lever most likely to raise the
retrieval ceiling that Gap 1 showed everything rests on — and it responds to
data, so more (and real) pairs should extend it.

**Caveats.** 190 LLM-generated training pairs; single base model; offline q2q
eval (not deployed to Vespa); 47 eval queries; modest magnitude.

---

## 2026-07-27 — Clean-room corpus: rebuild the data, re-measure everything

**Question.** Exps 8–13 ran on 940 answers parsed from the published security
questionnaires of four major cloud vendors. Preparing to publish surfaced the
problem: that questionnaire's question set is copyrighted by the industry body
that publishes it, and its licence — which the scraper had helpfully pulled into
the corpus verbatim — says it "may not be redistributed" and is for "personal,
informational, non-commercial use". 397 distinct questions and ~428 KB of vendor
answer text were sitting in git. Anonymising the tenant labels would not have helped:
89% of the answers name their own vendor *inside the prose*. So: can the study
stand on a corpus we own outright?

> Caught before publication: the repository was private throughout, and the
> corpus was replaced before anything was pushed or shared. No commit reachable
> from `main` has ever contained that material — the published history starts
> from the clean-room corpus.

**Design.** Build a clean-room replacement and re-run every experiment against
it, carrying over nothing.

- Our own control taxonomy — 17 domains, our codes (GOV/AUD/ACC/CRY/…, no CCM
  codes), 197 controls, 261 questions written to it.
- Four fictional vendors: Tanager Geospatial, Kestrel Cloud, Orrery Software,
  Pellucid Data. Distinct products and voices so answers self-identify.
- Matched to the old corpus on every structural property that drives the
  findings: 940 docs, 261 questions, 197 controls, the same sub-question
  profile (144 single-part / 46 two / 5 three / 2 five), the same uneven
  coverage (258/214/234/234).
- `build_synthetic_rfq_corpus.py` generates questions, answers and the 47-query
  paraphrase set via gpt-4o-mini, caching every call so a rebuild is free.
- New tooling so nothing is ad hoc again: `feed_rfq.py` (purge + feed, 940 docs
  in ~180 s), `run_rfq.py` (every arm, one command), `score_rfq.py` (the tables
  in RESULTS.md).

**Three attempts to get the difficulty right.** Worth recording, because
calibrating a synthetic corpus turned out to be the hard part of the day.

1. *Questions generated per control.* Each of the 197 calls saw only the domain
   name, so sixteen ACC controls came back as sixteen wordings of the same
   question — **261 question ids sharing 155 distinct texts**. That silently
   corrupts the qrels: 25 of 47 paraphrase queries had a word-for-word
   identical question graded 0, punishing hardest the question-matching arms
   the study is about. Every number from this run was wrong.
2. *Questions generated per domain, instructed to be maximally distinct.* Fixed
   the duplication (261/261 distinct) but overshot: the questions became so
   semantically separated that matching a paraphrase to its stored question was
   trivial. q2q-semantic hit nDCG **0.952** at MRR **0.986** — the gold answer
   essentially always at rank 1, every arm bunched 0.71–0.95, no headroom to
   tell techniques apart. This is exactly the near-ceiling failure that made
   the original study abandon the small corpus, reproduced.
3. *Questions confusable but distinct, and paraphrases written as a procurement
   analyst would type them* — plain business language, no security jargon,
   looser than the original. That produced the spread below. The paraphrase
   generator, not the question generator, turned out to be the difficulty dial.

`build_synthetic_rfq_corpus.py` now refuses outright to build a corpus with
duplicate question text, so failure mode 1 cannot recur silently.

**Results.** All of Exps 8–13 and Gaps 1–3 re-measured. Paraphrase set,
tenant-filtered nDCG@10, old corpus in brackets:

| arm | new | old |
|---|---|---|
| BM25 | 0.279 | 0.327 |
| blueprint hybrid | 0.436 | 0.390 |
| RRF fusion | 0.489 | 0.453 |
| semantic | 0.505 | 0.377 |
| ColBERT | 0.517 | 0.752 |
| cross-encoder | 0.561–0.578 | 0.717 |
| **q2q-semantic (library model)** | **0.635** | 0.648 |

New: on verbatim queries the entire ranking stack is redundant — BM25 alone
scores 0.957, and hybrid (0.963), the library model (0.961) and all three
cross-encoder variants (0.960-0.961) land within 0.006 of it. Half a real
workload may look like this, which argues for routing by query style rather
than sending everything down the expensive path.

Replicated: the library model as the dominant structural win; the tenant filter
as the largest single intervention (+0.149 to +0.294, ~350 leaks per arm);
BM25-wins-verbatim / semantic-wins-paraphrase, more sharply than before (0.957
vs 0.279 for BM25 across the two styles); contrastive fine-tuning +0.031 (vs
+0.029); binarisation costing quality; RRF beating the blueprint's trained
linear ranker; the generator faithful (1.00) and abstaining 8/8.

Changed: ColBERT no longer beats the cross-encoder. The headline claim inverted
— see below. And **the pointwise/pairwise result reversed**: the pointwise-GBDT
student reached 0.604 against pairwise-linear's 0.553, where Exps 2 and 4 both
found the opposite. That comparison confounds objective with model class (tree
vs linear), so the correct conclusion is that the objective advantage is real
but smaller than "confirmed 3×" implied, and can be outweighed by capacity.
Recorded as a retraction in RESULTS.md finding 28 rather than quietly dropped —
this is the same failure mode Exp 6 caught with the bootstrap, one level up.

**Conclusion.** The story sharpened. It used to be "the expensive stages earn
their keep — cross-encoder 0.717, ColBERT 0.752, full stack 0.774." It is now:
**the cheapest arm wins, and the expensive reranker makes it worse.** The
library model at 0.635 beats every reranker tested, and stacking a
cross-encoder on top drops it to 0.578 at ~300× the query cost — because
rescoring query-against-*answer* reintroduces exactly the near-duplicate
confusion that matching question-to-question removes. Four vendors' answers to
one question are near-identical prose; a reranker reading that prose cannot
separate them as well as a question-to-question cosine can.

**What this costs in honesty.** The corpus is now LLM-generated, which is a
worse validity threat than the one it fixed — one model wrote all 940 answers,
so the register is more uniform than real filed text. And as attempts 1–3 show,
its difficulty is a *dial I set*, not a property of the world: I chose the
paraphrase prompt that produced a useful spread. That is a legitimate research
hazard and is recorded as limitation 1 in RESULTS.md, now the study's biggest
stated caveat. Where old and new disagree, trust neither number — the
disagreement marks where a conclusion was corpus-dependent, not architectural.

**Next.** A genuinely real, licensable RFP corpus spanning more than security
compliance — the recursive lesson again: the small corpus was near-ceiling so a
harder one was needed; this one is authored, so a real one is needed next.

---

## 2026-08-01 — Experiment 14: HyDE — transform the query, in both directions

**Question.** The study has changed the *ranking* (Exps 9–13) and changed *what
is indexed* (the library model, finding 22), but never the *query
representation*. HyDE is the canonical query-side intervention: have an LLM
write the hypothetical document that would answer the query, and embed that
instead. Classic HyDE maps the query into **answer space** — which findings 22
and 24 say is the wrong space for this corpus (near-duplicate answers, generic
boilerplate as a cosine attractor). The symmetric variant maps it into
**question space**: write the standard questionnaire wording the buyer is
paraphrasing, then run the library model on it. Pre-registered predictions:
answer-space HyDE lands near semantic's 0.505 and below the library model;
question-space HyDE competitive with or above 0.635. Either outcome is
informative — this is the library model's strongest challenger so far, because
it uses the same insight (get into question space) with a general-purpose
technique instead of a schema change.

**Design.** `hyde_generate.py`: one LLM call per paraphrase query returns both
a hypothetical vendor answer (80–120 words, first person plural, no company
names) and the formal SIG/CAIQ-style wording of the question. The generator is
`google/gemini-2.5-flash` — deliberately a *different model family* from the
gpt-4o-mini that wrote the corpus, to blunt the same-generator register
confound (limitation 1). Generations cached in `hyde_generations.jsonl`; four
query files emitted with unchanged query_ids/tenants, so `qrels-rfq-para.tsv`
and `evaluate.py` apply as-is: hypothetical answer alone, hypothetical question
alone, and a query2doc-style concatenation of each (original paraphrase +
hypothetical text). Answer variants run through the existing `semantic` arm,
question variants through `q2q-semantic`; single sample per query at
temperature 0.7 (not the paper's multi-sample embedding average — Vespa embeds
`@query` server-side, so averaging would need client-side embedding). Runs in
`results/exp14/`. Generation cost: mean **1.99 s**, max 2.78 s per query —
HyDE lives in the cross-encoder's latency class (~2.5 s/q), not the fast class.

**Results (nDCG@10, 47 paraphrases, tenant-filtered; paired bootstrap over
per-query nDCG vs each arm's own baseline, 10k resamples, two-sided).**

| arm | recall@10 | MRR@10 | nDCG@10 | delta | p |
|---|---|---|---|---|---|
| semantic — embed the answer (Exp 8 baseline) | 0.617 | 0.475 | 0.505 | — | — |
| HyDE hypothetical answer | 0.741 | 0.493 | 0.538 | +0.033 | 0.63 |
| HyDE answer + original query | 0.773 | 0.558 | 0.604 | +0.099 | 0.084 |
| q2q-semantic — library model (Exp 10 baseline) | 0.684 | 0.649 | 0.635 | — | — |
| HyDE hypothetical question | 0.720 | 0.628 | 0.640 | +0.005 | 0.93 |
| **HyDE question + original query** | **0.777** | **0.725** | **0.719** | +0.084 | 0.064 |

Cross-class comparison: HyDE question+query vs the cross-encoder over library
recall (ce-q2q 0.578, the best reranker, same ~2.5 s/q latency class):
**+0.141, p≈0.007**. And answer-space HyDE at its best still loses to the plain
library model with no LLM at all: 0.604 vs 0.635 (p≈0.65).

**Finding 34 — the target space matters more than the technique, again.**
Answer-space HyDE spends an LLM call to move the query toward answer prose and
tops out at 0.604 — below a zero-cost schema change into question space
(0.635). The prediction from finding 22 held: an LLM call into the wrong space
is worth less than a field change into the right one. Finding 24's mechanism
also shows up on the query side — the hypothetical answer is exactly the
generic near-duplicate prose the corpus is full of, which is why recall jumps
(0.617 → 0.741) while MRR barely moves (0.475 → 0.493): it pulls in more
same-topic material but cannot separate the four vendors' near-identical
answers.

**Finding 35 — normalise-and-keep: paraphrase + reconstructed standard wording
is the best retrieval number on the paraphrase set (0.719), suggestive but not
confirmed at 47 queries (p≈0.064).** The hypothetical question *alone* only
ties the library model (0.640 vs 0.635): the LLM reliably recovers the formal
register but sometimes reconstructs the *wrong control*, and then the query is
thrown entirely. Concatenating the original paraphrase hedges that failure —
the reconstruction supplies the vocabulary the paraphrase lacks, the paraphrase
keeps the intent the reconstruction sometimes loses. Recall@10 rises 0.684 →
0.777, which attacks finding 27's binding constraint (gold never surfacing in
the top-10) — the one thing no reranker can fix, and the reason this beats
every reranker tested (vs ce-q2q: +0.141, p≈0.007, at the same latency). Exp
6's lesson applies before celebrating: +0.084 at n=47 with p≈0.064 is exactly
the shape of result that has died at scale in this study before. It needs a
scaled paraphrase set before it graduates from suggestive to confirmed.

**Caveats.** All three texts in play — stored question, paraphrase,
hypothetical reconstruction — are LLM prose; a different generator family
(gemini vs gpt-4o-mini) blunts but does not remove the affinity confound, and a
real buyer's wording may sit farther from what an LLM reconstructs than these
paraphrases do. Single sample per query, one generator, one prompt. 47 queries.
Query cost is ~2 s of LLM generation on top of a fast Vespa query — cacheable
across repeat questions, and generation could overlap other work, but it is
not the fast class the library model lives in. Verbatim set not run: BM25 is
at 0.957 there and a query rewrite can only dilute exact overlap.

> **Superseded in part by Exp 15 (same day).** The scaled paraphrase set did to
> finding 35 what Exp 6 did to the pairwise claim: the +0.084 edge over the
> library model attenuated to +0.023 out-of-sample (p≈0.24) and is **not
> confirmed**. Findings 34 and the ce-comparison half of 35 survived. See the
> Exp 15 entry below.

---

## 2026-08-01 — Experiment 15: scale the paraphrase set 47 → 232 (revalidate finding 35)

**Question.** Finding 35's headline — normalise-and-keep at 0.719, +0.084 over
the library model — sat at p≈0.064 on 47 queries, exactly the shape of result
Exp 6 killed at scale. Does it survive a bigger instrument? The original 47
paraphrases also formed the hypothesis, so the honest test is on queries that
played no part in it.

**Design.** `scale_para.py` gives every verbatim query a paraphrase twin:
232 total — the original 47 copied through byte-identical, plus 185 new ones
generated with `build_synthetic_rfq_corpus.gen_paraphrase` **unchanged** (same
model, same prompt, same temperature — the paraphrase prompt is the corpus's
difficulty dial and a new prompt would silently move it). New queries carry
cluster `questionnaire-para-new` so the out-of-sample subset scores
separately; structural qrels regenerate under the same rule and reproduce the
original 47's lines exactly (asserted in the script). HyDE texts for the 185
via `hyde_generate.py` (same generator, mean 1.97 s/query). All fast arms plus
ce-q2q and the student re-run on the full 232 (`results/exp15/`); ColBERT and
the other two cross-encoder variants were not re-run. Trained-component evals
(Gap 2 fine-tune) must keep using the original 47 — the scaled set overlaps
their training question_ids.

**Results (nDCG@10, tenant-filtered).**

| arm | orig-47 | new-185 (out-of-sample) | full-232 |
|---|---|---|---|
| BM25 | 0.312* | 0.354 | 0.346 |
| semantic | 0.505 | 0.604 | 0.584 |
| hybrid | 0.416* | 0.502 | 0.484 |
| RRF | 0.480* | 0.555 | 0.539 |
| **q2q-semantic (library model)** | 0.635 | 0.766 | **0.740** |
| distilled student | 0.556* | 0.684 | 0.658 |
| ce-q2q (cross-encoder) | 0.578 | 0.710 | 0.683 |
| HyDE answer | 0.538 | 0.577 | 0.569 |
| HyDE answer + query | 0.604 | 0.637 | 0.631 |
| HyDE question | 0.640 | 0.693 | 0.682 |
| HyDE question + query | 0.719 | 0.789 | 0.775 |

Paired bootstrap (10k resamples, two-sided):

| comparison | full-232 | new-185 only |
|---|---|---|
| HyDE q+query vs q2q | +0.035, p≈0.046 | **+0.023, p≈0.24** |
| HyDE question vs q2q | −0.058, p≈0.015 | −0.073, p≈0.003 |
| HyDE q+query vs ce-q2q | +0.091, p<0.0001 | +0.079, p≈0.001 |
| HyDE ans+query vs q2q | −0.109, p<0.0001 | −0.129, p<0.0001 |
| HyDE q+query vs HyDE question | +0.093, p<0.0001 | +0.096, p<0.0001 |
| q2q vs ce-q2q | +0.056, p≈0.001 | — |
| q2q vs student | +0.081, p<0.0001 | — |

**Finding 36 — scale strikes again: finding 35's headline does not survive
out-of-sample confirmation.** On the 185 queries that played no part in
forming the hypothesis, the normalise-and-keep edge over the library model is
+0.023 at p≈0.24. The full-232 p≈0.046 is dragged under 0.05 by the original
47 — the queries the claim was built on — which is exactly the in-sample
flattery this study keeps re-learning to distrust (Exp 6: +0.006 at 19q died
at 69q; the "confirmed 3×" pairwise claim died at Exp 13). Demoted from "best
number in the study" to *unconfirmed ~+0.02–0.03 tendency at ~2 s/query*. What
**did** survive, all significant out-of-sample: (1) HyDE-into-question-space
**beats the cross-encoder at the same latency** (+0.079, p≈0.001) — if you are
paying LLM-class latency per query, rewrite the query rather than rerank the
answers; (2) **pure rewriting is actively harmful** (−0.073, p≈0.003) — the
reconstruction throws away intent often enough that q2q on the raw paraphrase
beats q2q on the "cleaned-up" one; keeping the original query is worth +0.096
(p<0.0001); (3) **answer-space HyDE never competes** (−0.129 vs q2q,
p<0.0001) — finding 34 confirmed at scale. The practical recommendation is
unchanged and now better-supported: the library model, alone, on the raw
query.

**Finding 37 — the library model's dominance replicates at scale, and the
scaled instrument exposed its own drift.** On 232 queries q2q (0.740)
significantly beats the cross-encoder stacked on it (0.683, p≈0.001) and the
distilled student (0.658, p<0.0001) — findings 22/24 graduate from 47-query
observations to confirmed results. Two instrument notes recorded honestly:
*(a)* every arm scores higher on the new 185 than on the original 47 (q2q
0.766 vs 0.635, ~3σ against sampling noise) — same prompt, same nominal
model, five days apart, yet the new paraphrases run measurably easier
(content-word Jaccard to the original question 0.077 vs 0.070). The
difficulty dial of limitation 1 not only exists, it **drifts** between
generation runs; cross-set absolute comparisons are unsafe, paired
within-query comparisons unaffected. *(b)* re-running the committed lexical
arms reproduces the semantic numbers exactly but shifts BM25/hybrid/RRF by
±0.01–0.03 on the same queries and qrels — the live index's BM25 scores
differ from whatever state produced the July run files (a re-feed since is
the likely cause). All Exp 15 comparisons are within a single index state.

**Caveats.** The out-of-sample subset shares the paraphrase generator with the
original set, so "out-of-sample" means new queries, not a new distribution;
ColBERT/ce-semantic/ce-hybrid not re-run at scale; the drift observation is
n=1 (one regeneration event, cause unresolved — provider routing, model
update, or an unlucky original sample).

---

## 2026-08-01 — Gap 4: the mixed workload, and fusing the best arms

**Question.** Two analyses computable entirely from committed run files.
*(a)* Finding 25b suggested routing by query style; nobody measured what a
single configuration scores on realistic blended traffic, or what a router
would actually buy. *(b)* Does RRF-fusing the two best arms beat both?

**Design.** *(a)* Blend = 232 verbatim + 232 scaled paraphrases (Exp 15's
twins), tenant-filtered, per-arm mean over the 464-query union; oracle router
= best fixed arm per style chosen with hindsight (an upper bound no real
router reaches). *(b)* Client-side reciprocal rank fusion (k=60) over top-50
runs (`results/gap4/`, re-run with `--hits 50`) of q2q, the HyDE
question+query rewrite, and the student.

**Results.** *(a)* Blend-464 nDCG@10: q2q **0.850**; student 0.811; ce-q2q
0.822; hybrid 0.724; BM25 0.652. Oracle router (student on verbatim / q2q on
paraphrase) reaches 0.852 — **router headroom +0.001**. q2q is within 0.003
of the best verbatim arm (0.961 vs 0.964) while dominating paraphrases, so
one config covers the mixed workload. *(b)* Every fusion lands at or below
its best parent: q2q+hydeqcat 0.765 vs hydeqcat alone 0.775 (p≈0.44);
q2q+student 0.731 vs q2q 0.740 (p≈0.55); the triple 0.758 (p≈0.36).

**Finding 38 — no router, no fusion: serve the library model alone.** The
routing idea from finding 25b dissolves on measurement — the library model is
already at the verbatim ceiling, so styling-aware dispatch buys +0.001. And
rank fusion of the best arms never beats the best arm (finding 23's
weak-signal lesson extends to fusing two *strong* arms whose rankings mostly
agree — the fusion can only blur the better one). The deployment story gets
simpler, not more clever.

---

## 2026-08-01 — Gap 5: serving the fine-tuned embedder (Gap 2, completed)

**Question.** Gap 2 fine-tuned MiniLM offline and never deployed it — yet
finding 11's "fine-tune the representation" and finding 22's "index the
question" had never been combined in the engine. Does the served fine-tune
catch the deployed nomic library model?

**Design.** `gap5_export_minilm.py` re-runs the Gap 2 training reproducibly
(seed 42; base 0.591 exactly reproduces `results/gap2/result.json`, tuned
0.617 vs July's 0.622 — within the seed band), saves the checkpoint Gap 2
discarded, and exports base + tuned MiniLM to ONNX. Served as two
`hugging-face-embedder` components (`minibase`, `minift`), two float-tensor
question-embedding fields with angular NN, and two rank profiles
(`q2q-mini`, `q2q-ft`). 940 docs re-fed; eval on the **original 47 only**
(the scaled set overlaps the training question_ids). Runs in `results/gap5/`.

**Results (nDCG@10, 47 paraphrases, tenant-filtered).**

| served arm | nDCG@10 |
|---|---|
| q2q — nomic ModernBERT (Exp 10) | **0.635** |
| q2q — fine-tuned MiniLM | 0.595 |
| q2q — base MiniLM | 0.573 |

Fine-tune delta served: +0.023 (p≈0.21; offline +0.026). Tuned vs nomic:
−0.039 (p≈0.29). Served numbers sit ~0.02 under their offline equivalents
(approximate NN + targetHits vs exact within-tenant scan).

**Finding 39 — embedder choice dominates embedder fine-tuning.** The
contrastive fine-tune reproduces and survives serving, but it moves a small
embedder +0.023 while the *gap to the larger zero-shot embedder* is +0.062.
Finding 11 is true within a model family and misleading across them: before
fine-tuning anything, pick the strongest base embedder — then fine-tune
*that*. The unfinished follow-up writes itself: contrastively fine-tune the
nomic embedder on the same 195 pairs (heavier — ModernBERT on MPS — and
untested here).

---

## 2026-08-01 — Gap 6: a modern cross-encoder (was finding 24 about reranking, or about one model?)

**Question.** Findings 24/29 — "the cross-encoder makes the library model
*worse*; retire it" — rest entirely on ms-marco MiniLM-L-6, a 6-layer
2020-era model. Is the sharpest negative result in the study a fact about
reranking, or about that reranker?

**Design.** `gap6_modern_ce.py`: BAAI/bge-reranker-v2-m3 (568M,
current-generation), offline on MPS, reranking the same tenant-filtered
top-50 pools (library-model and hybrid first phases, `results/gap4/`),
reading stored question + answer text like the deployed `ce_tokens` input.
Scored on the original 47 and the full scaled 232; per-query scores in
`results/gap6/`. ~1.7 s/query on MPS — same cost class as the deployed
cross-encoder and the HyDE rewrite.

**Results (nDCG@10, tenant-filtered).**

| arm | orig-47 | full-232 | new-185 (oos) |
|---|---|---|---|
| q2q first phase (free) | 0.635 | 0.740 | 0.766 |
| old ce-q2q (ms-marco MiniLM) | 0.578 | 0.683 | 0.710 |
| **bge-reranker over q2q recall** | 0.678 | 0.765 | 0.787 |
| HyDE question+query (Exp 15) | 0.719 | 0.775 | 0.789 |

Paired bootstrap: bge vs old ce **+0.082, p<0.0001** (full-232) — the modern
model is categorically better. bge vs the free library model: +0.043 p≈0.33
(47), +0.025 p≈0.11 (232), +0.021 p≈0.21 (oos) — never significant. bge vs
the HyDE rewrite at the same cost: −0.010 p≈0.59 (232) — a statistical tie.

**Finding 40 — "actively harmful" is retracted; "doesn't earn its cost"
survives.** The harm in finding 24 was a property of the shipped model, not
of reranking: a modern cross-encoder over the same recall is at-or-above the
library model on every subset. But it is *never significantly above it* —
~1.7 s/query of 568M-parameter compute buys a +0.02–0.04 tendency that the
paired bootstrap cannot distinguish from zero, and it exactly ties the LLM
query rewrite at equal cost. The refined through-line: on this corpus the
expensive stage is no longer poison, and still not worth buying — the free
library model remains the recommendation. (Both expensive techniques now
show the same unconfirmed ~+0.02 at scale; if either is ever worth it, a
much larger query set will have to say so.)

---

## 2026-08-02 — Gap 7: prefix symmetry and an embedder bake-off

**Question.** Two threads from the embedder discussion. *(a)* The library
model is a **symmetric** task — question matched against question — but the
nomic embedder conditions its two sides asymmetrically (`search_query:` vs
`search_document:` prepends, inherited from the blueprint). Is the
document-side prefix costing accuracy on a task that has no documents?
*(b)* Every number in the study rides on one small 2024 embedder; do
current-generation models (bge-m3, Qwen3-Embedding-0.6B — the latter
instruction-conditioned, aimed squarely at capturing HyDE's
normalise-toward-question-space gain inside the forward pass) beat it?

**Design.** *(a)* Served: a second nomic component (`nomicsym`) identical
except both prepends are `search_query:`; stored questions re-embedded into
`title_emb_sym`; two rank profiles (`q2q-float` / `q2q-sym`) scoring
identical float cosine over the whole tenant, so the *only* difference is the
document-side prefix. Runs in `results/gap7/`, scaled paraphrase set +
verbatim. *(b)* Offline (`gap7_embed_bakeoff.py`, same exact-scan harness as
Gap 2): nomic asym/sym, bge-m3, Qwen3-0.6B raw and with a task instruction;
per-query scores in `results/gap7/bakeoff.json`. Ops note for the record:
the third embedder pushed the 6 GB container past Vespa's default feed-block
limits (HTTP 507 NO_SPACE mid-feed — disk was at 77% against the 0.75
default); fixed with explicit `resource-limits` (0.88/0.85) in services.xml,
then a clean 940/940 re-feed.

**Results (served, nDCG@10, tenant-filtered).**

| arm | orig-47 | new-185 (oos) | full-232 | verbatim |
|---|---|---|---|---|
| q2q — binary titles, asym prefixes (Exp 15 baseline) | 0.635 | 0.766 | 0.740 | 0.961 |
| q2q — float titles, asym prefixes | 0.662 | 0.788 | 0.762 | 0.964 |
| **q2q — float titles, symmetric prefixes** | **0.703** | **0.827** | **0.802** | **0.965** |

Paired bootstrap: symmetric vs asymmetric float (the pure prefix effect)
**+0.039, p<0.0001 — identical out-of-sample**. Float vs binary titles
+0.023 (p≈0.017 full / 0.045 oos). Combined vs the study's headline arm:
**+0.062, p<0.0001**. Verbatim 0.965 is the best verbatim number in the
study; the 464-query blend rises 0.850 → **0.883**, and the oracle router
question closes itself — the symmetric arm is now the best arm in *both*
styles, so router headroom is zero by construction. Against the HyDE rewrite
it is +0.027 (p≈0.11) — the free arm is at-or-above the ~2 s/query one,
which is thereby strictly dominated.

Offline bake-off (full-232, delta vs nomic-asym, paired bootstrap):

| arm | full-232 | delta | p (full / oos) |
|---|---|---|---|
| nomic asym (baseline) | 0.763 | — | — |
| **nomic symmetric** | **0.805** | **+0.042** | **<0.0001 / <0.0001** |
| bge-m3 | 0.766 | +0.003 | 0.84 / 0.82 |
| Qwen3-0.6B raw | 0.793 | +0.030 | 0.060 / 0.11 |
| Qwen3-0.6B instructed | 0.797 | +0.034 | 0.035 / 0.067 |

**Finding 41 — the library model was leaving free, significant accuracy on
the table: match a symmetric task symmetrically.** Embedding the stored
question with the *query* prefix is worth +0.039 served (p<0.0001, the first
intervention since the tenant filter to clear significance out-of-sample),
plus +0.023 for float title vectors over binarized ones (940 titles × 3 KB —
binarization saves nothing worth having here). The upgraded free arm:
paraphrase 0.802, verbatim 0.965, blend 0.883 — best confirmed numbers in
the study on all three, at zero query cost. The recipe gains six words:
*query prefix on both sides, float titles.*

**Finding 42 — configuration beat model shopping.** Same-model symmetric
conditioning (+0.042, p<0.0001) beat switching to bge-m3 (+0.003, a tie) and
beat Qwen3-0.6B (+0.030 raw; instruction conditioning adds only +0.004 over
raw and is not significant out-of-sample) — the instructed-embedder route to
HyDE's gain mostly wasn't there, at least at 0.6B. Consistent with the
study's oldest through-line: before buying a newer model, check what the
current one is being asked to do.

**Caveats.** Bake-off is offline (exact scan; Gap 5 measured serving ≈
−0.02) and single-seed; Qwen3's larger variants and API embedders untested;
the prefix result is one model family — other prefix-conditioned embedders
(e5, Cohere input_type) should replicate it before it's called general; all
of limitation 1 applies.
