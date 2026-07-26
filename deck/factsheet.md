# Deck factsheet — accurate content & numbers (do NOT invent anything beyond this)

Project: a stress-test of a production-shaped retrieval architecture. The
workload is a **fictional test case** (`spec/target-architecture.md`): a
multi-tenant platform answering RFPs and security questionnaires from a
vendor's own past approved answers. We deployed the stack it implies,
built an adversarial benchmark, and ran 13 experiments to find where it breaks
and what fixes it.

## Slide 1 — Title / hero
- Title idea: "A retrieval stack, stress-tested" / subtitle "Which parts of a
  phased RAG pipeline actually earn their cost — and 13 experiments answering it"
- Anchor stat: retrieval quality journey **0.505 → 0.635 nDCG@10** on the hard case —
  achieved by changing *what you match against*, not by adding a model
- Context tags: Vespa · hybrid retrieval · cross-encoder · ColBERT · distillation

## Slide 2 (deck s3) — What Vespa is: not a vector database, a ranking engine
A document holds **tensors**; ranking is an expression the engine evaluates
next to the data. Compared against a vector index + app code (pgvector as the
reference point — verified against its README, Aug 2026):
- **Vectors per document**: pgvector stores one vector per column, so a
  40-chunk document is 40 rows regrouped in app code. Vespa:
  `tensor<int8>(chunk{}, x[96])` — every chunk vector inside the document,
  with rank expressions reducing over them.
- **Lexical + semantic**: pgvector is vector-only; BM25 comes from Postgres
  full-text search or a separate extension, fused in app code. Vespa does both
  in one query, fused in the rank profile.
- **Reranking**: candidates cross the network before the app can rerank them.
  Vespa does phased ranking in-engine.
- **Model inference**: pgvector has none — embeddings are produced outside and
  inserted. Vespa runs ONNX in-engine (embedder, cross-encoder, ColBERT,
  LightGBM); this study ran three at once on an 8GB laptop.
- **Filtering**: with approximate indexes pgvector applies the filter *after*
  the index scan (iterative scans mitigate). Vespa attributes filter *during*
  matching.
- Payoff line: late interaction, per-chunk scoring and top-k chunk selection
  are rank expressions you write, not features you wait for.

## Slide 3 (deck s4) — The Vespa RAG Blueprint, and what it ships by default
Vespa's reference RAG application, deployed here as-is. Defaults, all verified
against the committed app files:
- **Schema & chunking**: chunked in-schema (`chunk fixed-length 1024`) into a
  BM25-indexed `chunks` array *and* a per-chunk embedding tensor
- **Embeddings**: ModernBERT embedder in-engine at index and query time;
  `pack_bits` to 1 bit/dimension (int8, hamming); float query vector kept for
  cosine rescoring
- **Retrieval**: one YQL query — `userInput` OR ANN on the title vector OR ANN
  on the chunk vectors, `targetHits:100` each; shipped as the hybrid / rag /
  deepresearch query profiles
- **Ranking**: first-phase learned linear over 7 features, weights shipped as
  constants in the query profile; optional second-phase LightGBM GBDT
- **Chunk selection**: `top(3, chunk_sim_scores)` via `select-elements-by` —
  picks which paragraphs reach the LLM
- **Generation**: the `rag` profile chains an LLM client, streaming over SSE
- **Not included**: no tenant concept, no cross-encoder, no late interaction —
  those are this study's additions

## Slide 4 (deck s5) — The test harness we built
- **Two corpora**: a small adversarial set (39 LLM-written docs, 3 tenants,
  one query cluster per known failure mode) + a **questionnaire corpus** (940
  answers to a 261-question security questionnaire from 4 fictional vendors —
  our taxonomy, our questions, generated answers; all committed)
- **Query sets**: 20→80 adversarial queries (11 failure-mode clusters);
  232 verbatim + 47 paraphrase questionnaire queries
- **Graded TREC qrels** (0/1/2) incl. **paragraph-level** judgments
- **evaluate.py**: nDCG@10, Recall@10, MRR@10, per-cluster; plus
  **cross-tenant leakage accounting** and **unanswerable-query** handling
- **Statistical rigour**: **paired bootstrap** significance + **grouped 5-fold
  cross-validation**
- Ran on local Vespa in Docker (M1, 8GB), 3 ONNX models resident
- Carries the **operating philosophy**: *"diagnose before you optimise, ship
  incrementally, don't fool yourself with noisy results"* — it sits with the
  measurement machinery it justifies

## Slide 5 — The 13 experiments (map)
Synthetic (1–7):
1. Run blueprint → shipped LightGBM *degrades* on a foreign corpus (0.846 vs 0.950 semantic)
2. Pointwise retrain → *loses* to zero-shot (0.902 vs 0.950); +5pt overfitting illusion
3. Paragraph chunking → fixes chunk selection, invisible to doc-level metrics
4. Pairwise training → better objective
5. Dedup-then-prefer-fresh rerank → fixes staleness (+0.011, p≈0.011)
6. Scale 20→80 → **retracts** the pairwise-beats-zero-shot claim (p≈0.98); tenant filter +0.035 (p<0.001)
7. Abstention on unanswerables → AUC 0.884, first-pass only
Questionnaire corpus (8–13):
8. Questionnaire corpus + RRF → query style flips the winner (verbatim BM25 **0.957** / paraphrase **0.279**); tenant crowding an order of magnitude above the small set
9. Cross-encoder rerank → 0.505 → **0.576**, but at ~2500 ms/query
10. Library model (match the question, not the answer) → 0.505 → **0.635** — the best arm tested, and free
11. ColBERT late interaction → 0.517 at ~60 ms; between fusion and the cross-encoder for 1/40th the cost
12. Cross-encoder *on top of* the library model → **0.578, worse than 0.635 without it**
13. Distillation → student **0.604 beats the 0.561 teacher**; deployed native, **~8 ms server-side vs ~2500 ms**

## Slide 6 — Results: the cost/quality frontier (paraphrase set, nDCG@10, tenant-filtered)
| technique | nDCG@10 | query cost |
|---|---|---|
| BM25 (lexical) | 0.279 | fast |
| blueprint hybrid | 0.436 | fast |
| RRF fusion | 0.489 | fast |
| semantic (embed the answer) | 0.505 | fast |
| ColBERT late interaction | 0.517 | ~60 ms |
| cross-encoder rerank | 0.576 | ~2500 ms |
| library recall + cross-encoder | 0.578 | ~2500 ms |
| **q2q library (embed the question)** | **0.635** | **fast** |

## Slide 7 — The through-lines (findings)
- **Structure beats learned ranking, and it is not close.** Every durable win
  was structural: hard tenant filter (+0.15 to +0.29 here, +0.035 on the small
  set), paragraph chunking, dedup-freshness, the library model. Learned rankers
  repeatedly *tied* zero-shot.
- **The cheapest arm wins.** Matching question-to-question (0.635) beats every
  reranker tested, including a cross-encoder costing ~300× more per query.
- **And the expensive stage actively hurts.** Stacking the cross-encoder on the
  library model drops it to 0.578 — rescoring against near-identical answer
  text reintroduces the confusion that matching on questions removes.
- **Pointwise loses, pairwise wins — confirmed 3×** (Exps 2, 4, 13).
- **The cross-encoder is retirable outright.** Distilled, the student *beats*
  it (0.604 vs 0.561) and serves in ~8 ms instead of ~2500 ms.
- **A three-times-confirmed finding reversed.** Pointwise beat pairwise in
  distillation (0.604 vs 0.553) — the comparison confounds objective with model
  class, so the objective advantage is real but smaller than claimed.
- **Fine-tuning the representation was the only training that beat zero-shot**
  (+0.031). Ranker training tied; embedder training worked.
- **The measurement infrastructure was the real product** — leakage accounting,
  grouped CV, paired bootstrap, chunk-level qrels found (and killed) findings.

## Slide 8 — Final thoughts
- We validated the **search-quality half** of the architecture in depth; the
  **production-systems half** (cluster sizing, multi-region reliability,
  observability, hosted model serving) needs real production infrastructure.
- Honest limitation: **author-graded ground truth.** Highest-value hardening =
  independent expert judgments validating the structural qrels.
- The operating philosophy — diagnose before optimise, don't fool yourself —
  is exactly what the study kept proving.

---

# Technology reference — the concept slides

Each technique gets one slide in the consolidated deck, in the format
**what it is · how it works · the tradeoff · what we saw**. Source copy below.

Audience: a reader learning the IR/RAG concepts the study spans.
Every concept slide follows the SAME template so it reads as a course:
  WHAT IT IS · HOW IT WORKS · THE TRADEOFF · WHAT WE SAW (our grounded result).
Use ONLY the numbers below — all are real results from this project.

---

## Slide 1 — Title / map
"The concepts, taught." A companion to the results deck: what each technique
*is*, how it works, and the tradeoff — so the numbers in the other deck make
sense. 16 slides, grouped: the problem → ways to match → ways to rank →
representation → RFP-specific tricks → how you know it works.

## Slide 2 — RAG, and why retrieval is the whole game
WHAT: Retrieval-Augmented Generation = find relevant passages, then let an LLM
write the answer from them.
HOW: two stages — a retriever pulls top-k passages; a generator conditions on
them. The generator can only be as right as what it's handed.
TRADEOFF: all the leverage is upstream. A perfect generator over bad retrieval
still produces wrong answers; a modest generator over good retrieval is fine.
WHAT WE SAW: the generator was flawless (faithfulness 1.00, zero hallucination,
declined 8/8 unanswerables) — correctness (~0.50) was capped entirely by
retrieval recall. The study's core premise, confirmed: get retrieval wrong and
every answer downstream is wrong too.

## Slide 3 — Two ways to match: lexical (BM25) vs semantic (embeddings)
WHAT: the core dichotomy. Lexical = match words. Semantic = match meaning.
HOW: BM25 scores by term overlap weighted by rarity (rare shared words count
more). Dense/semantic embeds query and doc into vectors; similarity = cosine.
TRADEOFF: BM25 nails exact terms, acronyms, rare tokens — but dies on
vocabulary mismatch (synonyms, paraphrase). Semantic bridges wording — but blurs
fine distinctions and collapses when many docs are near-identical.
WHAT WE SAW: query STYLE flipped the winner. Verbatim questionnaire re-runs:
BM25 0.957, semantic 0.915. Reworded questions: BM25 collapses to 0.279 and
semantic wins (0.505). Neither alone is enough.

## Slide 4 — Hybrid retrieval & fusion (RRF)
WHAT: combine lexical + semantic so each covers the other's blind spot.
HOW: run both, merge. Reciprocal Rank Fusion (RRF) adds 1/(k+rank) from each
list — rank-based, so it needs no score calibration and no training.
TRADEOFF: robust and free (no tuning), but untuned — a trained combiner can beat
it if you have labels; and it can't fix a signal that's simply absent.
WHAT WE SAW: on reworded questions, untrained RRF (0.489) beat the blueprint's
trained linear ranker (0.436). Structure beat a learned model — a recurring theme.

## Slide 5 — Chunking: how you cut documents
WHAT: docs are split into passages before embedding/indexing.
HOW: fixed-size (e.g. every 1024 chars) is simple; semantic/paragraph-aware
splits on natural boundaries.
TRADEOFF: big chunks keep context but blur what they're "about" (a chunk about
five things matches everything — a cosine attractor); small chunks are precise
but can sever an answer mid-thought. Fixed-size is easy but boundary-blind.
WHAT WE SAW: fixed-1024 split a password policy mid-sentence and let a generic
intro chunk win context slots. Paragraph chunking fixed both — invisible to
doc-level metrics, decisive at the chunk level. You only see it if you judge
chunks, not just docs.

## Slide 6 — Phased ranking: the compute funnel
WHAT: rank in stages, cheap-to-expensive, each stage re-ranking fewer docs.
HOW: first-phase (cheap, all candidates) → second-phase (costlier, top-N) →
global-phase (most expensive, top-few). Vespa names these literally.
TRADEOFF: lets you afford an expensive model by only running it on a shortlist —
but the shortlist is only as good as the cheap stage's recall. A weak first
phase forces a wide, expensive rerank window.
WHAT WE SAW: recall@10 runs 0.43-0.69 on paraphrases — on a third or more of
queries the gold answer never reaches the top-10 at all, and no reranker can
save what was never retrieved. First-phase recall is a real constraint here.

## Slide 7 — Learning to rank: the objective matters (pointwise vs pairwise)
WHAT: train a model to order results, over cheap features (BM25, similarities…).
HOW: pointwise = predict each doc's relevance score independently. Pairwise =
learn which of two docs should rank higher (RankNet-style). Listwise optimizes
the whole ordering.
TRADEOFF: pointwise is simple but optimizes the wrong thing (absolute scores, not
order) — it can rank worse than its own best feature. Pairwise matches the task
(ordering) and needs graded pairs.
WHAT WE SAW: on the small corpus, same data and features — pointwise LOST (0.902
vs zero-shot 0.950), pairwise WON (0.956). But in distillation on the
questionnaire corpus it REVERSED: pointwise 0.604, pairwise 0.553. That
comparison confounds objective with model class (tree vs linear), so the honest
lesson is that the objective matters but less than "confirmed 3×" implied.

## Slide 8 — Cross-encoders: the accurate, expensive reranker
WHAT: a transformer that reads query and document TOGETHER and outputs a
relevance score.
HOW: concatenate [query, doc], run one full transformer pass, read the score.
Joint attention sees exact interactions a vector-dot-product can't.
TRADEOFF: most accurate reranker available — but you pay a full model forward
pass PER (query,doc) pair, at query time. Latency scales with how many you rerank.
WHAT WE SAW: lifted the hard case 0.505→0.576 at ~2.5 s/query on CPU — a real
gain, but it is beaten outright by the library model (0.635) which costs nothing,
and stacking it ON that model makes things worse (0.578).

## Slide 9 — Late interaction (ColBERT): the sweet spot
WHAT: keep per-TOKEN embeddings for each doc; match at the token level.
HOW: MaxSim — for each query token, take its best-matching doc token, sum. Doc
token vectors are precomputed at INDEX time, so query time is a cheap tensor op,
no transformer pass per doc.
TRADEOFF: near-cross-encoder quality at bi-encoder speed — but stores many
vectors per doc (a token-matrix), so the index is larger and feeds are heavier.
WHAT WE SAW: 0.517 at ~60 ms/query — between fusion and the cross-encoder for
1/40th the cost. The cost moved to indexing (a third model, slower feeds). Often
the right answer to "the reranker is too expensive."

## Slide 10 — Knowledge distillation: pay once, serve cheap
WHAT: train a cheap "student" to imitate an expensive "teacher."
HOW: run the teacher (cross-encoder) offline to LABEL (query,doc) pairs; train a
fast student (linear/GBDT over features, or a small bi-encoder) to reproduce its
ranking. The teacher never runs at query time.
TRADEOFF: you get most of the teacher's quality at a fraction of the serving
cost — but the student is capped by its features, and (again) the training
objective matters.
WHAT WE SAW: the distilled student BEAT its teacher — 0.604 vs 0.561 — at
microsecond cost, and the deployed linear variant serves in ~8 ms vs ~2500 ms
(~300× less ranking compute). The expensive reranker becomes an offline teacher,
not a serving component.

## Slide 11 — Representation learning: bi-encoders & contrastive fine-tuning
WHAT: the embedding model itself — and teaching it your domain.
HOW: a bi-encoder embeds query and doc separately (fast, cacheable). Contrastive
fine-tuning pulls matching pairs together and pushes mismatches apart;
MultipleNegativesRankingLoss uses other items in the batch as free negatives.
TRADEOFF: fine-tuning directly attacks the retrieval ceiling and is the highest-
leverage representation lever — but needs domain pairs (real or synthesized) and
compute; over-fitting on little data is a risk.
WHAT WE SAW: fine-tuning a small MiniLM on 195 generated paraphrase pairs lifted
it 0.591→0.622 (+0.031) — the ONLY trained component to beat its zero-shot
baseline (the rankers all tied). Fixing the representation matched the actual
failure mode (vocabulary mismatch).

## Slide 12 — Quantization / binarization: the memory-quality dial
WHAT: shrink embeddings by storing them at lower precision.
HOW: float32 → int8, or 1-bit binary; compare with cheap distances (hamming for
binary). Optionally rescore the top-k in full precision.
TRADEOFF: massive storage/latency savings for a small, measurable quality loss —
and you can buy most of the quality back by rescoring finalists in float.
WHAT WE SAW: binary cost ~0.053 nDCG (~12%) for 32× smaller vectors on the hard
case. The right default at scale (thousands of docs × dozens of tenants);
recover the gap with binary recall + float rescoring of the top-k.

## Slide 13 — The library model (the RFP-specific insight)
WHAT: retrieve against past QUESTIONS, not past answers.
HOW: index your curated Q&A library keyed on the stored question text; embed the
incoming (reworded) question and match it to the closest stored question, then
return its vetted answer.
TRADEOFF: hugely effective when questions recur (security questionnaires) — but
depends on having a curated library and on question recurrence; bespoke,
one-off questions benefit less.
WHAT WE SAW: matching question→question lifted single-vector semantic
0.505→0.635, same embedder, only the indexed field changed — the best arm in the
whole study, and free. This is why real RFP tools are built around answer
libraries, not raw-document RAG.

## Slide 14 — The "boring" structural wins: tenancy, freshness, dedup
WHAT: the unglamorous mechanisms that mattered most.
HOW: tenant isolation = a hard metadata FILTER (never let ranking decide it).
Freshness = prefer the newer of near-duplicates (a tie-break), not a global
ranking feature. Dedup = collapse near-identical answers.
TRADEOFF: cheap, robust, and often the biggest wins — but they're plumbing, not
models, so they get overlooked. A global "freshness feature" actively misleads;
freshness only discriminates between duplicates on the same topic.
WHAT WE SAW: the hard tenant filter was the single biggest relevance
intervention — +0.035 (p<0.001) on the small set, and +0.15 to +0.29 on the
questionnaire corpus, removing ~350 cross-tenant leaks per arm. A global
freshness weight learned NEGATIVE; dedup-then-prefer-fresh worked (+0.011).
Structure beat learned ranking repeatedly.

## Slide 15 — How you know it works: evaluation & statistical rigour
WHAT: measuring retrieval honestly — the part that's easy to fool yourself on.
HOW: nDCG@10 (graded, position-weighted) over TREC qrels; grouped cross-
validation (never test on a query you trained on); paired bootstrap (is a
delta real or noise?); chunk-level judgments; leakage accounting.
TRADEOFF: rigorous eval is slow and unglamorous — but without it you ship noise.
In-sample numbers flatter; small samples lie.
WHAT WE SAW: an in-sample fit looked like a +5-point win that grouped CV
revealed as a LOSS; a "pairwise beats zero-shot" result evaporated (p≈0.98) when
we scaled the query set. The measurement rig caught (and killed) findings the
metrics alone would have shipped. "Don't fool yourself with noisy results."

## Slide 16 — The generation stage & the limits of it all
WHAT: drafting the answer, and knowing what this study does and doesn't prove.
HOW: prompt the LLM with retrieved context, instruct it to answer ONLY from that
context and to decline if it's not covered — faithfulness and abstention are the
safety properties for a compliance product.
TRADEOFF: a well-instructed generator turns retrieval misses into honest "not
covered" instead of confident wrong answers — but that's for atomic-answer
domains; synthesis/essay questions make the generator do real, harder work.
WHAT WE SAW: perfect faithfulness (1.00), zero hallucination, 8/8 abstention on
unanswerables. Correctness 0.50 is capped by retrieval, not by the generator.
AND the honest limit: our data is security-compliance, a standardized best case,
and the corpus is LLM-generated. The ARCHITECTURE (filter, library model, phased
ranking, distillation, fine-tuning) should port; the MAGNITUDES won't. Know the
limits of your eval.
