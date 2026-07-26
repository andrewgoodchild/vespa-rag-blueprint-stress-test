# Target architecture — multi-tenant RFP/questionnaire retrieval

> **This is a fictional scenario**, written to give the benchmark in this repo
> a concrete workload to be adversarial *against*. No real company, product,
> or customer is described here. The tenants in the dataset (`nimbus_sec`,
> `acme_sat`, `granite_cap`) are invented, and the architecture below is a
> plausible-but-hypothetical design brief — it exists so that "where does this
> break?" is a question with a testable answer.

## The scenario

A SaaS platform helps vendors answer the security questionnaires, RFPs, and
due-diligence checklists that buyers send out during procurement. A single
questionnaire can run to 900 questions. The platform's value is that it
answers most of them automatically by retrieving the vendor's *own* previously
approved answers and letting an LLM adapt them to the current phrasing.

That makes retrieval the whole product. Every generated answer is downstream
of one retrieval call, so a retrieval miss is not a degraded result — it is a
wrong answer shipped to a customer's prospect, in a compliance document.

### Workload shape

| property | value | why it matters |
|---|---|---|
| Corpus | thousands of past answers per tenant | small enough to brute-force per tenant, large enough that ranking matters |
| Multi-tenancy | many vendors, one index | tenant A's answer to "what is your uptime SLA?" is *wrong* for tenant B, and lexically near-identical |
| Near-duplication | extreme | every vendor answers the same standard questionnaires; the corpus is near-duplicates by construction |
| Query mix | verbatim re-runs **and** paraphrases | the same standard question arrives worded identically one week and rephrased the next |
| Staleness | high | answers accumulate for years; last year's insurance limit is confidently wrong today |
| Answerability | not guaranteed | some questions have never been answered by this vendor; the system must decline rather than invent |
| Latency | interactive | a 900-question form must not take a day to draft |

### Non-negotiables

1. **Tenant isolation.** A cross-tenant document reaching the LLM context is a
   compliance failure, not a relevance miss. It cannot be left to ranking.
2. **Don't fool yourself.** Ranking changes must be justified by measurement on
   held-out data, with significance testing — not by in-sample improvement.
3. **Diagnose before optimising.** Each stage of the pipeline should be
   measurable in isolation, so a quality problem can be attributed to a stage.

## The target stack

This brief specifies a multi-phase retrieval pipeline:

- **Hybrid retrieval** — BM25 + semantic (dense vector) recall, fused.
- **Phased ranking** — cheap scoring over everything, graduating to
  progressively more expensive scoring over progressively fewer candidates,
  ending in cross-encoder reranking.
- **Tensor-based chunk selection** — documents are chunked in-schema; a tensor
  expression picks which chunks are worth putting in the LLM's context window.
- **Learned ranking in Python** — linear regression and LightGBM models trained
  offline on relevance judgments, deployed into the ranking phases.
- **Embedding compression** — binarisation/quantisation of vectors, with
  full-precision rescoring where it pays for itself.
- **An evaluation harness** — offline metrics, diagnostics, and per-stage
  attribution as first-class infrastructure.

## Why this maps onto the Vespa RAG Blueprint

Every item above is something the [Vespa RAG
Blueprint](https://docs.vespa.ai/en/learn/tutorials/rag-blueprint.html)
([vespa-engine/sample-apps](https://github.com/vespa-engine/sample-apps))
implements directly — which is what makes it the natural stack to deploy and
then attack:

| Requirement | Vespa RAG Blueprint feature |
|---|---|
| hybrid retrieval (BM25 + semantic) | `bm25(title)`/`bm25(chunks)` + ANN over chunk embeddings |
| phased ranking, cheap → cross-encoder | first-phase learned-linear → second-phase GBDT → global-phase cross-encoder |
| tensor-based chunk selection | chunk embeddings as `tensor<int8>(chunk{}, x[96])`; a `top()` tensor expression picks the best chunks per doc (`chunks_top3` summary) |
| linear regression + LightGBM, Python | the blueprint's training pipeline: logistic/linear first-phase weights, LightGBM second phase, pyvespa `VespaFeatureCollector` |
| embedding binarisation/quantisation | binary (`pack_bits`) embeddings with hamming-distance retrieval, float query vector for rescoring |
| schema design, cluster sizing, resource budgeting | Vespa's operational model |

The one thing the blueprint does **not** ship is a tenant concept — which
turns out to matter more than any of the ranking machinery (see
[`../benchmark/RESULTS.md`](../benchmark/RESULTS.md), finding 2).

References:
[RAG Blueprint tutorial](https://docs.vespa.ai/en/learn/tutorials/rag-blueprint.html) ·
[blueprint overview](https://vespa.ai/solutions/retrieval-augmented-generation/the-rag-blueprint/) ·
[phased ranking](https://docs.vespa.ai/en/ranking/phased-ranking.html) ·
[LightGBM ranking](https://docs.vespa.ai/en/ranking/lightgbm.html) ·
[working with chunks](https://docs.vespa.ai/en/rag/working-with-chunks.html)

## What this repo does with it

Deploy the blueprint as specified, build a benchmark deliberately shaped like
the workload above, and find out which parts of the target architecture
actually earn their cost. See [`../README.md`](../README.md) for results.
