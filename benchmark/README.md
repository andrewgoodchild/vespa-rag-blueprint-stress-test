# RFP Retrieval Benchmark

A small, adversarial benchmark for a multi-tenant RFP/questionnaire RAG pipeline,
modelled on the target architecture in [`../spec/target-architecture.md`](../spec/target-architecture.md)
(a fictional scenario written for this study).

> **Status: 13 experiments + 3 follow-ups executed** against the Vespa RAG
> Blueprint deployed locally (adapted app in `../vespa-app/`). Chronology in
> [../LABBOOK.md](../LABBOOK.md); full findings and per-arm scores in
> [RESULTS.md](RESULTS.md); raw TREC runs in `results/`.
>
> Headlines, small adversarial set: the blueprint's shipped LightGBM model
> degrades quality on a foreign corpus (nDCG 0.846 vs 0.950 semantic-only);
> pointwise retraining on 19 queries also loses (0.902) but **pairwise training
> on the same data wins (0.956)** — the objective, not the label count, was the
> bottleneck; tenant filtering removes all 49–61 leaked docs at zero relevance
> cost; a global freshness feature learns *negative* even pairwise; paragraph
> chunking (`docp` schema) fixes chunk selection completely while doc-level
> metrics barely move.
>
> Headlines, questionnaire corpus: matching the incoming question against the
> **stored question** rather than the answer is the biggest win (0.505 →
> **0.635** on paraphrases), and beats a cross-encoder costing ~300× more — which
> actively *hurts* when stacked on top (0.578). On verbatim queries the whole
> ranking stack is redundant: BM25 alone scores 0.957 and nothing above it helps.
> The hard tenant filter is worth +0.15 to +0.29 nDCG on its own.

## The architecture hypothesis

The target architecture maps almost one-to-one onto **Vespa's RAG Blueprint**
(`vespa-engine/sample-apps/rag-blueprint`):

| Requirement | Vespa RAG Blueprint feature |
|---|---|
| hybrid retrieval (BM25 + semantic search) | `bm25(title)/bm25(chunks)` + ANN over chunk embeddings |
| phased ranking that graduates from cheap scoring to expensive cross-encoder reranking | first-phase learned-linear → second-phase GBDT → global-phase cross-encoder |
| **tensor-based chunk selection** | chunk embeddings stored as `tensor<int8>(chunk{}, x[96])`; `top()` tensor expression picks best chunks per doc (`chunks_top3` summary) |
| linear regression, LightGBM, evaluation scripts (Python) | the blueprint's exact training pipeline: logistic/linear first-phase weights, LightGBM second phase, pyvespa `VespaFeatureCollector` |
| binarisation/quantisation of embeddings | blueprint's binary (pack_bits) embeddings with hamming-distance retrieval, float query for rescoring |
| cluster sizing, resource budgeting, schema design | Vespa's operational model |

So the stack under test: Vespa, documents = past RFP answers
chunked in-schema, hybrid recall, learned-linear first phase, LightGBM second
phase, cross-encoder global phase, tensor chunk selection to pick which chunks
go into LLM context.

References: [RAG Blueprint tutorial](https://docs.vespa.ai/en/learn/tutorials/rag-blueprint.html) ·
[blueprint overview](https://vespa.ai/solutions/retrieval-augmented-generation/the-rag-blueprint/) ·
[phased ranking](https://docs.vespa.ai/en/ranking/phased-ranking.html) ·
[LightGBM ranking](https://docs.vespa.ai/en/ranking/lightgbm.html) ·
[working with chunks](https://docs.vespa.ai/en/rag/working-with-chunks.html) ·
[sample-apps repo](https://github.com/vespa-engine/sample-apps)

## What the dataset stresses

39 synthetic documents across three fictional tenants spanning the scenario's
customer archetypes — `nimbus_sec` (SaaS vendor, 29 docs), `acme_sat` (satellite
operator, 7), `granite_cap` (fund manager, 3) — and 20 queries in clusters, each targeting a
known failure mode of this specific architecture:

| Cluster | Queries | Failure mode targeted |
|---|---|---|
| `lexical-exact` | q01, q02 | Acronyms/standards (SOC 2 Type II, IRAP) where semantic-only retrieval substitutes the nearest compliance doc; hybrid weighting test |
| `paraphrase` | q03–q05 | Vocabulary mismatch where BM25 favours the wrong doc (onboarding vs offboarding) |
| `hard-negative` | q06–q08 | Lexically near-identical wrong docs (at-rest vs in-transit; retention vs backup) — measures what the cross-encoder phase actually buys |
| `tenant-isolation` | q09 | Same question, three tenants, three different SLA numbers. Any cross-tenant doc returned = compliance-grade failure, scored separately |
| `freshness` | q10, q11 | Near-duplicate stale answers that are confidently wrong today (old data-centre list, old insurance limits) — tests recency features in ranking |
| `negation-trap` | q12, q13 | Correct answer is a negative statement; policy docs share all the query vocabulary but answer the wrong question |
| `chunk-selection` | q14, q15 | Answer buried in one paragraph of a long doc — directly stresses the tensor `top()` chunk selection and whole-doc score dilution |
| `multi-doc` | q16 | Complete answer needs two docs; tests top-k diversity |
| `near-duplicates` | q17 | Four boilerplate company overviews (two stale) — dedup/diversity; they also act as soft distractors for every other query |
| `domain-vocab` | q18, q20 | Specialist vocabulary (RFI/uplink; trade-lifecycle SoD) outside typical embedding-model training distribution; small-tenant recall |
| `unanswerable` | q19 | No relevant doc exists — score calibration / does the generator decline |

## Files

- `docs.jsonl` — one doc per line: `id, synthetic, tenant, doc_type, title, created, tags, text`.
  Long docs use `\n\n` paragraph breaks. `synthetic` is provenance metadata, not a
  schema field — drop it when building the feed payload. Map the rest to the
  blueprint schema:
  `text` → chunked field, `created` → `modified_timestamp`, `tenant` → a
  filterable attribute (query-time `tenant_id` filter — never rely on ranking for isolation).
- `queries.jsonl` — `query_id, tenant, cluster, query, tests` (what each query probes).
- `qrels.tsv` — TREC format (`qid 0 docid grade`), graded 0/1/2. Grade-0 rows are
  deliberate hard negatives worth tracking, not omissions.
- `qrels-chunks.tsv` — paragraph-level judgments for q14/q15 (chunk-selection).
- `evaluate.py` — stdlib-only scorer: Recall@k / MRR@k / nDCG@k per cluster,
  plus tenant-leakage and unanswerable-query reporting. Run file is TREC format.

```
python benchmark/evaluate.py run.txt --k 10
```

## The questionnaire corpus (Experiments 8–13)

The second corpus is 940 answers to a 261-question security questionnaire,
from four fictional vendors. All of it is committed — corpus, query sets and
qrels — so nothing needs downloading or rebuilding to reproduce the runs.

It is generated by [`build_synthetic_rfq_corpus.py`](build_synthetic_rfq_corpus.py),
which owns the whole chain: a 17-domain control taxonomy (197 controls), the
questions written against it, four vendors' answers to each, and the hard
paraphrase set. Regenerating needs `OPENROUTER_API_KEY` and a few hundred LLM
calls; the cached generations make a re-run free.

| file | what it is |
|---|---|
| `docs-rfq.jsonl` | 940 answers — `title` is the question, `text` the vendor's answer |
| | Every record carries a `synthetic` field. The vendors, products and answers are invented; nothing here is a real security attestation, and the field is metadata rather than a schema field (`feed_rfq.py` ignores it). |
| `queries-rfq.jsonl` | 232 verbatim queries: the stored question, asked exactly |
| `queries-rfq-para.jsonl` | 47 paraphrases — the hard case, reworded to share little vocabulary |
| `qrels-rfq*.tsv` | structural judgments: same-tenant same-question 2, sibling sub-question 1 |
| `distill_train.csv` | 47,000 cross-encoder-labelled pairs for Exp 13 |
| `gap2_pairs.jsonl` | 195 contrastive fine-tuning pairs for Gap 2 |

The corpus is built to be hard in the ways the workload is hard: every tenant
answers the *same* questions (near-duplicate by construction), answers name
their own vendor (so cross-tenant hits are detectable compliance failures),
coverage is uneven (258/214/234/234 of 261 — "never answered" is a real case),
and multi-part controls make sibling sub-questions graded near-misses.

To retarget it at your own vendors or your own question set, edit `TAXONOMY`
and `TENANTS` at the top of the builder and re-run.

> An earlier version of this study used a published industry security
> questionnaire answered by four real vendors. That question set is copyrighted
> and cannot be redistributed, so it was replaced with the corpus above and
> every Exp 8–13 number was re-measured. None were carried over.

## Reproducing the run

1. Start Vespa: `docker run --detach --name rag-blueprint --hostname rag-blueprint \
   --publish 127.0.0.1:8080:8080 --publish 127.0.0.1:19071:19071 vespaengine/vespa`
   (Docker VM needs ≥6GB).
2. Deploy the adapted app: zip `../vespa-app/` contents and POST to
   `http://127.0.0.1:19071/application/v2/tenant/default/prepareandactivate`.
   First startup downloads the ModernBERT embedder ONNX model (~2 min).
3. Feed: convert `docs.jsonl` to `{"put": "id:doc:doc::<id>", "fields": {...}}`
   with epoch timestamps (see `run_queries.py` header or RESULTS.md for the
   field mapping) and POST each to `/document/v1/doc/doc/docid/<id>`.
4. `python3 run_queries.py --outdir results` then
   `python3 evaluate.py results/<arm>.txt --k 10` per arm.

The per-cluster table shows exactly which pipeline stage each failure mode
needs — the operating philosophy ("diagnose before you optimise") applied to
the stack it specifies. See [RESULTS.md](RESULTS.md) for what this actually
surfaced: learned-model transfer failure, free tenant filtering, staleness
blindness, chunk-boundary splits, and near-duplicate confusion.

Caveat: judgments assume whole-doc retrieval with k=10; if you change chunking,
re-derive `qrels-chunks.tsv` indices.
