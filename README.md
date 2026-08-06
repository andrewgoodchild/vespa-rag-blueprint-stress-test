# vespa-rag-blueprint-stress-test — 15 experiments on multi-tenant RFP retrieval

An empirical study of a retrieval architecture: deploy the
[Vespa RAG Blueprint](https://docs.vespa.ai/en/learn/tutorials/rag-blueprint.html)
([vespa-engine/sample-apps](https://github.com/vespa-engine/sample-apps)),
point it at a workload built to break it, and measure which parts of the stack
actually earn their cost.

> **TL;DR.** The deployable recipe is two lines: a **hard tenant filter**,
> plus **match the incoming question against the stored question** (the
> "library model") — on the raw query, embedding **both sides with the query
> prefix** (it's a symmetric task) and keeping float title vectors. The
> embedder is the blueprint's own `nomic modernbert-embed-base`: a bake-off
> found that conditioning it symmetrically beat swapping in newer models
> (bge-m3, Qwen3) and a learned-sparse SPLADE arm entirely. That free
> config lifted the hard paraphrase case 0.505 → **0.703** nDCG@10 and covers
> a blended verbatim+paraphrase workload at **0.883**, best-in-study on both
> styles at once (so a query-style router buys nothing). Everything more expensive
> failed to confirm a gain over it: learned rankers tied zero-shot, the
> shipped cross-encoder was *harmful* (a modern one merely ties), fusing the
> best arms never beat the best arm, a served fine-tuned embedder couldn't
> catch the bigger zero-shot one, and the HyDE query rewrite that briefly
> held the record at 0.719 **failed out-of-sample replication** when the
> query set was scaled — while the symmetric-prefix fix, a one-line config
> change, passed that same bar at p<0.0001 and beat swapping in newer
> embedders outright. The recipe is also engine-portable: at fixed model
> and config, Qdrant serves it at identical quality (Gap 9) — and the
> document layer is cheap to cross: corpora parsed back out of messy
> PDF/DOCX/XLSX renderings retrieve identically to clean text (Gap 10).
> The rest of the repo is the evidence chain for those
> two lines, and the measurement harness that kept killing everything else.
> Biggest caveat: the corpus is LLM-generated and its difficulty is a dial
> we set — validate the magnitudes on real data before trusting them.

## Why Vespa?

Most RAG stacks are a vector index plus application code. Vespa is a tensor-based
search engine: a document holds many vectors, and ranking is an expression the
engine evaluates next to the data. Lexical and semantic retrieval run in one
query, reranking is phased inside the engine, and models run there too — so
techniques that would be separate services elsewhere are a change to a text
file here. That makes it a good thing to experiment on.

Gap 9 closes the question from the other side: the *winning* recipe, held at
fixed model and config, serves at identical quality from Qdrant — the engine
mattered for the 15-experiment search, not for the destination it found.

## Test cases

The workload is a **fictional test case**
([`spec/target-architecture.md`](spec/target-architecture.md)): a multi-tenant
platform that answers security questionnaires and RFPs by retrieving a vendor's
own past approved answers. It's a good adversary for this architecture because
the corpus is near-duplicates by construction — every vendor answers the same
standard questionnaires — and a cross-tenant retrieval is a compliance failure,
not just a bad result.

The study runs on two datasets, both authored here and both committed — clone
and everything reproduces with no downloads.

**Small adversarial set** (Exps 1–7): 39 LLM-written multi-tenant docs, built
as a unit test for known failure modes — one query cluster per documented way
this architecture breaks.

**Questionnaire corpus** (Exps 8–15): 940 answers to a 261-question security
questionnaire, from four fictional vendors — Tanager Geospatial, Kestrel Cloud,
Orrery Software and Pellucid Data. The control taxonomy (17 domains, 197
controls) is ours, the questions are written to it, and the answers are
generated per vendor by [`build_synthetic_rfq_corpus.py`](benchmark/build_synthetic_rfq_corpus.py).

It is built to be hard the way the real workload is hard: every tenant answers
the *same* question set, so the corpus is near-duplicate by construction;
answers name their own vendor, so a cross-tenant hit is a real compliance
failure; coverage is uneven, so "this tenant never answered that" is a genuine
case; and multi-part controls make a sibling sub-question a graded near-miss
rather than a clean negative.

> **On provenance.** An earlier version ran on a published industry
> questionnaire answered by four real vendors. Its question set is copyrighted
> and may not be redistributed, so it was replaced by clean-rooming the public
> material into the corpus above — all vendor names fictional — and **every
> number was re-measured**. Nothing is carried over.
>
> The replacement matches the original on the structural properties that drive
> the findings, but it is LLM-generated and *its difficulty is a dial we set*:
> calibrating it took three attempts. That is the study's largest caveat, ahead
> of anything in the results — see
> [RESULTS.md limitation 1](benchmark/RESULTS.md).

## What's here

| path | contents |
|---|---|
| [`spec/target-architecture.md`](spec/target-architecture.md) | The fictional scenario and target stack, mapped onto blueprint features |
| [`LABBOOK.md`](LABBOOK.md) | Chronological lab notebook — question, design, results, conclusions per experiment (Exps 1–15) |
| [`benchmark/RESULTS.md`](benchmark/RESULTS.md) | Full findings (1–46) with per-query diagnostics |
| [`benchmark/`](benchmark/README.md) | Both datasets in full — 39 adversarial docs and the 940-answer questionnaire corpus — plus query sets, graded TREC qrels, scorer, runners, training scripts and every raw run |
| [`vespa-app/`](vespa-app/) | The blueprint app adapted for local Docker: tenant field, stage-isolation profiles, paragraph-chunked `docp` and real-data `rfq` schemas, cross-encoder + ColBERT + RRF rank profiles |
| [`deck/`](deck/) | `deck.html` — a 32-slide deck covering what was tested, the technique behind each stage, and what the measurements showed |

## Headline results

**Synthetic 80-query set** (nDCG@10, grouped CV where trained): the durable
gains were all structural — hard tenant filter **+0.035 (p<0.001)**,
dedup-freshness rerank **+0.011 (p≈0.011)** — while every learned ranker tied
zero-shot semantic. Best pipeline **0.921** (pairwise + tenant filter +
dedup-freshness), vs 0.874 for the unfiltered blueprint hybrid.

**Questionnaire paraphrase set** (47 queries, nDCG@10, tenant-filtered — the
hard case: a buyer asks in their own words, the library holds the standard
wording):

| technique | nDCG@10 | query cost |
|---|---|---|
| BM25 (lexical) | 0.279 | fast |
| blueprint hybrid | 0.436 | fast |
| RRF fusion (untrained) | 0.489 | fast |
| SPLADE learned sparse, q2q (offline, Gap 8) | 0.496 | fast |
| semantic (embed the answer) | 0.505 | fast |
| ColBERT late interaction | 0.517 | ~60 ms/q |
| cross-encoder rerank | 0.561–0.578 | ~2.5 s/q |
| HyDE rewrite into question space + original query | 0.719 | ~2 s/q |
| q2q-semantic (embed the question — the "library model") | 0.635 | fast |
| **library model + symmetric prefixes & float titles (Gap 7)** | **0.703** | fast |

The cheapest arm wins. Matching the incoming question against the *stored
question* rather than the stored answer beats every reranker tested, including
a cross-encoder costing ~300× more per query — and the cross-encoder applied on
top of it makes things *worse* (0.578 vs 0.635), because reranking on answer
text reintroduces exactly the near-duplicate confusion the library model
avoids.

The one arm that outscored it — a HyDE-style LLM rewrite of the query into the
standard questionnaire wording, concatenated with the original (0.719) — was
then put through the study's own discipline: scaling the paraphrase set 47 → 232
(Exp 15) attenuated its edge to **+0.023 out-of-sample (p≈0.24), unconfirmed**,
while the library model's wins over the cross-encoder and the distilled student
replicated with significance on 232 queries (p≤0.001). What the rewrite did
prove: at the same ~2 s/query cost it beats the cross-encoder outright
(p≈0.001), and rewriting *without* keeping the original query is actively
harmful (−0.073, p≈0.003).

On verbatim re-runs the ordering inverts completely: BM25 scores **0.957**,
ahead of semantic's 0.915, because the query *is* the stored question. In fact
the whole ranking stack is redundant there — hybrid 0.963, library model 0.961,
all three cross-encoder variants 0.960–0.961. A term-overlap score is already at
ceiling and nothing above it buys anything measurable. Hybrid retrieval is
forced by the workload mixing both styles, not chosen for elegance.

The single largest intervention is not a model at all. The hard tenant filter
is worth **+0.149 to +0.294 nDCG** depending on arm, and removes ~350
cross-tenant leaks per arm.

## What the study found

1. **Learned rankers don't transfer.** The blueprint's shipped LightGBM model
   — trained on its own demo corpus — is the *worst* configuration tested,
   demoting correct answers and resurfacing stale ones.
2. **…and retraining isn't automatically the fix.** Pointwise logistic
   regression on our own labels loses to or ties zero-shot semantic search at
   every scale tested, while looking great in-sample — a measured ~5-point
   overfitting illusion on the small set.
3. **Scale killed a seductive finding.** On 19 queries, pairwise training
   "beat" zero-shot (0.956 vs 0.950); at 69 queries the difference is p≈0.98.
   Pairwise remains the better objective (stabler coefficients, never worse
   than pointwise) — but no learned ranker beat zero-shot in this study.
4. **Tenant isolation is a relevance feature, not just compliance.** The hard
   filter removed 219 leaked cross-tenant docs *and* was the single biggest
   quality intervention (+0.035, p<0.001): other tenants' near-matches
   actively crowd out correct answers.
5. **Freshness is not a ranking feature.** Even pairwise, a global freshness
   term learns a *negative* weight. Staleness only discriminates between
   near-duplicates on the same topic — dedup-then-prefer-fresh implements
   exactly that and replicates significantly (+0.011, p≈0.011).
6. **Chunking quality is invisible to doc-level metrics.** Fixed 1024-char
   chunks split answers mid-sentence and made generic intro text a cosine
   attractor; paragraph chunks fixed both. Only chunk-level judgments
   (`qrels-chunks.tsv`) detect the difference — and the chunks are what the
   LLM actually reads.
7. **Retrieval scores can't fully detect "no answer exists" (AUC 0.884).**
   The hard failures are right-topic-missing-fact; abstention needs
   generation-stage answerability verification on top of a score threshold.
8. **Match the question, not the answer.** On the questionnaire corpus the
   biggest retrieval win is changing *what you index against*: matching an
   incoming question to the stored **question** lifts 0.505 → **0.635**, with
   the same embedder and no training. This is why real RFP tools are built
   around curated answer libraries rather than raw-document RAG.
9. **The expensive reranker is not just unnecessary — it is harmful here.**
   A cross-encoder over the library model scores 0.578 vs 0.635 without it, at
   ~300× the query cost. Reranking on answer text reintroduces the
   near-duplicate confusion that matching on questions avoids. Retrieve on the
   right field and the reranker has nothing left to add. *(Later narrowed by a
   follow-up: the harm is specific to the shipped 2020-era model — a modern
   reranker is at-or-above the library model, but still never significantly
   above it. "Harmful" retracted; "doesn't earn its cost" confirmed.)*
10. **Distillation beats its own teacher — and reversed a result this study
    had confirmed three times.** A cross-encoder teacher labelled 47,000 pairs;
    a distilled student then *outscored* it, 0.604 vs 0.561, at microsecond
    cost. But the winner was the **pointwise** student, where Exps 2 and 4 both
    found pointwise losing to pairwise (0.553 here). The comparison confounds
    objective with model class — pointwise is a gradient-boosted tree, pairwise
    a linear model — so the honest reading is that the objective advantage is
    real but smaller than claimed, and model capacity can outweigh it. The
    three-times-confirmed version of this finding was over-stated.
11. **Fine-tuning the embedder beats fine-tuning the ranker.** Contrastive
    fine-tuning on 195 synthesized paraphrase pairs moved MiniLM 0.591 →
    **0.622** (+0.031) — attacking the representation, which is where the
    failure actually was, while every learned *ranker* tied or lost.
12. **Every durable improvement came from measurement the stack doesn't
    ship:** leakage accounting, grouped cross-validation, paired bootstrap,
    chunk-level qrels, a scaled query set. The evaluation harness, not the
    ranking model, was the high-leverage artifact.
13. **Query rewriting (HyDE) validated the thesis, then obeyed the house
    rule.** Rewriting the query into *answer* space never competes; rewriting
    it into *question* space with the original kept alongside produced the
    study's best single number (0.719 on 47 queries) — which attenuated to an
    unconfirmed +0.023 (p≈0.24) on 185 out-of-sample paraphrases. The
    survivors, all significant at scale: the rewrite beats the equal-cost
    cross-encoder, discarding the original query is actively harmful, and the
    library model's edge over every reranker is confirmed on 232 queries.
14. **Three follow-ups simplified the deployment story further.** On a
    464-query verbatim+paraphrase blend, the library model alone scores 0.850
    and an oracle per-style router adds +0.001 — no router needed. RRF-fusing
    the best arms never beats the best arm. And serving the fine-tuned small
    embedder showed its +0.023 cannot close the −0.062 gap to the larger
    zero-shot embedder — pick the strongest base embedder before fine-tuning
    anything. Serve the library model, alone, on the raw query, with the best
    embedder you have.
15. **The library model was leaving free, significant accuracy on the
    table: match a symmetric task symmetrically.** The blueprint embeds
    queries and documents with different prefixes — but in the library model
    both sides are questions. Embedding the stored question with the *query*
    prefix is worth +0.039 (p<0.0001, identical out-of-sample — the first
    intervention since the tenant filter to clear that bar), and float title
    vectors add +0.023 over binarized. Combined: 0.635 → **0.703** on the
    hard case, 0.965 verbatim, 0.883 blended — best-in-study on all three at
    zero query cost, strictly dominating the HyDE rewrite. An embedder
    bake-off confirmed the config mattered more than the model: symmetric
    nomic beat bge-m3 and Qwen3-0.6B (instructed or not) outright.
16. **Learned sparse retrieval doesn't rescue lexical here.** SPLADE (Gap 8,
    offline) halves BM25's paraphrase collapse (0.279 → 0.496) but loses to
    every dense arm by wide, significant margins (−0.222 vs the symmetric
    library model, p<0.0001) and finds no refuge at the verbatim ceiling
    (0.958 vs 0.966). Match-the-question replicates in its third retrieval
    family (+0.109, p<0.0001): on a near-duplicate corpus the discriminating
    signal is phrasing nuance — dense geometry's home turf, not term
    weighting's.
17. **The engine didn't matter; the recipe did.** Fed the same embeddings
    and config (Gap 9), Qdrant reproduces the offline scan identically on
    all 464 queries, shows zero ANN-under-filter loss, and ties served
    Vespa (+0.003; only 19/232 queries differ at all — the embedding path,
    not the index). The deployable recipe is portable to any vector store
    with payload filtering; what Vespa bought was the *exploration* — the
    15-experiment search ran as rank-profile edits in one engine.
18. **The document layer is cheap to cross — and retrieval metrics are
    blind to what it breaks.** Rendering the corpus into four flavours of
    messy vendor documents (page-split PDF tables, DOCX callout boxes,
    SIG-style XLSX, two-column prose with no IDs) and parsing it back
    (Gap 10): format-aware parsing retrieves *query-for-query identically*
    to clean text; a naive text-dump loses only where whole units vanish
    (−0.014, all from 16 wrapped table refs). But naive also returned 31
    answers as empty text at zero nDCG cost — the library model matches on
    the question, so answer-side parse damage surfaces only at generation.
    Ingestion QA needs its own fidelity metric; no retrieval metric will
    raise the alarm. (Digital text layer only — scans/OCR untested.)

## Reproducing

Requires Docker (≥6GB VM memory) and Python 3.10+ with scikit-learn.

```bash
# 1. Fetch the two ONNX rerankers (not in git — see vespa-app/models/README.md)
(cd vespa-app/models \
  && curl -sL -o colbert.onnx 'https://huggingface.co/colbert-ir/colbertv2.0/resolve/main/model.onnx' \
  && curl -sL -o cross_encoder.onnx 'https://huggingface.co/Xenova/ms-marco-MiniLM-L-6-v2/resolve/main/onnx/model.onnx')

# 2. Start Vespa and deploy the app
docker run --detach --name rag-blueprint --hostname rag-blueprint \
  --publish 127.0.0.1:8080:8080 --publish 127.0.0.1:19071:19071 vespaengine/vespa
(cd vespa-app && zip -qr /tmp/app.zip . && curl -s -H "Content-Type: application/zip" \
  --data-binary @/tmp/app.zip http://127.0.0.1:19071/application/v2/tenant/default/prepareandactivate)
# first deploy downloads the ModernBERT embedder ONNX (~2 min)

# 3. Feed docs (both schemas), run arms, score
#    see benchmark/README.md for the feed field mapping
python3 benchmark/run_queries.py --outdir runs
python3 benchmark/evaluate.py runs/hybrid.txt --k 10

# 4. Questionnaire corpus (Exps 8-15): feed, run every arm, score them all
python3 benchmark/feed_rfq.py --purge
python3 benchmark/run_rfq.py --queries queries-rfq-para.jsonl --outdir benchmark/results/exp8
python3 benchmark/score_rfq.py          # the tables in RESULTS.md

# 5. Retraining, distillation and fine-tuning experiments
python3 benchmark/train_linear.py     --outdir benchmark/results/exp2
python3 benchmark/train_pairwise.py   --outdir benchmark/results/exp4
python3 benchmark/distill_collect.py  # ~40 min: teacher-labels 47k pairs
python3 benchmark/distill_pairwise.py # trains + writes student_expr.json
python3 benchmark/gap2_finetune.py    # contrastive fine-tune, offline

# 6. HyDE query rewriting + the scaled paraphrase set (Exps 14-15)
#    the generated query files are committed, so these two need no API key:
python3 benchmark/run_rfq.py --queries queries-rfq-para-all-hyde-q-cat.jsonl \
  --tag para-all-hydeqcat --arms q2q-semantic --outdir benchmark/results/exp15
python3 benchmark/run_rfq.py --queries queries-rfq-para-all.jsonl \
  --tag para-all --outdir benchmark/results/exp15
#    regenerating the rewrites / scaling further needs OPENROUTER_API_KEY:
#    hyde_generate.py --queries <set>.jsonl, scale_para.py

# 7. Learned sparse (Gap 8): SPLADE vs dense, offline — downloads the
#    SPLADE++ checkpoint (~450 MB) on first run, no Vespa needed
python3 benchmark/gap8_splade.py

# 8. Engine bake-off (Gap 9): the winning recipe served from Qdrant
docker run -d --name qdrant-bakeoff -p 127.0.0.1:6333:6333 qdrant/qdrant
python3 benchmark/gap9_qdrant.py

# 9. The document layer (Gap 10): render messy PDF/DOCX/XLSX vendor docs,
#    parse them back three ways, score retrieval over what survived
pip install reportlab openpyxl pymupdf pdfplumber python-docx
python3 benchmark/gap10_render_docs.py
python3 benchmark/gap10_parse.py --arms naive,structured
python3 benchmark/gap10_eval.py
#    the docling arm downgrades transformers — run it from its own venv:
#    python -m venv /tmp/dv && /tmp/dv/bin/pip install docling
#    /tmp/dv/bin/python benchmark/gap10_parse.py --arms docling
```

The questionnaire corpus is committed, so steps 4–5 need no data preparation.
To rebuild or retarget it — your own vendors, your own questions — see
[`build_synthetic_rfq_corpus.py`](benchmark/build_synthetic_rfq_corpus.py)
(regenerating costs a few hundred LLM calls and needs `OPENROUTER_API_KEY`).

Both datasets use fictional companies and are sized so a laptop run takes
minutes. Every query cluster in the small set targets a specific, documented
failure mode of this architecture; the questionnaire corpus is adversarial in a
different way — near-duplicate by construction, because every tenant answers
the same questions.

## License

[Apache-2.0](LICENSE) — the same licence as the Vespa RAG Blueprint
([sample-apps](https://github.com/vespa-engine/sample-apps)) that `vespa-app/` is
adapted from, so one licence covers the tree. Twelve upstream files are committed
unmodified and two are modified; those carry Vespa.ai's copyright notices, and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) lists each one and what
changed. The committed ColBERT tokenizer is MIT.

The ONNX weights (ColBERT, MiniLM cross-encoder) and the ModernBERT embedder are
fetched at setup, not redistributed, each under its upstream terms.
