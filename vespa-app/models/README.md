# Models

The tokenizer JSONs and the LightGBM ranker are committed. The two ONNX weight
files are **not** — they're downloaded artifacts, not authored source, and at
~520 MB combined they don't belong in git. Fetch them before deploying:

```bash
cd vespa-app/models

# ColBERT v2 — late-interaction reranker (Exp 11), ~416 MB
curl -sL -o colbert.onnx \
  'https://huggingface.co/colbert-ir/colbertv2.0/resolve/main/model.onnx'

# MiniLM cross-encoder — global-phase reranker (Exps 9–12), ~87 MB
curl -sL -o cross_encoder.onnx \
  'https://huggingface.co/Xenova/ms-marco-MiniLM-L-6-v2/resolve/main/onnx/model.onnx'
```

Their tokenizers (`colbert_tok.json`, `ce_tokenizer.json`) are already here, so
no further setup is needed — zip and deploy `vespa-app/` as normal.

| file | in git | source |
|---|---|---|
| `colbert.onnx` | no | `colbert-ir/colbertv2.0` |
| `cross_encoder.onnx` | no | `Xenova/ms-marco-MiniLM-L-6-v2` |
| `colbert_tok.json` | yes | same repo, `raw/main/tokenizer.json` |
| `ce_tokenizer.json` | yes | same repo, `resolve/main/tokenizer.json` |
| `lightgbm_model.json` | yes | trained here (`benchmark/train_*.py`) |

The ModernBERT embedder (`nomic modernbert-embed-base`) is not in this
directory at all — Vespa downloads it in-container on first deploy, per the
`services.xml` model reference.
