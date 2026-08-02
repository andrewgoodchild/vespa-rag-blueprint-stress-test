#!/usr/bin/env python3
"""Gap 5 step 1: train the Gap 2 fine-tune reproducibly, save it, and export
base + tuned MiniLM to ONNX so Vespa can serve the library model with them.

Gap 2 measured the contrastive fine-tune offline only (and never saved the
model). This script re-runs that training with a fixed seed, saves the
checkpoint, re-reports the offline base/tuned numbers on the held-out 47
paraphrases, and exports both models for the app:

  ../vespa-app/models/minilm_base.onnx
  ../vespa-app/models/minilm_ft.onnx
  ../vespa-app/models/minilm_tokenizer.json

The ONNX takes input_ids/attention_mask/token_type_ids and emits
last_hidden_state — Vespa's hugging-face-embedder defaults (mean pooling)
match sentence-transformers' MiniLM pooling. Eval of the served arms must use
the ORIGINAL 47 paraphrases only: the training pairs are rewordings of the
non-eval question_ids, which the Exp 15 scaled set overlaps.

  python3 gap5_export_minilm.py
"""
import json
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from gap2_finetune import BASE, DEVICE, evaluate, load_eval

HERE = Path(__file__).parent
CKPT = HERE / "checkpoints" / "model" / "ft-minilm"
MODELS = HERE.parent / "vespa-app" / "models"


class _Encoder(torch.nn.Module):
    """Pin the ONNX signature to exactly the three inputs Vespa sends."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.model(input_ids=input_ids, attention_mask=attention_mask,
                          token_type_ids=token_type_ids).last_hidden_state


def export_onnx(src, out):
    model = _Encoder(AutoModel.from_pretrained(src))
    model.eval()
    tok = AutoTokenizer.from_pretrained(src)
    enc = tok(["a dummy question"], return_tensors="pt")
    args = (enc["input_ids"], enc["attention_mask"], enc["token_type_ids"])
    dyn = {0: "batch", 1: "seq"}
    torch.onnx.export(
        model, args, str(out),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["last_hidden_state"],
        dynamic_axes={"input_ids": dyn, "attention_mask": dyn,
                      "token_type_ids": dyn, "last_hidden_state": dyn},
        opset_version=17, dynamo=False)
    print(f"exported {out.name} ({out.stat().st_size // 1_000_000} MB)")


def main():
    torch.manual_seed(42)
    docs, queries, qrels = load_eval()
    pairs = [json.loads(l) for l in open(HERE / "gap2_pairs.jsonl")]

    base = SentenceTransformer(BASE, device=DEVICE)
    print(f"BASE offline q2q nDCG@10 = {evaluate(base, docs, queries, qrels):.3f}")

    if (CKPT / "model.safetensors").exists():
        model = SentenceTransformer(str(CKPT), device=DEVICE)
        print(f"reusing checkpoint {CKPT}")
    else:
        model = SentenceTransformer(BASE, device=DEVICE)
        examples = [InputExample(texts=[p["anchor"], p["positive"]]) for p in pairs]
        loader = DataLoader(examples, batch_size=32, shuffle=True)
        loss = losses.MultipleNegativesRankingLoss(model)
        model.fit(train_objectives=[(loader, loss)], epochs=3,
                  warmup_steps=int(0.1 * len(loader) * 3), show_progress_bar=False)
        CKPT.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(CKPT))
        print(f"saved checkpoint to {CKPT}")
    print(f"TUNED offline q2q nDCG@10 = {evaluate(model, docs, queries, qrels):.3f}")

    export_onnx(BASE, MODELS / "minilm_base.onnx")
    export_onnx(str(CKPT), MODELS / "minilm_ft.onnx")
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.backend_tokenizer.save(str(MODELS / "minilm_tokenizer.json"))
    print("saved tokenizer")


if __name__ == "__main__":
    main()
