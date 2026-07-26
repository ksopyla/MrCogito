#!/usr/bin/env python3
"""STS-B zero-shot semantic probe for `backbone_concept` checkpoints (E10/E16 family).

The generic `evaluate_on_benchmark.py` sentence-pair route is built around
`ConceptEncoder` + `ConceptEncoderConfig` and does not yet accept
`BackboneConceptLM`. This script is the Tier-2 probe for that family:

  sentence → block-recurrent encode_concepts → mean-pool C concepts → cosine

Also reports Gemma trivial floors (token-embed mean, teacher-hidden mean) so the
Pearson number is interpretable. Not a registered E16b gate — mechanism Tier-1
already closed; this answers "do used concepts carry transferable similarity?".

Usage:
  uv run python evaluation/evaluate_backbone_concept_stsb_zero_shot.py \\
    --model_path Cache/Training/<run>/checkpoint-7900 \\
    --source_training_run_id <run_id>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, default_data_collator

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nn.backbone_concept_lm import BackboneConceptLM

REPORTS_DIR = Path("Cache/Evaluation_reports")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_path", required=True)
    p.add_argument("--tokenizer_name", default=None)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--source_training_run_id", default=None)
    p.add_argument("--skip_floors", action="store_true")
    p.add_argument("--output_json", default=None)
    return p.parse_args()


def _tokenize_pairs(dataset, tokenizer, max_length):
    def preprocess(examples):
        a = tokenizer(
            examples["sentence1"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        b = tokenizer(
            examples["sentence2"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        return {
            "input_ids_a": a["input_ids"],
            "attention_mask_a": a["attention_mask"],
            "input_ids_b": b["input_ids"],
            "attention_mask_b": b["attention_mask"],
            "labels": [float(x) for x in examples["label"]],
        }

    return dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset.column_names,
        desc="tokenize STS-B",
    )


@torch.no_grad()
def _mean_pool_concepts(model, input_ids, attention_mask):
    out = model.encode_concepts(input_ids=input_ids, attention_mask=attention_mask)
    concepts = out.last_hidden_state  # [B, C, H]
    return concepts.mean(dim=1)


@torch.no_grad()
def _score_cosine(encode_fn, dataloader, device):
    preds, labels = [], []
    for batch in dataloader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        z_a = encode_fn(batch["input_ids_a"], batch["attention_mask_a"])
        z_b = encode_fn(batch["input_ids_b"], batch["attention_mask_b"])
        cos = F.cosine_similarity(z_a.float(), z_b.float(), dim=-1)
        preds.append(cos.cpu())
        labels.append(batch["labels"].cpu())
    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    return {
        "pearsonr": float(pearsonr(preds, labels)[0]),
        "spearmanr": float(spearmanr(preds, labels)[0]),
    }


def _floor_encode(teacher, embedding, variant):
    def encode(input_ids, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        if variant == "token_embed_mean":
            h = embedding(input_ids)
        else:
            h = teacher(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)

    return encode


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading BackboneConceptLM from {args.model_path}")
    model = BackboneConceptLM.from_pretrained(args.model_path)
    model = model.to(device).eval()
    tok_name = args.tokenizer_name or getattr(model.config, "tokenizer_name", None) or model.config.backbone_model
    tokenizer = AutoTokenizer.from_pretrained(tok_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token

    raw = load_dataset("glue", "stsb")
    eval_ds = _tokenize_pairs(raw["validation"], tokenizer, args.max_length)
    loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=default_data_collator,
    )

    print("=== concept mean-pool cosine ===")
    concept_scores = _score_cosine(
        lambda ids, mask: _mean_pool_concepts(model, ids, mask),
        loader,
        device,
    )
    for k, v in concept_scores.items():
        print(f"  {k}: {v:.4f}")

    floors = {}
    if not args.skip_floors:
        backbone_name = model.config.backbone_model
        print(f"=== trivial floors on {backbone_name} ===")
        teacher = AutoModel.from_pretrained(backbone_name).to(device).eval()
        emb = teacher.get_input_embeddings()
        for variant in ("token_embed_mean", "teacher_hidden_mean"):
            floors[variant] = _score_cosine(
                _floor_encode(teacher, emb, variant),
                loader,
                device,
            )
            print(f"  [{variant}] pearson={floors[variant]['pearsonr']:.4f} "
                  f"spearman={floors[variant]['spearmanr']:.4f}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_id = args.source_training_run_id or Path(args.model_path).parent.name
    ckpt = Path(args.model_path).name
    out = {
        "model_path": args.model_path,
        "run_id": run_id,
        "checkpoint": ckpt,
        "tokenizer": tok_name,
        "max_length": args.max_length,
        "pool": "concept_mean",
        "concept_stsb_zero_shot": concept_scores,
        "floors": floors,
        "delta_vs_token_embed_mean": (
            concept_scores["pearsonr"] - floors["token_embed_mean"]["pearsonr"]
            if "token_embed_mean" in floors else None
        ),
        "delta_vs_teacher_hidden_mean": (
            concept_scores["pearsonr"] - floors["teacher_hidden_mean"]["pearsonr"]
            if "teacher_hidden_mean" in floors else None
        ),
    }
    out_path = Path(args.output_json) if args.output_json else (
        REPORTS_DIR / f"e16b_{run_id}_{ckpt}_stsb_zero_shot_{stamp}.json"
    )
    out_path.write_text(json.dumps(out, indent=2))
    pd.DataFrame([{
        "run_id": run_id,
        "checkpoint": ckpt,
        "pearsonr": concept_scores["pearsonr"],
        "spearmanr": concept_scores["spearmanr"],
        "token_embed_mean_pearson": floors.get("token_embed_mean", {}).get("pearsonr"),
        "teacher_hidden_mean_pearson": floors.get("teacher_hidden_mean", {}).get("pearsonr"),
    }]).to_csv(out_path.with_suffix(".csv"), index=False)
    print(f"Wrote {out_path}")
    print("__BACKBONE_STSB_EXIT__=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
