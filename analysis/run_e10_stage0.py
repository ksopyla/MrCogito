#!/usr/bin/env python
"""E10 Stage 0 — measure the long-range CE gap G BEFORE any training (the spec's go/no-go gate).

Scores held-out long documents with the UNTRAINED graft wrapper (concept_num=0, no LoRA) in
two modes:
  * full_attention — intact backbone (global layers see everything): the upper baseline.
  * blockwise      — the E10 training protocol without concepts: block-recurrent forwards
                     with a one-block carry, every layer window-masked. History is HARD
                     truncated at carry+block (~2K tokens back at K=512); note a single
                     window-masked forward would NOT measure this (stacked SWA layers widen
                     the receptive field by ~(W-1) per layer, ≈13K tokens for 26 layers).
The per-position-bucket difference (blockwise − full) is the gap G the concepts must close.
Spec gate: G >= 0.05 nats averaged over positions >= 1024 at seq 2048, else re-scope before
training. Spec: docs/experiments_specs/E10_gemma_backbone_concept_memory.md

Usage (GPU server):
  uv run python analysis/run_e10_stage0.py --seq_lens 2048 8192 --num_docs 64 --output Cache/e10_stage0.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from datasets import load_dataset            # noqa: E402
from transformers import AutoTokenizer       # noqa: E402

from nn.backbone_concept_lm import BackboneConceptConfig, BackboneConceptLM  # noqa: E402

BUCKETS = [(0, 512), (512, 1024), (1024, 2048), (2048, 4096), (4096, 8192)]


def collect_docs(tokenizer, seq_len: int, num_docs: int, dataset: str, subset: str):
    """Stream docs, keep those with >= seq_len tokens, truncate to seq_len."""
    ds = load_dataset(dataset, name=subset, split="train", streaming=True)
    rows = []
    for ex in ds:
        ids = tokenizer(ex["text"], truncation=True, max_length=seq_len)["input_ids"]
        if len(ids) >= seq_len:
            rows.append(ids[:seq_len])
        if len(rows) >= num_docs:
            break
    if len(rows) < num_docs:
        print(f"WARNING: only {len(rows)} docs with >= {seq_len} tokens found")
    return torch.tensor(rows, dtype=torch.long)


@torch.no_grad()
def score(model, input_ids, mode: str, batch_size: int, device) -> torch.Tensor:
    out = []
    for s in range(0, input_ids.shape[0], batch_size):
        batch = input_ids[s : s + batch_size].to(device)
        out.append(model.per_position_ce(batch, mode=mode).cpu())
    return torch.cat(out, dim=0)   # [N, S]


def bucket_means(pos_ce: torch.Tensor) -> dict:
    res = {}
    for lo, hi in BUCKETS:
        if lo >= pos_ce.shape[1]:
            continue
        seg = pos_ce[:, lo : min(hi, pos_ce.shape[1])]
        if not torch.isnan(seg).all():
            res[f"[{lo},{min(hi, pos_ce.shape[1])})"] = float(seg.nanmean().item())
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="google/gemma-3-1b-pt")
    p.add_argument("--seq_lens", type=int, nargs="+", default=[2048, 8192])
    p.add_argument("--num_docs", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset_subset", default="sample-10BT")
    p.add_argument("--concept_block", type=int, default=512)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    config = BackboneConceptConfig(
        backbone_model=args.backbone,
        concept_num=0,                    # measurement uses the bare backbone
        concept_block=args.concept_block,
        lora_r=0,
    )
    print(f"Loading {args.backbone} (dtype={dtype}, device={device})")
    model = BackboneConceptLM.from_pretrained_backbone(config, dtype=dtype)
    model.to(device).eval()

    report = {"backbone": args.backbone, "num_docs": args.num_docs, "seq_lens": {}}
    for seq_len in args.seq_lens:
        print(f"\n=== seq_len {seq_len}: collecting {args.num_docs} docs ===")
        input_ids = collect_docs(tokenizer, seq_len, args.num_docs, args.dataset, args.dataset_subset)
        ce_full = score(model, input_ids, "full_attention", args.batch_size, device)
        ce_win = score(model, input_ids, "blockwise", args.batch_size, device)

        full_b, win_b = bucket_means(ce_full), bucket_means(ce_win)
        beyond = slice(1024, seq_len)
        gap_beyond = float((ce_win[:, beyond].nanmean() - ce_full[:, beyond].nanmean()).item())
        entry = {
            "full_attention": full_b,
            "windowed": win_b,
            "gap_per_bucket": {k: round(win_b[k] - full_b[k], 4) for k in full_b if k in win_b},
            "G_beyond_1024": round(gap_beyond, 4),
        }
        report["seq_lens"][str(seq_len)] = entry

        print(f"{'bucket':>14} {'full':>8} {'windowed':>9} {'gap':>7}")
        for k in full_b:
            print(f"{k:>14} {full_b[k]:8.4f} {win_b.get(k, float('nan')):9.4f} "
                  f"{win_b.get(k, float('nan')) - full_b[k]:7.4f}")
        verdict = "GO (>= 0.05)" if gap_beyond >= 0.05 else "NO-GO (< 0.05) — re-scope seq/eval"
        print(f"G (positions >= 1024) = {gap_beyond:.4f} nats → {verdict}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
