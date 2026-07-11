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
from pathlib import Path

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from datasets import load_dataset            # noqa: E402
from transformers import AutoTokenizer       # noqa: E402

from nn.backbone_concept_lm import BackboneConceptConfig, BackboneConceptLM  # noqa: E402
from analysis.run_e10_comparison import load_eval_rows  # noqa: E402

BUCKETS = [(0, 512), (512, 1024), (1024, 2048), (2048, 4096), (4096, 8192),
           (8192, 16384), (16384, 32768)]


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
    p.add_argument("--seq_lens", type=int, nargs="+", default=[2048, 4096, 8192, 16384])
    p.add_argument("--num_docs", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset_subset", default="sample-10BT")
    p.add_argument(
        "--eval_manifest",
        default=None,
        help="Frozen pretokenized eval-only manifest. When set, all sequence lengths use "
             "the same long documents truncated to length (paired and train-disjoint).",
    )
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
    paired_rows = None
    if args.eval_manifest:
        paired_rows = load_eval_rows(
            Path(args.eval_manifest), max(args.seq_lens), args.num_docs, seed=42
        )
        report["eval_manifest"] = args.eval_manifest
        report["paired_documents_across_lengths"] = True
    for seq_len in args.seq_lens:
        print(f"\n=== seq_len {seq_len}: collecting {args.num_docs} docs ===")
        input_ids = (
            paired_rows[:, :seq_len]
            if paired_rows is not None
            else collect_docs(tokenizer, seq_len, args.num_docs, args.dataset, args.dataset_subset)
        )
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

        # Checkpoint after each length so an OOM/crash at a longer seq still leaves
        # the shorter-length results on disk.
        if args.output:
            with open(args.output, "w") as f:
                json.dump(report, f, indent=2)

    # Decision-tree summary: classify the G-vs-length curve and emit a recommendation.
    # Logic mirrors the discussion in chat (2026-07-09): the spec's training seq should
    # be the shortest length at which the concept mechanism could plausibly matter, and
    # the eval horizon should be 2-4x that. A steep G growth past the training horizon
    # argues for a length-curriculum amendment; a flat G everywhere means concepts have
    # nothing to close and the hypothesis must be re-scoped before any GPU spend.
    print("\n=== G curve (beyond-1024 gap vs length) ===")
    curve = [(int(s), entry["G_beyond_1024"]) for s, entry in report["seq_lens"].items()]
    for s, g in curve:
        flag = "(above gate)" if g >= 0.05 else "(below gate)"
        print(f"  seq {s:>5}: G = {g:.4f} nats  {flag}")
    report["G_curve"] = [{"seq_len": s, "G_beyond_1024": g} for s, g in curve]

    seqs_above = [s for s, g in curve if g >= 0.05]
    recommendation = {}
    if not seqs_above:
        msg = ("G < 0.05 at every measured length. The windowed backbone doesn't hurt "
               "Gemma even at the longest eval. Concepts have nothing to close; re-scope "
               "the hypothesis (or push seq/eval longer) before any training spend.")
        action = "REScope"
        recommendation["shortest_viable_seq"] = None
    else:
        shortest_viable = min(seqs_above)
        recommendation["shortest_viable_seq"] = shortest_viable
        if shortest_viable <= 2048:
            g_2k = next((g for s, g in curve if s == 2048), None)
            g_8k = next((g for s, g in curve if s == 8192), None)
            ratio_8k = (g_8k / g_2k) if (g_2k and g_2k > 0 and g_8k is not None) else None
            recommendation["G_8K_over_G_2K"] = round(ratio_8k, 2) if ratio_8k else None
            if ratio_8k is not None and ratio_8k >= 3.0:
                msg = (f"G crosses 0.05 at seq {shortest_viable} but grows steeply "
                       f"(G(8K)/G(2K) = {ratio_8k:.2f}x). Recommend a 2K-majority + "
                       "4-8K-tail length curriculum (LongLoRA-style) so the recurrence "
                       "learns to stay long. Eval horizon stays at 8K.")
                action = "ADD_CURRICULUM"
            else:
                msg = ("G crosses 0.05 at seq 2048 and grows gently with length. Spec "
                       "as-written (train 2K, eval 8K) is a clean extrapolation bet.")
                action = "KEEP_SPEC"
        else:
            msg = (f"2K is too short (gate fails at 2048). Bump TRAINING seq to "
                   f"{shortest_viable}; eval horizon should be 2-4x that.")
            action = "BUMP_TRAIN_SEQ"
    recommendation["action"] = action
    recommendation["rationale"] = msg
    report["recommendation"] = recommendation
    print(f"\nVERDICT [{action}]: {msg}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
