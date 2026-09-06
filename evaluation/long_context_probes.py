#!/usr/bin/env python
"""Long-context probes for the Perceiver AR v2 family (E18 pilot gates P2/P3).

Three probes, all teacher-forced (no generation needed, so they run at 32k on a 3090):

  * position-bucketed CE on long documents  — does context beyond 8k lower the loss?
  * passkey retrieval                        — argmax accuracy over the 5 answer digits
  * copy task                                — token accuracy on the mirrored second half

Usage:
  uv run python evaluation/long_context_probes.py --checkpoint <dir> --probe buckets \
      --manifest <eval manifest> --max_seq_length 32768 --buckets 8192,32768
  uv run python evaluation/long_context_probes.py --checkpoint <dir> --probe passkey \
      --manifest <eval manifest> --context_lengths 4096,8192,16384,32768
  uv run python evaluation/long_context_probes.py --checkpoint <dir> --probe copy \
      --copy_dataset <arrow dir>
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from nn.perceiver_ar_lm import PerceiverARConfig, PerceiverARLM  # noqa: E402


def load_model(checkpoint: str, device: str, attn_backend: str | None = None) -> PerceiverARLM:
    cfg = PerceiverARConfig.from_pretrained(checkpoint)
    if attn_backend:
        cfg.attn_backend = attn_backend
    model = PerceiverARLM.from_pretrained(checkpoint, config=cfg)
    model.to(device).eval()
    if device.startswith("cuda"):
        model.to(torch.bfloat16)
    return model


def load_eval_rows(manifest: str, min_len: int, max_rows: int) -> list[list[int]]:
    from datasets import load_from_disk

    man = json.loads(Path(manifest).read_text())
    rows: list[list[int]] = []
    for src in man["sources"]:
        ds = load_from_disk(src["eval_path"])
        for r in ds:
            ids = r["input_ids"]
            if len(ids) >= min_len:
                rows.append(list(ids))
            if len(rows) >= max_rows:
                return rows
    return rows


@torch.no_grad()
def per_token_ce(model, ids: list[int], device: str, max_len: int) -> torch.Tensor:
    x = torch.tensor(ids[:max_len], device=device)[None]
    _, per, valid = model(input_ids=x, labels=x.clone(), return_per_token_loss=True)
    return per[0][valid[0]].float().cpu()


def probe_buckets(model, args, device) -> dict:
    edges = [int(x) for x in args.buckets.split(",")]
    rows = load_eval_rows(args.manifest, min_len=edges[-1], max_rows=args.max_rows)
    if not rows:
        raise SystemExit(f"no eval rows with >= {edges[-1]} tokens in {args.manifest}")
    sums = [0.0] * len(edges)
    counts = [0] * len(edges)
    for ids in rows:
        per = per_token_ce(model, ids, device, edges[-1])
        lo = 0
        for bi, hi in enumerate(edges):
            seg = per[lo:hi]
            sums[bi] += float(seg.sum())
            counts[bi] += int(seg.numel())
            lo = hi
    out = {"rows": len(rows)}
    lo = 0
    for bi, hi in enumerate(edges):
        out[f"ce[{lo},{hi})"] = sums[bi] / max(counts[bi], 1)
        lo = hi
    return out


def build_passkey(tokenizer, filler_ids: list[int], context_len: int, depth: float, rng) -> tuple[list[int], list[int]]:
    key = f"{rng.randint(0, 99999):05d}"
    needle = tokenizer.encode(f" The pass key is {key}. Remember it. ", add_special_tokens=False)
    question = tokenizer.encode(" What is the pass key? The pass key is", add_special_tokens=False)
    answer = tokenizer.encode(f" {key}", add_special_tokens=False)
    budget = context_len - len(needle) - len(question) - len(answer)
    filler = filler_ids[:budget]
    cut = int(len(filler) * depth)
    ids = filler[:cut] + needle + filler[cut:] + question + answer
    return ids, answer


def probe_passkey(model, args, device) -> dict:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    rows = load_eval_rows(args.manifest, min_len=max(int(x) for x in args.context_lengths.split(",")), max_rows=args.max_rows)
    if not rows:
        raise SystemExit("no filler rows long enough")
    rng = random.Random(args.seed)
    results = {}
    for L in (int(x) for x in args.context_lengths.split(",")):
        correct = total = 0
        for depth in (0.1, 0.3, 0.5, 0.7, 0.9):
            for trial in range(args.trials):
                filler = rows[(trial * 7 + int(depth * 10)) % len(rows)]
                ids, answer = build_passkey(tok, filler, L, depth, rng)
                x = torch.tensor(ids, device=device)[None]
                logits = model(input_ids=x).logits[0]
                n = len(answer)
                pred = logits[-n - 1 : -1].argmax(-1).tolist()
                correct += int(pred == answer)
                total += 1
        results[f"passkey@{L}"] = correct / total
    return results


def probe_copy(model, args, device) -> dict:
    from datasets import load_from_disk

    ds = load_from_disk(args.copy_dataset)
    correct = total = 0
    for r in ds:
        x = torch.tensor(r["input_ids"], device=device)[None]
        labels = torch.tensor(r["labels"], device=device)[None]
        logits = model(input_ids=x).logits[0]
        tgt = labels[0, 1:]
        pred = logits[:-1].argmax(-1)
        m = tgt != -100
        correct += int((pred[m] == tgt[m]).sum())
        total += int(m.sum())
    return {"copy_token_accuracy": correct / max(total, 1), "rows": len(ds)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--probe", choices=["buckets", "passkey", "copy"], required=True)
    p.add_argument("--manifest", default=None)
    p.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM3-3B")
    p.add_argument("--buckets", default="8192,32768")
    p.add_argument("--context_lengths", default="4096,8192,16384,32768")
    p.add_argument("--trials", type=int, default=4)
    p.add_argument("--max_rows", type=int, default=64)
    p.add_argument("--copy_dataset", default=None)
    p.add_argument("--attn_backend", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="JSON output path")
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint, device, args.attn_backend)
    fn = {"buckets": probe_buckets, "passkey": probe_passkey, "copy": probe_copy}[args.probe]
    res = fn(model, args, device)
    res["checkpoint"] = args.checkpoint
    print(json.dumps(res, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
