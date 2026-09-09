#!/usr/bin/env python
"""Long-context probes for the Perceiver AR v2 family (E18 pilot gates P2/P3).

Three probes, all teacher-forced (no generation needed, so they run at 32k on a 3090):

  * position-bucketed CE on long documents  — does context beyond 8k lower the loss?
  * passkey retrieval                        — argmax accuracy over the 5 answer digits
  * copy task                                — token accuracy on the mirrored second half
  * reach ablation (paired)                  — same rows, same weights, the global read restricted
                                               to swa(W) for several W: does the loss at far
                                               positions depend on DIRECT access to far keys?
                                               Confound-free (no second run, no position-difficulty
                                               gradient); reports paired Δ with a standard error.

Any probe accepts `--reach_window W` (e.g. the P2 copy model with W below the copy offset is the
positive control: accuracy must collapse if the global read is the retrieval channel).

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
def argmax_tokens(model, x: torch.Tensor, start: int, end: int, chunk: int = 2048) -> torch.Tensor:
    """Greedy next-token predictions for hidden positions [start, end) of x [1, S].

    Uses `model.hidden_states` + a chunked head projection so memory stays O(chunk × V)
    instead of O(S × V): full logits at 32k are 16 GB, which is why the passkey / copy probes
    OOM'd next to a training run. Argmax is invariant to the tanh soft-cap, so it is skipped.
    """
    h = model.hidden_states(x)[0]  # [S, d]
    S = h.shape[0]
    start = start % S if start < 0 else start
    end = S if end is None else (end % S if end < 0 else end)
    weight = model.lm_head.weight
    preds = []
    for lo in range(start, end, chunk):
        hi = min(lo + chunk, end)
        preds.append(torch.nn.functional.linear(h[lo:hi], weight).argmax(-1))
    return torch.cat(preds) if preds else h.new_zeros(0, dtype=torch.long)


@torch.no_grad()
def per_token_ce(model, ids: list[int], device: str, max_len: int) -> torch.Tensor:
    x = torch.tensor(ids[:max_len], device=device)[None]
    _, per, valid = model(input_ids=x, labels=x.clone(), return_per_token_loss=True)
    return per[0][valid[0]].float().cpu()


def bucket_means(per: torch.Tensor, edges: list[int]) -> list[float]:
    """Mean per-token CE inside consecutive position buckets [0,e0), [e0,e1), ... ."""
    out, lo = [], 0
    for hi in edges:
        seg = per[lo:hi]
        out.append(float(seg.mean()) if seg.numel() else float("nan"))
        lo = hi
    return out


def _bucket_labels(edges: list[int]) -> list[str]:
    labels, lo = [], 0
    for hi in edges:
        labels.append(f"[{lo},{hi})")
        lo = hi
    return labels


def probe_reach(model, args, device) -> dict:
    """Paired reach ablation over `--reach_windows` (comma list; 'full' = unrestricted).

    For every window W the same rows are scored with every `full` layer restricted to swa(W);
    Δ(W) = CE(W) − CE(full) per row and bucket, reported as mean ± standard error over rows.
    Buckets entirely below W are computed from exactly the same keys, so their Δ is the numerical
    noise floor of the backend (exactly 0 on the sdpa/fp32 path).
    """
    edges = [int(x) for x in args.buckets.split(",")]
    rows = load_eval_rows(args.manifest, min_len=edges[-1], max_rows=args.max_rows)
    if not rows:
        raise SystemExit(f"no eval rows with >= {edges[-1]} tokens in {args.manifest}")
    windows: list[int | None] = []
    for w in args.reach_windows.split(","):
        w = w.strip()
        windows.append(None if w == "full" else int(w))
    if None not in windows:
        windows.append(None)
    labels = _bucket_labels(edges)
    per_row: dict[str, list[list[float]]] = {}
    touched: list[int] = []
    for w in windows:
        key = "full" if w is None else str(w)
        with model.reach_override(w) as t:
            touched = t or touched
            per_row[key] = [bucket_means(per_token_ce(model, ids, device, edges[-1]), edges) for ids in rows]
    base = torch.tensor(per_row["full"])  # [rows, buckets]
    out: dict = {"rows": len(rows), "buckets": labels, "windows": [k for k in per_row],
                 "touched_layers": touched, "ce": {}, "delta_vs_full": {}}
    for key, vals in per_row.items():
        t = torch.tensor(vals)
        out["ce"][key] = {lab: float(t[:, b].mean()) for b, lab in enumerate(labels)}
        if key == "full":
            continue
        d = t - base
        n = d.shape[0]
        out["delta_vs_full"][key] = {
            lab: {
                "mean": float(d[:, b].mean()),
                "se": float(d[:, b].std(unbiased=True) / (n ** 0.5)) if n > 1 else float("nan"),
                "n": n,
            }
            for b, lab in enumerate(labels)
        }
    out["per_row"] = per_row
    return out


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
                n = len(answer)
                pred = argmax_tokens(model, x, x.shape[1] - n - 1, x.shape[1] - 1).tolist()
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
        tgt = labels[0, 1:]
        pred = argmax_tokens(model, x, 0, x.shape[1] - 1)
        m = tgt != -100
        correct += int((pred[m] == tgt[m]).sum())
        total += int(m.sum())
    return {"copy_token_accuracy": correct / max(total, 1), "rows": len(ds)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--probe", choices=["buckets", "passkey", "copy", "reach"], required=True)
    p.add_argument("--reach_window", type=int, default=None,
                   help="restrict every full layer to swa(W) for this probe (positive-control runs)")
    p.add_argument("--reach_windows", default="512,2048,8192,full",
                   help="--probe reach: comma list of windows to sweep ('full' = unrestricted)")
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
    fn = {"buckets": probe_buckets, "passkey": probe_passkey, "copy": probe_copy, "reach": probe_reach}[args.probe]
    with model.reach_override(args.reach_window) as touched:
        res = fn(model, args, device)
    res["checkpoint"] = args.checkpoint
    if args.reach_window is not None:
        res["reach_window"] = args.reach_window
        res["touched_layers"] = touched
    print(json.dumps(res, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
