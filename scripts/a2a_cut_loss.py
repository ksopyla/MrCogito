#!/usr/bin/env python
"""E37c: what a cut costs, by distance after it (docs/experiments_specs/done_failed/E37c_two_agent_split_training.md).

Held-out LongDocs-lite rows, each scored by the plain forward three ways:
  plain      no cut (one agent)
  every      a cut every --every tokens: a local layer reads its own side of a cut only, the reading layer still
             reads the whole document (`cut_ids`, nn/perceiver_ar_lm.py) — every side is an agent B that starts cold
  train      cuts drawn as the split phase draws them (--train_gaps MIN,MAX, log-uniform gaps; one fixed draw per row):
             the loss training on cut rows starts from

Reported: the mean next-token loss of each way, and for `every` the loss of the positions 0-7, 8-31, 32-127 and 128+
tokens after a cut next to the same positions without cuts (the paired difference and its standard error over tokens).
The position right at a cut has an empty local window; the reading layer has no position code, so the word order just
before the cut is not addressable and the first bin cannot close fully. The 32-127 bin is the one training should
close.

Usage (GPU): uv run python scripts/a2a_cut_loss.py --checkpoint RUN/final --data DATA/longdocs_lite_gpt2 --out out.json
Laptop pipeline check: add --device cpu --backend sdpa --length 2048 --every 512 --rows 1.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BINS = ((0, 8), (8, 32), (32, 128), (128, 1 << 30))


def bin_name(lo: int, hi: int) -> str:
    return f"{lo}-{hi - 1}" if hi < (1 << 30) else f"{lo}+"


@torch.no_grad()
def per_token(model, ids, doc, cuts=None):
    """Loss of every position's next-token prediction [S − 1] (float) and which of them count."""
    _, per, valid = model(input_ids=ids, labels=ids, doc_ids=doc, return_per_token_loss=True, cut_ids=cuts)
    return per[0].float().cpu(), valid[0].cpu()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True, help="LongDocs-lite dir with <source>_val_*.bin and manifest.json")
    ap.add_argument("--length", type=int, default=16384)
    ap.add_argument("--every", type=int, default=1024, help="a cut every this many tokens")
    ap.add_argument("--train_gaps", default="32,4096", help="MIN,MAX of the split phase's gaps ('' = skip)")
    ap.add_argument("--rows", type=int, default=8, help="rows per source")
    ap.add_argument("--backend", default="flex")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from data.bin_token_dataset import BinRowDataset
    from evaluation.lm_eval_perceiver_ar import load_perceiver_ar_for_eval
    from nn.perceiver_ar_lm import draw_cuts

    model = load_perceiver_ar_for_eval(a.checkpoint, device=a.device, attn_backend=a.backend)
    S, E = a.length, a.every
    gaps = tuple(int(x) for x in a.train_gaps.split(",")) if a.train_gaps else None
    every = torch.zeros(1, S, dtype=torch.bool)
    every[0, E::E] = True
    t = torch.arange(S - 1)                      # position t predicts token t + 1
    dist = torch.where(t >= E, t % E, torch.full_like(t, -1))  # tokens since the last cut (−1: before the first)
    gen = torch.Generator().manual_seed(1234)
    srcs = sorted(json.loads((Path(a.data) / "manifest.json").read_text())["sources"])
    tot = {k: [0.0, 0] for k in ("plain", "every", "train")}
    by = {bin_name(*b): {"plain": 0.0, "cut": 0.0, "d2": 0.0, "n": 0} for b in BINS}
    rows, t0 = 0, time.time()
    for src in srcs:
        try:
            ds = BinRowDataset([{"name": src, "files": [str(Path(a.data) / f"{src}_val_*.bin")]}], S, "doc", seed=7)
        except (ValueError, FileNotFoundError):
            continue
        for i in range(min(a.rows, len(ds))):
            item = ds[i]
            ids, doc = item["input_ids"][None].to(a.device), item["doc_ids"][None].to(a.device)
            p0, valid = per_token(model, ids, doc)
            p1, _ = per_token(model, ids, doc, every.to(a.device))
            tot["plain"][0] += float(p0[valid].sum())
            tot["every"][0] += float(p1[valid].sum())
            tot["plain"][1] += int(valid.sum())
            tot["every"][1] += int(valid.sum())
            if gaps:
                p2, _ = per_token(model, ids, doc, draw_cuts(1, S, gaps[0], gaps[1], 1.0, generator=gen).to(a.device))
                tot["train"][0] += float(p2[valid].sum())
                tot["train"][1] += int(valid.sum())
            for lo, hi in BINS:
                m = valid & (dist >= lo) & (dist < hi)
                b = by[bin_name(lo, hi)]
                d = p1[m] - p0[m]
                b["plain"] += float(p0[m].sum())
                b["cut"] += float(p1[m].sum())
                b["d2"] += float((d * d).sum())
                b["n"] += int(m.sum())
            rows += 1
    res = {"checkpoint": a.checkpoint, "length": S, "every": E, "train_gaps": gaps, "rows": rows,
           "seconds": round(time.time() - t0, 1),
           "loss": {k: (v[0] / v[1] if v[1] else None) for k, v in tot.items()}, "after_a_cut": {}}
    for name, b in by.items():
        n = b["n"]
        if not n:
            continue
        extra = (b["cut"] - b["plain"]) / n
        var = max(0.0, b["d2"] / n - extra * extra)
        res["after_a_cut"][name] = {"tokens": n, "plain": b["plain"] / n, "cut": b["cut"] / n, "extra": extra,
                                    "se": (var / n) ** 0.5}
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps({"loss": res["loss"], "extra_after_a_cut": {k: round(v["extra"], 4)
                                                                 for k, v in res["after_a_cut"].items()}}))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
