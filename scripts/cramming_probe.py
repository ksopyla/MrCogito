#!/usr/bin/env python
"""Channel capacity of a receiver, the "Cramming" way (arXiv 2502.13063): per text, optimize K input vectors placed
before it so that the FROZEN decoder reproduces the text; report the longest text reproduced exactly and the
information the vectors add.

E35 (docs/experiments_specs/ahead/E35_latent_messages.md) uses it as the ceiling a trained transmitter is compared
against: how many tokens can K message vectors carry into this receiver at all?

For each length N and each of `--n_texts` held-out story rows: vectors [K, d] start at random (embedding scale),
AdamW (lr 0.01) for up to `--steps` steps, stop when every token is predicted (teacher-forced argmax). Recorded per
text: token accuracy, cross-entropy without / with the vectors, information gain = Σ (CE_without − CE_with) in bits.
Texts are optimized in parallel (one set of vectors per text).

  uv run python scripts/cramming_probe.py --checkpoint <decoder or A2A final> --data <text_checks data dir> \\
      --lengths 16 32 64 128 256 --vectors 1 --n_texts 8 --out Cache/e35/cramming_dense.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def load_decoder(checkpoint: str, device: str, backend: str):
    from nn.perceiver_families import checkpoint_family, load_perceiver_lm

    m = load_perceiver_lm(checkpoint, device, backend, dtype=torch.float32)
    if checkpoint_family(checkpoint) == "a2a":
        m = m.receiver
    for p in m.parameters():
        p.requires_grad_(False)
    return m.eval()


def run(model, rows: torch.Tensor, K: int, steps: int, lr: float, device) -> dict:
    """rows [T, N] token ids (BOS first). Returns per-text accuracy, CE without / with vectors, steps used."""
    T, N = rows.shape
    d = model.config.hidden_size
    labels = rows.clone()
    with torch.no_grad():
        base = model(input_ids=rows, labels=labels, return_per_token_loss=True)
        ce0 = (base[1].float() * base[2]).sum(-1) / base[2].sum(-1)  # nats per token, no vectors
    vec = (torch.randn(T, K, d, device=device) * 1.0).requires_grad_(True)  # RMS ≈ 1, like the normed embeddings
    opt = torch.optim.AdamW([vec], lr=lr, weight_decay=0.0)
    done = torch.zeros(T, dtype=torch.bool, device=device)
    used = torch.full((T,), steps, device=device)
    for s in range(steps):
        out = model(input_ids=rows, labels=labels, prefix_embeds=vec, return_logits=True)
        logits = out.logits[:, :-1].float()
        tgt = rows[:, 1:]
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="none").view(T, -1)
        acc = (logits.argmax(-1) == tgt).float().mean(-1)
        newly = (acc >= 1.0) & ~done
        used[newly] = s
        done |= acc >= 1.0
        if bool(done.all()):
            break
        loss = (ce.mean(-1) * (~done).float()).sum()  # finished texts stop moving
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        out = model(input_ids=rows, labels=labels, prefix_embeds=vec, return_logits=True)
        logits = out.logits[:, :-1].float()
        tgt = rows[:, 1:]
        ce1 = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="none").view(T, -1).mean(-1)
        acc = (logits.argmax(-1) == tgt).float().mean(-1)
    n_pred = N - 1
    gain_bits = (ce0 - ce1) * n_pred / math.log(2)
    return {"acc": acc.tolist(), "ce_without": ce0.tolist(), "ce_with": ce1.tolist(),
            "info_gain_bits": gain_bits.tolist(), "steps": used.tolist()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True, help="text-checks data dir (stories/eval rows are the texts)")
    p.add_argument("--lengths", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    p.add_argument("--vectors", type=int, nargs="+", default=[1])
    p.add_argument("--n_texts", type=int, default=8)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--device", default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    from datasets import load_from_disk

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    backend = "sdpa"
    model = load_decoder(a.checkpoint, device, backend)
    ds = load_from_disk(os.path.join(a.data, "stories", "eval"))
    res = {"checkpoint": a.checkpoint, "steps": a.steps, "lr": a.lr, "cells": []}
    t0 = time.time()
    for K in a.vectors:
        for N in a.lengths:
            rows = torch.tensor([ds[i]["input_ids"][:N] for i in range(a.n_texts)], device=device)
            r = run(model, rows, K, a.steps, a.lr, device)
            exact = sum(x >= 0.99 for x in r["acc"]) / len(r["acc"])
            cell = {"vectors": K, "tokens": N, "share_reproduced": exact, "mean_acc": sum(r["acc"]) / len(r["acc"]),
                    "mean_info_gain_bits": sum(r["info_gain_bits"]) / len(r["info_gain_bits"]), **r}
            res["cells"].append(cell)
            print(f"K={K} N={N:5d}  reproduced {exact:4.0%}  token acc {cell['mean_acc']:.3f}  "
                  f"info gain {cell['mean_info_gain_bits']:7.1f} bits  ({time.time() - t0:.0f}s)", flush=True)
            os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
            json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
