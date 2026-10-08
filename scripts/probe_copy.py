#!/usr/bin/env python
"""Copy probe: can a trained checkpoint repeat a random token sequence it has already seen?

A block of `span` random tokens is shown, then shown again. A model that can look back (an
"induction" ability every causal transformer learns early) predicts the second copy almost
perfectly; one limited to a short window cannot. Reports next-token loss (nats/token) on the first
and second copy, for a few gaps between them (filler tokens in between).

  uv run python scripts/probe_copy.py --checkpoint <run>/final --device mps cpu
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402


@torch.no_grad()
def probe(model, vocab: int, span: int, gap: int, device: str, n: int = 8, seed: int = 0,
          text_rows=None) -> tuple[float, float]:
    """`text_rows`: token rows of real text; blocks and filler are cut from them instead of random ids."""
    g = torch.Generator().manual_seed(seed)
    first, second = 0.0, 0.0
    for i in range(n):
        if text_rows is not None:
            row = torch.tensor(text_rows[i % len(text_rows)][1:])
            block, filler = row[:span], row[span: span + gap]
        else:
            block = torch.randint(10, vocab, (span,), generator=g)
            filler = torch.randint(10, vocab, (gap,), generator=g)
        ids = torch.cat([torch.tensor([1]), block, filler, block]).unsqueeze(0).to(device)
        out = model(input_ids=ids)
        logits = out.logits if hasattr(out, "logits") else out[0]
        lp = torch.log_softmax(logits[0, :-1].float(), -1)
        nll = -lp.gather(-1, ids[0, 1:].unsqueeze(-1)).squeeze(-1).cpu()
        first += float(nll[1:span].mean())
        s0 = 1 + span + gap
        second += float(nll[s0: s0 + span - 1].mean())
    return first / n, second / n


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", nargs="+", default=["cpu"])
    p.add_argument("--span", type=int, default=64)
    p.add_argument("--gaps", type=int, nargs="+", default=[0, 16, 100, 400])
    p.add_argument("--attn_backend", default="sdpa")
    p.add_argument("--text", default=None, help="a text-checks data dir (its held-out story rows) or a tokenized split: copy real text")
    a = p.parse_args()
    from nn.perceiver_families import load_perceiver_lm

    rows = None
    if a.text:
        from datasets import load_from_disk

        ds = load_from_disk(os.path.join(a.text, "stories", "eval") if os.path.isdir(os.path.join(a.text, "stories")) else a.text)
        rows = [ds[i]["input_ids"] for i in range(min(16, len(ds)))]
    for dev in a.device:
        model = load_perceiver_lm(a.checkpoint, dev, a.attn_backend).eval()
        vocab = model.config.vocab_size if hasattr(model, "config") else 4096
        for gap in a.gaps:
            f, s = probe(model, vocab, a.span, gap, dev, text_rows=rows)
            print(f"{'text' if rows else 'random'} {dev:<4} gap {gap:>4}: first copy {f:5.2f} nats/token, second copy "
                  f"{s:5.2f}  ({'copies' if s < 0.5 * f else 'does NOT copy'})", flush=True)


if __name__ == "__main__":
    main()
