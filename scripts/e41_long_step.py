"""E41: can one card train on a very long row through the tree? Time and memory of a training step by row length.

The stages so far stop at rows of 65,536 tokens, where the concept read ranks every concept for every query (the
cost grows with the square of the row). A bounded tree read follows a fixed number of nodes per level, so its cost
per query hardly grows with the row. This script measures a real training step (forward, loss, backward, gradient
checkpointing, bf16) of a checkpoint read through the tree, at several row lengths and row mixes, on random tokens
(the cost does not depend on the text).

  uv run python scripts/e41_long_step.py --checkpoint CK --lengths 65536,131072,262144 \
      --mixes 1,0,0:0,0.6,0.4 --follow 16 --steps 3 --out step.json
A mix is normal,bounded,cut (shares of rows: the whole read; a bounded read; a read cut at a level).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--lengths", default="65536,131072,262144")
    ap.add_argument("--mixes", default="1,0,0:0,0.6,0.4")
    ap.add_argument("--follow", type=int, default=16, help="nodes followed per level on a bounded row")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--wide", action="store_true", help="wide summaries on (a fresh writer)")
    ap.add_argument("--top_down", action="store_true", help="the top-down read (tree_sparse): every row bounded")
    ap.add_argument("--scan", action="store_true", help="every concept scored for opening (tree_scan)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from nn.perceiver_ar_lm import PerceiverARConfig, PerceiverARLM

    out = {"checkpoint": a.checkpoint, "follow": a.follow, "top_down": bool(a.top_down), "rows": []}
    for mix in a.mixes.split(":"):
        cfg = PerceiverARConfig.from_pretrained(a.checkpoint)
        cfg.concept_tree, cfg.tree_probes, cfg.tree_fan = True, 8, 8
        cfg.tree_row_mix, cfg.tree_train_follow, cfg.tree_row_draw = mix, a.follow, "step"
        cfg.tree_wide = bool(a.wide)
        cfg.tree_sparse = bool(a.top_down)
        cfg.tree_scan = bool(a.scan)
        model = PerceiverARLM.from_pretrained(a.checkpoint, config=cfg).cuda().train()
        model.gradient_checkpointing_enable()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-6)
        vocab = int(cfg.vocab_size)
        for S in (int(x) for x in a.lengths.split(",")):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            g = torch.Generator().manual_seed(S)
            times, err = [], None
            try:
                for i in range(a.steps):
                    ids = torch.randint(3, vocab, (1, S), generator=g).cuda()
                    ids[:, 0] = 0
                    torch.cuda.synchronize()
                    t0 = time.time()
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss = model(ids, labels=ids).loss
                    loss.backward()
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    torch.cuda.synchronize()
                    times.append(time.time() - t0)
            except torch.OutOfMemoryError as e:  # the row does not fit this card
                err = str(e)[:160]
                opt.zero_grad(set_to_none=True)
            row = {"mix": mix, "length": S, "seconds": times, "peak_gb": torch.cuda.max_memory_allocated() / 2 ** 30,
                   "tokens_per_s": (S / min(times[1:] or times)) if times else None, "error": err}
            out["rows"].append(row)
            print(json.dumps(row), flush=True)
            Path(a.out).write_text(json.dumps(out, indent=2))
        del model, opt


if __name__ == "__main__":
    main()
