"""E41: where a forward's time goes on a long row, by read. One row of random tokens per length, no gradient, bf16.

For each length and read: seconds per forward, the far read's share of it (the reading layer's tree read, summed
over its loop passes), and peak memory. Reads: `table` (the flex kernel over per-query tables: every concept scored
for every query), `stream` (the same, chunk by chunk without tables), `top_down` (only the children of followed
nodes are scored).

  uv run python scripts/e41_read_profile.py --checkpoint CK --lengths 16384,65536,262144 --reads table,top_down \
      --follow 16 --out profile.json
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
    ap.add_argument("--lengths", default="16384,65536,262144")
    ap.add_argument("--reads", default="table,top_down")
    ap.add_argument("--follow", type=int, default=16)
    ap.add_argument("--scan", action="store_true", help="every concept scored for opening (tree_scan)")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import nn.concept_tree as ct
    from nn.perceiver_ar_lm import PerceiverARConfig, PerceiverARLM

    cfg = PerceiverARConfig.from_pretrained(a.checkpoint)
    cfg.concept_tree, cfg.tree_probes, cfg.tree_fan = True, 8, 8
    cfg.tree_follow, cfg.tree_route, cfg.tree_scan = a.follow, "descent", bool(a.scan)
    model = PerceiverARLM.from_pretrained(a.checkpoint, config=cfg).cuda().eval()
    trees = [m for m in model.modules() if isinstance(getattr(m, "tree", None), dict)]
    spent = {"t": 0.0, "calls": 0, "shape": None}

    def timed(fn):
        def run(*args, **kw):
            torch.cuda.synchronize()
            t0 = time.time()
            out = fn(*args, **kw)
            torch.cuda.synchronize()
            spent["t"] += time.time() - t0
            spent["calls"] += 1
            return out
        return run

    orig = {n: getattr(ct, n) for n in ("tree_read_sparse", "tree_read_stream", "tree_choice", "attend_tree")}
    import nn.perceiver_ar_lm as pm  # the model imports the functions inside its read: patch the module they live in

    for n, f in orig.items():
        setattr(ct, n, timed(f))
    out = {"checkpoint": a.checkpoint, "follow": a.follow, "rows": []}
    for S in (int(x) for x in a.lengths.split(",")):
        ids = torch.randint(3, int(cfg.vocab_size), (1, S), generator=torch.Generator().manual_seed(S)).cuda()
        ids[:, 0] = 0
        for read in a.reads.split(","):
            for m in trees:
                m.tree["sparse"] = read == "top_down"
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            row = {"length": S, "read": read, "seconds": [], "far_read_seconds": [], "error": None}
            try:
                for _ in range(a.repeats + 1):
                    spent["t"], spent["calls"] = 0.0, 0
                    torch.cuda.synchronize()
                    t0 = time.time()
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        with model.open_impl("list" if read == "stream" else None):
                            model.hidden_states(ids)
                    torch.cuda.synchronize()
                    row["seconds"].append(time.time() - t0)
                    row["far_read_seconds"].append(spent["t"])
                row["calls"] = spent["calls"]
            except (torch.OutOfMemoryError, RuntimeError) as e:  # the row does not fit this read
                row["error"] = str(e)[:160]
            row["peak_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
            out["rows"].append(row)
            print(json.dumps(row), flush=True)
            Path(a.out).write_text(json.dumps(out, indent=2))
    del pm


if __name__ == "__main__":
    main()
