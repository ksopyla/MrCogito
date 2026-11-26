"""E41: a tree-read view of a concept-memory checkpoint, and the checks that go with it.

`view`   writes a checkpoint folder whose config turns the concept tree on (the weights are linked, not copied), so
         every evaluation script can read a trained concept-memory model through the tree read (bounded reads and
         cuts are memory modes of evaluation/long_context_use.py: tree16, descent16, cut2, …). With --arm the folder
         gets the config of a run_e33n arm (tree, tree_noaux, tree_read): the reasoner's weights are then new.
`check`  loads the source and the view, runs real rows and prints: the largest logit difference with every node
         followed (the identity the spec's K4 asks for), the loss of bounded reads and cuts on the same rows, the
         entries each read holds, and seconds per forward.

  uv run python scripts/e41_tree_view.py view  SRC DST [--arm tree]
  uv run python scripts/e41_tree_view.py check SRC DST --shard FILE.bin [--len 16384 --rows 2]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TREE_READ = dict(concept_tree=True, tree_probes=8, tree_fan=8)


def arm_fields(arm: str) -> dict:
    """The tree_* fields of a run_e33n arm, as config values."""
    from scripts.run_e33n import ARMS

    out = {}
    for k, v in ARMS[arm].items():
        if k == "concept_tree" or k.startswith("tree_"):
            out[k] = (str(v).lower() == "true") if str(v).lower() in ("true", "false") else v
    return out


def view(src: Path, dst: Path, arm: str | None) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((src / "config.json").read_text())
    if str(cfg.get("par_global_read")) != "concept":
        raise SystemExit(f"{src}: par_global_read is {cfg.get('par_global_read')!r}, the tree extends 'concept'")
    cfg.update(arm_fields(arm) if arm else TREE_READ)
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))
    for f in src.iterdir():
        if f.name == "config.json":
            continue
        link = dst / f.name
        if link.exists() or link.is_symlink():
            link.unlink()
        os.symlink(f.resolve(), link)
    print(f"view of {src} at {dst}: " + ", ".join(f"{k}={v}" for k, v in sorted(cfg.items())
                                                   if k == "concept_tree" or k.startswith("tree_")))


def rows_from_shard(shard: Path, length: int, rows: int):
    import numpy as np
    import torch

    tok = np.fromfile(shard, dtype=np.uint16, offset=256 * 4)
    if tok.size < length * rows:
        raise SystemExit(f"{shard}: {tok.size} tokens, need {length * rows}")
    return torch.from_numpy(tok[: length * rows].astype(np.int64)).view(rows, length)


def check(src: Path, dst: Path, shard: Path, length: int, rows: int) -> None:
    import torch

    from evaluation.long_context_use import memory_mode
    from nn.perceiver_ar_lm import PerceiverARLM

    dev = "cuda"
    ids = rows_from_shard(shard, length, rows).to(dev)
    base = PerceiverARLM.from_pretrained(src).to(dev).eval()
    tree = PerceiverARLM.from_pretrained(dst).to(dev).eval()
    att = tree.layers[tree.config.global_layer_index].attn

    def run(m, mode=None):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            if mode is None:
                out = m(ids, labels=ids, return_logits=True)
            else:
                with memory_mode(m, mode):
                    out = m(ids, labels=ids, return_logits=True)
        torch.cuda.synchronize()
        return out, time.time() - t0

    ob, _ = run(base)                                              # the first calls compile the kernels
    ob, tb = run(base)
    ot, _ = run(tree)
    ot, tt = run(tree)
    diff = float((ob.logits.float() - ot.logits.float()).abs().max())
    print(f"rows {rows} × {length} tokens from {shard.name}")
    print(f"identity (every node followed): max |logit difference| {diff:.2e}; loss {float(ob.loss):.4f} vs "
          f"{float(ot.loss):.4f}; forward {tb:.2f} s vs {tt:.2f} s")
    print(f"{'read':12} {'loss':>8} {'vs flat':>8} {'entries':>8} {'tokens in summaries':>20} {'seconds':>8}")
    for mode in ("tree64", "tree16", "tree8", "tree4", "descent16", "descent8", "cut1", "cut2", "cut3", "nosum",
                 "off"):
        att.tree_stats = {} if mode not in ("nosum", "off") else None
        try:
            run(tree, mode)
            o, sec = run(tree, mode)
        except Exception as e:  # a level the row does not have, a switch the read lacks
            print(f"{mode:12} skipped: {type(e).__name__}: {str(e)[:80]}")
            continue
        st = att.tree_stats or {}
        att.tree_stats = None
        print(f"{mode:12} {float(o.loss):8.4f} {float(o.loss) - float(ot.loss):+8.4f} "
              f"{st.get('tree/entries', float('nan')):8.0f} {st.get('tree/summarised_tokens', float('nan')):20.0f} "
              f"{sec:8.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="what", required=True)
    v = sub.add_parser("view")
    v.add_argument("src", type=Path)
    v.add_argument("dst", type=Path)
    v.add_argument("--arm", default=None)
    c = sub.add_parser("check")
    c.add_argument("src", type=Path)
    c.add_argument("dst", type=Path)
    c.add_argument("--shard", type=Path, required=True)
    c.add_argument("--len", type=int, default=16384)
    c.add_argument("--rows", type=int, default=2)
    a = ap.parse_args()
    if a.what == "view":
        view(a.src, a.dst, a.arm)
    else:
        check(a.src, a.dst, a.shard, a.len, a.rows)


if __name__ == "__main__":
    main()
