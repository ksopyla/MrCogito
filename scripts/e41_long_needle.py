"""E41: one number hidden in a very long text (64k to 1M tokens and beyond), read through the tree.

The card protocol stops at 128k (no held-out document is longer, and the concept read's per-query table is 4 GB
there). The tree read has a path without tables (`open_impl list`, automatic past the table's limit), so this
script measures what the vision asks for: does a reader that holds about a thousand entries still find one fact
in a text of a million tokens?

A row: held-out documents of a built mix (SmolLM2 tokens), joined into one text of the asked length (no document
of the test set is 1M tokens long, so the filler is several books in a row: fine for a lookup, not a measure of
book understanding); RULER's number needle ("One of the special magic numbers for <word> is: <7 digits>.") planted
at a depth; RULER's question at the end. The answer is scored teacher-forced: the model's greedy prediction at
each answer position given the true prefix (all digits right; first token right).

  uv run python scripts/e41_long_needle.py --checkpoint CK --mix DIR_WITH_VAL_SHARDS --lengths 65536,262144,1048576 \
      --depths 8 --modes on,tree64,tree16,descent16 --out needle.json
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

WORDS = ["amber", "willow", "copper", "falcon", "harbor", "meadow", "quartz", "saffron", "thistle", "velvet",
         "juniper", "lantern", "marble", "nectar", "orchid", "pebble"]


def filler(mix: Path, sources: list[str], need: int, eot: int) -> np.ndarray:
    """`need` tokens of held-out text: the sources' val shards in turn, document marks dropped."""
    parts, have = [], 0
    files = [f for s in sources for f in sorted(glob.glob(str(mix / f"{s}_val_*.bin")))]
    if not files:
        raise SystemExit(f"no val shards for {sources} under {mix}")
    while have < need:
        for f in files:
            tok = np.fromfile(f, dtype=np.uint16, offset=256 * 4)
            tok = tok[tok != eot]
            parts.append(tok)
            have += tok.size
            if have >= need:
                break
    return np.concatenate(parts)[:need].astype(np.int64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--mix", required=True, type=Path, help="a built mix folder with <source>_val_*.bin (SmolLM2)")
    ap.add_argument("--sources", default="books,arxiv,doab,usgpo,hansard")
    ap.add_argument("--lengths", default="65536,262144")
    ap.add_argument("--depths", type=int, default=8)
    ap.add_argument("--modes", default="on,tree64,tree16,descent16")
    ap.add_argument("--list_from", type=int, default=131072, help="rows this long or longer read without tables")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--top_down", action="store_true",
                    help="bounded reads by descent go through the top-down path (tree_sparse) whatever the "
                         "checkpoint's config says")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from transformers import AutoTokenizer

    from evaluation.long_context_use import memory_mode
    from nn.perceiver_ar_lm import PerceiverARLM

    tok = AutoTokenizer.from_pretrained(a.checkpoint)
    model = PerceiverARLM.from_pretrained(a.checkpoint).cuda().eval()
    eot = int(model.config.eos_token_id if model.config.eos_token_id is not None else 0)
    if a.top_down:
        for mod in model.modules():
            if isinstance(getattr(mod, "tree", None), dict):
                mod.tree["sparse"] = True
    rng = random.Random(a.seed)
    lengths = [int(x) for x in a.lengths.split(",")]
    text = filler(a.mix, a.sources.split(","), max(lengths) + 4096, eot)
    out = {"checkpoint": a.checkpoint, "lengths": {}, "modes": a.modes.split(","), "depths": a.depths}

    def enc(s: str) -> list[int]:
        return tok(s, add_special_tokens=False)["input_ids"]

    for S in lengths:
        res = {m: {"all": 0, "first": 0, "seconds": 0.0, "n": 0} for m in out["modes"]}
        for d in range(a.depths):
            word = rng.choice(WORDS)
            number = "".join(rng.choice("0123456789") for _ in range(7))
            needle = enc(f" One of the special magic numbers for {word} is: {number}.")
            ask = enc(f"\nWhat is the special magic number for {word} mentioned in the provided text? "
                      f"The special magic number for {word} mentioned in the provided text is")
            ans = enc(f" {number}")
            body = S - 1 - len(needle) - len(ask) - len(ans)
            off = rng.randrange(0, max(len(text) - body, 1))
            fill = text[off:off + body].tolist()
            at = int((d + 0.5) / a.depths * body)
            ids = [eot] + fill[:at] + needle + fill[at:] + ask + ans
            x = torch.tensor(ids, device="cuda")[None]
            n = len(ans)
            for m in out["modes"]:
                torch.cuda.synchronize()
                t0 = time.time()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    with model.open_impl("list" if S >= a.list_from else None):
                        if m == "on":
                            h = model.hidden_states(x)[0]
                        else:
                            with memory_mode(model, m):
                                h = model.hidden_states(x)[0]
                    pred = torch.nn.functional.linear(h[S - n - 1:S - 1].float(),
                                                      model.lm_head.weight.float()).argmax(-1).tolist()
                torch.cuda.synchronize()
                r = res[m]
                r["all"] += int(pred == ans)
                r["first"] += int(pred[0] == ans[0])
                r["seconds"] += time.time() - t0
                r["n"] += 1
            print(f"L={S} depth {d + 1}/{a.depths}: " + "  ".join(
                f"{m} {res[m]['all']}/{res[m]['n']} ({res[m]['seconds'] / res[m]['n']:.0f}s)" for m in out["modes"]),
                flush=True)
        out["lengths"][str(S)] = {m: {"all_digits": r["all"] / max(r["n"], 1), "first_token": r["first"] / max(r["n"], 1),
                                      "seconds_per_row": r["seconds"] / max(r["n"], 1), "rows": r["n"]}
                                  for m, r in res.items()}
        out["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
        Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out["lengths"], indent=1))


if __name__ == "__main__":
    main()
