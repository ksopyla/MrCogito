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


def planted(rng, word: str, kind: str) -> tuple[str, str, str]:
    """The planted sentence, the question and the answer text. `number`: 7 digits; `uuid`: a version-4 UUID (36
    characters, RULER's niah_single_3 value: about 20 tokens that also occur all over ordinary text)."""
    import uuid

    if kind == "uuid":
        value, noun = str(uuid.UUID(int=rng.getrandbits(128), version=4)), "uuid"
    else:
        value, noun = "".join(rng.choice("0123456789") for _ in range(7)), "number"
    return (f" One of the special magic {noun}s for {word} is: {value}.",
            f"\nWhat is the special magic {noun} for {word} mentioned in the provided text? "
            f"The special magic {noun} for {word} mentioned in the provided text is", f" {value}")


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
    ap.add_argument("--stream_from", type=int, default=0,
                    help="rows this long or longer are read window by window (hidden_states_streamed: the far "
                         "read's keys and values carried along), so a 1M row fits a 24 GB card; 0 = never. Needs "
                         "bounded reads by descent (modes descent<N>, or a top-down model's own read)")
    ap.add_argument("--window", type=int, default=65536)
    ap.add_argument("--overlap", type=int, default=8192)
    ap.add_argument("--scan", action="store_true",
                    help="tokens are opened from the best concepts of the whole row (tree_scan) whatever the "
                         "checkpoint's config says; the followed nodes bound the summaries")
    ap.add_argument("--span", type=int, default=-1,
                    help="an opened token brings this many tokens after it (tree_span; top-down read); default: "
                         "the checkpoint's own")
    ap.add_argument("--top_down", action="store_true",
                    help="bounded reads by descent go through the top-down path (tree_sparse) whatever the "
                         "checkpoint's config says")
    ap.add_argument("--top_k", type=int, default=0, help="tokens a query may open (0: the checkpoint's own, 512)")
    ap.add_argument("--width", type=int, default=0, help="concepts a query ranks for opening (0: the checkpoint's own)")
    ap.add_argument("--rerank", type=int, default=-1,
                    help="the scan's best n concepts are re-ranked by their best member token (tree_rerank); "
                         "default: the checkpoint's own")
    ap.add_argument("--value", default="number", choices=("number", "uuid"),
                    help="what is planted: a 7-digit number, or a UUID (a long value to copy)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from transformers import AutoTokenizer

    from evaluation.long_context_use import memory_mode
    from nn.perceiver_ar_lm import PerceiverARLM

    tok = AutoTokenizer.from_pretrained(a.checkpoint)
    model = PerceiverARLM.from_pretrained(a.checkpoint).cuda().eval()
    eot = int(model.config.eos_token_id if model.config.eos_token_id is not None else 0)
    for mod in model.modules():
        c_ = getattr(mod, "concept", None)
        if isinstance(c_, dict) and "top_k" in c_:
            if a.top_k:
                c_["top_k"] = a.top_k
            if a.width:
                c_["width"] = a.width
        if isinstance(getattr(mod, "tree", None), dict):
            if a.top_down:
                mod.tree["sparse"] = True
            if a.rerank >= 0:
                mod.tree["rerank"] = a.rerank
            if a.scan:
                mod.tree["scan"] = True
            if a.span >= 0:
                mod.tree["span"] = a.span
    rng = random.Random(a.seed)
    lengths = [int(x) for x in a.lengths.split(",")]
    text = filler(a.mix, a.sources.split(","), max(lengths) + 4096, eot)
    out = {"checkpoint": a.checkpoint, "lengths": {}, "modes": a.modes.split(","), "depths": a.depths}

    def enc(s: str) -> list[int]:
        return tok(s, add_special_tokens=False)["input_ids"]

    for S in lengths:
        res = {m: {"all": 0, "first": 0, "rest": 0, "said": {}, "seconds": 0.0, "n": 0, "tok": 0, "toks": 0}
               for m in out["modes"]}
        for d in range(a.depths):
            word = rng.choice(WORDS)
            needle, ask, ans = (enc(t) for t in planted(rng, word, a.value))
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
                def states():  # the states that predict the answer's tokens: positions S − n − 1 … S − 2
                    if a.stream_from and S >= a.stream_from:
                        return model.hidden_states_streamed(x, window=a.window, overlap=a.overlap,
                                                            keep_last=n + 1)[0][:-1]
                    return model.hidden_states(x)[0][S - n - 1:S - 1]

                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    with model.open_impl("list" if S >= a.list_from else None):
                        if m == "on":
                            h = states()
                        else:
                            with memory_mode(model, m):
                                h = states()
                    pred = torch.nn.functional.linear(h.float(), model.lm_head.weight.float()).argmax(-1).tolist()
                torch.cuda.synchronize()
                r = res[m]
                r["all"] += int(pred == ans)
                r["first"] += int(pred[0] == ans[0])
                # `rest`: every token after the first right (the first one given). The planted sentence reads
                # "... is: 1234567." and the question ends "... is": a reader that copies the sentence says ":"
                # first and then the digits, which `all` counts as a miss although the number was found and read
                # (RULER, which generates and looks for the number in the output, counts it as read). `said`:
                # what was said first when it was not the number's first token.
                r["rest"] += int(pred[1:] == ans[1:])
                r["tok"] += sum(int(p == t) for p, t in zip(pred[1:], ans[1:]))
                r["toks"] += len(ans) - 1
                if pred[0] != ans[0]:
                    w = tok.decode([pred[0]])
                    r["said"][w] = r["said"].get(w, 0) + 1
                r["seconds"] += time.time() - t0
                r["n"] += 1
            print(f"L={S} depth {d + 1}/{a.depths}: " + "  ".join(
                f"{m} {res[m]['all']}/{res[m]['n']} ({res[m]['seconds'] / res[m]['n']:.0f}s)" for m in out["modes"])
                + "   after the first: " + " ".join(f"{res[m]['rest']}" for m in out["modes"])
                + "   later tokens right: " + " ".join(f"{res[m]['tok']}/{res[m]['toks']}" for m in out["modes"]),
                flush=True)
        out["lengths"][str(S)] = {m: {"all_digits": r["all"] / max(r["n"], 1), "first_token": r["first"] / max(r["n"], 1),
                                      "digits_after_first": r["rest"] / max(r["n"], 1), "said_first_instead": r["said"],
                                      "later_tokens_right": r["tok"] / max(r["toks"], 1), "later_tokens": r["toks"],
                                      "seconds_per_row": r["seconds"] / max(r["n"], 1), "rows": r["n"]}
                                  for m, r in res.items()}
        out["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
        Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out["lengths"], indent=1))


if __name__ == "__main__":
    main()
