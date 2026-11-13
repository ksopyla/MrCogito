"""Cached vs uncached greedy decoding on a real checkpoint: same tokens, and the wall time of each (2026-11-12).

A RULER-like prompt per length (filler sentences, one planted 7-digit number a third of the way in, the question at
the end) goes through the harness adapter's `_model_generate` twice: with the cache (nn/incremental.py) and with
`PERCEIVER_AR_GEN_CACHE=0` (the loop that re-runs the row per token). Prints, per length, whether the answers are
equal, both times and their ratio, and the needle and the answer.

Usage (GPU): uv run python evaluation/bench_cached_decoding.py --checkpoint RUN/final --lengths 16384,65536 --new 32
Laptop pipeline check: --device mps --backend sdpa --dtype float32 (sdpa's dense masks run out of memory past ~8k).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _sync(dev: str) -> None:
    if dev.startswith("cuda"):
        torch.cuda.synchronize()
    elif dev == "mps":
        torch.mps.synchronize()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", default="", help="default: the checkpoint directory")
    ap.add_argument("--lengths", default="4096,16384")
    ap.add_argument("--new", type=int, default=32, help="answer tokens (RULER caps at 32)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--backend", default="flex")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--skip_uncached_above", type=int, default=0, help="time only the cached path past this length")
    ap.add_argument("--out", default="")
    ap.add_argument("--warmup", type=int, default=1, help="1: one short untimed run of each path per length first")
    a = ap.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from transformers import AutoTokenizer

    from evaluation.lm_eval_perceiver_ar import PerceiverARLMEval, load_perceiver_ar_for_eval
    from nn.incremental import decode_kind

    tok = AutoTokenizer.from_pretrained(a.tokenizer or a.checkpoint)
    m = load_perceiver_ar_for_eval(a.checkpoint, device=a.device, dtype=getattr(torch, a.dtype),
                                   attn_backend=a.backend, attn_pad_multiple=1)
    try:
        kind = decode_kind(m)
    except NotImplementedError as e:
        kind = f"uncached only ({e})"
    print(f"{a.checkpoint}: {kind}", flush=True)
    rng = random.Random(0)
    filler = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
    fake = SimpleNamespace(model=m, tokenizer=tok, GEN_PAD=PerceiverARLMEval.GEN_PAD)
    rows = []
    for L in (int(x) for x in a.lengths.split(",")):
        num = rng.randint(1_000_000, 9_999_999)
        needle = tok(f" One of the special magic numbers is: {num}. ")["input_ids"]
        q = tok(" What is the special magic number mentioned in the text above? The special magic number is")["input_ids"]
        body = tok(filler * (L // 20 + 1))["input_ids"][: L - len(needle) - len(q)]
        at = len(body) // 3
        ids = torch.tensor([body[:at] + needle + body[at:] + q], device=a.device)
        L0, P = ids.shape[1], fake.GEN_PAD
        # warm-up: the fewest answer tokens that give the same padded row, so flex attention's graph for this shape
        # is compiled before either timed run (otherwise the first path timed pays for the compile)
        warm = max(1, -(-(L0 + a.new) // P) * P - P + 1 - L0)
        res, row = {}, dict(length=L, needle=num)
        try:
            for cached in (True, False):
                if not cached and a.skip_uncached_above and L > a.skip_uncached_above:
                    continue
                os.environ["PERCEIVER_AR_GEN_CACHE"] = "1" if cached else "0"
                if a.warmup:
                    PerceiverARLMEval._model_generate(fake, ids.clone(), L0 + warm, stop=None)
                if a.device.startswith("cuda"):
                    torch.cuda.reset_peak_memory_stats()
                _sync(a.device)
                t0 = time.time()
                g = PerceiverARLMEval._model_generate(fake, ids.clone(), L0 + a.new, stop=None)
                _sync(a.device)
                res[cached] = (g[0, L0:].tolist(), time.time() - t0)
                if a.device.startswith("cuda"):
                    row["cached_peak_gb" if cached else "uncached_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
        except torch.cuda.OutOfMemoryError as e:  # a long row past the card: keep what was timed, go on to the next length
            row["error"] = f"out of memory ({str(e)[:80]})"
            torch.cuda.empty_cache()
        if True in res:
            row.update(cached_s=round(res[True][1], 2), answer=tok.decode(res[True][0]))
        if True in res and False in res:
            row.update(uncached_s=round(res[False][1], 2), same=res[True][0] == res[False][0],
                       speedup=round(res[False][1] / max(res[True][1], 1e-9), 1))
        rows.append(row)
        print(json.dumps(row), flush=True)
        if a.out:  # after every length, so a crash at a long one keeps the shorter ones
            Path(a.out).write_text(json.dumps(dict(checkpoint=a.checkpoint, kind=kind, rows=rows), indent=1))


if __name__ == "__main__":
    main()
