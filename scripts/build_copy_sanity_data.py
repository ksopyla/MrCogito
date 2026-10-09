#!/usr/bin/env python
"""Training-pipeline sanity data: every row is a random token block shown twice.

A model that trains correctly learns to predict the second copy within a few million tokens. Used to
tell "the task is too hard at this budget" from "training cannot learn to look back at all". Writes the
same layout as the text-checks data (manifest, tokenizer copy, a minimal meta file), so the runner can
train on it unchanged.

  uv run python scripts/build_copy_sanity_data.py --tokenizer Cache/text_checks/v0_hub/tokenizer \
      --out Cache/text_checks/copy_sanity
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.build_text_checks_data import _row, _save  # noqa: E402


def rows(n: int, span: int, vocab: int, bos: int, eos: int, seed: int):
    rng = random.Random(seed)
    for _ in range(n):
        block = [rng.randrange(10, vocab) for _ in range(span)]
        yield _row(block + block, bos, eos)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--span", type=int, default=64)
    p.add_argument("--n_train", type=int, default=120_000)
    p.add_argument("--n_dev", type=int, default=512)
    a = p.parse_args()
    from transformers import AutoTokenizer

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    tok.save_pretrained(str(out / "tokenizer"))
    v, b, e = len(tok), tok.bos_token_id, tok.eos_token_id
    _save(lambda: rows(a.n_train, a.span, v, b, e, 1), out / "copy" / "train")
    _save(lambda: rows(a.n_dev, a.span, v, b, e, 2), out / "copy" / "dev")
    L = 2 * a.span + 2
    manifest = {"mix_id": "copy_sanity", "max_seq_length": L, "objective": "causal_lm", "seed": 0,
                "tokenizer": str((out / "tokenizer").resolve()),
                "sources": [{"name": "copy", "train_path": str((out / "copy" / "train").resolve()),
                             "eval_path": str((out / "copy" / "dev").resolve()), "weight": 1.0, "in_eval": True}]}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "text_checks_meta.json").write_text(json.dumps({"version": "copy-sanity", "mix": {"mean_row_tokens": L}}))
    print(f"wrote {a.n_train} rows of {L} tokens to {out}")


if __name__ == "__main__":
    main()
