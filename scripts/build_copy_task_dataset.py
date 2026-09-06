#!/usr/bin/env python
"""Build the Perceiver AR mirrored-copy dataset (E18 gate P2) as a pretokenized manifest.

Each row: [BOS] + n random tokens + mirror(n random tokens) + [EOS]; labels are -100 on the
first half so the loss (and the accuracy probe) only score the mirrored half. Token ids are
drawn from a contiguous slice of the vocabulary so the task is tokenizer-agnostic.

  uv run python scripts/build_copy_task_dataset.py --context 32768 --n_train 20000 --n_eval 200 \
      --out_dir $DATASETS_TOK_DIR/copy_32k --manifest $DATASETS_TOK_DIR/copy_32k_manifest.json
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

from datasets import Dataset


def make_rows(n_rows: int, context: int, vocab_lo: int, vocab_hi: int, bos: int, eos: int, seed: int):
    rng = random.Random(seed)
    half = (context - 2) // 2
    rows = []
    for _ in range(n_rows):
        a = [rng.randint(vocab_lo, vocab_hi - 1) for _ in range(half)]
        ids = [bos] + a + a[::-1] + [eos]
        labels = [-100] * (1 + half) + a[::-1] + [eos]
        rows.append({"input_ids": ids, "labels": labels})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--context", type=int, default=32768)
    p.add_argument("--n_train", type=int, default=20000)
    p.add_argument("--n_eval", type=int, default=200)
    p.add_argument("--vocab_lo", type=int, default=1000)
    p.add_argument("--vocab_hi", type=int, default=1256)
    p.add_argument("--bos", type=int, default=128000)
    p.add_argument("--eos", type=int, default=128001)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    out = Path(args.out_dir)
    if out.exists():
        if not args.overwrite:
            raise SystemExit(f"{out} exists (use --overwrite)")
        shutil.rmtree(out)
    train_path, eval_path = out / "train", out / "eval"
    Dataset.from_list(make_rows(args.n_train, args.context, args.vocab_lo, args.vocab_hi, args.bos, args.eos, args.seed)).save_to_disk(str(train_path))
    Dataset.from_list(make_rows(args.n_eval, args.context, args.vocab_lo, args.vocab_hi, args.bos, args.eos, args.seed + 1)).save_to_disk(str(eval_path))
    manifest = {
        "mix_id": f"copy_{args.context}",
        "objective": "causal_lm",
        "max_seq_length": args.context,
        "seed": args.seed,
        "sources": [
            {"name": "copy", "weight": 1.0, "train_path": str(train_path), "eval_path": str(eval_path),
             "num_train_rows": args.n_train, "num_eval_rows": args.n_eval}
        ],
    }
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
