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

import numpy as np
from datasets import Dataset, Features, Sequence, Value

FEATURES = Features({"input_ids": Sequence(Value("int32")), "labels": Sequence(Value("int32"))})


def iter_rows(n_rows: int, context: int, vocab_lo: int, vocab_hi: int, bos: int, eos: int, seed: int, task: str = "mirror"):
    """Yield rows one at a time (numpy RNG; int32 columns) so arrow never sees one giant list."""
    rng = np.random.default_rng(seed)
    half = (context - 2) // 2
    for _ in range(n_rows):
        a = rng.integers(vocab_lo, vocab_hi, size=half, dtype=np.int32)
        # "mirror" (paper-style reversal; position-varying offset) or "copy" (plain forward copy at a
        # fixed offset of `half` tokens — the retrieval the single global read must implement; see
        # verification/e18_copy_tiny.py for why mirror is not learnable quickly by RoPE-only models).
        mirrored = a[::-1] if task == "mirror" else a
        ids = np.concatenate([[bos], a, mirrored, [eos]]).astype(np.int32)
        labels = np.concatenate([np.full(1 + half, -100, dtype=np.int32), mirrored, [eos]]).astype(np.int32)
        yield {"input_ids": ids.tolist(), "labels": labels.tolist()}


def make_rows(n_rows: int, context: int, vocab_lo: int, vocab_hi: int, bos: int, eos: int, seed: int, task: str = "mirror"):
    return list(iter_rows(n_rows, context, vocab_lo, vocab_hi, bos, eos, seed, task))


def save_rows(path, n_rows, context, vocab_lo, vocab_hi, bos, eos, seed, task="mirror"):
    ds = Dataset.from_generator(
        iter_rows,
        features=FEATURES,
        gen_kwargs=dict(n_rows=n_rows, context=context, vocab_lo=vocab_lo, vocab_hi=vocab_hi, bos=bos, eos=eos, seed=seed, task=task),
        keep_in_memory=False,
    )
    ds.save_to_disk(str(path))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--context", type=int, default=32768)
    p.add_argument("--n_train", type=int, default=20000)
    p.add_argument("--n_eval", type=int, default=200)
    p.add_argument("--task", choices=["mirror", "copy"], default="mirror",
                   help="mirror = reversed second half (paper); copy = plain forward copy (fixed offset)")
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
    save_rows(train_path, args.n_train, args.context, args.vocab_lo, args.vocab_hi, args.bos, args.eos, args.seed, args.task)
    save_rows(eval_path, args.n_eval, args.context, args.vocab_lo, args.vocab_hi, args.bos, args.eos, args.seed + 1, args.task)
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
