#!/usr/bin/env python
"""Build symbolic long-context task shards (DNA-like alphabet) as a pretokenized manifest.

Each task in `data/symbolic_tasks.py` puts the information needed for the supervised tokens at
a controlled distance, over an alphabet small enough that the cross-entropy floor for a model
that cannot reach the evidence is known in closed form. The manifest records that floor, so a
gate can cite a *derived* ceiling instead of an aspiration.

One source per `--task`, so a single manifest can carry the whole diagnostic suite (or, with
`--lm_columns_only` and `--sym_lo`, be merged into a text mix as E23's dense-label rows).

  # standalone diagnostic suite, pure symbolic vocabulary
  uv run python scripts/build_symbolic_dataset.py --task recall far_copy chain count \
      --seq_len 4096 --min_gap 1024 --n_train 40000 --n_eval 512 \
      --out_dir $DATASETS_TOK_DIR/sym_4k --manifest $DATASETS_TOK_DIR/sym_4k_manifest.json

  # rows destined for a text mix: reserve an id slice and keep the plain LM shard schema
  uv run python scripts/build_symbolic_dataset.py --task far_copy chain --seq_len 32768 \
      --min_gap 4096 --sym_lo 128100 --lm_columns_only --n_train 8000 --n_eval 128 \
      --out_dir $DATASETS_TOK_DIR/sym_32k --manifest $DATASETS_TOK_DIR/sym_32k_manifest.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from datasets import Dataset, Features, Sequence, Value

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.symbolic_tasks import (  # noqa: E402
    TASKS,
    SymbolicTaskConfig,
    chance_accuracy,
    floor_nats,
    generate_row,
)

DIAG_FEATURES = Features(
    {
        "input_ids": Sequence(Value("int32")),
        "labels": Sequence(Value("int32")),
        "gap": Value("int32"),
    }
)
# Must match the columns written by `scripts/pretokenize_mix.py` so `interleave_datasets` can
# mix these rows with text shards.
LM_FEATURES = Features(
    {
        "input_ids": Sequence(Value("int32")),
        "attention_mask": Sequence(Value("int8")),
        "special_tokens_mask": Sequence(Value("int8")),
    }
)


def _iter_rows(cfg_kwargs: dict, n_rows: int, seed: int, lm_columns_only: bool):
    cfg = SymbolicTaskConfig(**cfg_kwargs)
    rng = np.random.default_rng(seed)
    n_symbols, sym_lo = cfg.n_symbols, cfg.sym_lo
    for _ in range(n_rows):
        row = generate_row(cfg, rng)
        ids = row.input_ids.astype(np.int32)
        if lm_columns_only:
            is_control = (ids < sym_lo) | (ids >= sym_lo + n_symbols)
            yield {
                "input_ids": ids.tolist(),
                "attention_mask": np.ones(ids.size, dtype=np.int8).tolist(),
                "special_tokens_mask": is_control.astype(np.int8).tolist(),
            }
        else:
            yield {
                "input_ids": ids.tolist(),
                "labels": row.labels.astype(np.int32).tolist(),
                "gap": int(row.gap),
            }


def _save(path: Path, cfg_kwargs: dict, n_rows: int, seed: int, lm_columns_only: bool) -> None:
    Dataset.from_generator(
        _iter_rows,
        features=LM_FEATURES if lm_columns_only else DIAG_FEATURES,
        gen_kwargs=dict(
            cfg_kwargs=cfg_kwargs, n_rows=n_rows, seed=seed, lm_columns_only=lm_columns_only
        ),
        keep_in_memory=False,
    ).save_to_disk(str(path))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", nargs="+", choices=TASKS, default=["recall"])
    p.add_argument("--seq_len", type=int, default=4096)
    p.add_argument("--n_symbols", type=int, default=4, help="content alphabet size (4 = DNA-like)")
    p.add_argument("--sym_lo", type=int, default=0, help="id of symbol 0; reserve a slice when mixing into text")
    p.add_argument("--min_gap", type=int, default=1024,
                   help="guaranteed distance from evidence to answer; set >= the decoder's raw window")
    p.add_argument("--key_len", type=int, default=4)
    p.add_argument("--value_len", type=int, default=4)
    p.add_argument("--span_len", type=int, default=32)
    p.add_argument("--n_distractors", type=int, default=7)
    p.add_argument("--hops", type=int, default=3)
    p.add_argument("--count_mod", type=int, default=4)
    p.add_argument("--n_train", type=int, default=20000)
    p.add_argument("--n_eval", type=int, default=256)
    p.add_argument("--floor_window", type=int, default=None,
                   help="raw window to report the floor for (default: min_gap - 1, the largest "
                        "window still guaranteed not to see the evidence)")
    p.add_argument("--lm_columns_only", action="store_true",
                   help="emit input_ids/attention_mask/special_tokens_mask only (mixable with text "
                        "shards); supervise via --loss_span_markers instead of a labels column")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--mix_id", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    out = Path(args.out_dir)
    if out.exists():
        if not args.overwrite:
            raise SystemExit(f"{out} exists (use --overwrite)")
        shutil.rmtree(out)

    shared = dict(
        seq_len=args.seq_len,
        n_symbols=args.n_symbols,
        sym_lo=args.sym_lo,
        min_gap=args.min_gap,
        key_len=args.key_len,
        value_len=args.value_len,
        span_len=args.span_len,
        n_distractors=args.n_distractors,
        hops=args.hops,
        count_mod=args.count_mod,
    )
    window = args.floor_window if args.floor_window is not None else args.min_gap - 1
    sources = []
    for i, task in enumerate(args.task):
        cfg_kwargs = {**shared, "task": task}
        cfg = SymbolicTaskConfig(**cfg_kwargs)
        train_path, eval_path = out / task / "train", out / task / "eval"
        _save(train_path, cfg_kwargs, args.n_train, args.seed + 2 * i, args.lm_columns_only)
        _save(eval_path, cfg_kwargs, args.n_eval, args.seed + 2 * i + 1, args.lm_columns_only)
        sources.append(
            {
                "name": f"sym_{task}",
                "weight": 1.0,
                "train_path": str(train_path),
                "eval_path": str(eval_path),
                "num_train_rows": args.n_train,
                "num_eval_rows": args.n_eval,
                "mean_row_tokens": args.seq_len,
                "task": task,
                "answer_len": cfg.answer_len,
                "supervised_tokens_per_row": cfg.answer_len,
                # The contract a gate can cite: a model with a raw window of `floor_window` and no
                # working long-range channel cannot score below this on the supervised tokens.
                "floor_window": window,
                "floor_nats_per_supervised_token": round(floor_nats(cfg, window), 6),
                "chance_accuracy": round(chance_accuracy(cfg), 6),
            }
        )

    vocab = SymbolicTaskConfig(**{**shared, "task": args.task[0]}).vocab
    manifest = {
        "mix_id": args.mix_id or f"symbolic_{'_'.join(args.task)}_{args.seq_len}",
        "objective": "causal_lm",
        "max_seq_length": args.seq_len,
        "seed": args.seed,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "symbolic_meta": {
            "n_symbols": args.n_symbols,
            "sym_lo": args.sym_lo,
            "vocab_size": vocab.vocab_size,
            "min_gap": args.min_gap,
            "lm_columns_only": args.lm_columns_only,
            # Pass these to training as `--loss_span_markers` when there is no labels column.
            "markers": [vocab.control("answer"), vocab.control("end")],
            "generator": "data/symbolic_tasks.py",
        },
        "label_policy": "span_markers" if args.lm_columns_only else "answer_only",
        "sources": sources,
    }
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
