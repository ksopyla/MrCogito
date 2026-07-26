#!/usr/bin/env python
"""Build an immutable pretokenized delayed-recall dataset and training manifest."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from datasets import Dataset
from dotenv import load_dotenv
from transformers import AutoConfig, AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.dataset_preprocess import configure_text_tokenizer_for_model_vocab
from data.delayed_recall import (
    build_delayed_recall_rows,
    select_delayed_recall_token_pools,
)


def _save_rows(rows: list[dict], path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Dataset already exists: {path}")
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).save_to_disk(str(path))


def build_dataset(args: argparse.Namespace) -> dict:
    if args.manifest.exists() and not args.overwrite:
        raise FileExistsError(f"Manifest already exists: {args.manifest}")
    if args.train_rows % args.value_count:
        raise ValueError("--train_rows must be divisible by --value_count.")
    eval_rows = 2 * args.eval_pairs
    if eval_rows % args.value_count:
        raise ValueError("2 * --eval_pairs must be divisible by --value_count.")
    diagnostic_rows = 2 * args.diagnostic_pairs
    if diagnostic_rows and diagnostic_rows % args.value_count:
        raise ValueError("2 * --diagnostic_pairs must be divisible by --value_count.")
    if args.sequence_length % args.block_size:
        raise ValueError("--sequence_length must be divisible by --block_size.")
    num_blocks = args.sequence_length // args.block_size
    if num_blocks < 4:
        raise ValueError("Delayed-recall build requires at least four blocks.")

    config = AutoConfig.from_pretrained(args.tokenizer)
    model_vocab_size = int(config.vocab_size)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    split_special_tokens = configure_text_tokenizer_for_model_vocab(
        tokenizer, model_vocab_size
    )
    pools = select_delayed_recall_token_pools(
        tokenizer,
        model_vocab_size=model_vocab_size,
        value_count=args.value_count,
        key_count=args.key_count,
        noise_count=args.noise_count,
        seed=args.seed,
    )

    output_dir = args.output_dir.resolve()
    train_path = output_dir / "train"
    eval_path = output_dir / "eval_block4"
    train_rows = build_delayed_recall_rows(
        tokenizer,
        pools,
        split="train",
        num_rows=args.train_rows,
        sequence_length=args.sequence_length,
        block_size=args.block_size,
        query_block=4,
        seed=args.seed + 1,
    )
    eval_block4_rows = build_delayed_recall_rows(
        tokenizer,
        pools,
        split="eval",
        num_rows=eval_rows,
        sequence_length=args.sequence_length,
        block_size=args.block_size,
        query_block=4,
        seed=args.seed + 100,
    )
    if {row["pair_id"] for row in train_rows} & {
        row["pair_id"] for row in eval_block4_rows
    }:
        raise ValueError("Train and eval pair ids overlap.")

    _save_rows(train_rows, train_path, overwrite=args.overwrite)
    _save_rows(eval_block4_rows, eval_path, overwrite=args.overwrite)

    eval_views = {"block4": str(eval_path)}
    for query_block in (2, 3):
        if not args.diagnostic_pairs:
            break
        rows = build_delayed_recall_rows(
            tokenizer,
            pools,
            split="eval",
            num_rows=diagnostic_rows,
            sequence_length=args.sequence_length,
            block_size=args.block_size,
            query_block=query_block,
            seed=args.seed + 100,
        )
        path = output_dir / f"eval_block{query_block}"
        _save_rows(rows, path, overwrite=args.overwrite)
        eval_views[f"block{query_block}"] = str(path)

    manifest = {
        "mix_id": "delayed_recall_v1",
        "dataset_contract_version": 1,
        "tokenizer": args.tokenizer,
        "model_vocab_size": model_vocab_size,
        "split_special_tokens": split_special_tokens,
        "max_seq_length": args.sequence_length,
        "block_size": args.block_size,
        "objective": "causal_lm",
        "label_policy": "answer_only",
        "seed": args.seed,
        "created": datetime.now(timezone.utc).isoformat(),
        "value_token_ids": list(pools.value_ids),
        "decoded_values": list(pools.decoded_values),
        "key_token_ids": list(pools.key_ids),
        "eval_views": eval_views,
        "sources": [
            {
                "name": "delayed_recall_v1",
                "weight": 1.0,
                "train_path": str(train_path),
                "eval_path": str(eval_path),
                "train_rows": len(train_rows),
                "eval_rows": len(eval_block4_rows),
            }
        ],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default="google/gemma-3-1b-pt")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--train_rows", type=int, default=4608)
    parser.add_argument("--eval_pairs", type=int, default=2048)
    parser.add_argument(
        "--diagnostic_pairs",
        type=int,
        default=256,
        help="Held-out pairs per block-2/3 memory-age view; 0 disables them.",
    )
    parser.add_argument("--sequence_length", type=int, default=2048)
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--value_count", type=int, default=64)
    parser.add_argument("--key_count", type=int, default=256)
    parser.add_argument("--noise_count", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    manifest = build_dataset(parse_args())
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
