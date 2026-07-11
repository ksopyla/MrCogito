#!/usr/bin/env python
"""Compute and cache exact token statistics for a pretokenized mix manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.dataset_preprocess import load_pretokenized_mix


def compute_stats(manifest_path: Path, target_tokens: int, effective_batch: int) -> dict:
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    cache_path = manifest_path.with_suffix(manifest_path.suffix + ".token_stats.json")
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        if (
            cached.get("manifest_sha256") == manifest_hash
            and cached.get("target_tokens") == target_tokens
            and cached.get("effective_batch") == effective_batch
        ):
            return cached

    train_ds, eval_ds = load_pretokenized_mix(manifest_path)
    train_tokens = 0
    for batch in train_ds.iter(batch_size=8192):
        train_tokens += sum(len(ids) for ids in batch["input_ids"])
    if train_tokens <= 0:
        raise ValueError(f"Manifest has no training tokens: {manifest_path}")

    epochs = target_tokens / train_tokens
    optimizer_steps = math.ceil(len(train_ds) * epochs / effective_batch)
    stats = {
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "train_rows": len(train_ds),
        "eval_rows": len(eval_ds),
        "full_epoch_tokens": train_tokens,
        "average_tokens_per_row": train_tokens / len(train_ds),
        "target_tokens": target_tokens,
        "epochs_for_target": epochs,
        "effective_batch": effective_batch,
        "estimated_optimizer_steps": optimizer_steps,
    }
    cache_path.write_text(json.dumps(stats, indent=2))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target_tokens", type=int, required=True)
    parser.add_argument("--effective_batch", type=int, required=True)
    parser.add_argument(
        "--field",
        choices=["epochs_for_target", "estimated_optimizer_steps"],
        default=None,
        help="Print one machine-readable value instead of the JSON report.",
    )
    args = parser.parse_args()
    stats = compute_stats(args.manifest, args.target_tokens, args.effective_batch)
    if args.field:
        print(stats[args.field])
    else:
        print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
