#!/usr/bin/env python
"""Compute and cache exact token statistics for a pretokenized mix manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.dataset_preprocess import load_pretokenized_mix


def _count_token_batch(batch: dict) -> dict:
    """Reduce one map batch to one scalar row (picklable for Dataset.map workers)."""
    return {"token_count": [sum(len(ids) for ids in batch["input_ids"])]}


def compute_stats(
    manifest_path: Path,
    target_tokens: int,
    effective_batch: int,
    num_proc: int | None = None,
) -> dict:
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    cache_path = manifest_path.with_suffix(manifest_path.suffix + ".token_stats.json")
    cached: dict | None = None
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        if (
            cached.get("manifest_sha256") == manifest_hash
            and cached.get("target_tokens") == target_tokens
            and cached.get("effective_batch") == effective_batch
            and "epochs_for_target" in cached
            and "estimated_optimizer_steps" in cached
        ):
            return cached
        # Reuse the expensive exact token count when only effective_batch (or
        # derived step math) changes — critical for batch-size calibration sweeps.
        if (
            cached.get("manifest_sha256") == manifest_hash
            and isinstance(cached.get("full_epoch_tokens"), (int, float))
            and cached.get("full_epoch_tokens") > 0
            and isinstance(cached.get("train_rows"), int)
            and cached.get("train_rows") > 0
        ):
            train_tokens = int(cached["full_epoch_tokens"])
            train_rows = int(cached["train_rows"])
            epochs = target_tokens / train_tokens
            optimizer_steps = math.ceil(train_rows * epochs / effective_batch)
            stats = {
                **cached,
                "target_tokens": target_tokens,
                "epochs_for_target": epochs,
                "effective_batch": effective_batch,
                "estimated_optimizer_steps": optimizer_steps,
            }
            temporary_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
            temporary_path.write_text(json.dumps(stats, indent=2))
            temporary_path.replace(cache_path)
            print(
                f"Reused cached token count ({train_tokens:,}); "
                f"recomputed steps for effective_batch={effective_batch} → {optimizer_steps}.",
                file=sys.stderr,
                flush=True,
            )
            return stats

    train_ds, eval_ds = load_pretokenized_mix(manifest_path)
    workers = max(1, min(num_proc or min(8, os.cpu_count() or 1), len(train_ds)))
    started = time.monotonic()
    print(
        f"Counting tokens across {len(train_ds):,} interleaved rows "
        f"with {workers} workers...",
        file=sys.stderr,
        flush=True,
    )
    token_ds = train_ds.select_columns(["input_ids"])
    partials = token_ds.map(
        _count_token_batch,
        batched=True,
        batch_size=8192,
        num_proc=workers,
        remove_columns=token_ds.column_names,
        keep_in_memory=True,
        load_from_cache_file=False,
        desc="Exact token count",
    )
    train_tokens = sum(partials["token_count"])
    elapsed = time.monotonic() - started
    print(
        f"Counted {train_tokens:,} tokens in {elapsed:.1f}s "
        f"({len(train_ds) / max(elapsed, 1e-9):,.0f} rows/s).",
        file=sys.stderr,
        flush=True,
    )
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
    temporary_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(stats, indent=2))
    temporary_path.replace(cache_path)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target_tokens", type=int, required=True)
    parser.add_argument("--effective_batch", type=int, required=True)
    parser.add_argument(
        "--num_proc",
        type=int,
        default=None,
        help="Parallel Dataset.map workers (default: min(8, CPU count)).",
    )
    parser.add_argument(
        "--field",
        choices=["epochs_for_target", "estimated_optimizer_steps"],
        default=None,
        help="Print one machine-readable value instead of the JSON report.",
    )
    args = parser.parse_args()
    stats = compute_stats(
        args.manifest,
        args.target_tokens,
        args.effective_batch,
        num_proc=args.num_proc,
    )
    if args.field:
        print(stats[args.field])
    else:
        print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
