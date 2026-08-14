#!/usr/bin/env python3
"""Build or inspect the Hugging Face datasets length sidecar for a mix manifest.

Usage:
    uv run python scripts/manifest_length_cache.py --manifest /path/to/mix_manifest.json
    uv run python scripts/manifest_length_cache.py --manifest /path/to/mix_manifest.json --force
    uv run python scripts/manifest_length_cache.py --manifest /path/to/mix_manifest.json --num_proc 32
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.dataset_preprocess import load_pretokenized_mix
from data.length_cache import (
    compute_or_load_interleaved_lengths,
    length_cache_paths,
    load_length_cache,
)


def _report(lengths: np.ndarray, cache_dir: Path, meta_path: Path) -> None:
    print(
        json.dumps(
            {
                "lengths_path": str(cache_dir),
                "metadata_path": str(meta_path),
                "format": "hf_datasets_arrow",
                "column": "length",
                "rows": int(lengths.size),
                "min": int(lengths.min()) if lengths.size else None,
                "mean": float(lengths.mean()) if lengths.size else None,
                "p50": float(np.percentile(lengths, 50)) if lengths.size else None,
                "p90": float(np.percentile(lengths, 90)) if lengths.size else None,
                "p99": float(np.percentile(lengths, 99)) if lengths.size else None,
                "max": int(lengths.max()) if lengths.size else None,
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--num_proc",
        type=int,
        default=None,
        help="datasets.map workers (default: min(32, cpu_count-2))",
    )
    args = parser.parse_args()

    manifest = args.manifest.expanduser().resolve()
    if not manifest.is_file():
        raise SystemExit(f"Manifest not found: {manifest}")

    cache_dir, meta_path = length_cache_paths(manifest)
    if not args.force:
        existing = load_length_cache(manifest)
        if existing is not None:
            _report(existing, cache_dir, meta_path)
            return

    print(f"Loading interleaved mix from {manifest} ...", file=sys.stderr, flush=True)
    train_ds, _ = load_pretokenized_mix(manifest)
    lengths = compute_or_load_interleaved_lengths(
        manifest,
        train_ds=train_ds,
        force_recompute=args.force,
        num_proc=args.num_proc,
    )
    _report(lengths, cache_dir, meta_path)


if __name__ == "__main__":
    main()
