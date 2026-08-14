#!/usr/bin/env python
"""Build or inspect the sequence-length sidecar for a pretokenized manifest."""

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
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    train_ds, _ = load_pretokenized_mix(args.manifest)
    lengths = compute_or_load_interleaved_lengths(
        args.manifest,
        train_ds=train_ds,
        force_recompute=args.force,
    )
    npz_path, meta_path = length_cache_paths(args.manifest)
    print(
        json.dumps(
            {
                "lengths_path": str(npz_path),
                "metadata_path": str(meta_path),
                "rows": int(lengths.size),
                "min": int(lengths.min()),
                "mean": float(lengths.mean()),
                "p50": float(np.percentile(lengths, 50)),
                "p90": float(np.percentile(lengths, 90)),
                "p99": float(np.percentile(lengths, 99)),
                "max": int(lengths.max()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
