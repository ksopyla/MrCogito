"""Persistent sequence lengths aligned to a pretokenized manifest's train rows."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def length_cache_paths(manifest_path: str | Path) -> tuple[Path, Path]:
    """Return the dense length array and human-readable metadata sidecars."""
    manifest = Path(manifest_path)
    return (
        Path(f"{manifest}.lengths.npz"),
        Path(f"{manifest}.lengths.meta.json"),
    )


def _manifest_metadata(manifest_path: Path, n_rows: int) -> dict[str, Any]:
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "n_rows": n_rows,
        "seed": int(manifest.get("seed", 42)),
        "max_seq_length": manifest.get("max_seq_length"),
    }


def _valid_cached_lengths(
    npz_path: Path,
    meta_path: Path,
    expected: dict[str, Any],
) -> np.ndarray | None:
    if not npz_path.exists() or not meta_path.exists():
        return None
    try:
        metadata = json.loads(meta_path.read_text())
        for key in ("manifest_sha256", "n_rows", "seed", "max_seq_length"):
            if metadata.get(key) != expected.get(key):
                return None
        with np.load(npz_path, allow_pickle=False) as archive:
            lengths = archive["lengths"]
        if lengths.dtype != np.int32 or lengths.shape != (expected["n_rows"],):
            return None
        if lengths.size and bool(np.any(lengths < 1)):
            return None
        return lengths
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def _compute_lengths(train_ds, *, batch_size: int = 8192) -> np.ndarray:
    lengths = np.empty(len(train_ds), dtype=np.int32)
    offset = 0
    for batch in train_ds.select_columns(["input_ids"]).iter(batch_size=batch_size):
        batch_lengths = np.fromiter(
            (len(input_ids) for input_ids in batch["input_ids"]),
            dtype=np.int32,
            count=len(batch["input_ids"]),
        )
        lengths[offset : offset + len(batch_lengths)] = batch_lengths
        offset += len(batch_lengths)
    if offset != len(train_ds):
        raise RuntimeError(
            f"Length scan visited {offset} rows but the train dataset has {len(train_ds)}."
        )
    if lengths.size and bool(np.any(lengths < 1)):
        raise ValueError("Pretokenized training rows must contain at least one token.")
    return lengths


def compute_or_load_interleaved_lengths(
    manifest_path: str | Path,
    *,
    train_ds,
    force_recompute: bool = False,
) -> np.ndarray:
    """Load or compute int32 lengths in the manifest's interleaved index space."""
    manifest = Path(manifest_path)
    if not manifest.exists():
        raise FileNotFoundError(f"Pretokenized manifest not found: {manifest}")
    expected = _manifest_metadata(manifest, len(train_ds))
    npz_path, meta_path = length_cache_paths(manifest)
    if not force_recompute:
        cached = _valid_cached_lengths(npz_path, meta_path, expected)
        if cached is not None:
            return cached

    lengths = _compute_lengths(train_ds)
    npz_tmp = Path(f"{npz_path}.tmp")
    meta_tmp = Path(f"{meta_path}.tmp")
    with npz_tmp.open("wb") as handle:
        np.savez_compressed(handle, lengths=lengths)
    metadata = {
        **expected,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "min_length": int(lengths.min()) if lengths.size else None,
        "mean_length": float(lengths.mean()) if lengths.size else None,
        "max_length": int(lengths.max()) if lengths.size else None,
    }
    meta_tmp.write_text(json.dumps(metadata, indent=2) + "\n")
    npz_tmp.replace(npz_path)
    meta_tmp.replace(meta_path)
    return lengths
