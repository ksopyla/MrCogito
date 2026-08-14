"""Sequence-length cache aligned to a pretokenized manifest's train rows.

Lengths are computed with Hugging Face ``datasets.Dataset.map`` (batched,
multiprocess) and stored with ``save_to_disk`` as a one-column Arrow dataset.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any

from datasets import Dataset, Features, Value, load_from_disk
from transformers.utils import logging
import numpy as np


logger = logging.get_logger(__name__)

_LENGTH_FEATURES = Features({"length": Value("int32")})


def length_cache_paths(manifest_path: str | Path) -> tuple[Path, Path]:
    """Return the Arrow length dataset directory and metadata sidecar."""
    manifest = Path(manifest_path)
    return Path(f"{manifest}.lengths"), Path(f"{manifest}.lengths.meta.json")


def cache_path_for_manifest(manifest_path: str | Path) -> Path:
    """Return the on-disk Hugging Face dataset directory for this manifest."""
    return length_cache_paths(manifest_path)[0]


def _manifest_metadata(manifest_path: Path, n_rows: int) -> dict[str, Any]:
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "n_rows": n_rows,
        "seed": int(manifest.get("seed", 42)),
        "max_seq_length": manifest.get("max_seq_length"),
        "format": "hf_datasets_arrow",
        "column": "length",
    }


def _default_num_proc(n_rows: int, requested: int | None) -> int:
    cpus = os.cpu_count() or 1
    auto = min(32, max(1, cpus - 2))
    workers = auto if requested is None else requested
    if workers < 1:
        raise ValueError("num_proc must be >= 1.")
    return max(1, min(workers, n_rows))


def _length_batch(batch: dict) -> dict:
    """Picklable batched map: one int32 length per ``input_ids`` row."""
    return {"length": [len(ids) for ids in batch["input_ids"]]}


def _lengths_from_dataset(length_ds: Dataset) -> np.ndarray:
    return np.asarray(length_ds.with_format("numpy")["length"], dtype=np.int32)


def _valid_cached_lengths(
    dataset_dir: Path,
    meta_path: Path,
    expected: dict[str, Any],
) -> np.ndarray | None:
    if not dataset_dir.is_dir() or not meta_path.exists():
        return None
    try:
        metadata = json.loads(meta_path.read_text())
        for key in ("manifest_sha256", "n_rows", "seed", "max_seq_length"):
            if metadata.get(key) != expected.get(key):
                return None
        cached = load_from_disk(str(dataset_dir))
        if "length" not in cached.column_names:
            return None
        lengths = _lengths_from_dataset(cached)
        if lengths.shape != (expected["n_rows"],):
            return None
        if lengths.size and bool(np.any(lengths < 1)):
            return None
        return lengths
    except (OSError, ValueError, KeyError, json.JSONDecodeError, TypeError):
        return None


def _write_metadata(meta_path: Path, metadata: dict[str, Any]) -> None:
    meta_tmp = Path(f"{meta_path}.tmp")
    meta_tmp.write_text(json.dumps(metadata, indent=2) + "\n")
    meta_tmp.replace(meta_path)


def _delete_stale_npz(manifest: Path) -> None:
    stale_npz = Path(f"{manifest}.lengths.npz")
    if stale_npz.exists():
        stale_npz.unlink()


def _atomic_save_length_dataset(length_ds: Dataset, dest: Path) -> None:
    tmp = Path(f"{dest}.tmp")
    old = Path(f"{dest}.old")
    if tmp.exists():
        shutil.rmtree(tmp)
    length_ds.save_to_disk(str(tmp), num_shards=1)
    if dest.exists():
        if old.exists():
            shutil.rmtree(old)
        dest.rename(old)
    tmp.rename(dest)
    if old.exists():
        shutil.rmtree(old, ignore_errors=True)


def _length_dataset_from_array(lengths: np.ndarray) -> Dataset:
    return Dataset.from_dict({"length": np.asarray(lengths, dtype=np.int32)}, features=_LENGTH_FEATURES)


def save_length_cache(manifest_path: str | Path, lengths: np.ndarray) -> None:
    """Write an Arrow length dataset + metadata sidecar for tests and rebuilds."""
    manifest = Path(manifest_path)
    lengths = np.asarray(lengths, dtype=np.int32).reshape(-1)
    expected = _manifest_metadata(manifest, int(lengths.size))
    dataset_dir, meta_path = length_cache_paths(manifest)
    metadata = {
        **expected,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "min_length": int(lengths.min()) if lengths.size else None,
        "mean_length": float(lengths.mean()) if lengths.size else None,
        "max_length": int(lengths.max()) if lengths.size else None,
        "num_proc": None,
    }
    _atomic_save_length_dataset(_length_dataset_from_array(lengths), dataset_dir)
    _write_metadata(meta_path, metadata)
    _delete_stale_npz(manifest)


def load_length_cache(manifest_path: str | Path) -> np.ndarray | None:
    """Return cached lengths if the sidecar matches this manifest identity."""
    manifest = Path(manifest_path)
    if not manifest.exists():
        return None
    dataset_dir, meta_path = length_cache_paths(manifest)
    if not meta_path.exists():
        return None
    try:
        n_rows = int(json.loads(meta_path.read_text())["n_rows"])
    except (OSError, KeyError, json.JSONDecodeError, TypeError, ValueError):
        return None
    expected = _manifest_metadata(manifest, n_rows)
    return _valid_cached_lengths(dataset_dir, meta_path, expected)


def _compute_lengths(
    train_ds,
    *,
    num_proc: int,
    batch_size: int = 8192,
) -> Dataset:
    token_ds = train_ds.select_columns(["input_ids"])
    logger.info(
        f"Computing sequence lengths over {len(token_ds):,} rows "
        f"with datasets.map num_proc={num_proc}, batch_size={batch_size}."
    )
    started = time.monotonic()
    map_kwargs: dict[str, Any] = {
        "function": _length_batch,
        "batched": True,
        "batch_size": batch_size,
        "remove_columns": token_ds.column_names,
        "features": _LENGTH_FEATURES,
        "load_from_cache_file": False,
        "keep_in_memory": True,
        "desc": "Sequence lengths",
    }
    if num_proc > 1:
        map_kwargs["num_proc"] = num_proc
    length_ds = token_ds.map(**map_kwargs)
    elapsed = time.monotonic() - started
    logger.info(
        f"Computed sequence lengths in {elapsed:.1f}s "
        f"({len(token_ds) / max(elapsed, 1e-9):,.0f} rows/s)."
    )
    return length_ds


def compute_or_load_interleaved_lengths(
    manifest_path: str | Path,
    *,
    train_ds,
    num_proc: int | None = None,
    force_recompute: bool = False,
    force: bool | None = None,
) -> np.ndarray:
    """Return int32[N] lengths aligned to ``load_pretokenized_mix(train)`` indices."""
    if force is not None:
        force_recompute = bool(force)
    manifest = Path(manifest_path)
    if not manifest.exists():
        raise FileNotFoundError(f"Pretokenized manifest not found: {manifest}")
    expected = _manifest_metadata(manifest, len(train_ds))
    dataset_dir, meta_path = length_cache_paths(manifest)
    if not force_recompute:
        cached = _valid_cached_lengths(dataset_dir, meta_path, expected)
        if cached is not None:
            logger.info(
                f"Loaded sequence-length cache from {dataset_dir} "
                f"({len(cached):,} rows)."
            )
            return cached

    workers = _default_num_proc(len(train_ds), num_proc)
    length_ds = _compute_lengths(train_ds, num_proc=workers)
    lengths = _lengths_from_dataset(length_ds)
    if lengths.shape != (expected["n_rows"],):
        raise RuntimeError(
            f"Length map produced {lengths.size} rows but the train dataset has "
            f"{expected['n_rows']}."
        )
    if lengths.size and bool(np.any(lengths < 1)):
        raise ValueError("Pretokenized training rows must contain at least one token.")

    metadata = {
        **expected,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "min_length": int(lengths.min()) if lengths.size else None,
        "mean_length": float(lengths.mean()) if lengths.size else None,
        "max_length": int(lengths.max()) if lengths.size else None,
        "num_proc": workers,
    }
    _atomic_save_length_dataset(length_ds, dataset_dir)
    _write_metadata(meta_path, metadata)
    _delete_stale_npz(manifest)
    logger.info(f"Wrote sequence-length cache to {dataset_dir}.")
    return lengths
