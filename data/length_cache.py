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
    """Zero-copy Arrow int32 column → numpy. Avoid Dataset.__getitem__ formatting."""
    column = length_ds.data.table.column("length")
    if hasattr(column, "combine_chunks"):
        column = column.combine_chunks()
    return np.asarray(column.to_numpy(), dtype=np.int32)


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


def _lengths_from_list_offsets(train_ds) -> np.ndarray | None:
    """Row lengths straight from the Arrow list offsets of ``input_ids`` — no token bytes read.

    A pretokenized mix is ``concatenate_datasets(shards).select(indices)``: the table holds
    every source row once and ``_indices`` is the interleave order. The list offsets are a
    4- or 8-byte-per-row buffer, so this takes seconds where the batched ``map`` over the
    token column takes ~40 min for a 2.7M-row / 7.8B-token mix. Returns None when the
    layout is not a plain (large) list column so the caller can fall back to the scan.
    """
    try:
        import pyarrow as pa

        table = train_ds.data.table
        column = table.column("input_ids")
        parts: list[np.ndarray] = []
        for chunk in column.chunks:
            if not pa.types.is_list(chunk.type) and not pa.types.is_large_list(chunk.type):
                return None
            if chunk.null_count:
                return None
            offsets = np.asarray(chunk.offsets.to_numpy(zero_copy_only=False), dtype=np.int64)
            parts.append(offsets[1:] - offsets[:-1])
        base = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)
        indices = getattr(train_ds, "_indices", None)
        if indices is not None:
            idx = np.asarray(indices.column(0).to_numpy(zero_copy_only=False), dtype=np.int64)
            base = base[idx]
        if base.shape != (len(train_ds),) or (base.size and base.max() > np.iinfo(np.int32).max):
            return None
        return base.astype(np.int32)
    except Exception as e:  # noqa: BLE001 — any layout surprise falls back to the exact scan
        logger.warning(f"Fast length path from Arrow offsets unavailable ({type(e).__name__}: {e}); scanning.")
        return None


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
        "desc": "Sequence lengths",
    }
    if num_proc > 1:
        map_kwargs["num_proc"] = num_proc
    try:
        length_ds = token_ds.map(**map_kwargs)
    except RuntimeError as e:
        # Forked map workers can die at random on some hosts ("abruptly died"); fall back to
        # a single-process pass rather than failing the launch.
        if num_proc > 1 and "abruptly died" in str(e):
            logger.warning("datasets.map workers died computing lengths; retrying single-process")
            map_kwargs.pop("num_proc", None)
            length_ds = token_ds.map(**map_kwargs)
        else:
            raise
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

    workers: int | None = None
    started = time.monotonic()
    lengths = _lengths_from_list_offsets(train_ds)
    if lengths is not None:
        logger.info(
            f"Computed {lengths.size:,} sequence lengths from Arrow list offsets in "
            f"{time.monotonic() - started:.1f}s."
        )
        length_ds = _length_dataset_from_array(lengths)
    else:
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
        "method": "arrow_list_offsets" if workers is None else "datasets_map",
    }
    _atomic_save_length_dataset(length_ds, dataset_dir)
    _write_metadata(meta_path, metadata)
    _delete_stale_npz(manifest)
    logger.info(f"Wrote sequence-length cache to {dataset_dir}.")
    return lengths
