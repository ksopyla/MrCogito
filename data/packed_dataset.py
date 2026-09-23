"""Document packing for long-context causal LM training (E18 / Perceiver AR v2).

Why packing, not length grouping
--------------------------------
`length_group` keeps one document per row and sorts similar lengths together. That drives
the pad ratio down (~5% at 8k), but every short document still occupies its own row, so a
32k / 256k window is mostly *unused* rather than padded: the model never sees a full-length
sequence unless a single document is that long. Packing concatenates whole documents into
one sequence up to `capacity` tokens, so every training sequence exercises the full window
and the per-step compute shape is constant.

Cross-document leakage is prevented in the model, not here: each packed item carries
per-token document ids and `PerceiverARLM.forward(doc_ids=...)` masks attention across
documents, resets RoPE positions per document and keeps hashed n-grams inside documents.
The collator sets the label at every document start to -100, so the last token of one
document is never trained to predict the first token of the next.

Determinism / caching
---------------------
Bins are built once from the cached per-row lengths (same sidecar as `length_group`) with a
fixed seed and stored next to the manifest, so all DDP ranks and every restart see the same
bins. Epochs > 1 reshuffle bin *order* (HF random sampler) but keep bin *composition*; the
E18 budgets are < 1 epoch, so this is a documented simplification, not a limitation we hit.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

PACK_CACHE_VERSION = 1


def pack_rows(
    lengths: Sequence[int] | np.ndarray,
    capacity: int,
    *,
    seed: int,
    open_bins: int = 64,
    close_slack: int | None = None,
) -> list[np.ndarray]:
    """Best-fit online packing of whole rows into bins of at most `capacity` tokens.

    Rows are visited in a seeded random order. A pool of `open_bins` partially filled bins is
    kept; each row goes into the open bin with the *least* remaining room that still fits it
    (best fit). If none fits, the fullest open bin is closed to make room. A bin whose
    remaining room drops to <= `close_slack` tokens (default capacity // 512, i.e. 16 at 8k)
    is closed immediately. Rows longer than
    `capacity` are counted as `capacity` (the collator truncates them) and end up alone.

    Returns a list of int64 index arrays; every row index appears exactly once.
    """
    lengths = np.asarray(lengths, dtype=np.int64).reshape(-1)
    if capacity < 1:
        raise ValueError("capacity must be positive.")
    if open_bins < 1:
        raise ValueError("open_bins must be positive.")
    if close_slack is None:
        close_slack = capacity // 512
    n = int(lengths.size)
    if n == 0:
        return []
    order = np.random.default_rng(seed).permutation(n)
    clipped = np.minimum(lengths, capacity)

    rem = np.full(open_bins, capacity, dtype=np.int64)
    members: list[list[int]] = [[] for _ in range(open_bins)]
    bins: list[np.ndarray] = []

    def close(j: int) -> None:
        if members[j]:
            bins.append(np.asarray(members[j], dtype=np.int64))
        members[j] = []
        rem[j] = capacity

    for idx in order:
        length = int(clipped[idx])
        fits = rem >= length
        if fits.any():
            # smallest remaining room that still fits -> best fit
            candidates = np.where(fits, rem, capacity + 1)
            j = int(np.argmin(candidates))
        else:
            j = int(np.argmin(rem))  # fullest bin
            close(j)
        members[j].append(int(idx))
        rem[j] -= length
        if rem[j] <= close_slack:
            close(j)
    for j in range(open_bins):
        close(j)
    return bins


def packing_stats(bins: list[np.ndarray], lengths: np.ndarray, capacity: int) -> dict[str, float]:
    lengths = np.asarray(lengths, dtype=np.int64)
    clipped = np.minimum(lengths, capacity)
    tokens = float(clipped.sum())
    slots = float(len(bins) * capacity)
    rows_per_bin = np.asarray([len(b) for b in bins], dtype=np.float64)
    return {
        "num_bins": float(len(bins)),
        "num_rows": float(lengths.size),
        "fill_ratio": tokens / slots if slots else 0.0,
        "mean_rows_per_bin": float(rows_per_bin.mean()) if rows_per_bin.size else 0.0,
        "max_rows_per_bin": float(rows_per_bin.max()) if rows_per_bin.size else 0.0,
        "truncated_rows": float((lengths > capacity).sum()),
    }


# --------------------------------------------------------------------------------------
# cache next to the manifest
# --------------------------------------------------------------------------------------
def packed_cache_path(manifest_path: str | Path, capacity: int, seed: int) -> Path:
    return Path(f"{manifest_path}.packed_c{capacity}_s{seed}.npz")


def _flatten(bins: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    sizes = np.asarray([len(b) for b in bins], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    flat = np.concatenate(bins) if bins else np.zeros(0, dtype=np.int64)
    return offsets, flat


def _unflatten(offsets: np.ndarray, flat: np.ndarray) -> list[np.ndarray]:
    return [flat[offsets[i] : offsets[i + 1]] for i in range(len(offsets) - 1)]


def _identity(lengths: np.ndarray, capacity: int, seed: int, open_bins: int) -> dict[str, Any]:
    lengths = np.asarray(lengths, dtype=np.int64)
    return {
        "version": PACK_CACHE_VERSION,
        "n_rows": int(lengths.size),
        "sum_lengths": int(lengths.sum()),
        "capacity": int(capacity),
        "seed": int(seed),
        "open_bins": int(open_bins),
    }


def build_or_load_packed_bins(
    manifest_path: str | Path,
    lengths: np.ndarray,
    *,
    capacity: int,
    seed: int,
    open_bins: int = 64,
) -> list[np.ndarray]:
    """Return cached bins for (manifest, lengths, capacity, seed) or build and cache them."""
    path = packed_cache_path(manifest_path, capacity, seed)
    want = _identity(lengths, capacity, seed, open_bins)
    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as z:
                meta = json.loads(str(z["meta"]))
                if meta == want:
                    return _unflatten(z["offsets"], z["indices"])
                logger.info("Packed-bin cache %s is stale (%s != %s); rebuilding.", path, meta, want)
        except Exception as exc:  # noqa: BLE001 - any corrupt cache is rebuilt
            logger.warning("Packed-bin cache %s unreadable (%s); rebuilding.", path, exc)
    bins = pack_rows(lengths, capacity, seed=seed, open_bins=open_bins)
    offsets, flat = _flatten(bins)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, offsets=offsets, indices=flat, meta=np.array(json.dumps(want)))
    tmp.replace(path)
    return bins


# --------------------------------------------------------------------------------------
# dataset wrapper
# --------------------------------------------------------------------------------------
class PackedDataset(Dataset):
    """Map-style view over `source` where item i is the concatenation of bin i.

    Yields ``{"input_ids": [..], "doc_ids": [..]}`` (plus ``"labels"`` when
    ``preserve_labels=True``). ``doc_ids`` are 0..k-1 per bin; the collator turns them into a
    padded ``[B, S]`` tensor with -1 at padding.
    """

    def __init__(
        self,
        source,
        bins: list[np.ndarray],
        *,
        capacity: int,
        preserve_labels: bool = False,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive.")
        self.source = source
        self.bins = bins
        self.capacity = capacity
        self.preserve_labels = preserve_labels

    def __len__(self) -> int:
        return len(self.bins)

    def __getitem__(self, i: int) -> dict[str, list[int]]:
        rows = self.bins[i].tolist()
        batch = self.source[rows]  # HF datasets: list index -> dict of column lists
        ids_col = batch["input_ids"]
        input_ids: list[int] = []
        doc_ids: list[int] = []
        labels: list[int] = []
        for k, ids in enumerate(ids_col):
            ids = list(ids)[: self.capacity]
            input_ids.extend(int(t) for t in ids)
            doc_ids.extend([k] * len(ids))
            if self.preserve_labels:
                row_labels = list(batch["labels"][k])[: self.capacity]
                if len(row_labels) != len(ids):
                    raise ValueError("Precomputed labels must have the same length as input_ids.")
                labels.extend(int(t) for t in row_labels)
        if len(input_ids) > self.capacity:
            raise RuntimeError(
                f"bin {i} holds {len(input_ids)} tokens > capacity {self.capacity}; "
                "bins must be built from the same lengths as the source rows."
            )
        item: dict[str, list[int]] = {"input_ids": input_ids, "doc_ids": doc_ids}
        if self.preserve_labels:
            item["labels"] = labels
        return item
