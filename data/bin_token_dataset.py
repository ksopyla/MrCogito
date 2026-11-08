"""Fixed-length training rows straight from GPT-2 token shards (modded-nanogpt `.bin` format), memory-mapped.

Format of a shard: 256 int32 header [20240520, 1, n_tokens, ...], then n_tokens uint16 ids; <|endoftext|> (50256)
starts every document (FineWeb-Edu `kjj0/finewebedu10B-gpt2`; LongDocs-lite from scripts/build_longdocs_lite.py).

Two row rules, one per training phase of the E33n protocol (LSA arXiv 2610.42147):
  * "stream" (trunk, FineWeb-Edu at 4k): row i is the i-th contiguous window of the concatenated shards; documents
    are packed and may be cut at row edges; per-token doc_ids (0..k-1, a new id after each <|endoftext|>) stop
    attention across documents.
  * "doc" (long decay, LongDocs at 16k): a row is one contiguous span of ONE document (LSA: "a row one contiguous span
    of one document"); spans are consecutive and non-overlapping; a document shorter than the row is skipped.

A manifest JSON names the sources and their weights:
  {"seq_len": 4096, "rule": "stream", "eot": 50256, "seed": 42,
   "train": [{"name": "fineweb_edu", "files": ["/data/fw/finewebedu_train_*.bin"], "weight": 1.0}],
   "eval":  [{"name": "fineweb_edu", "files": ["/data/fw/finewebedu_val_*.bin"], "rows": 512}]}
Rows of several sources are interleaved by weight in one fixed, seeded order. Items are
{"input_ids": LongTensor[S], "doc_ids": LongTensor[S]} — the contract of data.packed_dataset.PackedDataset, read by
data.data_collators.DataCollatorForCausalLM (which masks the label at every document start).
"""
from __future__ import annotations

import glob
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

MAGIC = 20240520
HEADER_INTS = 256


def write_shard(path: str | Path, tokens) -> None:
    """Write uint16 tokens in the shard format (tests, tiny local pipelines; the builders write the same layout)."""
    arr = np.asarray(tokens, dtype=np.uint16)
    header = np.zeros(HEADER_INTS, dtype=np.int32)
    header[0], header[1], header[2] = MAGIC, 1, len(arr)
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(arr.tobytes())


def open_shard(path: str) -> np.memmap:
    header = np.fromfile(path, dtype=np.int32, count=HEADER_INTS)
    if int(header[0]) != MAGIC:
        raise ValueError(f"{path}: not a token shard (magic {header[0]})")
    n = int(header[2])
    return np.memmap(path, dtype=np.uint16, mode="r", offset=HEADER_INTS * 4, shape=(n,))


def expand(files: list[str]) -> list[str]:
    out: list[str] = []
    for f in files:
        hits = sorted(glob.glob(f))
        if not hits:
            raise FileNotFoundError(f"no shard matches {f}")
        out.extend(hits)
    return out


@dataclass
class _Source:
    name: str
    shards: list[np.memmap]
    rows: np.ndarray  # [n_rows, 2] int64: (shard index, start offset)


def _stream_rows(shards: list[np.memmap], S: int) -> np.ndarray:
    rows = [(si, off) for si, sh in enumerate(shards) for off in range(0, len(sh) - S + 1, S)]
    return np.asarray(rows, dtype=np.int64).reshape(-1, 2)


def _doc_rows(shards: list[np.memmap], S: int, eot: int) -> np.ndarray:
    rows = []
    for si, sh in enumerate(shards):
        starts = np.flatnonzero(np.asarray(sh) == eot)
        ends = np.append(starts[1:], len(sh))
        for a, b in zip(starts, ends):
            for off in range(int(a), int(b) - S + 1, S):
                rows.append((si, off))
    return np.asarray(rows, dtype=np.int64).reshape(-1, 2)


class BinRowDataset(Dataset):
    """Weighted, seeded interleave of fixed-length rows from one or more shard sources."""

    def __init__(self, sources: list[dict], seq_len: int, rule: str = "stream", eot: int = 50256, seed: int = 42,
                 max_rows: int | None = None):
        if rule not in ("stream", "doc"):
            raise ValueError("rule must be 'stream' or 'doc'")
        self.S, self.eot, self.rule = int(seq_len), int(eot), rule
        self.sources: list[_Source] = []
        for spec in sources:
            shards = [open_shard(p) for p in expand(spec["files"])]
            rows = _stream_rows(shards, self.S) if rule == "stream" else _doc_rows(shards, self.S, self.eot)
            if len(rows) == 0:
                raise ValueError(f"source {spec['name']}: no row of {self.S} tokens")
            self.sources.append(_Source(spec["name"], shards, rows))
        rng = np.random.default_rng(seed)
        weights = np.asarray([float(s.get("weight", 1.0)) for s in sources])
        counts = np.asarray([len(s.rows) for s in self.sources])
        # rows per source in proportion to its weight, never more than it holds (a source runs out → it is used once)
        total = int(min(counts / np.maximum(weights / weights.sum(), 1e-12)))
        take = np.minimum(np.floor(weights / weights.sum() * total).astype(np.int64), counts)
        if max_rows is not None:
            scale = min(1.0, max_rows / max(1, int(take.sum())))
            take = np.maximum(1, np.floor(take * scale).astype(np.int64))
        index = []
        for k, (src, n) in enumerate(zip(self.sources, take)):
            pick = rng.permutation(len(src.rows))[: int(n)]
            index.append(np.stack([np.full(len(pick), k), pick], axis=1))
        self.index = np.concatenate(index)[rng.permutation(int(take.sum()))]
        self.rows_per_source = {s.name: int(n) for s, n in zip(self.sources, take)}

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        k, r = self.index[i]
        src = self.sources[int(k)]
        si, off = src.rows[int(r)]
        ids = torch.from_numpy(np.asarray(src.shards[int(si)][int(off): int(off) + self.S], dtype=np.int64))
        doc = torch.cumsum((ids == self.eot).long(), dim=0)
        doc = doc - doc[0]  # 0-based within the row
        return {"input_ids": ids, "doc_ids": doc}


def load_bin_manifest(path: str) -> tuple[BinRowDataset, BinRowDataset | None, dict]:
    m = json.loads(Path(path).read_text())
    S, rule, eot, seed = int(m["seq_len"]), m.get("rule", "stream"), int(m.get("eot", 50256)), int(m.get("seed", 42))
    train = BinRowDataset(m["train"], S, rule, eot, seed)
    ev = None
    if m.get("eval"):
        rows = sum(int(s.get("rows", 256)) for s in m["eval"])
        ev = BinRowDataset(m["eval"], S, rule, eot, seed + 1, max_rows=rows)
    return train, ev, m
