#!/usr/bin/env python
"""Parallel download + tokenize a dataset mix into save_to_disk cache.

Replaces the serial `load_dataset` bottleneck (which downloads shards one at a
time and pulls whole datasets even when `max_samples` is set). This script:

  1. resolves a mix recipe,
  2. for each source lists parquet files via the HF Hub API,
  3. caps to `max_shards` (or explicit `data_files`),
  4. downloads parquet shards in parallel (thread pool),
  5. loads via `load_dataset("parquet", data_files=...)` (no full-dataset scan),
  6. normalizes to a `text` column, splits off an eval slice, tokenizes with
     `num_proc` workers, and `save_to_disk` to a stable cache path,
  7. writes a manifest JSON the training loader consumes via `load_from_disk`.

Sources can be processed concurrently (`--jobs`) so download of the next source
overlaps tokenization of the current one. Raw parquet is deleted after each
source is tokenized to keep disk usage bounded.

Training reuses the cache: see `load_pretokenized_mix` in
`data/dataset_preprocess.py` and the `--pretokenized_manifest` knob.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# Allow `from data...` imports when run as `python scripts/...py`.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import urllib.request
from datasets import load_dataset
from transformers import AutoTokenizer

from data.dataset_preprocess import (
    _normalize_to_text_column,
    _make_tokenize_fn,
)

logger = logging.getLogger("pretokenize_mix")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

HF_API = "https://huggingface.co/api/datasets/{repo}/tree/main/{path}"
HF_RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"


def _api_list(repo: str, subpath: str, timeout: int = 60) -> list[dict]:
    url = HF_API.format(repo=repo, path=subpath.lstrip("/"))
    req = urllib.request.Request(url, headers={"User-Agent": "pretokenize_mix/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def list_parquet_urls(spec: dict) -> list[str]:
    """Return the parquet file URLs for a source, capped to max_shards."""
    repo = spec["hf_id"]
    subset = spec.get("subset") or ""
    explicit = spec.get("data_files")
    if explicit:
        return [u for u in explicit if u.endswith(".parquet")][: spec.get("max_shards") or len(explicit)]

    # Find the subpath that contains train parquets.
    parquet_glob = spec.get("parquet_glob")
    if parquet_glob:
        # e.g. "sample/10BT/*.parquet" or "data/cot-*.parquet"
        parent = str(Path(parquet_glob).parent)
        name = Path(parquet_glob).name
        entries = _api_list(repo, parent)
        if name == "*.parquet":
            files = [e["path"] for e in entries if e.get("path", "").endswith(".parquet")]
        elif name.endswith("*.parquet"):
            stem = name[: -len("*.parquet")]  # e.g. "cot-"
            files = [
                e["path"]
                for e in entries
                if e.get("path", "").endswith(".parquet") and Path(e["path"]).name.startswith(stem)
            ]
        else:
            files = [e["path"] for e in entries if e.get("path", "") == parquet_glob]
    elif subset:
        entries = _api_list(repo, subset)
        files = [e["path"] for e in entries if e.get("path", "").endswith(".parquet")]
    else:
        entries = _api_list(repo, "")
        files = [e["path"] for e in entries if e.get("path", "").endswith(".parquet")]

    cap = spec.get("max_shards")
    if cap:
        files = files[:cap]
    if not files:
        raise RuntimeError(f"No parquet files found for {repo} subset={subset!r} glob={parquet_glob!r}")
    return [HF_RESOLVE.format(repo=repo, path=f) for f in files]


def _download_one(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "pretokenize_mix/1.0"})
    with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    tmp.rename(dest)
    return dest


def download_parquets(urls: list[str], raw_dir: Path, workers: int) -> list[Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(_download_one, url, raw_dir / Path(url).name): url
            for url in urls
        }
        for i, fut in enumerate(as_completed(futs), 1):
            p = fut.result()
            paths.append(p)
            logger.info(f"  downloaded [{i}/{len(urls)}] {p.name} ({p.stat().st_size // (1 << 20)} MB)")
    return sorted(paths)


def tokenize_source(
    spec: dict,
    tokenizer,
    max_seq_length: int,
    append_eos_token_id,
    cache_dir: Path,
    raw_dir: Path,
    test_size_percent: float,
    seed: int,
    train_num_proc: int,
    test_num_proc: int,
    download_workers: int,
) -> dict:
    name = spec.get("name", spec["hf_id"])
    tok_dir = cache_dir / f"{name}"
    train_dir = tok_dir / "train"
    eval_dir = tok_dir / "eval"
    manifest_entry: dict[str, Any] = {"name": name, "weight": spec.get("weight", 1.0), "path": str(tok_dir)}

    if train_dir.exists() and (train_dir / "dataset_info.json").exists():
        logger.info(f"[{name}] tokenized cache exists at {tok_dir} — skipping")
        manifest_entry["train_path"] = str(train_dir)
        manifest_entry["eval_path"] = str(eval_dir)
        return manifest_entry

    logger.info(f"[{name}] resolving parquet URLs")
    urls = list_parquet_urls(spec)
    logger.info(f"[{name}] {len(urls)} parquet files to download")
    paths = download_parquets(urls, raw_dir, download_workers)

    logger.info(f"[{name}] loading {len(paths)} parquet files")
    ds = load_dataset("parquet", data_files=[str(p) for p in paths], split="train")
    ds = _normalize_to_text_column(ds, spec.get("text_columns"))

    max_samples = spec.get("max_samples")
    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))
        logger.info(f"[{name}] capped to max_samples={max_samples}")

    n = len(ds)
    eval_size = max(1, min(int(n * test_size_percent), n - 1, 5000))
    split_ds = ds.train_test_split(test_size=eval_size, seed=seed)
    src_train, src_eval = split_ds["train"], split_ds["test"]

    tokenize_fn = _make_tokenize_fn(tokenizer, max_seq_length, append_eos_token_id)
    ntr = max(1, min(train_num_proc, len(src_train)))
    nte = max(1, min(test_num_proc, len(src_eval)))
    logger.info(f"[{name}] tokenizing train={len(src_train):,} (proc={ntr}) eval={len(src_eval):,} (proc={nte})")
    src_train = src_train.map(tokenize_fn, batched=True, num_proc=ntr, remove_columns=["text"])
    src_eval = src_eval.map(tokenize_fn, batched=True, num_proc=nte, remove_columns=["text"])

    # save_to_disk cannot save an interleave; here we save train and eval separately.
    src_train.save_to_disk(str(train_dir))
    src_eval.save_to_disk(str(eval_dir))
    manifest_entry["train_path"] = str(train_dir)
    manifest_entry["eval_path"] = str(eval_dir)
    manifest_entry["train_rows"] = len(src_train)
    manifest_entry["eval_rows"] = len(src_eval)

    # Free raw parquet to keep disk bounded.
    for p in paths:
        try:
            p.unlink()
        except OSError:
            pass
    logger.info(f"[{name}] DONE train={len(src_train):,} eval={len(src_eval):,} → {tok_dir}")
    return manifest_entry


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mix", required=True, help="Recipe id/path or registered mix name")
    p.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM2-135M")
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--cache_dir", default=None, help="Tokenized cache root (default: $HF_DATASETS_CACHE/../datasets_tok)")
    p.add_argument("--raw_dir", default=None, help="Raw parquet root (default: $HF_DATASETS_CACHE/../datasets_raw)")
    p.add_argument("--manifest", default=None, help="Manifest JSON path (default: <cache_dir>/<mix>_manifest.json)")
    p.add_argument("--test_size_percent", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_num_proc", type=int, default=8)
    p.add_argument("--test_num_proc", type=int, default=4)
    p.add_argument("--download_workers", type=int, default=8, help="Parallel shard downloads per source")
    p.add_argument("--jobs", type=int, default=1, help="Concurrent sources (1 = sequential, safer)")
    p.add_argument("--only", default=None, help="Comma-separated source names to process")
    p.add_argument("--objective", default="prefix_suffix", choices=["prefix_suffix", "reconstruction", "reconstruction+contrastive"])
    args = p.parse_args()

    from training.train_perceiver_denoise import resolve_append_eos_token_id

    hf_cache = os.environ.get("HF_DATASETS_CACHE", os.path.expanduser("~/dev/hf_home/datasets"))
    cache_root = Path(args.cache_dir or os.path.join(os.path.dirname(hf_cache), "datasets_tok"))
    raw_root = Path(args.raw_dir or os.path.join(os.path.dirname(hf_cache), "datasets_raw"))
    cache_root.mkdir(parents=True, exist_ok=True)
    (raw_root).mkdir(parents=True, exist_ok=True)

    from data.dataset_preprocess import _resolve_mix_sources
    sources, meta = _resolve_mix_sources(args.mix)
    mix_id = meta.get("mix_id", "mix")
    manifest_path = Path(args.manifest or cache_root / f"{mix_id}_manifest.json")

    if args.only:
        wanted = set(s.strip() for s in args.only.split(","))
        sources = [s for s in sources if s.get("name") in wanted]

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    append_eos = resolve_append_eos_token_id(
        args.objective, is_causal_ar=True, eos_token_id=tokenizer.eos_token_id
    )

    logger.info(f"Mix: {mix_id} | sources: {[s.get('name') for s in sources]} | seq={args.max_seq_length}")
    logger.info(f"Cache: {cache_root} | raw: {raw_root} | manifest: {manifest_path}")

    entries: list[dict] = []
    if args.jobs <= 1:
        for spec in sources:
            t0 = time.time()
            try:
                entry = tokenize_source(
                    spec, tokenizer, args.max_seq_length, append_eos,
                    cache_root, raw_root / spec.get("name", spec["hf_id"]),
                    args.test_size_percent, args.seed, args.train_num_proc,
                    args.test_num_proc, args.download_workers,
                )
                entries.append(entry)
                logger.info(f"[{entry['name']}] elapsed {time.time()-t0:.0f}s")
            except Exception as e:
                logger.exception(f"[{spec.get('name')}] FAILED: {e}")
                raise
    else:
        from concurrent.futures import ProcessPoolExecutor
        # NOTE: concurrent sources each spawn their own num_proc tokenize workers;
        # keep jobs * train_num_proc <= core count to avoid overload.
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = {}
            for spec in sources:
                futs[ex.submit(
                    _proc_worker, spec, args, cache_root, raw_root, append_eos,
                )] = spec.get("name")
            for fut in as_completed(futs):
                entries.append(fut.result())

    manifest = {
        "mix_id": mix_id,
        "mix_origin": meta.get("mix_origin"),
        "mix_recipe_path": meta.get("mix_recipe_path"),
        "tokenizer": args.tokenizer,
        "max_seq_length": args.max_seq_length,
        "objective": args.objective,
        "append_eos_token_id": append_eos,
        "seed": args.seed,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "sources": entries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info(f"Manifest written: {manifest_path}")
    logger.info(f"Total sources: {len(entries)} | rows: {sum(e.get('train_rows',0) for e in entries):,} train")


def _proc_worker(spec, args, cache_root, raw_root, append_eos):
    """Process-pool worker for --jobs>1."""
    from transformers import AutoTokenizer
    from training.train_perceiver_denoise import resolve_append_eos_token_id as _r
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenize_source(
        spec, tokenizer, args.max_seq_length, append_eos,
        cache_root, raw_root / spec.get("name", spec["hf_id"]),
        args.test_size_percent, args.seed, args.train_num_proc,
        args.test_num_proc, args.download_workers,
    )


if __name__ == "__main__":
    main()
