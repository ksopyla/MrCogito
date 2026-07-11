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
from transformers import AutoConfig, AutoTokenizer

from data.dataset_preprocess import (
    _normalize_to_text_column,
    _make_tokenize_fn,
    configure_text_tokenizer_for_model_vocab,
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


def _list_recursive(repo: str, subpath: str, timeout: int = 60) -> list[dict]:
    """Recursively walk a directory tree, returning all file entries.

    Used for datasets whose shards are nested (e.g. DCLM:
    global-shard_*/local-shard_*/*.jsonl.zst). The HF tree API is non-recursive,
    so we descend into subdirectories.
    """
    entries = _api_list(repo, subpath, timeout=timeout)
    files: list[dict] = []
    for e in entries:
        if e.get("type") == "directory":
            files.extend(_list_recursive(repo, e["path"], timeout=timeout))
        elif e.get("type") in (None, "file") or e.get("path", "").endswith((".parquet", ".zst", ".jsonl")):
            files.append(e)
    return files


def list_file_urls(spec: dict) -> list[tuple[str, int | None]]:
    """Return (url, expected_size) pairs for a source, capped to max_shards.

    `expected_size` is the HF Hub-reported file size in bytes when available
    (used to validate downloads), else None.

    Supports:
      - explicit `data_files` (filtered to parquet/zst; size unknown -> None)
      - `parquet_glob` / `file_glob` like "sample/10BT/*.parquet" or "data/cot-*.parquet"
      - a `subset` dir (listed, optionally recursive via `recursive: true`)
      - `file_suffix` to select non-parquet files (e.g. ".jsonl.zst")
    """
    repo = spec["hf_id"]
    subset = spec.get("subset") or ""
    explicit = spec.get("data_files")
    suffixes = (".parquet", ".jsonl.zst", ".zst")
    file_suffix = spec.get("file_suffix")  # e.g. ".jsonl.zst"
    if file_suffix:
        suffixes = (file_suffix,)

    def _pick(entries: list[dict]) -> list[tuple[str, int | None]]:
        out = []
        for e in entries:
            p = e.get("path", "")
            if any(p.endswith(s) for s in suffixes):
                out.append((p, e.get("size")))
        return out

    picked: list[tuple[str, int | None]] = []
    if explicit:
        picked = [(u, None) for u in explicit if u.endswith(suffixes)]
    else:
        # Support one glob (file_glob/parquet_glob) or a list (file_globs).
        file_globs = spec.get("file_globs")
        if file_globs is None:
            single = spec.get("file_glob") or spec.get("parquet_glob")
            file_globs = [single] if single else []
        if file_globs:
            for fg in file_globs:
                parent = str(Path(fg).parent)
                name = Path(fg).name
                recursive = spec.get("recursive", False)
                entries = _list_recursive(repo, parent) if recursive else _api_list(repo, parent)
                if "*" not in name:
                    picked.extend(
                        (e["path"], e.get("size")) for e in entries if e.get("path", "") == fg
                    )
                else:
                    star = name.index("*")
                    prefix, suf = name[:star], name[star + 1:]
                    picked.extend(
                        (e["path"], e.get("size"))
                        for e in entries
                        if any(e.get("path", "").endswith(s) for s in suffixes)
                        and Path(e["path"]).name.startswith(prefix)
                        and Path(e["path"]).name.endswith(suf)
                    )
            # Dedup by path preserving order.
            seen = set()
            picked = [x for x in picked if not (x[0] in seen or seen.add(x[0]))]
        elif spec.get("recursive"):
            picked = _pick(_list_recursive(repo, subset))
        elif subset:
            picked = _pick(_api_list(repo, subset))
        else:
            picked = _pick(_api_list(repo, ""))

    cap = spec.get("max_shards")
    if cap:
        picked = picked[:cap]
    if not picked:
        raise RuntimeError(
            f"No files ({suffixes}) found for {repo} subset={subset!r} globs={file_globs!r}"
        )
    return [(HF_RESOLVE.format(repo=repo, path=f), sz) for f, sz in picked]


# Back-compat alias.
list_parquet_urls = list_file_urls


def _download_one(url: str, dest: Path, expected_size: int | None = None, retries: int = 5) -> Path:
    """Download `url` to `dest` with Content-Length validation and retry.

    Verifies the final size against the HTTP Content-Length when available, and
    retries on short reads / connection resets. A partial `.part` file is always
    removed before a retry so we never append to a truncated body.
    """
    if dest.exists() and dest.stat().st_size > 0:
        size = dest.stat().st_size
        if expected_size is None or size == expected_size:
            return dest
        logger.warning(f"  existing {dest.name} is short ({size} < {expected_size}) — re-downloading")
        dest.unlink()

    tmp = dest.with_suffix(dest.suffix + ".part")
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        if tmp.exists():
            tmp.unlink()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "pretokenize_mix/1.0"})
            with urllib.request.urlopen(req, timeout=300) as r:
                clen = int(r.headers.get("Content-Length", 0)) or None
                got = 0
                with open(tmp, "wb") as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        got += len(chunk)
                target = expected_size or clen
                if target and got != target:
                    raise IOError(f"short read: got {got} of {target} bytes")
                if target is None and got == 0:
                    raise IOError("empty response")
            tmp.rename(dest)
            return dest
        except Exception as e:
            last_err = e
            logger.warning(f"  download {dest.name} attempt {attempt}/{retries} failed: {e}")
            time.sleep(2 * attempt)
    raise RuntimeError(f"Failed to download {url} after {retries} attempts: {last_err}")


def download_parquets(items: list[tuple[str, int | None]], raw_dir: Path, workers: int) -> list[Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(_download_one, url, raw_dir / Path(url).name, sz): url
            for url, sz in items
        }
        for i, fut in enumerate(as_completed(futs), 1):
            url = futs[fut]
            try:
                p = fut.result()
                paths.append(p)
                logger.info(f"  downloaded [{i}/{len(items)}] {p.name} ({p.stat().st_size // (1 << 20)} MB)")
            except Exception as e:
                # A few flaky shards (DCLM CDN short-reads) must not kill the whole
                # source — skip and continue. We over-cap by a few shards in the recipe
                # so losing 1-2 still meets max_samples.
                failed.append(url)
                logger.error(f"  SKIPPED {Path(url).name} after retries: {e}")
    if failed:
        logger.warning(f"  {len(failed)} file(s) skipped: {[Path(u).name for u in failed]}")
    if not paths:
        raise RuntimeError(f"All {len(items)} downloads failed for {raw_dir}")
    return sorted(paths)


def _archive_source_raw(spec: dict, name: str, raw_archive_dir: Path, download_workers: int) -> None:
    """Download a source's raw files and move them to the NAS archive (tokenizer-agnostic).

    Skips files already present in the archive with a matching size, so re-runs are
    cheap. Used both for cached sources (tokenize skipped) and as a standalone pass.
    """
    archive_dst = raw_archive_dir / name
    archive_dst.mkdir(parents=True, exist_ok=True)
    logger.info(f"[{name}] archiving raw to {archive_dst}")
    items = list_file_urls(spec)
    # Filter out files already archived with the right size.
    to_fetch = []
    for url, sz in items:
        fname = Path(url).name
        dst = archive_dst / fname
        if dst.exists() and (sz is None or dst.stat().st_size == sz):
            continue
        to_fetch.append((url, sz))
    if not to_fetch:
        logger.info(f"[{name}] raw already fully archived — nothing to do")
        return
    logger.info(f"[{name}] {len(to_fetch)}/{len(items)} files to fetch for archive")
    tmp_raw = raw_archive_dir.parent / "_raw_staging" / name
    paths = download_parquets(to_fetch, tmp_raw, download_workers)
    for p in paths:
        dst = archive_dst / p.name
        try:
            p.replace(dst)
        except OSError:
            import shutil
            shutil.copy2(p, dst)
            p.unlink(missing_ok=True)
    logger.info(f"[{name}] raw archive complete ({len(paths)} files)")


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
    raw_archive_dir: Path | None = None,
) -> dict:
    name = spec.get("name", spec["hf_id"])
    tok_dir = cache_dir / f"{name}"
    train_dir = tok_dir / "train"
    eval_dir = tok_dir / "eval"
    manifest_entry: dict[str, Any] = {"name": name, "weight": spec.get("weight", 1.0), "path": str(tok_dir)}

    if train_dir.exists() and (train_dir / "dataset_info.json").exists():
        logger.info(f"[{name}] tokenized cache exists at {tok_dir} — skipping tokenize")
        manifest_entry["train_path"] = str(train_dir)
        manifest_entry["eval_path"] = str(eval_dir)
        # Even when tokenize is skipped, ensure raw is archived if requested.
        if raw_archive_dir is not None:
            _archive_source_raw(spec, name, raw_archive_dir, download_workers)
        return manifest_entry

    logger.info(f"[{name}] resolving file URLs")
    items = list_file_urls(spec)
    logger.info(f"[{name}] {len(items)} files to download")
    paths = download_parquets(items, raw_dir, download_workers)

    # Pick the loader by file extension: parquet -> parquet, .jsonl.zst/.zst/.jsonl -> json.
    is_json = any(p.suffix in (".zst", ".jsonl") or str(p).endswith(".jsonl.zst") for p in paths)
    if is_json:
        logger.info(f"[{name}] loading {len(paths)} jsonl.zst files (zstd-decompressed by datasets)")
        ds = load_dataset("json", data_files=[str(p) for p in paths], split="train")
    else:
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

    # Pre-truncate gigantic web/PDF docs so the Fast tokenizer never scans a huge string
    # (it would OOM/crash a num_proc worker even though truncation=max_seq_length discards
    # all but ~8k chars). Env-overridable; default 100k chars >> 2048 tokens.
    max_chars = int(os.environ.get("PRETOKENIZE_MAX_CHARS", "100000"))
    tokenize_fn = _make_tokenize_fn(tokenizer, max_seq_length, append_eos_token_id, max_chars=max_chars)

    def _map_resilient(ds, num_proc, **kw):
        """map() with a num_proc=1 fallback: if a worker dies opaquely ('subprocess
        abruptly died'), retry single-process so the real error (MemoryError / bad row)
        surfaces instead of the generic multiprocessing message."""
        try:
            return ds.map(tokenize_fn, batched=True, num_proc=num_proc, **kw)
        except RuntimeError as e:
            if num_proc and num_proc > 1 and "abruptly died" in str(e):
                logger.warning(f"[{name}] multiprocessing map died ({e}); retrying num_proc=1 to surface the real error")
                return ds.map(tokenize_fn, batched=True, num_proc=1, **kw)
            raise

    ntr = max(1, min(train_num_proc, len(src_train)))
    nte = max(1, min(test_num_proc, len(src_eval)))
    logger.info(f"[{name}] tokenizing train={len(src_train):,} (proc={ntr}, max_chars={max_chars}) eval={len(src_eval):,} (proc={nte})")
    src_train = _map_resilient(src_train, ntr, remove_columns=["text"])
    src_eval = _map_resilient(src_eval, nte, remove_columns=["text"])

    # save_to_disk cannot save an interleave; here we save train and eval separately.
    src_train.save_to_disk(str(train_dir))
    src_eval.save_to_disk(str(eval_dir))
    manifest_entry["train_path"] = str(train_dir)
    manifest_entry["eval_path"] = str(eval_dir)
    manifest_entry["train_rows"] = len(src_train)
    manifest_entry["eval_rows"] = len(src_eval)

    # Archive raw parquet/zst to NAS (tokenizer-agnostic) so a future tokenizer
    # switch can re-tokenize without re-downloading; then free NVMe. If no archive
    # dir is configured, delete the raw files to keep NVMe bounded.
    if raw_archive_dir is not None:
        archive_src = raw_archive_dir / name
        archive_dst = raw_archive_dir / name
        archive_dst.mkdir(parents=True, exist_ok=True)
        for p in paths:
            dst = archive_dst / p.name
            if dst.exists() and p.resolve() == dst.resolve():
                # Tokenizing straight from the archive (--raw_dir == --raw_archive_dir):
                # source IS the archive file — unlinking would destroy the archive.
                continue
            if dst.exists() and dst.stat().st_size == p.stat().st_size:
                p.unlink(missing_ok=True)
                continue
            logger.info(f"[{name}] archiving raw {p.name} → {dst}")
            try:
                p.replace(dst)
            except OSError:
                # Cross-device (NVMe → NAS) move falls back to copy+delete.
                import shutil
                shutil.copy2(p, dst)
                p.unlink(missing_ok=True)
        logger.info(f"[{name}] raw archived to {archive_dst}")
    else:
        for p in paths:
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
    logger.info(f"[{name}] DONE train={len(src_train):,} eval={len(src_eval):,} → {tok_dir}")
    return manifest_entry


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mix", required=True, help="Recipe id/path or registered mix name")
    p.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM2-135M")
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--cache_dir", default=None, help="Tokenized cache root (default: $DATASETS_TOK_DIR from remote_paths.sh, or $HF_HOME/datasets_tok — the canonical pre-tokenized corpora tree, per remote-servers SKILL.md)")
    p.add_argument("--raw_dir", default=None, help="Raw parquet root (default: $DATASETS_RAW_DIR, or $HF_HOME/datasets_raw)")
    p.add_argument("--raw_archive_dir", default=None, help="If set, move raw parquet/zst here (per-source subdir) after tokenizing instead of deleting — a tokenizer-agnostic archive for future re-tokenization. E.g. /nas/ml_data/mrcogito/hf_datasets/raw")
    p.add_argument("--archive_raw_only", action="store_true", help="Only download + archive raw files to --raw_archive_dir for ALL sources (no tokenize). Use to populate the NAS archive for sources already tokenized under another tokenizer.")
    p.add_argument("--manifest", default=None, help="Manifest JSON path (default: <cache_dir>/<mix>_manifest.json)")
    p.add_argument("--test_size_percent", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_num_proc", type=int, default=8)
    p.add_argument("--test_num_proc", type=int, default=4)
    p.add_argument("--download_workers", type=int, default=8, help="Parallel shard downloads per source")
    p.add_argument("--jobs", type=int, default=1, help="Concurrent sources (1 = sequential, safer)")
    p.add_argument("--only", default=None, help="Comma-separated source names to process")
    p.add_argument("--objective", default="prefix_suffix", choices=["prefix_suffix", "reconstruction", "reconstruction+contrastive", "causal_lm"])
    args = p.parse_args()

    from training.train_perceiver_denoise import resolve_append_eos_token_id

    # Load .env (HF_TOKEN, HF_HOME on local macOS) so direct invocation without a bash
    # launcher still resolves the same paths. Existing env vars take precedence
    # (load_dotenv does not overwrite), so launcher-set HF_HOME wins on the servers.
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    # Single source of truth: HF_HOME. Everything else is a sibling subdir of it.
    # On the servers remote_paths.sh exports HF_HOME=/home/ksopyla/dev/hf_home and the
    # sibling trees DATASETS_TOK_DIR / DATASETS_RAW_DIR; on local macOS .env sets an
    # absolute HF_HOME (e.g. /Users/.../MrCogito/Cache/hf_home). Do NOT hardcode
    # server paths here.
    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        raise RuntimeError(
            "HF_HOME is not set. Source scripts/remote_paths.sh (servers) or ensure "
            ".env defines HF_HOME (local)."
        )
    hf_home = os.path.abspath(hf_home)

    cache_root = Path(args.cache_dir or os.environ.get("DATASETS_TOK_DIR") or os.path.join(hf_home, "datasets_tok"))
    raw_root = Path(args.raw_dir or os.environ.get("DATASETS_RAW_DIR") or os.path.join(hf_home, "datasets_raw"))
    cache_root.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)
    raw_archive_root = Path(args.raw_archive_dir) if args.raw_archive_dir else None
    if raw_archive_root is not None:
        raw_archive_root.mkdir(parents=True, exist_ok=True)
        logger.info(f"Raw archive (tokenizer-agnostic): {raw_archive_root}")

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
    model_vocab_size = AutoConfig.from_pretrained(args.tokenizer).vocab_size
    split_special_tokens = configure_text_tokenizer_for_model_vocab(
        tokenizer, model_vocab_size
    )
    if split_special_tokens:
        logger.warning(
            "Tokenizer has ids beyond the text-model vocabulary "
            f"({len(tokenizer)} > {model_vocab_size}); splitting literal special-token "
            "strings to prevent invalid embedding indices."
        )
    append_eos = resolve_append_eos_token_id(
        args.objective, is_causal_ar=True, eos_token_id=tokenizer.eos_token_id
    )

    logger.info(f"Mix: {mix_id} | sources: {[s.get('name') for s in sources]} | seq={args.max_seq_length}")
    logger.info(f"Cache: {cache_root} | raw: {raw_root} | manifest: {manifest_path}")

    if args.archive_raw_only:
        if raw_archive_root is None:
            parser.error("--archive_raw_only requires --raw_archive_dir")
        logger.info(f"=== ARCHIVE-RAW-ONLY: populating {raw_archive_root} (no tokenize) ===")
        for spec in sources:
            name = spec.get("name", spec["hf_id"])
            t0 = time.time()
            _archive_source_raw(spec, name, raw_archive_root, args.download_workers)
            logger.info(f"[{name}] archive elapsed {time.time()-t0:.0f}s")
        logger.info("Archive-only pass complete.")
        return

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
                    raw_archive_dir=raw_archive_root,
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
                    _proc_worker, spec, args, cache_root, raw_root, append_eos, raw_archive_root,
                )] = spec.get("name")
            for fut in as_completed(futs):
                entries.append(fut.result())

    manifest = {
        "mix_id": mix_id,
        "mix_origin": meta.get("mix_origin"),
        "mix_recipe_path": meta.get("mix_recipe_path"),
        "tokenizer": args.tokenizer,
        "model_vocab_size": model_vocab_size,
        "split_special_tokens": split_special_tokens,
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


def _proc_worker(spec, args, cache_root, raw_root, append_eos, raw_archive_root=None):
    """Process-pool worker for --jobs>1."""
    from transformers import AutoConfig, AutoTokenizer
    from training.train_perceiver_denoise import resolve_append_eos_token_id as _r
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    configure_text_tokenizer_for_model_vocab(
        tokenizer, AutoConfig.from_pretrained(args.tokenizer).vocab_size
    )
    return tokenize_source(
        spec, tokenizer, args.max_seq_length, append_eos,
        cache_root, raw_root / spec.get("name", spec["hf_id"]),
        args.test_size_percent, args.seed, args.train_num_proc,
        args.test_num_proc, args.download_workers,
        raw_archive_dir=raw_archive_root,
    )


if __name__ == "__main__":
    main()
