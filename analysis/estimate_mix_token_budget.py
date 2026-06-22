"""Estimate token budgets for dataset mixes using HF map + tokenizer lengths.

Designed for long-context pretraining planning:
  - uses map-style datasets (same family as training pipeline)
  - uses tokenizer(length) via batched map with num_proc workers
  - writes results to Cache/Evaluation_reports/token_budget

Outputs include, per source:
  - measured rows and token counts on processed rows
  - estimated tokens at max_samples
  - estimated tokens for full split
  - train/eval rows matching load_and_preprocess_dataset_mix split logic

And per mix:
  - baseline no-oversample 1-epoch train token budget
  - interleave(all_exhausted) 1-epoch train token budget estimate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import load_dataset, load_dataset_builder
from transformers import AutoTokenizer

# Allow "from data..." imports when run as `python analysis/...py`.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset_preprocess import resolve_mix_sources

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Estimate token budgets for dataset mixes")
    p.add_argument(
        "--mix",
        nargs="+",
        required=True,
        help=(
            "Mix identifiers to analyze. Each can be a registered DATASET_MIXES key "
            "(e.g. long_2k_base_v1) or a recipe id/path (e.g. smollm3_inspired_2k)."
        ),
    )
    p.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM2-135M")
    p.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Training sequence length for truncated token budget estimates.",
    )
    p.add_argument(
        "--append_eos",
        action="store_true",
        default=True,
        help="Assume training pipeline appends EOS to each sample (default true).",
    )
    p.add_argument(
        "--rows_per_source",
        type=int,
        default=200_000,
        help=(
            "Rows to tokenize per source for average-length estimation. "
            "Set <=0 to use full max_samples rows (can be very expensive)."
        ),
    )
    p.add_argument(
        "--test_size_percent",
        type=float,
        default=0.1,
        help="Must match load_and_preprocess_dataset_mix split logic.",
    )
    p.add_argument("--num_proc", type=int, default=max(1, (os.cpu_count() or 8) - 2))
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--cache_dir", default="./Cache/Datasets")
    p.add_argument("--out_dir", default="./Cache/Evaluation_reports/token_budget")
    p.add_argument(
        "--hf_token_env",
        default="HF_TOKEN",
        help="Env var containing HF token for gated/private datasets.",
    )
    return p.parse_args()


def _eval_size_for_loader(n_rows: int, test_size_percent: float) -> int:
    # Mirrors data.dataset_preprocess.load_and_preprocess_dataset_mix logic.
    return max(1, min(int(n_rows * test_size_percent), n_rows - 1, 5000))


def _load_source_dataset(
    spec: dict[str, Any],
    cache_dir: str,
    split_override: str | None = None,
    cap_rows: int | None = None,
    token: str | None = None,
):
    split = split_override or spec.get("split", "train")
    if cap_rows is not None and cap_rows > 0:
        split = f"{split}[:{int(cap_rows)}]"

    common_kwargs: dict[str, Any] = {"split": split, "cache_dir": cache_dir}
    if token:
        common_kwargs["token"] = token
    if spec.get("revision"):
        common_kwargs["revision"] = spec["revision"]

    data_files = spec.get("data_files")
    if data_files:
        ds = load_dataset("parquet", data_files=data_files, **common_kwargs)
    else:
        subset = spec.get("subset") or None
        dataset_kwargs: dict[str, Any] = {}
        if spec.get("data_dir"):
            dataset_kwargs["data_dir"] = spec["data_dir"]
        if spec.get("trust_remote_code") is not None:
            dataset_kwargs["trust_remote_code"] = bool(spec["trust_remote_code"])
        ds = load_dataset(spec["hf_id"], subset, **dataset_kwargs, **common_kwargs)
    return ds


def _get_full_split_rows(spec: dict[str, Any], cache_dir: str, token: str | None = None) -> int:
    """Best-effort full split row count without downloading full data."""
    split = spec.get("split", "train")
    try:
        builder_kwargs: dict[str, Any] = {"cache_dir": cache_dir}
        if token:
            builder_kwargs["token"] = token
        if spec.get("revision"):
            builder_kwargs["revision"] = spec["revision"]

        data_files = spec.get("data_files")
        if data_files:
            builder = load_dataset_builder("parquet", data_files=data_files, **builder_kwargs)
        else:
            subset = spec.get("subset") or None
            if spec.get("data_dir"):
                builder_kwargs["data_dir"] = spec["data_dir"]
            if spec.get("trust_remote_code") is not None:
                builder_kwargs["trust_remote_code"] = bool(spec["trust_remote_code"])
            builder = load_dataset_builder(spec["hf_id"], subset, **builder_kwargs)

        if builder.info and builder.info.splits and split in builder.info.splits:
            num = builder.info.splits[split].num_examples
            if num is not None:
                return int(num)
    except Exception as ex:
        logger.warning(
            "Falling back to loading split for row count (%s/%s): %s",
            spec.get("hf_id"),
            spec.get("subset"),
            ex,
        )

    # Fallback: load split and ask len(). This can be expensive.
    ds = _load_source_dataset(spec, cache_dir, token=token)
    return int(len(ds))


def _normalize_text_columns(ds, text_columns: list[str]):
    cols = [c for c in (text_columns or ["text"]) if c in ds.column_names]
    if not cols:
        raise ValueError(f"text columns not found in source. available={ds.column_names}")
    if cols == ["text"]:
        # Some sources contain non-string values (e.g. NaN floats) in text.
        # Normalize to string so tokenizer() never receives invalid types.
        def _cast_text(example):
            value = example.get("text")
            if value is None:
                return {"text": ""}
            return {"text": value if isinstance(value, str) else str(value)}

        return ds.select_columns(["text"]).map(_cast_text)

    def _join(example):
        parts = []
        for c in cols:
            v = example.get(c)
            if v is None:
                continue
            if isinstance(v, str):
                parts.append(v)
            else:
                parts.append(str(v))
        return {"text": "\n\n".join(p for p in parts if p)}

    return ds.map(_join, remove_columns=ds.column_names)


def _source_signature(spec: dict[str, Any], tokenizer_name: str, max_seq_length: int, append_eos: bool) -> str:
    payload = {
        "hf_id": spec.get("hf_id"),
        "subset": spec.get("subset"),
        "split": spec.get("split", "train"),
        "data_files": spec.get("data_files"),
        "text_columns": spec.get("text_columns"),
        "tokenizer": tokenizer_name,
        "max_seq_length": max_seq_length,
        "append_eos": append_eos,
    }
    raw = json.dumps(payload, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _compute_token_stats(
    ds_text,
    tokenizer,
    *,
    max_seq_length: int,
    append_eos: bool,
    num_proc: int,
    batch_size: int,
):
    n = len(ds_text)
    workers = max(1, min(num_proc, n))
    truncate_limit = max_seq_length - 1 if append_eos else max_seq_length
    eos_add = 1 if append_eos else 0

    def _count_tokens(batch):
        enc = tokenizer(
            batch["text"],
            add_special_tokens=False,
            truncation=False,
            return_length=True,
        )
        lengths = enc["length"]
        train_lengths = [min(l, truncate_limit) + eos_add for l in lengths]
        return {
            "token_count_full": lengths,
            "token_count_train": train_lengths,
        }

    counted = ds_text.map(
        _count_tokens,
        batched=True,
        batch_size=batch_size,
        num_proc=workers,
        load_from_cache_file=True,
        desc="token_count_map",
    )

    full_counts = counted["token_count_full"]
    train_counts = counted["token_count_train"]
    total_full = int(sum(full_counts))
    total_train = int(sum(train_counts))
    avg_full = float(total_full / n) if n else 0.0
    avg_train = float(total_train / n) if n else 0.0
    return {
        "rows_measured": n,
        "total_tokens_full_measured": total_full,
        "total_tokens_train_measured": total_train,
        "avg_tokens_full": avg_full,
        "avg_tokens_train_truncated": avg_train,
    }


def analyze_mix(
    mix_id: str,
    tokenizer,
    args: argparse.Namespace,
    token: str | None,
    per_source_cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    sources = resolve_mix_sources(mix_id)
    if not sources:
        raise ValueError(f"Mix '{mix_id}' resolved to no sources.")

    source_rows: list[dict[str, Any]] = []
    weights = [float(s.get("weight", 1.0)) for s in sources]
    wsum = sum(weights)
    probs = [w / wsum for w in weights]

    for spec, p in zip(sources, probs):
        name = spec.get("name", spec.get("hf_id", "unknown_source"))
        sig = _source_signature(
            spec,
            tokenizer_name=tokenizer.name_or_path,
            max_seq_length=args.max_seq_length,
            append_eos=args.append_eos,
        )

        # Load full split length from metadata where possible.
        full_rows = _get_full_split_rows(spec, args.cache_dir, token=token)

        max_samples = int(spec.get("max_samples", full_rows))
        capped_rows = min(max_samples, full_rows)
        rows_for_stats = capped_rows if args.rows_per_source <= 0 else min(capped_rows, args.rows_per_source)

        cache_key = f"{sig}:{rows_for_stats}"
        if cache_key in per_source_cache:
            stats = per_source_cache[cache_key]
        else:
            ds_measured = _load_source_dataset(spec, args.cache_dir, cap_rows=rows_for_stats, token=token)
            ds_measured = _normalize_text_columns(ds_measured, spec.get("text_columns", ["text"]))
            stats = _compute_token_stats(
                ds_measured,
                tokenizer,
                max_seq_length=args.max_seq_length,
                append_eos=args.append_eos,
                num_proc=args.num_proc,
                batch_size=args.batch_size,
            )
            per_source_cache[cache_key] = stats

        eval_rows = _eval_size_for_loader(capped_rows, args.test_size_percent)
        train_rows = capped_rows - eval_rows

        avg_full = stats["avg_tokens_full"]
        avg_train = stats["avg_tokens_train_truncated"]

        est_tokens_max_samples_full = int(round(avg_full * capped_rows))
        est_tokens_max_samples_train = int(round(avg_train * capped_rows))
        est_tokens_train_split = int(round(avg_train * train_rows))
        est_tokens_full_split = int(round(avg_full * full_rows))

        source_rows.append(
            {
                "name": name,
                "hf_id": spec.get("hf_id"),
                "subset": spec.get("subset"),
                "split": spec.get("split", "train"),
                "weight": float(spec.get("weight", 1.0)),
                "probability": p,
                "max_samples": max_samples,
                "full_rows_split": full_rows,
                "capped_rows_used_by_loader": capped_rows,
                "rows_measured_for_stats": stats["rows_measured"],
                "avg_tokens_full": avg_full,
                "avg_tokens_train_truncated": avg_train,
                "total_tokens_full_measured": stats["total_tokens_full_measured"],
                "total_tokens_train_measured": stats["total_tokens_train_measured"],
                "estimated_tokens_at_capped_rows_full": est_tokens_max_samples_full,
                "estimated_tokens_at_capped_rows_train_truncated": est_tokens_max_samples_train,
                "train_rows_after_split": train_rows,
                "eval_rows_after_split": eval_rows,
                "estimated_tokens_train_rows_after_split": est_tokens_train_split,
                "estimated_tokens_full_split": est_tokens_full_split,
                "capped_to_full_row_ratio": round(capped_rows / full_rows, 6) if full_rows else 0.0,
                "notes": spec.get("notes"),
            }
        )

    # Epoch token budget estimates.
    no_oversample_tokens = sum(r["estimated_tokens_train_rows_after_split"] for r in source_rows)
    target_examples = max(
        r["train_rows_after_split"] / r["probability"] for r in source_rows if r["probability"] > 0
    )
    interleave_tokens = sum(
        target_examples * r["probability"] * r["avg_tokens_train_truncated"] for r in source_rows
    )
    interleave_examples = int(round(target_examples))

    summary = {
        "mix_id": mix_id,
        "tokenizer": tokenizer.name_or_path,
        "max_seq_length": args.max_seq_length,
        "append_eos": args.append_eos,
        "rows_per_source_for_stats": args.rows_per_source,
        "num_proc": args.num_proc,
        "batch_size": args.batch_size,
        "test_size_percent": args.test_size_percent,
        "source_count": len(source_rows),
        "sum_train_rows_after_split": int(sum(r["train_rows_after_split"] for r in source_rows)),
        "sum_eval_rows_after_split": int(sum(r["eval_rows_after_split"] for r in source_rows)),
        "estimated_train_tokens_no_oversample_one_epoch": int(round(no_oversample_tokens)),
        "estimated_train_examples_interleave_all_exhausted_one_epoch": interleave_examples,
        "estimated_train_tokens_interleave_all_exhausted_one_epoch": int(round(interleave_tokens)),
        "estimated_train_tokens_interleave_all_exhausted_03_epoch": int(round(0.3 * interleave_tokens)),
        "estimated_train_tokens_interleave_all_exhausted_1_epoch": int(round(1.0 * interleave_tokens)),
        "estimated_train_tokens_interleave_all_exhausted_5_epoch": int(round(5.0 * interleave_tokens)),
    }

    return {"mix": summary, "sources": source_rows}


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)

    token = os.environ.get(args.hf_token_env)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, cache_dir=args.cache_dir)

    per_source_cache: dict[str, dict[str, Any]] = {}
    results: list[dict[str, Any]] = []
    for mix in args.mix:
        print(f"\n=== analyzing mix: {mix} ===")
        mix_result = analyze_mix(mix, tokenizer, args, token, per_source_cache)
        results.append(mix_result)
        print(
            f"mix={mix_result['mix']['mix_id']} "
            f"tokens_epoch_all_exhausted={mix_result['mix']['estimated_train_tokens_interleave_all_exhausted_one_epoch']:,}"
        )

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": os.uname().nodename if hasattr(os, "uname") else "unknown",
        "args": vars(args),
        "mix_results": results,
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    detailed_path = out_dir / f"token_budget_summary_{ts}.json"
    latest_path = out_dir / "token_budget_summary_latest.json"
    with detailed_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with latest_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # compact table for notebooks / quick reads
    compact = []
    for item in results:
        m = item["mix"]
        compact.append(
            {
                "mix_id": m["mix_id"],
                "source_count": m["source_count"],
                "sum_train_rows_after_split": m["sum_train_rows_after_split"],
                "estimated_train_tokens_no_oversample_one_epoch": m["estimated_train_tokens_no_oversample_one_epoch"],
                "estimated_train_tokens_interleave_all_exhausted_one_epoch": m[
                    "estimated_train_tokens_interleave_all_exhausted_one_epoch"
                ],
                "estimated_train_tokens_interleave_all_exhausted_03_epoch": m[
                    "estimated_train_tokens_interleave_all_exhausted_03_epoch"
                ],
                "estimated_train_tokens_interleave_all_exhausted_5_epoch": m[
                    "estimated_train_tokens_interleave_all_exhausted_5_epoch"
                ],
            }
        )
    compact_path = out_dir / "token_budget_compact_latest.json"
    with compact_path.open("w", encoding="utf-8") as f:
        json.dump(compact, f, indent=2)

    print(f"\nSaved detailed summary: {detailed_path}")
    print(f"Saved latest summary : {latest_path}")
    print(f"Saved compact summary: {compact_path}")


if __name__ == "__main__":
    main()

