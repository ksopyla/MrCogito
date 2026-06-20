"""Compute sequence-length distributions for HF datasets.

CPU-only and streaming by default, so it is safe to run beside GPU training
when sampling a modest number of rows. Prefer a real tokenizer for datasets
without a precomputed token-count column.

Examples:
    uv run python analysis/dataset_seqlen_distribution.py \
        --dataset HuggingFaceFW/fineweb-edu --subset sample-10BT \
        --length_column token_count --max_docs 200000

    uv run python analysis/dataset_seqlen_distribution.py \
        --dataset HuggingFaceFW/finepdfs_100BT --text_column text \
        --tokenizer HuggingFaceTB/SmolLM2-135M --max_docs 10000

    uv run python analysis/dataset_seqlen_distribution.py \
        --candidates analysis/long_dataset_candidates.json \
        --tokenizer HuggingFaceTB/SmolLM2-135M --max_docs 5000 \
        --shuffle --shuffle_buffer_size 10000 --seed 42
"""

import argparse
import json
import os
from collections import Counter
from datetime import datetime
from typing import Any

from datasets import load_dataset
from transformers import AutoTokenizer


# Thresholds that matter for our experiments.
# At 512 tokens we run E01/E02; the goal is long-range forcing at 4k/8k.
THRESHOLDS = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
DEFAULT_TEXT_COLUMNS = [
    "text",
    "content",
    "document",
    "problem",
    "prompt",
    "question",
    "response",
    "answer",
    "completion",
    "code",
]


def parse_args():
    p = argparse.ArgumentParser(description="Dataset sequence-length distribution")
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--subset", default="sample-10BT")
    p.add_argument("--split", default="train")
    p.add_argument(
        "--candidates",
        help=(
            "JSON file with a list of candidate objects. Each object may define "
            "name, dataset, subset, split, text_column, text_columns, "
            "length_column, data_files, trust_remote_code."
        ),
    )
    p.add_argument("--text_column")
    p.add_argument(
        "--text_columns",
        nargs="+",
        help="Concatenate these columns before tokenization.",
    )
    p.add_argument(
        "--length_column",
        help="Use an existing token/word/length column instead of tokenizing text.",
    )
    p.add_argument(
        "--tokenizer",
        default=None,
        help="Tokenizer name/path. If omitted, falls back to chars_per_token.",
    )
    p.add_argument(
        "--max_docs",
        type=int,
        default=500_000,
        help="Number of documents to sample (streaming). -1 for full dataset.",
    )
    p.add_argument(
        "--chars_per_token",
        type=float,
        default=4.0,
        help="Fallback: chars-per-token ratio when no token_count column exists.",
    )
    p.add_argument("--cache_dir", default="./Cache/Datasets")
    p.add_argument("--out_dir", default="./Cache/Evaluation_reports")
    p.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the iterable dataset before taking max_docs (approximate for streaming datasets).",
    )
    p.add_argument(
        "--shuffle_buffer_size",
        type=int,
        default=10_000,
        help="Buffer size for streaming shuffle when --shuffle is set.",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed for streaming shuffle.")
    p.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Pass trust_remote_code=True to datasets.load_dataset.",
    )
    p.add_argument(
        "--stop_on_error",
        action="store_true",
        help="Abort when one candidate fails instead of continuing to the next.",
    )
    return p.parse_args()


def assign_bin(n: int, thresholds: list[int]) -> str:
    for t in thresholds:
        if n <= t:
            return f"<={t}"
    return f">{thresholds[-1]}"


def _get_nested(row: dict[str, Any], key: str) -> Any:
    value: Any = row
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _stringify_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        # Common SFT shape: {"role": "...", "content": "..."}.
        role = value.get("role")
        content = value.get("content")
        if content is not None:
            prefix = f"{role}: " if role else ""
            return prefix + _stringify_value(content)
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, list):
        return "\n".join(_stringify_value(item) for item in value if item is not None)
    return str(value)


def _extract_text(row: dict[str, Any], candidate: dict[str, Any]) -> str:
    text_columns = candidate.get("text_columns")
    if isinstance(text_columns, str):
        text_columns = [text_columns]
    if not text_columns and candidate.get("text_column"):
        text_columns = [candidate["text_column"]]

    if text_columns:
        parts = [_stringify_value(_get_nested(row, col)) for col in text_columns]
        return "\n\n".join(part for part in parts if part)

    for col in DEFAULT_TEXT_COLUMNS:
        if col in row:
            text = _stringify_value(row[col])
            if text:
                return text

    # Chat-style datasets often use messages/conversations as the only payload.
    for col in ("messages", "conversations", "dialogue", "turns"):
        if col in row:
            text = _stringify_value(row[col])
            if text:
                return text

    raise ValueError(
        "Could not infer text field. Set text_column or text_columns for this dataset. "
        f"Available columns: {sorted(row.keys())}"
    )


def _load_stream(candidate: dict[str, Any], args: argparse.Namespace):
    data_files = candidate.get("data_files")
    if data_files:
        return load_dataset(
            "parquet",
            data_files=data_files,
            split=candidate.get("split", args.split),
            streaming=True,
            cache_dir=args.cache_dir,
        )

    subset = candidate.get("subset")
    return load_dataset(
        candidate["dataset"],
        subset if subset else None,
        split=candidate.get("split", args.split),
        streaming=True,
        cache_dir=args.cache_dir,
        trust_remote_code=bool(candidate.get("trust_remote_code", args.trust_remote_code)),
    )


def _candidate_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "dataset": args.dataset,
        "subset": args.subset,
        "split": args.split,
        "text_column": args.text_column,
        "text_columns": args.text_columns,
        "length_column": args.length_column,
        "trust_remote_code": args.trust_remote_code,
    }


def _load_candidates(args: argparse.Namespace) -> list[dict[str, Any]]:
    if not args.candidates:
        return [_candidate_from_args(args)]

    with open(args.candidates) as f:
        candidates = json.load(f)
    if not isinstance(candidates, list):
        raise ValueError("--candidates must point to a JSON list of candidate objects")
    return candidates


def _length_for_row(
    row: dict[str, Any],
    candidate: dict[str, Any],
    tokenizer: Any,
    chars_per_token: float,
) -> int:
    length_column = candidate.get("length_column")
    if length_column:
        value = _get_nested(row, length_column)
        if value is None:
            raise ValueError(f"length_column '{length_column}' not found in row")
        return int(value)

    text = _extract_text(row, candidate)
    if tokenizer is None:
        return int(len(text) / chars_per_token)
    return len(tokenizer.encode(text, add_special_tokens=False))


def _summarize_lengths(lengths: list[int]) -> dict[str, float | int]:
    lengths.sort()
    n = len(lengths)
    return {
        "min": lengths[0],
        "mean": round(sum(lengths) / n, 1),
        "p50": lengths[int(n * 0.50)],
        "p75": lengths[int(n * 0.75)],
        "p90": lengths[int(n * 0.90)],
        "p95": lengths[int(n * 0.95)],
        "p99": lengths[int(n * 0.99)],
        "max": lengths[-1],
    }


def _safe_slug(candidate: dict[str, Any]) -> str:
    base = candidate.get("name") or candidate["dataset"]
    subset = candidate.get("subset")
    if subset:
        base = f"{base}_{subset}"
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in base)


def analyze_candidate(
    candidate: dict[str, Any],
    args: argparse.Namespace,
    tokenizer: Any,
) -> dict[str, Any]:
    subset = candidate.get("subset")
    split = candidate.get("split", args.split)
    name = candidate.get("name", candidate["dataset"])
    print(
        f"Loading {candidate['dataset']}"
        + (f" ({subset})" if subset else "")
        + f" [{split}] in streaming mode..."
    )

    ds = _load_stream(candidate, args)
    sample_method = "streaming first rows"
    if args.shuffle:
        ds = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer_size)
        sample_method = f"streaming shuffle(buffer={args.shuffle_buffer_size}, seed={args.seed})"

    counts: Counter = Counter()
    total = 0
    lengths: list[int] = []
    skipped = 0
    length_column = candidate.get("length_column")
    mode = (
        f"{length_column} column"
        if length_column
        else (f"tokenizer:{args.tokenizer}" if tokenizer is not None else f"len(text)/{args.chars_per_token:.1f} chars/tok")
    )
    print(f"Length source: {mode}")
    print(f"Sampling up to {args.max_docs:,} documents ({sample_method}) ...\n")

    for i, row in enumerate(ds):
        if args.max_docs > 0 and i >= args.max_docs:
            break

        try:
            n = _length_for_row(row, candidate, tokenizer, args.chars_per_token)
        except Exception as exc:
            skipped += 1
            if skipped <= 3:
                print(f"  skipping row {i}: {exc}")
            continue

        lengths.append(n)
        counts[assign_bin(n, THRESHOLDS)] += 1
        total += 1

        if total % 50_000 == 0:
            print(f"  ... processed {total:,} documents", flush=True)

    # Sort bin labels in a sensible order.
    bin_order = [f"<={t}" for t in THRESHOLDS] + [f">{THRESHOLDS[-1]}"]
    bin_order = [b for b in bin_order if b in counts]

    if not lengths:
        raise RuntimeError(f"No rows could be measured for {name}")

    stats = _summarize_lengths(lengths)

    print(f"\n{'='*60}")
    print(f"Dataset : {name}")
    print(f"HF path : {candidate['dataset']}" + (f" / {subset}" if subset else ""))
    print(f"Split   : {split}   |   Documents sampled: {total:,}   |   Skipped: {skipped:,}")
    print(f"Source  : {mode}")
    print(f"{'='*60}")
    print(
        "  "
        + "  ".join(
            f"{key}={value}" for key, value in stats.items()
        )
    )
    print()
    print(f"{'Bin':<15}  {'Count':>10}  {'%':>8}  {'Cumulative%':>12}")
    print("-" * 50)
    cum = 0
    for b in bin_order:
        c = counts[b]
        cum += c
        print(f"{b:<15}  {c:>10,}  {100*c/total:>7.2f}%  {100*cum/total:>11.2f}%")
    print()

    # Long-range forcing: fraction of docs that would require cross-window memory.
    print("Long-range forcing potential (docs that span beyond window):")
    for t in [512, 1024, 2048, 4096, 8192]:
        longer = sum(length > t for length in lengths)
        frac = 100 * longer / total
        print(f"  > {t:>5} tokens : {longer:>10,}  ({frac:.2f}% of docs)")

    print()
    result = {
        "name": name,
        "dataset": candidate["dataset"],
        "subset": subset,
        "split": split,
        "total_docs": total,
        "skipped_docs": skipped,
        "length_source": mode,
        "sample_method": sample_method,
        "stats": stats,
        "bins": {b: counts[b] for b in bin_order},
        "bins_pct": {b: round(100 * counts[b] / total, 3) for b in bin_order},
        "longer_than_pct": {
            str(t): round(100 * sum(length > t for length in lengths) / total, 3)
            for t in [512, 1024, 2048, 4096, 8192, 16384, 32768]
        },
        "timestamp": datetime.now().isoformat(),
    }

    slug = _safe_slug(candidate)
    out_path = os.path.join(args.out_dir, f"seqlen_dist_{slug}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {out_path}")
    return result


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer = None
    if args.tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    candidates = _load_candidates(args)
    results = []
    for idx, candidate in enumerate(candidates, start=1):
        print(f"\n### Candidate {idx}/{len(candidates)}")
        try:
            results.append(analyze_candidate(candidate, args, tokenizer))
        except Exception as exc:
            if args.stop_on_error:
                raise
            print(f"Candidate failed: {candidate.get('name', candidate.get('dataset'))}: {exc}")

    if len(results) > 1:
        out_path = os.path.join(args.out_dir, "seqlen_dist_summary.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSummary saved to {out_path}")


if __name__ == "__main__":
    main()
