"""Generation-quality runner (E09 Stage-0 diagnostic + repetition/diversity).

Loads a `concept_ar` checkpoint and computes the two generation-quality axes:

1. **Suffix-CE by position** (`compute_suffix_ce_by_position`) — the E09 Stage-0
   kill-gate diagnostic. For each batch, splits at `--prefix_ratio`, encodes the
   prefix, teacher-forces the suffix, and bins CE by suffix position. A rising
   CE-intact curve past the K-window (or a growing intact-vs-shuffled gap) quantifies
   the "frozen snapshot + K-window cannot sustain prediction" wall. A *flat* curve
   falsifies E09's hypothesis at no training cost.

2. **Free-running generation + diversity metrics** — generates from a small bank of
   prompts and reports distinct-1/2/3, repetition-1/2, REP-3, and a length-binned
   diversity profile aligned to the decoder's window. Repetition-rate → 1.0 is the
   E05/E02-long loop signature.

Both axes are independent of the experiment-evaluate tiered suite — they answer
orthogonal questions (long-form generation coherence + cross-window memory) and run
cheaply (CPU or one GPU, ~minutes on minipile + a fixed prompt bank).

Usage:
    # Full Stage-0 read on a checkpoint
    uv run python analysis/run_generation_quality.py \\
        --model_path Cache/Training/MODEL/checkpoint-XXXX \\
        --model_type concept_ar \\
        --output_json Cache/Evaluation_reports/MODEL_generation_quality.json

    # Just the suffix-CE diagnostic (no free-running generation, no prompt bank)
    uv run python analysis/run_generation_quality.py \\
        --model_path Cache/Training/MODEL/checkpoint-XXXX \\
        --no_free_generation

    # Just the repetition/diversity read on the prompt bank (no held-out batches)
    uv run python analysis/run_generation_quality.py \\
        --model_path Cache/Training/MODEL/checkpoint-XXXX \\
        --no_suffix_ce --free_generation_max_new_tokens 512
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

# Make the repo importable when run as a standalone script.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nn.concept_encoder import ConceptEncoderConfig  # noqa: E402
from nn.concept_encoder_perceiver import (  # noqa: E402
    ConceptEncoderForConditionalLM,
    ConceptEncoderForDenoisingPerceiver,
)
from nn.concept_encoder_weighted import ConceptEncoderForMaskedLMWeighted  # noqa: E402
from evaluation.generation_quality import (  # noqa: E402
    compute_suffix_ce_by_position,
    generate_free_running,
    summarize_generation,
)

MODEL_CLASSES = {
    "perceiver_denoise": ConceptEncoderForDenoisingPerceiver,
    "concept_ar": ConceptEncoderForConditionalLM,
    "weighted_mlm": ConceptEncoderForMaskedLMWeighted,
}

# Default prompt bank: continuation-style (not instructions — these are base
# concept-LMs, not SFT'd). Tuned to probe both topical and stylistic generation.
DEFAULT_PROMPTS = [
    "The future of renewable energy depends on",
    "Once upon a time, in a quiet village",
    "The most important skill in software engineering is",
    "Scientists have long wondered whether",
    "In machine learning, a gradient is",
    "The meaning of life is",
    "Long ago, the kingdom of",
    "The most surprising discovery of the last decade was",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_path", required=True)
    p.add_argument("--model_type", default="concept_ar", choices=list(MODEL_CLASSES))
    p.add_argument("--output_json", default=None,
                   help="Where to save the JSON results. Default: stdout only.")
    # Suffix-CE-by-position knobs.
    p.add_argument("--no_suffix_ce", action="store_true",
                   help="Skip the suffix-CE-by-position diagnostic.")
    p.add_argument("--dataset", default="JeanKaddour/minipile")
    p.add_argument("--dataset_config", default=None)
    p.add_argument("--num_batches", type=int, default=10,
                   help="Held-out batches for suffix-CE binning.")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_seq_length", type=int, default=2048,
                   help="Tokenizer truncation length. E05 models trained at 2048.")
    p.add_argument("--prefix_ratio", type=float, default=0.4,
                   help="Train/eval prefix→suffix split fraction (E02-long/E05: 0.35-0.45).")
    p.add_argument("--bin_size", type=int, default=128,
                   help="Suffix-position bin size. Default = E05 K-window.")
    p.add_argument("--window_k", type=int, default=None,
                   help="First-bin edge (= decoder_context_window). Default: from checkpoint config.")
    # Free-generation knobs.
    p.add_argument("--no_free_generation", action="store_true",
                   help="Skip the free-running generation + diversity read.")
    p.add_argument("--free_generation_max_new_tokens", type=int, default=256)
    p.add_argument("--free_generation_prompts", nargs="*", default=None,
                   help="Override the default prompt bank. Pass continuation-style prompts, not instructions.")
    p.add_argument("--greedy", action="store_true", default=True,
                   help="Greedy decode (default; reproducible across models).")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--top_p", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _load_held_out_batches(args, tokenizer, device) -> List[tuple]:
    """Stream `args.num_batches` of `args.batch_size` from the dataset, tokenize,
    pad to the longest in the batch. Returns CPU tensors (will be moved per-batch
    by the diagnostic), matching `run_concept_analysis.py`'s convention."""
    if args.dataset_config:
        ds = load_dataset(args.dataset, args.dataset_config, split="train", streaming=True)
    else:
        ds = load_dataset(args.dataset, split="train", streaming=True)

    batches: List[tuple] = []
    batch_texts: List[str] = []
    for sample in ds:
        text = sample.get("text", "") or ""
        if len(text.strip()) < 50:  # filter stubs (keeps short rows out of the suffix split)
            continue
        batch_texts.append(text)
        if len(batch_texts) == args.batch_size:
            enc = tokenizer(
                batch_texts,
                max_length=args.max_seq_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            batches.append((enc["input_ids"], enc["attention_mask"]))
            batch_texts = []
            if len(batches) >= args.num_batches:
                break
    return batches


def _print_suffix_ce(result: Dict, *, indent: str = "  ") -> None:
    wk = result.get("window_k")
    bin_size = result["bin_size"]
    print(f"{indent}Prefix→suffix split ratio: {result['prefix_ratio']}  "
          f"(bin size {bin_size}, window_k {wk}, batches {result['n_batches']})")
    print(f"{indent}Early-position (first {result['early_k']} suffix tokens):")
    print(f"{indent}  CE intact      = {result['ce_intact_early']:.4f}")
    print(f"{indent}  Δshuffle (early) = {result['delta_shuffle_early']:+.4f}")
    print(f"{indent}  Δzero    (early) = {result['delta_zero_early']:+.4f}")
    print(f"{indent}Per-bin suffix-CE (the E09 Stage-0 curve):")
    intact = {b["bin_index"]: b for b in result["ce_intact_by_bin"]}
    deltas = {b["bin_index"]: b for b in result["delta_by_bin"]}
    print(f"{indent}  {'bin':>4}  {'range':>14}  {'CE intact':>10}  {'Δshuffle':>9}  {'Δzero':>9}  {'#tok':>7}")
    # Reconstruct display ranges: bin 0 = [0, wk), bin i>0 = [wk+(i-1)*bs, wk+i*bs).
    for idx in sorted(intact):
        b = intact[idx]
        if idx == 0:
            start, end = 0, (wk if wk else bin_size)
        else:
            base = wk if wk else 0
            start = base + (idx - 1) * bin_size
            end = start + bin_size
        rng = f"[{start}, {end})"
        d = deltas.get(idx, {})
        ds = d.get("delta_shuffle", float("nan"))
        dz = d.get("delta_zero", float("nan"))
        print(f"{indent}  {idx:>4}  {rng:>14}  {b['ce']:>10.4f}  {ds:>+9.4f}  {dz:>+9.4f}  {b['n_tokens']:>7}")
    print(f"{indent}Read: a rising CE-intact past window_k, OR a growing Δshuffle, is the "
          f"frozen-memory wall.")
    print(f"{indent}     A flat curve falsifies E09's hypothesis (no writable memory needed).")


def _print_generation(samples: List[Dict], *, indent: str = "  ") -> None:
    for s in samples:
        summary = s["summary"]
        print(f"{indent}- Prompt ({s['prompt_n_tokens']} tok): {s['prompt']!r}")
        print(f"{indent}  Generated {summary['n_tokens']} tok: "
              f"distinct-1={summary['distinct_1']:.3f}  distinct-2={summary['distinct_2']:.3f}  "
              f"rep-1={summary['repetition_1']:.3f}  rep-3={summary['rep_3']:.3f}")
        # First ~120 chars of the generation, single-line.
        preview = " ".join(s["text"][:160].split())
        print(f"{indent}  Text: {preview!r}")
        # Per-bin distinct-1 profile (compact one-liner) — falls past window_k = loop.
        bins = summary["length_binned_diversity"]["bins"]
        prof = " ".join(f"{b['distinct_n_by_n'][1]:.2f}" for b in bins[:8])
        print(f"{indent}  distinct-1 by {summary['length_binned_diversity']['bin_size']}-tok bin "
              f"({len(bins)} bins): {prof} ...")


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading model from: {args.model_path}")
    print(f"Model type: {args.model_type}")

    model_class = MODEL_CLASSES[args.model_type]
    model = model_class.from_pretrained(args.model_path).to(device).eval()

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

    window_k = args.window_k or getattr(model.config, "decoder_context_window", None)
    output: Dict = {
        "model_path": args.model_path,
        "model_type": args.model_type,
        "decoder_context_window": window_k,
    }

    # ---- Suffix-CE-by-position diagnostic ----
    if not args.no_suffix_ce and args.model_type == "concept_ar":
        print(f"\nLoading held-out batches ({args.num_batches} × {args.batch_size} "
              f"from {args.dataset}) ...")
        batches = _load_held_out_batches(args, tokenizer, device)
        print(f"Computing suffix-CE by position (prefix_ratio={args.prefix_ratio}, "
              f"bin_size={args.bin_size}, window_k={window_k}) ...")
        suffix_ce = compute_suffix_ce_by_position(
            model, batches, device,
            prefix_ratio=args.prefix_ratio,
            bin_size=args.bin_size,
            window_k=window_k,
        )
        output["suffix_ce_by_position"] = suffix_ce
        print()
        _print_suffix_ce(suffix_ce)
    elif not args.no_suffix_ce:
        print(f"\nSkipping suffix-CE: model_type {args.model_type} is not concept_ar.")

    # ---- Free-running generation + diversity ----
    if not args.no_free_generation and args.model_type == "concept_ar":
        prompts = args.free_generation_prompts or DEFAULT_PROMPTS
        print(f"\nFree-running generation on {len(prompts)} prompts "
              f"(max {args.free_generation_max_new_tokens} new tok, "
              f"{'greedy' if args.greedy else 'sampled'}) ...")
        samples: List[Dict] = []
        for prompt in prompts:
            gen = generate_free_running(
                model, tokenizer, prompt, device,
                max_new_tokens=args.free_generation_max_new_tokens,
                greedy=args.greedy,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                seed=args.seed,
            )
            summary = summarize_generation(gen["ids"], decoder_window_k=window_k or 128)
            samples.append({
                "prompt": prompt,
                "prompt_n_tokens": gen["prompt_n_tokens"],
                "text": gen["text"],
                "n_tokens": gen["n_tokens"],
                "summary": summary,
            })
        output["free_generation"] = samples
        print()
        _print_generation(samples)
    elif not args.no_free_generation:
        print(f"\nSkipping free generation: model_type {args.model_type} is not concept_ar.")

    # ---- Save ----
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
