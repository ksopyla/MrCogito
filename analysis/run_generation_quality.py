"""Generation-quality runner (diversity + concept ablation + length sweep).

Supports:
  - `concept_ar` — encode→BOS free-decode (`generate_free_running`) + optional
    suffix-CE-by-position (E09 Stage-0).
  - `backbone_concept` — true causal continuation (`generate_continuation`) with
    concept_mode ablations (real / zero / shuffle / static).

Both families share the same token-level diversity metrics (distinct-n, REP-3,
length-binned profile). A single long generation is cut at
`--length_cutoffs` so one decode answers 512 / 1K / 2K / … quality.

Usage:
    # E16b backbone, continuation + chat, concept ablation, length cutoffs
    uv run python analysis/run_generation_quality.py \\
        --model_path Cache/Training/.../checkpoint-7900 \\
        --model_type backbone_concept \\
        --no_suffix_ce \\
        --free_generation_max_new_tokens 2048 \\
        --length_cutoffs 512 1024 2048 4096 8192 16384 \\
        --prompt_styles continuation chat \\
        --concept_modes real zero shuffle static \\
        --output_json Cache/Evaluation_reports/e16b_generation_quality.json

    # E05 concept_ar baseline (same metrics, different decode contract)
    uv run python analysis/run_generation_quality.py \\
        --model_path Cache/Training/.../checkpoint-69142 \\
        --model_type concept_ar \\
        --no_suffix_ce \\
        --free_generation_max_new_tokens 512 \\
        --length_cutoffs 64 128 256 512 \\
        --output_json Cache/Evaluation_reports/e05_generation_quality.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# Make the repo importable when run as a standalone script.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Load local .env (HF_HOME, PYTORCH_ENABLE_MPS_FALLBACK) before torch/HF init.
try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from nn.backbone_concept_lm import BackboneConceptLM  # noqa: E402
from nn.concept_encoder_perceiver import (  # noqa: E402
    ConceptEncoderForConditionalLM,
    ConceptEncoderForDenoisingPerceiver,
)
from nn.concept_encoder_weighted import ConceptEncoderForMaskedLMWeighted  # noqa: E402
from evaluation.generation_quality import (  # noqa: E402
    compute_suffix_ce_by_position,
    format_gemma_chat,
    generate_continuation,
    generate_free_running,
    summarize_generation,
    summarize_generation_at_lengths,
)

MODEL_CLASSES = {
    "perceiver_denoise": ConceptEncoderForDenoisingPerceiver,
    "concept_ar": ConceptEncoderForConditionalLM,
    "weighted_mlm": ConceptEncoderForMaskedLMWeighted,
    "backbone_concept": BackboneConceptLM,
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

# Instruction-shaped counterparts for the chat-template probe (same topics).
DEFAULT_CHAT_USER_MESSAGES = [
    "Continue this sentence in fluent English: The future of renewable energy depends on",
    "Write a short fairy-tale continuation: Once upon a time, in a quiet village",
    "Explain briefly: The most important skill in software engineering is",
    "Continue scientifically: Scientists have long wondered whether",
    "Explain: In machine learning, a gradient is",
    "Reflect briefly: The meaning of life is",
    "Continue the story: Long ago, the kingdom of",
    "Describe: The most surprising discovery of the last decade was",
]

DEFAULT_LENGTH_CUTOFFS = [512, 1024, 2048, 4096, 8192, 16384]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_path", required=True)
    p.add_argument("--model_type", default="concept_ar", choices=list(MODEL_CLASSES))
    p.add_argument("--output_json", default=None,
                   help="Where to save the JSON results. Default: stdout only.")
    p.add_argument("--dtype", default="auto",
                   choices=["auto", "float32", "float16", "bfloat16"],
                   help="Load dtype. auto: bf16 on CUDA, float32 elsewhere (MPS-safe).")
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
                   help="Override the default continuation prompt bank.")
    p.add_argument("--max_prompts", type=int, default=None,
                   help="Use only the first N prompts (smoke / cost control).")
    p.add_argument("--prompt_styles", nargs="+", default=["continuation"],
                   choices=["continuation", "chat"],
                   help="continuation = raw base prompts; chat = Gemma turn template.")
    p.add_argument("--concept_modes", nargs="+", default=["real"],
                   help="backbone_concept only: real zero shuffle static one_block. "
                        "Ignored for concept_ar.")
    p.add_argument("--length_cutoffs", nargs="*", type=int, default=DEFAULT_LENGTH_CUTOFFS,
                   help="Report diversity on generation prefixes at these lengths.")
    p.add_argument("--greedy", action="store_true", default=True,
                   help="Greedy decode (default; reproducible across models).")
    p.add_argument("--sample", action="store_true",
                   help="Use sampling instead of greedy (sets greedy=False).")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--top_p", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_prompt_len", type=int, default=1024)
    return p.parse_args()


def _resolve_dtype(name: str, device: str):
    if name == "auto":
        if device == "cuda":
            return torch.bfloat16
        return torch.float32
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


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
        mode = s.get("concept_mode", "n/a")
        style = s.get("prompt_style", "continuation")
        print(f"{indent}- [{style}|{mode}] Prompt ({s['prompt_n_tokens']} tok): {s['prompt']!r}")
        print(f"{indent}  Generated {summary['n_tokens']} tok in {s.get('seconds', float('nan')):.1f}s: "
              f"distinct-1={summary['distinct_1']:.3f}  distinct-2={summary['distinct_2']:.3f}  "
              f"rep-1={summary['repetition_1']:.3f}  rep-3={summary['rep_3']:.3f}")
        preview = " ".join(s["text"][:160].split())
        print(f"{indent}  Text: {preview!r}")
        bins = summary["length_binned_diversity"]["bins"]
        prof = " ".join(f"{b['distinct_n_by_n'][1]:.2f}" for b in bins[:8])
        print(f"{indent}  distinct-1 by {summary['length_binned_diversity']['bin_size']}-tok bin "
              f"({len(bins)} bins): {prof} ...")
        by_len = s.get("by_length") or {}
        if by_len:
            parts = []
            for L in sorted(by_len, key=int):
                m = by_len[L]
                parts.append(f"{L}:d1={m['distinct_1']:.2f}/r3={m['rep_3']:.2f}")
            print(f"{indent}  by_length: " + " | ".join(parts))


def _aggregate_by_length(samples: Sequence[Dict]) -> Dict[str, Dict[str, float]]:
    """Mean diversity metrics across samples that reached each cutoff."""
    buckets: Dict[str, List[Dict]] = {}
    for s in samples:
        for L, m in (s.get("by_length") or {}).items():
            buckets.setdefault(L, []).append(m)
    out: Dict[str, Dict[str, float]] = {}
    for L, rows in buckets.items():
        keys = ("distinct_1", "distinct_2", "distinct_3", "repetition_1", "repetition_2", "rep_3")
        out[L] = {k: sum(r[k] for r in rows) / len(rows) for k in keys}
        out[L]["n_samples"] = float(len(rows))
    return out


def _build_prompt_bank(args) -> List[Dict[str, str]]:
    """List of {style, prompt, user_message?} entries."""
    cont = args.free_generation_prompts or DEFAULT_PROMPTS
    if args.max_prompts is not None:
        cont = cont[: args.max_prompts]
    bank: List[Dict[str, str]] = []
    if "continuation" in args.prompt_styles:
        for p in cont:
            bank.append({"prompt_style": "continuation", "prompt": p})
    if "chat" in args.prompt_styles:
        chat_msgs = DEFAULT_CHAT_USER_MESSAGES
        if args.free_generation_prompts:
            # Mirror custom continuation prompts into chat wrappers.
            chat_msgs = [f"Continue fluently: {p}" for p in cont]
        elif args.max_prompts is not None:
            chat_msgs = chat_msgs[: args.max_prompts]
        for msg in chat_msgs:
            bank.append({
                "prompt_style": "chat",
                "prompt": format_gemma_chat(msg),
                "user_message": msg,
            })
    return bank


def _run_free_generation(args, model, tokenizer, device, window_k: Optional[int]) -> List[Dict]:
    bank = _build_prompt_bank(args)
    greedy = not args.sample
    is_backbone = args.model_type == "backbone_concept"
    modes = args.concept_modes if is_backbone else ["real"]
    bin_k = window_k or (getattr(model.config, "concept_block", None) or 128)

    print(f"\nGeneration on {len(bank)} prompts × {len(modes)} concept_mode(s) "
          f"(max {args.free_generation_max_new_tokens} new tok, "
          f"{'greedy' if greedy else 'sampled'}, styles={args.prompt_styles}) ...")

    samples: List[Dict] = []
    for entry in bank:
        for mode in modes:
            t0 = time.time()
            if is_backbone:
                gen = generate_continuation(
                    model, tokenizer, entry["prompt"], device,
                    max_new_tokens=args.free_generation_max_new_tokens,
                    max_prompt_len=args.max_prompt_len,
                    greedy=greedy,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p if args.top_p else 1.0,
                    seed=args.seed,
                    concept_mode=mode,
                )
            else:
                if mode != "real":
                    continue  # concept_ar free-gen path has no concept_mode yet
                gen = generate_free_running(
                    model, tokenizer, entry["prompt"], device,
                    max_new_tokens=args.free_generation_max_new_tokens,
                    max_prompt_len=args.max_prompt_len,
                    greedy=greedy,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    seed=args.seed,
                )
            dt = time.time() - t0
            summary = summarize_generation(gen["ids"], decoder_window_k=bin_k)
            by_length = summarize_generation_at_lengths(
                gen["ids"], args.length_cutoffs, decoder_window_k=bin_k,
            )
            samples.append({
                "prompt": entry["prompt"],
                "prompt_style": entry["prompt_style"],
                "user_message": entry.get("user_message"),
                "prompt_n_tokens": gen["prompt_n_tokens"],
                "text": gen["text"],
                "n_tokens": gen["n_tokens"],
                "concept_mode": mode,
                "seconds": dt,
                "summary": summary,
                "by_length": by_length,
            })
            print(f"  done [{entry['prompt_style']}|{mode}] "
                  f"{gen['n_tokens']} tok in {dt:.1f}s  "
                  f"d1={summary['distinct_1']:.3f} r3={summary['rep_3']:.3f}")
    return samples


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = _resolve_dtype(args.dtype, device)
    print(f"Device: {device} | dtype: {dtype}")
    print(f"Loading model from: {args.model_path}")
    print(f"Model type: {args.model_type}")

    model_class = MODEL_CLASSES[args.model_type]
    load_kw = {"local_files_only": True}
    if args.model_type == "backbone_concept":
        model = model_class.from_pretrained(args.model_path, dtype=dtype, **load_kw)
    else:
        model = model_class.from_pretrained(args.model_path, **load_kw)
        if dtype != torch.float32 and device == "cuda":
            model = model.to(dtype=dtype)
    model = model.to(device).eval()

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    window_k = args.window_k
    if window_k is None:
        window_k = getattr(model.config, "decoder_context_window", None)
    if window_k is None:
        window_k = getattr(model.config, "concept_block", None)

    output: Dict = {
        "model_path": args.model_path,
        "model_type": args.model_type,
        "decoder_context_window": window_k,
        "length_cutoffs": args.length_cutoffs,
        "prompt_styles": args.prompt_styles,
        "concept_modes": args.concept_modes if args.model_type == "backbone_concept" else ["real"],
        "max_new_tokens": args.free_generation_max_new_tokens,
        "dtype": str(dtype).replace("torch.", ""),
        "device": device,
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

    # ---- Free-running / continuation generation + diversity ----
    if not args.no_free_generation:
        samples = _run_free_generation(args, model, tokenizer, device, window_k)
        output["free_generation"] = samples
        output["aggregate_by_length"] = _aggregate_by_length(samples)
        # Also aggregate by (style, mode) × length for concept-impact reads.
        grouped: Dict[str, List[Dict]] = {}
        for s in samples:
            key = f"{s['prompt_style']}|{s['concept_mode']}"
            grouped.setdefault(key, []).append(s)
        output["aggregate_by_condition"] = {
            k: _aggregate_by_length(v) for k, v in grouped.items()
        }
        print()
        _print_generation(samples)
        print("\nAggregate by length (mean over prompts that reached the cutoff):")
        for L, m in sorted(output["aggregate_by_length"].items(), key=lambda kv: int(kv[0])):
            print(f"  L={L:>5}: distinct-1={m['distinct_1']:.3f}  distinct-2={m['distinct_2']:.3f}  "
                  f"rep-1={m['repetition_1']:.3f}  rep-3={m['rep_3']:.3f}  n={int(m['n_samples'])}")
        print("\nAggregate by condition × length:")
        for cond, by_len in sorted(output["aggregate_by_condition"].items()):
            print(f"  [{cond}]")
            for L, m in sorted(by_len.items(), key=lambda kv: int(kv[0])):
                print(f"    L={L:>5}: d1={m['distinct_1']:.3f}  r3={m['rep_3']:.3f}  "
                      f"n={int(m['n_samples'])}")
    else:
        print("\nSkipping free generation (--no_free_generation).")

    # ---- Save ----
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
