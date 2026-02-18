"""
Standalone concept analysis runner.

Loads a pretrained ConceptEncoder checkpoint, runs a batch of Minipile text
through it, and computes geometry metrics on the resulting concept representations.
Results are printed to stdout and optionally saved to a JSON file.

Usage:
    python analysis/run_concept_analysis.py \
        --model_path Cache/Training/MODEL/MODEL \
        --model_type perceiver_mlm \
        [--output_json /tmp/results.json] \
        [--num_batches 20] \
        [--batch_size 32]
"""

import sys
import os
import argparse
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer
from datasets import load_dataset

from nn.concept_encoder import ConceptEncoderConfig
from nn.concept_encoder_perceiver import (
    ConceptEncoderForMaskedLMPerceiver,
    ConceptEncoderForMaskedLMPerceiverPosOnly,
)
from nn.concept_encoder_weighted import ConceptEncoderForMaskedLMWeighted
from analysis.concept_analysis import compute_concept_geometry_metrics


MODEL_CLASSES = {
    "perceiver_mlm": ConceptEncoderForMaskedLMPerceiver,
    "perceiver_posonly_mlm": ConceptEncoderForMaskedLMPerceiverPosOnly,
    "weighted_mlm": ConceptEncoderForMaskedLMWeighted,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--model_type", default="perceiver_mlm", choices=list(MODEL_CLASSES))
    p.add_argument("--output_json", default=None)
    p.add_argument("--num_batches", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--dataset", default="JeanKaddour/minipile")
    p.add_argument("--max_seq_length", type=int, default=512)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading model from: {args.model_path}")
    print(f"Model type: {args.model_type}")

    model_class = MODEL_CLASSES[args.model_type]
    model = model_class.from_pretrained(args.model_path)
    model = model.to(device).eval()

    print(f"Loading tokenizer from: {args.model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

    print(f"Loading dataset: {args.dataset}")
    ds = load_dataset(args.dataset, split="train", streaming=True)

    all_metrics = []
    concept_reprs = []
    n = 0

    print(f"Running {args.num_batches} batches of size {args.batch_size} ...")
    batch_texts = []

    with torch.no_grad():
        for sample in ds:
            text = sample.get("text", "") or ""
            if len(text.strip()) < 20:
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
                input_ids = enc["input_ids"].to(device)
                attention_mask = enc["attention_mask"].to(device)

                # Forward pass — grab concepts from encoder directly
                encoder_out = model.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
                concepts = encoder_out.last_hidden_state.float()  # [B, C, H]

                batch_metrics = compute_concept_geometry_metrics(concepts.cpu())
                all_metrics.append(batch_metrics)
                concept_reprs.append(concepts.cpu())

                n += 1
                batch_texts = []
                if n >= args.num_batches:
                    break

    print(f"\nAnalysed {n} batches, {n * args.batch_size} total samples.")

    # Aggregate metrics
    agg = {}
    for key in all_metrics[0]:
        vals = [m[key] for m in all_metrics if not (isinstance(m[key], float) and m[key] != m[key])]
        if vals:
            agg[key] = sum(vals) / len(vals)

    # Cross-batch effective rank using stacked concepts
    all_concepts = torch.cat(concept_reprs, dim=0)  # [N*B, C, H]
    concept_mean = all_concepts.mean(dim=0)          # [C, H]
    try:
        _, S, _ = torch.svd(concept_mean)
        global_eff_rank = (S.sum() / (S.max() + 1e-8)).item()
        global_eff_rank_norm = global_eff_rank / min(concept_mean.shape[0], concept_mean.shape[1])
        singular_values = S.tolist()
    except Exception:
        global_eff_rank = float("nan")
        global_eff_rank_norm = float("nan")
        singular_values = []

    agg["global_effective_rank"] = global_eff_rank
    agg["global_effective_rank_normalized"] = global_eff_rank_norm

    # --- Print report ---
    print("\n" + "=" * 65)
    print("CONCEPT SPACE GEOMETRY REPORT")
    print("=" * 65)
    print(f"Model           : {args.model_path}")
    print(f"Model type      : {args.model_type}")
    print(f"Concepts (C)    : {concept_mean.shape[0]}")
    print(f"Hidden dim (H)  : {concept_mean.shape[1]}")
    print(f"Batches analysed: {n}")
    print()

    grade = lambda v, lo, hi: ("✓ GOOD" if v >= hi else ("△ OK" if v >= lo else "✗ POOR"))

    def row(name, val, lo, hi, fmt=".4f", unit=""):
        g = grade(val, lo, hi)
        print(f"  {name:<40s} {val:{fmt}}{unit}   {g}")

    print("─── Collapse Detection ─────────────────────────────────────")
    row("Global effective rank (raw)",
        global_eff_rank, 40, 90)
    row("Global effective rank (normalized 0-1)",
        global_eff_rank_norm, 0.3, 0.7, fmt=".3f")
    row("Participation ratio (normalized)",
        agg.get("participation_ratio_normalized", float("nan")), 0.1, 0.3, fmt=".3f")
    row("Dimensions needed for 95% variance",
        agg.get("dimensions_for_95_variance", float("nan")), 10, 50, fmt=".1f")
    row("Collapsed dimensions (ratio)",
        1.0 - agg.get("collapsed_dimensions_ratio", 1.0), 0.9, 0.99, fmt=".3f",
        unit="  (fraction active)")
    row("Isotropy (min/max eigenvalue ratio)",
        agg.get("isotropy", float("nan")), 0.001, 0.01, fmt=".5f")

    print()
    print("─── Concept Diversity ──────────────────────────────────────")
    mean_sim = agg.get("mean_concept_similarity", float("nan"))
    max_sim = agg.get("max_concept_similarity", float("nan"))
    # Lower similarity = more diverse (grade: <0.3 good, <0.5 ok)
    sim_grade = "✓ GOOD" if mean_sim < 0.3 else ("△ OK" if mean_sim < 0.5 else "✗ POOR (concepts correlated)")
    print(f"  {'Mean pairwise concept similarity':<40s} {mean_sim:.4f}   {sim_grade}")
    max_grade = "✓ GOOD" if max_sim < 0.6 else ("△ OK" if max_sim < 0.8 else "✗ POOR")
    print(f"  {'Max pairwise concept similarity':<40s} {max_sim:.4f}   {max_grade}")
    uni = agg.get("uniformity_loss", float("nan"))
    uni_grade = "✓ GOOD" if uni < 0.3 else ("△ OK" if uni < 0.6 else "✗ POOR (clustered)")
    print(f"  {'Uniformity loss (lower = more spread)':<40s} {uni:.4f}   {uni_grade}")

    print()
    print("─── Dimension Utilization ──────────────────────────────────")
    row("Mean dimension std",
        agg.get("mean_dimension_std", float("nan")), 0.3, 0.8, fmt=".4f")
    row("Min dimension std",
        agg.get("min_dimension_std", float("nan")), 0.01, 0.1, fmt=".5f")

    print()
    print("─── Concept Norms ──────────────────────────────────────────")
    print(f"  Mean concept L2 norm : {agg.get('mean_concept_norm', float('nan')):.4f}")
    print(f"  Std concept L2 norm  : {agg.get('std_concept_norm', float('nan')):.4f}")

    print()
    print("─── Top-5 Singular Values (concept mean matrix) ────────────")
    top5 = [f"{v:.3f}" for v in singular_values[:5]]
    print(f"  {', '.join(top5)}")
    if len(singular_values) > 1:
        dom_ratio = singular_values[0] / (sum(singular_values) + 1e-8)
        dom_grade = "✓ GOOD" if dom_ratio < 0.3 else ("△ OK" if dom_ratio < 0.5 else "✗ POOR (1 concept dominates)")
        print(f"  Top-1 dominance ratio: {dom_ratio:.3f}   {dom_grade}")

    print()
    print("─── Recommendations ────────────────────────────────────────")
    if global_eff_rank_norm < 0.3:
        print("  → CRITICAL: Effective rank < 30% — concepts are collapsed.")
        print("    Add VICReg or t_regs_mst loss BEFORE scaling data.")
    elif global_eff_rank_norm < 0.5:
        print("  → effective rank 30-50% — add 'combined' loss to improve utilization.")
    else:
        print("  → Effective rank OK — proceed with data scaling.")

    if mean_sim > 0.5:
        print("  → Mean concept similarity > 0.5 — add orthogonality or uniformity loss.")
    if agg.get("min_dimension_std", 1.0) < 0.01:
        print("  → Some dimensions near-zero — add variance or VICReg loss.")

    print("=" * 65)

    result = {
        "model_path": args.model_path,
        "model_type": args.model_type,
        "n_batches": n,
        "n_samples": n * args.batch_size,
        **agg,
        "global_effective_rank": global_eff_rank,
        "global_effective_rank_normalized": global_eff_rank_norm,
        "top5_singular_values": singular_values[:5],
    }

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to: {args.output_json}")

    return result


if __name__ == "__main__":
    main()
