"""
Standalone concept analysis runner (Tier 1 of the evaluation pipeline).

Loads a pretrained ConceptEncoder checkpoint, runs text batches through it, and
computes:
  * concept-space geometry metrics (effective rank, collapse, diversity) — all families;
  * concept-ablation ΔCE (zero / shuffle / no-concept floor) — `concept_ar` only;
  * qualitative autoregressive generation samples — `concept_ar` only.

Geometry answers "are concepts collapsed?"; ablation answers "does the AR decoder
actually use the concepts?" (E01/E02 primary gate); samples answer "is the generated
text coherent?". Results print to stdout and optionally save to a JSON file.

Usage:
    # Geometry only (any family)
    uv run python analysis/run_concept_analysis.py \
        --model_path Cache/Training/MODEL/checkpoint-XXXX \
        --model_type perceiver_denoise \
        --output_json Cache/Evaluation_reports/MODEL_concept_analysis.json

    # Geometry + AR ablation + generation samples (E01/E02), on FineWeb-Edu held-out text
    uv run python analysis/run_concept_analysis.py \
        --model_path Cache/Training/MODEL/checkpoint-XXXX \
        --model_type concept_ar \
        --dataset HuggingFaceFW/fineweb-edu --dataset_config sample-10BT \
        --output_json Cache/Evaluation_reports/MODEL_concept_analysis.json \
        --ablation_batches 8 --num_samples 4

Notes:
  * Geometry uses the encoder only, so it works for every maintained family.
  * Ablation / generation require an AR model (`concept_ar`); they are skipped otherwise.
  * `concept_ablation_ce` here covers the E01 reconstruction contract (encoder sees the
    clean sequence). The full E02 prefix→suffix ablation is reported inside training; use
    the training eval log for the suffix-CE deltas on prefix/suffix runs.
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
    ConceptEncoderForConditionalLM,
    ConceptEncoderForDenoisingPerceiver,
)
from nn.concept_encoder_weighted import ConceptEncoderForMaskedLMWeighted
from analysis.concept_analysis import (
    compute_concept_geometry_metrics,
    compute_representation_manifold_metrics,
    compute_within_sample_concept_rank,
)
from analysis.concept_generation_eval import (
    compute_roundtrip_recovery,
    compute_latent_specificity,
)


# Diffusion families (diffusion_mlm, prefix_diffusion) are parked in `parked/`;
# revive their MODEL_CLASSES entries alongside the parked model code if needed.
# concept_ar (E01) exposes .encoder like the others, so geometry analysis just works.
MODEL_CLASSES = {
    "perceiver_denoise": ConceptEncoderForDenoisingPerceiver,
    "concept_ar": ConceptEncoderForConditionalLM,
    "weighted_mlm": ConceptEncoderForMaskedLMWeighted,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--model_type", default="perceiver_denoise", choices=list(MODEL_CLASSES))
    p.add_argument("--output_json", default=None)
    p.add_argument("--num_batches", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--dataset", default="JeanKaddour/minipile")
    p.add_argument("--dataset_config", default=None,
                   help="Dataset config/subset, e.g. 'sample-10BT' for HuggingFaceFW/fineweb-edu.")
    p.add_argument("--max_seq_length", type=int, default=512)
    # concept_ar-only knobs (ignored for other families)
    p.add_argument("--ablation_batches", type=int, default=5,
                   help="concept_ar: number of held-out batches for concept-ablation ΔCE.")
    p.add_argument("--num_samples", type=int, default=4,
                   help="concept_ar: number of qualitative AR generation samples to dump.")
    p.add_argument("--max_new_tokens", type=int, default=64,
                   help="concept_ar: max tokens to greedily generate per sample.")
    p.add_argument("--generation_eval", action="store_true", default=True,
                   help="concept_ar: run L1/L3 round-trip recovery + compression curve + specificity.")
    p.add_argument("--no_generation_eval", dest="generation_eval", action="store_false",
                   help="Disable the L1/L3 generation/compression faithfulness eval.")
    p.add_argument("--free_running_examples", type=int, default=8,
                   help="concept_ar: number of free-running greedy round-trip examples (cost O(N*tokens)).")
    return p.parse_args()


@torch.no_grad()
def compute_ar_concept_ablation(model, batches, device):
    """Average concept_ablation_ce over a few held-out reconstruction batches.

    `batches` is a list of (input_ids, attention_mask) tensors already on CPU.
    Labels mask padding POSITIONALLY via attention_mask (labels[mask==0] = -100),
    never by token id: with SmolLM2-style pad=eos tokenizers, masking by id would
    silently drop every real eos target and skew CE vs the training-eval numbers.
    Returns the averaged ablation dict (ce_intact, ce_zero, ce_shuffle, deltas,
    plus ce_intact_wd / gap_clean_vs_wd when the model trained with word-dropout).

    Note: the encoder here sees the FULL clean sequence (no TSDAE deletion), so
    absolute CE values are not comparable with the trainer's eval metrics, which
    corrupt the encoder input. Deltas remain internally consistent.
    """
    sums = {}
    n = 0
    for input_ids, attention_mask in batches:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        m = model.concept_ablation_ce(input_ids, attention_mask, labels)
        for k, v in m.items():
            sums[k] = sums.get(k, 0.0) + v
        n += 1
    if n == 0:
        return {}
    return {k: v / n for k, v in sums.items()}


@torch.no_grad()
def generate_ar_samples(model, tokenizer, texts, device, max_new_tokens=64):
    """Greedy autoregressive decode conditioned on concepts from clean text.

    Encodes each text fully visible (attention all ones) to concepts, then greedily
    generates from the start token until eos or max_new_tokens. Returns a list of
    {prompt_preview, generated} dicts for qualitative coherence inspection.
    """
    cfg = model.config
    start_id = cfg.bos_token_id or cfg.eos_token_id or cfg.pad_token_id or 0
    eos_id = cfg.eos_token_id
    samples = []
    for text in texts:
        enc = tokenizer(text, max_length=cfg.max_seq_length if hasattr(cfg, "max_seq_length") else 512,
                        truncation=True, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = torch.ones_like(input_ids)
        concepts = model.encode_concepts(input_ids=input_ids, attention_mask=attention_mask,
                                          return_dict=True).last_hidden_state
        cur = torch.tensor([[start_id]], device=device)
        for _ in range(max_new_tokens):
            logits = model.decode_logits(concepts, cur)
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            cur = torch.cat([cur, next_id], dim=1)
            if eos_id is not None and next_id.item() == eos_id:
                break
        generated = tokenizer.decode(cur[0, 1:], skip_special_tokens=True)
        samples.append({
            "prompt_preview": text[:160].replace("\n", " "),
            "generated": generated.replace("\n", " "),
        })
    return samples


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

    print(f"Loading dataset: {args.dataset}" + (f" ({args.dataset_config})" if args.dataset_config else ""))
    if args.dataset_config:
        ds = load_dataset(args.dataset, args.dataset_config, split="train", streaming=True)
    else:
        ds = load_dataset(args.dataset, split="train", streaming=True)

    is_concept_ar = args.model_type == "concept_ar"

    all_metrics = []
    concept_reprs = []
    ablation_batches = []   # (input_ids, attention_mask) for concept_ar ΔCE
    sample_texts = []       # raw texts for concept_ar generation samples
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

                if is_concept_ar and len(ablation_batches) < args.ablation_batches:
                    ablation_batches.append((input_ids.cpu(), attention_mask.cpu()))
                if is_concept_ar and len(sample_texts) < args.num_samples:
                    sample_texts.extend(batch_texts[: args.num_samples - len(sample_texts)])

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

    # Per-sample representation manifold: SVD of the mean-pooled sentence embeddings
    # [N_samples, H] (exactly what zero-shot STS-B consumes). Unlike global_effective_rank
    # (which averages over the batch FIRST and so measures slot redundancy), this measures
    # how many directions DIFFERENT INPUTS span — the geometry downstream tasks ride on.
    pooled_embeddings = all_concepts.mean(dim=1)  # [N_samples, H]
    manifold = compute_representation_manifold_metrics(pooled_embeddings)
    agg.update(manifold)

    # PRIMARY de-collapse metric: per-sample within-set concept rank (how many
    # independent directions ONE input's C concepts span), averaged over inputs.
    # This is distinct from global_effective_rank (slot redundancy after batch-averaging)
    # and manifold_rankme (cross-sample embedding diversity).
    within = compute_within_sample_concept_rank(all_concepts)
    agg.update(within)

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

    print("─── De-collapse (PRIMARY: within-sample concept-set rank) ───")
    print("    RankMe of each input's [C, H] concepts, averaged over inputs.")
    print("    THE de-collapse metric: are one input's C concepts diverse?")
    row("Within-sample concept RankMe (mean)",
        agg.get("within_sample_rankme_mean", float("nan")), 16, 48, fmt=".2f")
    print(f"  {'  (std over inputs)':<40s} "
          f"{agg.get('within_sample_rankme_std', float('nan')):.2f}")

    print()
    print("─── Collapse Detection (SECONDARY diagnostics) ─────────────")
    print("    Global effective rank = SVD of the BATCH-AVERAGED slot matrix")
    print("    → measures slot redundancy, NOT per-input concept rank. Diagnostic only.")
    row("Slot-mean effective rank (raw, secondary)",
        global_eff_rank, 40, 90)
    row("Slot-mean effective rank (normalized, secondary)",
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
    print("─── Cross-sample Embedding Diversity (downstream/STS geometry) ─")
    print("    RankMe of mean-pooled sentence embeddings [N, H] across inputs.")
    print("    Embedding diversity across inputs — NOT concept-set rank (can exceed C).")
    row("Cross-sample embedding RankMe (entropy-based)",
        manifold.get("manifold_rankme", float("nan")), 16, 48, fmt=".2f")
    row("Participation ratio",
        manifold.get("manifold_participation_ratio", float("nan")), 16, 48, fmt=".2f")
    row("Dims for 95% across-sample variance",
        manifold.get("manifold_dims_for_95_variance", float("nan")), 10, 50, fmt=".0f")
    ani = manifold.get("manifold_anisotropy", float("nan"))
    ani_grade = "✓ GOOD" if ani < 0.3 else ("△ OK" if ani < 0.6 else "✗ POOR (narrow cone)")
    print(f"  {'Anisotropy (mean random-pair cosine)':<40s} {ani:.4f}   {ani_grade}")
    print(f"  {'Top-1 variance ratio (pooled)':<40s} "
          f"{manifold.get('manifold_top_1_variance_ratio', float('nan')):.4f}")

    print()
    print("─── Per-slot Input Activity (dead-register detection) ──────")
    row("Active slot fraction (std>1e-3 over inputs)",
        agg.get("active_slot_fraction", float("nan")), 0.5, 0.9, fmt=".3f")
    row("Mean slot input std",
        agg.get("mean_slot_input_std", float("nan")), 0.05, 0.2, fmt=".4f")

    print()
    print("─── Recommendations ────────────────────────────────────────")
    if global_eff_rank_norm < 0.3:
        print("  → CRITICAL: Effective rank < 30% — concepts are collapsed.")
        print("    Stop and check zero-shot STS-B before spending more GPU time.")
        print("    If this is a denoising run, only then try a light t_regs_mst or contrastive stage.")
    elif global_eff_rank_norm < 0.5:
        print("  → effective rank 30-50% — run zero-shot STS-B and a sentence-pair eval sweep next.")
    else:
        print("  → Effective rank OK — proceed to zero-shot STS-B, then fine-tuned evaluation.")

    if mean_sim > 0.5:
        print("  → Mean concept similarity > 0.5 — add orthogonality or uniformity loss.")
    if agg.get("min_dimension_std", 1.0) < 0.01:
        print("  → Some dimensions near-zero — add variance or VICReg loss.")

    print("=" * 65)

    # --- concept_ar: ablation ΔCE + qualitative generation samples ---
    ablation = {}
    samples = []
    gen_faith = {}
    if is_concept_ar:
        try:
            ablation = compute_ar_concept_ablation(model, ablation_batches, device)
        except Exception as e:  # never let AR extras kill the geometry report
            print(f"\n[concept_ar] concept-ablation skipped: {e}")
        if ablation:
            print()
            print("─── Concept-Ablation ΔCE (does the AR decoder use concepts?) ─")
            print(f"  CE intact            : {ablation['ce_intact']:.4f}")
            print(f"  CE zero (floor)      : {ablation['ce_zero']:.4f}")
            print(f"  CE shuffle           : {ablation['ce_shuffle']:.4f}")
            dz, dsh = ablation["delta_zero"], ablation["delta_shuffle"]
            gz = "✓ uses concepts" if dz >= 0.5 else "✗ near-collapse"
            gsh = "✓ uses concepts" if dsh >= 0.5 else "✗ near-collapse"
            print(f"  Δzero  (zero-intact) : {dz:.4f}   {gz}")
            print(f"  Δshuffle             : {dsh:.4f}   {gsh}   (stronger test)")
            print("  E01 gate: Δzero AND Δshuffle ≥ 0.5 nats.")
            # Early-position deltas: the sharper instrument. Concept reliance is strongest on
            # the first ~k targets; later positions are predictable from teacher-forced local
            # context regardless of concepts, diluting the all-position delta (the AR bypass).
            if "delta_zero_early" in ablation:
                dze, dshe = ablation["delta_zero_early"], ablation["delta_shuffle_early"]
                gze = "✓ uses concepts" if dze >= 0.5 else "✗ near-collapse"
                gshe = "✓ uses concepts" if dshe >= 0.5 else "✗ near-collapse"
                print(f"  Δzero  (early-pos)   : {dze:.4f}   {gze}   ← PRIMARY (less bypass dilution)")
                print(f"  Δshuffle (early-pos) : {dshe:.4f}   {gshe}   ← PRIMARY")
            if "ce_intact_wd" in ablation:
                gap = ablation["gap_clean_vs_wd"]
                gw = ("⚠ decoder specialized to word-dropped inputs — clean-input CE "
                      "understates quality" if gap > 0.5 else "✓ clean/train conditions agree")
                print(f"  CE intact (train-matched word-dropout): {ablation['ce_intact_wd']:.4f}")
                print(f"  Gap clean-vs-wd      : {gap:.4f}   {gw}")
        if args.generation_eval:
            try:
                recovery = compute_roundtrip_recovery(
                    model, ablation_batches, device,
                    concept_num=concept_mean.shape[0],
                    free_running_examples=args.free_running_examples,
                )
                specificity = compute_latent_specificity(model, ablation_batches, device)
                gen_faith = {**recovery, **specificity}
            except Exception as e:  # never let it kill the geometry report
                gen_faith = {}
                print(f"\n[concept_ar] generation/compression eval skipped: {e}")
            if gen_faith:
                print()
                print("─── L1/L3 — Generation & compression faithfulness ──────────")
                print(f"  Teacher-forced token acc : {gen_faith['teacher_forced_token_acc']:.4f}  "
                      "(recover input FROM concepts)")
                print(f"  Free-running exact match  : {gen_faith['free_running_exact_match']:.4f}  "
                      f"(greedy, n={gen_faith['free_running_n']})")
                print(f"  Free-running token-F1     : {gen_faith['free_running_token_f1']:.4f}")
                drop = gen_faith["specificity_acc_drop"]
                gdrop = "✓ input-specific" if drop >= 0.05 else "✗ not specific to input"
                print(f"  Specificity acc drop      : {drop:.4f}   {gdrop}  "
                      f"(matched {gen_faith['specificity_acc_matched']:.3f} vs "
                      f"shuffled {gen_faith['specificity_acc_shuffled']:.3f})")
                print(f"  Specificity symmetric-KL  : {gen_faith['specificity_symmetric_kl']:.4f}")
                curve = gen_faith.get("compression_curve", {})
                if curve:
                    print("  Compression curve (recovery vs ⌈seq_len/C⌉ ratio):")
                    for r in curve.values():
                        print(f"    ratio ~{r['compression_ratio']:>3}x : "
                              f"acc {r['teacher_forced_token_acc']:.3f}  ({r['n_tokens']} tok)")
        else:
            gen_faith = {}
        try:
            samples = generate_ar_samples(model, tokenizer, sample_texts, device,
                                          max_new_tokens=args.max_new_tokens)
        except Exception as e:
            print(f"\n[concept_ar] generation samples skipped: {e}")
        if samples:
            print()
            print("─── AR Generation Samples (greedy, concept-conditioned) ────")
            for i, s in enumerate(samples):
                print(f"  [{i}] prompt : {s['prompt_preview']}")
                print(f"      gen    : {s['generated'][:160]}")
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
    if ablation:
        result["concept_ablation"] = ablation
    if gen_faith:
        result["generation_faithfulness"] = gen_faith
    if samples:
        result["generation_samples"] = samples

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to: {args.output_json}")

    return result


if __name__ == "__main__":
    main()
