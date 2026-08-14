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

Data protocol (2026-07-07 upgrade — see docs/engineering_specs/concept_information_eval_upgrade.md):
  * `--eval_source holdout` (default) reproduces the TRAINING eval holdout
    (`train_test_split(seed=split_seed)`, identical to `data/dataset_preprocess.py`) so
    metrics are measured on genuinely held-out documents. The old behaviour — first N docs
    of the streaming train split, i.e. almost surely TRAINING data — is kept as
    `--eval_source stream` with a loud contamination warning.
  * `--eval_source pretokenized --pretokenized_manifest <path>` consumes the exact eval
    split of a pretokenized mix (the authoritative held-out source for E05+ 2K runs).
  * Batches are STRATIFIED by tokenized document length (`--length_buckets`), so geometry,
    ablation ΔCE, and the L3 compression curve are measured across length regimes instead
    of a single truncated length. Default max_seq_length is 2048 (the current 2K focus).
  * Everything is seeded (`--seed`); ablation deltas are reported as mean ± std over batches.

Usage:
    # Geometry + AR ablation + samples on the training holdout (2K protocol)
    uv run python analysis/run_concept_analysis.py \
        --model_path Cache/Training/MODEL/checkpoint-XXXX \
        --model_type concept_ar \
        --eval_source pretokenized \
        --pretokenized_manifest Cache/Datasets/pretokenized/smollm3_inspired_2k_e05/manifest.json \
        --output_json Cache/Evaluation_reports/MODEL_concept_analysis.json

    # Single-dataset holdout (replicates training split; args must match the run's
    # test_size_percent + seed)
    uv run python analysis/run_concept_analysis.py \
        --model_path Cache/Training/MODEL/checkpoint-XXXX \
        --model_type concept_ar \
        --dataset HuggingFaceFW/fineweb-edu --dataset_config sample-10BT \
        --eval_source holdout --split_seed 42 --test_size_percent 0.1 \
        --output_json Cache/Evaluation_reports/MODEL_concept_analysis.json

Notes:
  * Geometry uses the encoder only, so it works for every maintained family.
  * Ablation / generation require an AR model (`concept_ar`); they are skipped otherwise.
  * `concept_ablation_ce` here covers the E01 reconstruction contract (encoder sees the
    clean sequence). The full E02 prefix→suffix ablation is reported inside training; use
    the training eval log for the suffix-CE deltas on prefix/suffix runs.
  * Numbers produced BEFORE the 2026-07-07 data-protocol upgrade were measured on
    train-split data at seq 512 and are NOT comparable with post-upgrade numbers.
"""

import sys
import os
import argparse
import json
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer
from datasets import load_dataset

from nn.concept_encoder import ConceptEncoderConfig
from nn.concept_encoder_perceiver import (
    ConceptEncoderForConditionalLM,
    ConceptEncoderForDenoisingPerceiver,
)
from nn.backbone_concept_lm import BackboneConceptLM
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
from data.dataset_preprocess import _select_train_eval_splits, load_pretokenized_mix


# Diffusion families (diffusion_mlm, prefix_diffusion) are parked in `parked/`;
# revive their MODEL_CLASSES entries alongside the parked model code if needed.
# BackboneConceptLM exposes the same encode_concepts / concept_ablation_ce contracts,
# but intentionally has no separate encoder module or standalone generation decoder.
MODEL_CLASSES = {
    "perceiver_denoise": ConceptEncoderForDenoisingPerceiver,
    "concept_ar": ConceptEncoderForConditionalLM,
    "backbone_concept": BackboneConceptLM,
    "weighted_mlm": ConceptEncoderForMaskedLMWeighted,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--model_type", default="perceiver_denoise", choices=list(MODEL_CLASSES))
    p.add_argument("--tokenizer_name", default=None,
                   help="Explicit tokenizer to load. Default: the checkpoint's own tokenizer. "
                        "There is NO silent fallback — a wrong tokenizer produces garbage metrics.")
    p.add_argument("--output_json", default=None)
    p.add_argument("--seed", type=int, default=42,
                   help="Seeds torch+random: shuffle ablation permutation, anisotropy pair "
                        "sampling, and eval-split shuffling are all deterministic.")
    p.add_argument("--num_batches", type=int, default=24,
                   help="TOTAL batches across all length buckets (split evenly per bucket).")
    p.add_argument("--batch_size", type=int, default=8,
                   help="Default 8: at seq 2048 the 3× ablation decoder forwards are the "
                        "memory peak on 24 GB cards.")
    # --- data source ---
    p.add_argument("--eval_source", default="holdout",
                   choices=["holdout", "pretokenized", "stream"],
                   help="holdout: reproduce the training eval holdout via seeded "
                        "train_test_split (default, no train contamination). "
                        "pretokenized: exact eval split of a pretokenized mix manifest. "
                        "stream: legacy streaming train split — TRAIN-CONTAMINATED, kept "
                        "only for reproducing old numbers.")
    p.add_argument("--dataset", default="JeanKaddour/minipile")
    p.add_argument("--dataset_config", default=None,
                   help="Dataset config/subset, e.g. 'sample-10BT' for HuggingFaceFW/fineweb-edu.")
    p.add_argument("--pretokenized_manifest", default=None,
                   help="Manifest JSON from scripts/pretokenize_mix.py (eval_source=pretokenized).")
    p.add_argument("--split_seed", type=int, default=42,
                   help="holdout: seed of the training run's train_test_split (training default 42).")
    p.add_argument("--test_size_percent", type=float, default=0.1,
                   help="holdout: the training run's test_size_percent (training default 0.1).")
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--length_buckets", default="256,512,1024",
                   help="Comma-separated interior token-length bucket edges. With max_seq_length "
                        "2048 the default gives buckets (0,256] (256,512] (512,1024] (1024,2048]. "
                        "Pass '' to disable stratification (single bucket).")
    p.add_argument("--max_scan_docs", type=int, default=200_000,
                   help="Stop scanning the eval source after this many documents even if some "
                        "length buckets are underfilled (long-doc buckets can be rare).")
    # concept_ar-only knobs (ignored for other families)
    p.add_argument("--ablation_batches", type=int, default=8,
                   help="concept_ar: number of held-out batches for concept-ablation ΔCE "
                        "(taken round-robin across length buckets).")
    p.add_argument("--ablation_window_k", type=int, default=None,
                   help="E05: position boundary K for beyond-window concept-ablation ΔCE. "
                        "Defaults to the checkpoint's decoder_context_window; pass a fixed K to "
                        "compare a windowed checkpoint against its full-context control on the "
                        "same beyond-window positions.")
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


# ---------------------------------------------------------------------------
# Data protocol: eval sources + length-stratified batching
# ---------------------------------------------------------------------------

def iter_eval_token_rows(args, tokenizer):
    """Yield (input_ids: list[int], text: str) rows from the requested eval source.

    Tokenization is per-document without padding (padding happens per batch after
    length bucketing). Truncation at args.max_seq_length.
    """
    # Pre-truncate raw text before tokenizing (same guard as training's max_chars):
    # the fast tokenizer scans the WHOLE string even when truncation discards most of
    # it, so a 100MB web/PDF doc would stall the bucket scan. ~20x headroom over
    # max_seq_length at ~4 chars/token keeps this lossless for the kept tokens.
    max_chars = args.max_seq_length * 80

    def _tokenize(text):
        return tokenizer(text[:max_chars], max_length=args.max_seq_length, truncation=True)["input_ids"]

    if args.eval_source == "pretokenized":
        if not args.pretokenized_manifest:
            raise ValueError("--eval_source pretokenized requires --pretokenized_manifest.")
        _, test_ds = load_pretokenized_mix(args.pretokenized_manifest)
        # eval parts are concatenated per source; shuffle so batches mix sources.
        test_ds = test_ds.shuffle(seed=args.seed)
        for row in test_ds:
            ids = row["input_ids"][: args.max_seq_length]
            if len(ids) < 8:
                continue
            yield ids, tokenizer.decode(ids, skip_special_tokens=True)
        return

    if args.eval_source == "holdout":
        # Reproduce the training holdout exactly (data/dataset_preprocess.py):
        # same split function, same seed, same test_size cap. Requires the map-style
        # dataset (cached on the GPU servers from training).
        dataset = load_dataset(args.dataset, args.dataset_config)
        _, eval_ds = _select_train_eval_splits(
            dataset, args.test_size_percent, seed=args.split_seed
        )
        eval_ds = eval_ds.shuffle(seed=args.seed)
        for row in eval_ds:
            text = (row.get("text") or "").strip()
            if len(text) < 20:
                continue
            yield _tokenize(text), text
        return

    # stream: the legacy protocol — first docs of the streaming TRAIN split. Almost
    # every doc was trained on; keep only for reproducing pre-2026-07-07 numbers.
    print("\n" + "!" * 72)
    print("! WARNING: --eval_source stream reads the TRAIN split (contaminated).")
    print("! Metrics will be inflated by memorization. Use holdout/pretokenized.")
    print("!" * 72 + "\n")
    if args.dataset_config:
        ds = load_dataset(args.dataset, args.dataset_config, split="train", streaming=True)
    else:
        ds = load_dataset(args.dataset, split="train", streaming=True)
    for sample in ds:
        text = (sample.get("text") or "").strip()
        if len(text) < 20:
            continue
        yield _tokenize(text), text


def parse_length_buckets(spec: str, max_seq_length: int):
    """Return list of (lo, hi] token-length bucket bounds covering (0, max_seq_length]."""
    edges = [int(e) for e in spec.split(",") if e.strip()] if spec else []
    edges = sorted(e for e in edges if 0 < e < max_seq_length)
    bounds = []
    lo = 0
    for e in edges:
        bounds.append((lo, e))
        lo = e
    bounds.append((lo, max_seq_length))
    return bounds


def collect_stratified_batches(args, tokenizer):
    """Scan the eval source and build length-stratified padded batches.

    Returns a list of dicts {input_ids, attention_mask, bucket, texts} where
    input_ids/attention_mask are CPU tensors padded to the batch's longest row.
    Buckets that the corpus cannot fill (few long docs) are reported, not fatal.
    """
    buckets = parse_length_buckets(args.length_buckets, args.max_seq_length)
    n_buckets = len(buckets)
    quota = max(1, args.num_batches // n_buckets)
    quotas = [quota] * n_buckets
    for i in range(args.num_batches - quota * n_buckets):
        quotas[n_buckets - 1 - (i % n_buckets)] += 1  # remainder to the longest buckets

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("Tokenizer has no pad token even after eos fallback — cannot batch.")

    def _bucket_index(n_tokens):
        for bi, (lo, hi) in enumerate(buckets):
            if lo < n_tokens <= hi:
                return bi
        return None

    def _pad_batch(rows, bucket_label, texts):
        longest = max(len(r) for r in rows)
        input_ids = torch.full((len(rows), longest), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(rows), longest), dtype=torch.long)
        for i, r in enumerate(rows):
            input_ids[i, : len(r)] = torch.tensor(r, dtype=torch.long)
            attention_mask[i, : len(r)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask,
                "bucket": bucket_label, "texts": texts}

    buffers = [[] for _ in range(n_buckets)]
    text_buffers = [[] for _ in range(n_buckets)]
    done = [0] * n_buckets
    batches = []
    scanned = 0

    for ids, text in iter_eval_token_rows(args, tokenizer):
        scanned += 1
        bi = _bucket_index(len(ids))
        if bi is not None and done[bi] < quotas[bi]:
            buffers[bi].append(ids)
            text_buffers[bi].append(text)
            if len(buffers[bi]) == args.batch_size:
                label = f"({buckets[bi][0]},{buckets[bi][1]}]"
                batches.append(_pad_batch(buffers[bi], label, text_buffers[bi]))
                buffers[bi], text_buffers[bi] = [], []
                done[bi] += 1
        if all(d >= q for d, q in zip(done, quotas)) or scanned >= args.max_scan_docs:
            break

    # flush partial buffers (>=2 rows) so rare long buckets still contribute
    for bi in range(n_buckets):
        if len(buffers[bi]) >= 2 and done[bi] < quotas[bi]:
            label = f"({buckets[bi][0]},{buckets[bi][1]}]"
            batches.append(_pad_batch(buffers[bi], label, text_buffers[bi]))
            done[bi] += 1

    for bi, (lo, hi) in enumerate(buckets):
        status = "OK" if done[bi] >= quotas[bi] else f"UNDERFILLED ({done[bi]}/{quotas[bi]})"
        print(f"  length bucket ({lo:>4},{hi:>4}] : {done[bi]} batches   {status}")
    print(f"  scanned {scanned} documents from eval_source={args.eval_source}")
    return batches


@torch.no_grad()
def compute_ar_concept_ablation(model, batches, device, window_k=None):
    """Average concept_ablation_ce over held-out reconstruction batches.

    `batches` is a list of dicts {input_ids, attention_mask, bucket} with CPU tensors.
    Labels mask padding POSITIONALLY via attention_mask (labels[mask==0] = -100),
    never by token id: with SmolLM2-style pad=eos tokenizers, masking by id would
    silently drop every real eos target and skew CE vs the training-eval numbers.

    Returns the averaged ablation dict (ce_intact, ce_zero, ce_shuffle, deltas,
    plus ce_intact_wd / gap_clean_vs_wd when the model trained with word-dropout),
    with `<key>_std` per-batch standard deviations for the delta keys (the gates
    are hard thresholds, so run-to-run spread matters), and a `per_bucket`
    breakdown of ce_intact / delta_zero / delta_shuffle by length bucket.

    Note: the encoder here sees the FULL clean sequence (no TSDAE deletion), so
    absolute CE values are not comparable with the trainer's eval metrics, which
    corrupt the encoder input. Deltas remain internally consistent.
    """
    per_batch = []
    for batch in batches:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        m = model.concept_ablation_ce(input_ids, attention_mask, labels, window_k=window_k)
        # The classic concept_ar family names the reference path "intact"; E10 names it
        # "real". Normalize aliases here so aggregation/reporting works for both contracts.
        for suffix in ("", "_early", "_carry", "_beyond"):
            real_key = f"ce_real{suffix}"
            intact_key = f"ce_intact{suffix}"
            if real_key in m and intact_key not in m:
                m[intact_key] = m[real_key]
        m["_bucket"] = batch.get("bucket", "all")
        per_batch.append(m)
    if not per_batch:
        return {}

    # Union keys across batches: short sequences omit _beyond/_carry, so taking only
    # per_batch[0]'s keys silently drops the E10/E16b long-range gate when a short
    # bucket is sampled first.
    keys = sorted({k for m in per_batch for k in m if k != "_bucket"})
    out = {}
    for k in keys:
        vals = [m[k] for m in per_batch if k in m]
        out[k] = sum(vals) / len(vals)
        if k.startswith("delta_") and len(vals) > 1:
            mean = out[k]
            out[f"{k}_std"] = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5

    per_bucket = {}
    for m in per_batch:
        b = per_bucket.setdefault(m["_bucket"], {"n": 0})
        b["n"] += 1
        for k in ("ce_intact", "delta_zero", "delta_shuffle"):
            if k in m:
                b[k] = b.get(k, 0.0) + m[k]
    for b in per_bucket.values():
        for k in ("ce_intact", "delta_zero", "delta_shuffle"):
            if k in b:
                b[k] /= b["n"]
    out["per_bucket"] = per_bucket
    return out


@torch.no_grad()
def generate_ar_samples(model, tokenizer, texts, device, max_new_tokens=64, max_seq_length=None):
    """Greedy autoregressive decode conditioned on concepts from clean text.

    Encodes each text fully visible (attention all ones) to concepts, then greedily
    generates from the start token until eos or max_new_tokens. Returns a list of
    {prompt_preview, generated} dicts for qualitative coherence inspection.
    """
    cfg = model.config
    start_id = cfg.bos_token_id or cfg.eos_token_id or cfg.pad_token_id or 0
    eos_id = cfg.eos_token_id
    # Config attribute is max_sequence_length; the old hasattr(cfg, "max_seq_length")
    # check never matched and silently truncated every prompt to 512.
    if max_seq_length is None:
        max_seq_length = getattr(cfg, "max_sequence_length", 512)
    samples = []
    for text in texts:
        enc = tokenizer(text, max_length=max_seq_length,
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
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print(f"Loading model from: {args.model_path}")
    print(f"Model type: {args.model_type}")

    model_class = MODEL_CLASSES[args.model_type]
    model = model_class.from_pretrained(args.model_path)
    model = model.to(device).eval()

    tokenizer_src = args.tokenizer_name or args.model_path
    print(f"Loading tokenizer from: {tokenizer_src}")
    # No silent fallback: a wrong tokenizer yields garbage metrics that LOOK valid.
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Eval source: {args.eval_source}"
          + (f" | dataset: {args.dataset}"
             + (f" ({args.dataset_config})" if args.dataset_config else "")
             if args.eval_source != "pretokenized"
             else f" | manifest: {args.pretokenized_manifest}"))

    print(f"Collecting {args.num_batches} length-stratified batches of size {args.batch_size} "
          f"(max_seq_length={args.max_seq_length}) ...")
    batches = collect_stratified_batches(args, tokenizer)
    if not batches:
        raise RuntimeError("No batches collected — check eval_source/dataset arguments.")

    is_concept_ar = args.model_type == "concept_ar"
    has_ablation = args.model_type in {"concept_ar", "backbone_concept"}

    all_metrics = []
    concept_reprs = []
    bank_concept_reprs = {}
    bucket_of_batch = []
    ablation_batches = []   # dicts {input_ids, attention_mask, bucket} for concept_ar ΔCE
    sample_texts = []       # raw texts for concept_ar generation samples

    with torch.no_grad():
        for batch in batches:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Forward pass — use the public concept contract. The classic families expose
            # both .encoder and encode_concepts; E10 is recurrent encode==decode and only
            # has encode_concepts.
            if (
                hasattr(model, "encode_concept_banks")
                and getattr(model.config, "concept_io_mode", None) == "per_layer_banks"
            ):
                banks = model.encode_concept_banks(
                    input_ids=input_ids, attention_mask=attention_mask
                ).float()
                concepts = banks[:, -1]
                for bank_index in range(banks.shape[1]):
                    bank_concept_reprs.setdefault(bank_index, []).append(
                        banks[:, bank_index].cpu()
                    )
            else:
                encoder_out = model.encode_concepts(
                    input_ids=input_ids, attention_mask=attention_mask, return_dict=True
                )
                concepts = encoder_out.last_hidden_state.float()  # [B, C, H]

            batch_metrics = compute_concept_geometry_metrics(concepts.cpu())
            all_metrics.append(batch_metrics)
            concept_reprs.append(concepts.cpu())
            bucket_of_batch.append(batch["bucket"])

    if has_ablation:
        # Round-robin across buckets so ΔCE covers every length regime.
        by_bucket = {}
        for batch in batches:
            by_bucket.setdefault(batch["bucket"], []).append(batch)
        order = sorted(by_bucket)
        i = 0
        while len(ablation_batches) < args.ablation_batches and any(by_bucket.values()):
            bucket = order[i % len(order)]
            if by_bucket[bucket]:
                ablation_batches.append(by_bucket[bucket].pop(0))
            i += 1
        # Generation samples: one text per bucket, longest buckets first.
        for batch in reversed(batches):
            if len(sample_texts) >= args.num_samples:
                break
            if batch["texts"]:
                sample_texts.append(batch["texts"][0])

    n = len(batches)
    total_samples = sum(b["input_ids"].shape[0] for b in batches)
    print(f"\nAnalysed {n} batches, {total_samples} total samples.")

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
    per_bank_geometry = {}
    for bank_index, parts in sorted(bank_concept_reprs.items()):
        bank_concepts = torch.cat(parts, dim=0)
        bank_metrics = compute_concept_geometry_metrics(bank_concepts)
        bank_metrics.update(compute_within_sample_concept_rank(bank_concepts))
        per_bank_geometry[f"bank_{bank_index}"] = bank_metrics

    # Per-length-bucket within-sample RankMe: is de-collapse length-dependent?
    per_bucket_rankme = {}
    bucket_concepts = {}
    for reprs, bucket in zip(concept_reprs, bucket_of_batch):
        bucket_concepts.setdefault(bucket, []).append(reprs)
    for bucket, parts in sorted(bucket_concepts.items()):
        w = compute_within_sample_concept_rank(torch.cat(parts, dim=0))
        per_bucket_rankme[bucket] = {
            "within_sample_rankme_mean": w["within_sample_rankme_mean"],
            "within_sample_rankme_centered_mean": w["within_sample_rankme_centered_mean"],
            "n_samples": sum(p.shape[0] for p in parts),
        }

    # --- Print report ---
    print("\n" + "=" * 65)
    print("CONCEPT SPACE GEOMETRY REPORT")
    print("=" * 65)
    print(f"Model           : {args.model_path}")
    print(f"Model type      : {args.model_type}")
    print(f"Concepts (C)    : {concept_mean.shape[0]}")
    print(f"Hidden dim (H)  : {concept_mean.shape[1]}")
    print(f"Batches analysed: {n}")
    print(f"Eval source     : {args.eval_source} (seed {args.seed}, max_seq {args.max_seq_length})")
    print()

    grade = lambda v, lo, hi: ("✓ GOOD" if v >= hi else ("△ OK" if v >= lo else "✗ POOR"))

    def row(name, val, lo, hi, fmt=".4f", unit=""):
        g = grade(val, lo, hi)
        print(f"  {name:<40s} {val:{fmt}}{unit}   {g}")

    print("─── De-collapse (PRIMARY: within-sample concept-set rank) ───")
    print("    RankMe of each input's [C, H] concepts, averaged over inputs.")
    print("    THE de-collapse metric: are one input's C concepts diverse?")
    C = concept_mean.shape[0]
    # Grade thresholds scale with C (historical anchors 16/48 were tuned for C=128).
    ws_lo, ws_hi = C * 0.125, C * 0.375
    row("Within-sample concept RankMe (mean)",
        agg.get("within_sample_rankme_mean", float("nan")), ws_lo, ws_hi, fmt=".2f")
    print(f"  {'  (std over inputs)':<40s} "
          f"{agg.get('within_sample_rankme_std', float('nan')):.2f}")
    print(f"  {'Centered variant (shared offset removed)':<40s} "
          f"{agg.get('within_sample_rankme_centered_mean', float('nan')):.2f}")
    print("    raw low + centered high = shared-offset anisotropy, not collapse;")
    print("    low on BOTH = genuine collapse.")
    if len(per_bucket_rankme) > 1:
        print("    By length bucket (raw / centered):")
        for bucket, v in per_bucket_rankme.items():
            print(f"      {bucket:>12s} : {v['within_sample_rankme_mean']:.2f} / "
                  f"{v['within_sample_rankme_centered_mean']:.2f}   (n={v['n_samples']})")
    if per_bank_geometry:
        print("    By depth-private bank (raw / centered RankMe / mean cosine):")
        for bank_name, values in per_bank_geometry.items():
            print(
                f"      {bank_name:>12s} : "
                f"{values['within_sample_rankme_mean']:.2f} / "
                f"{values['within_sample_rankme_centered_mean']:.2f} / "
                f"{values['mean_concept_similarity']:.4f}"
            )

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
    # Verdict keyed on the PRIMARY de-collapse metric (within-sample RankMe), not the
    # secondary slot-mean rank: slot redundancy can be low while per-input concepts are
    # healthy (E02-long) and vice versa. Centered variant disambiguates shared-offset
    # anisotropy from genuine collapse.
    ws = agg.get("within_sample_rankme_mean", float("nan"))
    ws_c = agg.get("within_sample_rankme_centered_mean", float("nan"))
    ws_frac = ws / C if ws == ws else float("nan")
    if ws_frac < 0.125:
        if ws_c == ws_c and ws_c / C >= 0.25:
            print(f"  → Within-sample RankMe low ({ws:.1f}/{C}) but CENTERED rank is healthy "
                  f"({ws_c:.1f}/{C}) — shared-offset anisotropy, not full collapse.")
            print("    Check anisotropy + zero-shot STS-B; consider a centering/whitening readout.")
        else:
            print(f"  → CRITICAL: within-sample RankMe {ws:.1f}/{C} (<12.5%) — concepts are collapsed.")
            print("    Stop and check zero-shot STS-B before spending more GPU time.")
    elif ws_frac < 0.3:
        print(f"  → Within-sample RankMe {ws:.1f}/{C} (12.5–30%) — partially collapsed; run "
              "zero-shot STS-B and the ablation gates before drawing conclusions.")
    else:
        print(f"  → Within-sample RankMe {ws:.1f}/{C} OK — proceed to zero-shot STS-B, then "
              "fine-tuned evaluation.")
    if global_eff_rank_norm < 0.1:
        print("  → (diagnostic) slot-mean effective rank very low — slots are redundant on "
              "average; not a collapse verdict by itself.")

    if mean_sim > 0.5:
        print("  → Mean concept similarity > 0.5 — add orthogonality or uniformity loss.")
    if agg.get("min_dimension_std", 1.0) < 0.01:
        print("  → Some dimensions near-zero — add variance or VICReg loss.")

    print("=" * 65)

    # --- concept-ablation ΔCE; standalone generation extras remain concept_ar-only ---
    ablation = {}
    samples = []
    gen_faith = {}
    if has_ablation:
        # E05: beyond-window deltas. Explicit --ablation_window_k wins; else fall back to the
        # checkpoint's own window (None for full-context controls → no beyond-window metrics).
        window_k = args.ablation_window_k
        if window_k is None:
            window_k = getattr(model.config, "decoder_context_window", None)
        try:
            ablation = compute_ar_concept_ablation(
                model, ablation_batches, device, window_k=window_k
            )
        except Exception as e:  # never let AR extras kill the geometry report
            print(f"\n[concept_ar] concept-ablation skipped: {e}")
        if ablation:
            print()
            print("─── Concept-Ablation ΔCE (does the AR decoder use concepts?) ─")
            print(f"  CE intact            : {ablation['ce_intact']:.4f}")
            print(f"  CE zero (floor)      : {ablation['ce_zero']:.4f}")
            print(f"  CE shuffle           : {ablation['ce_shuffle']:.4f}")
            dz, dsh = ablation["delta_zero"], ablation["delta_shuffle"]
            dz_std = ablation.get("delta_zero_std", float("nan"))
            dsh_std = ablation.get("delta_shuffle_std", float("nan"))
            gz = "✓ uses concepts" if dz >= 0.5 else "✗ near-collapse"
            gsh = "✓ uses concepts" if dsh >= 0.5 else "✗ near-collapse"
            print(f"  Δzero  (zero-intact) : {dz:.4f} ± {dz_std:.4f}   {gz}")
            print(f"  Δshuffle             : {dsh:.4f} ± {dsh_std:.4f}   {gsh}   (stronger test)")
            if args.model_type == "backbone_concept":
                print("  E10 decisive gate is Δshuffle_beyond ≥ 0.1 at positions >=2K "
                      "(>=1024 for K=512); all-position deltas are diagnostic.")
            else:
                print("  E01 gate: Δzero AND Δshuffle ≥ 0.5 nats. (± = std over batches; a gate "
                      "cleared by less than one std is not decisively cleared.)")
            per_bucket = ablation.get("per_bucket", {})
            if len(per_bucket) > 1:
                print("  By length bucket (ce_intact / Δzero / Δshuffle):")
                for bucket, v in sorted(per_bucket.items()):
                    print(f"    {bucket:>12s} : {v.get('ce_intact', float('nan')):.3f} / "
                          f"{v.get('delta_zero', float('nan')):.3f} / "
                          f"{v.get('delta_shuffle', float('nan')):.3f}   (n={v['n']})")
            # Early-position deltas: the sharper instrument. Concept reliance is strongest on
            # the first ~k targets; later positions are predictable from teacher-forced local
            # context regardless of concepts, diluting the all-position delta (the AR bypass).
            if "delta_zero_early" in ablation:
                dze, dshe = ablation["delta_zero_early"], ablation["delta_shuffle_early"]
                gze = "✓ uses concepts" if dze >= 0.5 else "✗ near-collapse"
                gshe = "✓ uses concepts" if dshe >= 0.5 else "✗ near-collapse"
                qualifier = "" if args.model_type == "backbone_concept" else "   ← PRIMARY"
                print(f"  Δzero  (early-pos)   : {dze:.4f}   {gze}{qualifier}")
                print(f"  Δshuffle (early-pos) : {dshe:.4f}   {gshe}{qualifier}")
            if "delta_shuffle_beyond" in ablation:
                dzb = ablation["delta_zero_beyond"]
                dshb = ablation["delta_shuffle_beyond"]
                dst = ablation.get("delta_static_beyond", float("nan"))
                d1b = ablation.get("delta_one_block_beyond", float("nan"))
                gshb = "✓ content-bearing memory" if dshb >= 0.1 else "✗ no long-range content"
                boundary = 2 * getattr(model.config, "concept_block", 512)
                print(f"  Δzero  (>={boundary})      : {dzb:.4f}")
                print(f"  Δshuffle (>={boundary})    : {dshb:.4f}   {gshb}   ← E10 GATE")
                print(f"  Δstatic (>={boundary})     : {dst:.4f}   ← recurrent vs learned static state")
                print(f"  Δone-block (>={boundary})  : {d1b:.4f}   ← accumulated vs prior-block-only")
            # E05 long-range memory gate: beyond-window positions (t >= K) cannot reach
            # far-back tokens locally, so a large gap there = concepts carry cross-window memory.
            if "delta_zero_beyond_window" in ablation:
                wk = ablation.get("window_k")
                dzb, dshb = ablation["delta_zero_beyond_window"], ablation["delta_shuffle_beyond_window"]
                gzb = "✓ cross-window memory" if dzb >= 0.5 else "✗ no long-range use"
                gshb = "✓ cross-window memory" if dshb >= 0.5 else "✗ no long-range use"
                print(f"  CE intact (beyond K={wk}): {ablation['ce_intact_beyond_window']:.4f}  "
                      f"(within-window {ablation['ce_intact_within_window']:.4f})")
                print(f"  Δzero  (beyond-window): {dzb:.4f}   {gzb}   ← E05 GATE")
                print(f"  Δshuffle (beyond-win) : {dshb:.4f}   {gshb}   ← E05 GATE")
            if "ce_intact_wd" in ablation:
                gap = ablation["gap_clean_vs_wd"]
                gw = ("⚠ decoder specialized to word-dropped inputs — clean-input CE "
                      "understates quality" if gap > 0.5 else "✓ clean/train conditions agree")
                print(f"  CE intact (train-matched word-dropout): {ablation['ce_intact_wd']:.4f}")
                print(f"  Gap clean-vs-wd      : {gap:.4f}   {gw}")
        if is_concept_ar and args.generation_eval:
            # concept_generation_eval expects (input_ids, attention_mask) tuples.
            ablation_pairs = [(b["input_ids"], b["attention_mask"]) for b in ablation_batches]
            try:
                recovery = compute_roundtrip_recovery(
                    model, ablation_pairs, device,
                    concept_num=concept_mean.shape[0],
                    free_running_examples=args.free_running_examples,
                )
                specificity = compute_latent_specificity(model, ablation_pairs, device)
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
        if is_concept_ar:
            try:
                samples = generate_ar_samples(model, tokenizer, sample_texts, device,
                                              max_new_tokens=args.max_new_tokens,
                                              max_seq_length=args.max_seq_length)
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
        "data_protocol_version": "2026-07-07",   # held-out + length-stratified + seeded
        "eval_source": args.eval_source,
        "dataset": args.dataset if args.eval_source != "pretokenized" else args.pretokenized_manifest,
        "seed": args.seed,
        "max_seq_length": args.max_seq_length,
        "length_buckets": args.length_buckets,
        "n_batches": n,
        "n_samples": total_samples,
        **agg,
        "global_effective_rank": global_eff_rank,
        "global_effective_rank_normalized": global_eff_rank_norm,
        "top5_singular_values": singular_values[:5],
        "per_bucket_within_sample_rankme": per_bucket_rankme,
        "per_bank_geometry": per_bank_geometry,
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
