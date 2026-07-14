# E10 — 100M-token concept-arm pilot

**Date:** 2026-07-11
**Machine:** Odra (3× RTX 3090, 24 GB each)
**Run ID:** `backbone_concept_gemma_3_1b_pt_K512_concept_20260711_152847`
**WandB (training):** [Link](https://wandb.ai/ksopyla/MrCogito/runs/backbone_concept_gemma_3_1b_pt_K512_concept_20260711_152847) (group `E10_backbone_concept_gemma_3_1b_pt_K512`)
**Raw shell log:** `Cache/logs/shell_perceiver_denoise_20260711_151741.log`
**Best checkpoint:** `Cache/Training/backbone_concept_gemma_3_1b_pt_K512_concept_20260711_152847/checkpoint-1440` (copied to `final/`; final eval_loss 1.815)
**Git commit:** `6411356`
**Git tag:** —
**Related TODO:** E10 100M concept-arm pilot — [spec](../../experiments_specs/done_failed/E10_gemma_backbone_concept_memory.md)

---

## Goal

Run the first bounded E10 training probe before committing to the original 2B-token-per-arm budget. The pilot asks whether the global→concept read path and recurrent write begin carrying measurable information by 100M non-padding training tokens. The decisive concept-vs-control criterion remains unavailable until the matched `CONCEPT_NUM=0` arm runs.

## Configuration

| Item | Value |
|---|---|
| Family / backbone | `backbone_concept`; frozen `google/gemma-3-1b-pt` + LoRA r=16 |
| Concept mechanism | C=128, `global_kv` reads at Gemma global layers, recurrent BiXT write every K=512 tokens |
| Data / objective | Gemma-tokenized `smollm3_inspired_2k_e05`; seq 2048; causal next-token LM |
| Budget | **100M non-padding target tokens**; 0.010473 epoch; 1,449 optimizer steps |
| Effective batch | 8/GPU × 3 GPUs × accumulation 3 = **72** |
| Optimization | AdamW, LR 1e-4 cosine, warmup 50, clip 0.5, bf16, gradient checkpointing |
| Eval / save | 128 deterministic held-out samples every 180 steps; save every 360 steps |
| Runtime / throughput | 22,523 s (6 h 15 m); 0.064 steps/s; 4.63 samples/s |
| Compute | **18.77 GPU-h / 5.95 kWh / 0.214B max-token upper bound** (`compute/audit_state=finished`, flag `loss_fraction:unknown`; the run protocol targeted 100M non-padding tokens) |

## Training Outcome

Training completed all 1,449 steps without NaN, OOM, DDP failure, or CE divergence. Eval loss decreased monotonically from **1.845** at step 180 to **1.815** at step 1440. The aggregate W&B train loss of 6.251 is inflated by the known Transformers gradient-accumulation logging path: with accumulation 3, the comparable per-token train CE is about **2.08**; gradient scaling itself was correct.

The final checkpoint and final export were saved successfully. Short gradient-norm spikes resolved on the next log interval and did not coincide with eval regression.

## Concept Health

| Metric | Step 180 | Step 1440 | Gate / read |
|---|---:|---:|---|
| Within-sample RankMe mean | 116.4 | **77.1** | success floor 0.3·C = 38.4 — pass |
| Within-sample RankMe minimum | 113.5 | **72.3** | collapse kill <19.2 — pass |
| Centered within-sample RankMe | — | **123.1** | high centered rank: shared-offset anisotropy, not genuine collapse |
| Write gate | −0.018 | **−0.075** | opened modestly |
| Largest read gate (layer 5) | +0.015 | **+0.073** | opened modestly; other final read gates +0.032, −0.001, +0.037 |

Raw RankMe declined through training, but remained well above both collapse and success floors. The centered final rank stayed near C, so the decline primarily reflects a growing shared direction rather than loss of within-sample dimensionality.

## Evaluation

### Training-time recurrent-memory ablation at positions ≥1024

| Metric | Step 180 | Step 1440 | E10 target |
|---|---:|---:|---:|
| Real recurrent CE | 1.192 | **1.171** | lower is better |
| Static − recurrent CE | −0.00059 | **−0.00032** | ≥0.05 |
| Δshuffle | −0.00037 | **−0.00038** | ≥0.10 |
| Δzero | +0.00028 | **+0.00011** | diagnostic |
| Δone-block | +0.00002 | **−0.00082** | ≥0.02 at 8K in paired final protocol |

Every beyond-position ablation remained below **0.001 nats** in magnitude throughout the run. The gates moved away from zero, but changing, removing, shuffling, or truncating the recurrent state had no measurable CE cost. The model improved language-model CE without learning to use the content of its recurrent concept memory.

### Missing decisive evidence

- No matched 100M-token `CONCEPT_NUM=0` control exists, so the PRIMARY control-minus-concept recovery fraction is unresolved.
- The frozen paired 2K/8K `run_e10_comparison.py` evaluation requires that matched control and was not run.
- Generic STS-B/SICK/GLUE evaluation is not part of E10's mechanism gate.

## Interpretation

**Stable optimization, healthy concept-set dimensionality, but a null recurrent-memory mechanism at 100M tokens.** This is not collapse: final raw RankMe clears the success floor and centered RankMe remains 123/128. It is a usage failure. Despite nonzero read/write gates, real recurrence is indistinguishable from static, shuffled, zeroed, and previous-block-only states at the decisive beyond-local positions.

The result is therefore stronger than “under-observed” but weaker than a full E10 falsification. It fails the within-arm recurrence and Δshuffle targets at this pilot budget, while the experiment's primary paired criterion cannot be judged without the control arm. Extending this concept arm directly to 2B tokens is not justified by the observed trajectory.

## Decision

**Verdict: INCONCLUSIVE / NEGATIVE PILOT SIGNAL.** Preserve the checkpoints and run metadata; do not extend the concept arm to 2B. If E10 attribution is to be completed, run the matched 100M-token no-concept control and apply the paired 2K/8K comparison. Regardless of the control outcome, the recurrent-attribution co-primary is currently unmet and should be treated as the central finding.

*Related: `master_experiment_log.md`, [E10 spec](../../experiments_specs/done_failed/E10_gemma_backbone_concept_memory.md), `agenda.md`.*
