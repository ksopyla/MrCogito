# E05 attempt 2 — Windowed decoder diverged at step 40k — `concept_ar_prefix_H768L6C128D4_20260627_192407`

**Date:** 2026-06-28 (training killed; fast eval same day)
**Machine:** Odra (3× RTX 3090, 24 GB each)
**Run ID:** `concept_ar_prefix_H768L6C128D4_20260627_192407`
**WandB (training):** [Link](https://wandb.ai/ksopyla/MrCogito/runs/concept_ar_prefix_H768L6C128D4_20260627_192407)
**Raw shell log:** `Cache/logs/shell_perceiver_denoise_20260627_191816.log` (on Odra)
**Best checkpoint:** `Cache/Training/concept_ar_prefix_H768L6C128D4_20260627_192407/checkpoint-40000`
**Git commit:** `e44ad84`
**Git tag:** —
**Related TODO:** E05 in `docs/experiments_specs/done_failed/E05_windowed_decoder_concept_memory.md`

---

## Goal

Second proving attempt for the E05 windowed decoder (K=128 sliding-window causal mask on `concept_ar` + prefix→suffix). The 2026-06-26 first attempt (LR 3e-4 / warmup 500) had diverged at step ~20; this run lowered LR to 1e-4 / warmup 1500 to test whether the architecture is sound under a milder optimizer. Target was 1 epoch over the pretokenized `smollm3_inspired_2k_e05` mix.

## Configuration

| Item | Value |
|---|---|
| Family | `concept_ar`, `pretraining_objective=ar_prefix_suffix_generation` |
| Encoder | H768, L6, C128, BiXT |
| Decoder | causal AR, D4, `decoder_word_dropout=0.0`, RoPE, **`decoder_context_window=128`** |
| Token width | `token_embedding_dim=256` |
| Dataset | `smollm3_inspired_2k_e05` mix (pretokenized, 6.92M train / seq 2048) |
| Objective | prefix→suffix; prefix ratio 0.3–0.5, `split_strategy=sentence_boundary` |
| Epochs (target / actual) | 1.0 / 0.25 (killed at step 52,000) |
| Effective batch | 8 × grad_accum 2 × 3 GPUs = **48** |
| LR / warmup / clip / seed | 1e-4 / 1500 / **1.0 (HF default)** / 42 |
| LR schedule | cosine |
| Precision | bf16 |
| Throughput | 64.1 M tokens / GPU-h (2.65 step/s) |
| Compute | **81.3 GPU-h · 17.78 kWh · 5.21B max-tokens** (`compute/audit_state=finished`, flag `loss_fraction:prefix_suffix_approx`) |

## Training Outcome

Steady improvement through step 40,000 (epoch 0.19, 5.2B tokens seen), then **divergence onset**: eval_loss climbed 3.32 → 3.38 → 3.62 → 4.03 over the next 12k steps, while pre-clip `grad_norm` escalated 9 → 56 → 219 → **903** (top-10 grad norms all > 187). Cosine LR was still ~8.5e-5 at the divergence point — barely decayed from the 1e-4 peak — so the loss landscape got sharp under a still-hot optimizer.

`load_best_model_at_end` selected **checkpoint-40000** (eval_loss **3.317**). Training was killed manually at step 52,000 once the divergence signature was confirmed (3 consecutive eval points rising, grad norms escalating).

The signature is the same as the 2026-06-26 first attempt, just delayed because LR is lower (1e-4 vs 3e-4): rank halving (within-sample RankMe ~11 → ~5.5), eval-loss climb, gradient explosion.

## Concept Health

Fast eval (Tier 0 + Tier 1) on checkpoint-40000, run on Odra. JSON: `Cache/Evaluation_reports/concept_ar_prefix_H768L6C128D4_20260627_192407_best_concept_analysis.json`.

### Tier 0 — health
No NaN/Inf in 229 tensors; global weight range [-2.22, +2.48]; active slot fraction 1.000. (Script returned exit 1 only because `from_pretrained` doesn't yet recognize `concept_ar` model type — fell back to weight inspection, which is clean.)

### Tier 1 — geometry (on checkpoint-40000)

| Metric | Value | Verdict |
|---|---:|---|
| **Within-sample concept RankMe** (PRIMARY) | **59.78 / 128** | ✓ GOOD |
| Slot-mean effective rank (secondary) | 10.2 / 128 | ✗ slot redundancy |
| Active slot fraction | 1.000 | ✓ no dead slots |
| Mean pairwise concept sim | 0.188 | ✓ |
| Max pairwise concept sim | 0.998 | ✗ one near-duplicate pair |
| Participation ratio | 17.2 | △ OK |
| Anisotropy (mean random-pair cosine) | 0.291 | ✓ GOOD |

**Key insight:** the PRIMARY de-collapse metric (within-sample RankMe = 59.8) is genuinely good. The "low" `global_effective_rank` (~10) reported in training is the *secondary slot-redundancy* diagnostic — not concept collapse. Per the eval skill, the two measure different things.

### Tier 1 — AR ablation

| Metric | Value | Gate |
|---|---:|---|
| Δzero (early-pos) **PRIMARY** | **1.50** | ✓ ≥ 0.5 |
| Δshuffle (early-pos) **PRIMARY** | **0.85** | ✓ ≥ 0.5 |
| Δzero (beyond-window, K=128) | 1.45 | ✓ cross-window memory |
| **Δshuffle (beyond-window)** | **0.35** | ✗ target ≥ 0.5 (but ≥ Stage 1 floor 0.3) |

The decoder **does** use concepts. Beyond-window, zeroing concepts hurts a lot (1.45 nats) but **shuffling barely hurts** (0.35 nats) — the model relies on the *magnitude/presence* of concepts for long-range, but not their *content/ordering*. That is "concepts as a confidence signal" rather than "concepts as semantic memory" at this training maturity.

### Tier 1 — generation (expected weak at epoch 0.19)
Teacher-forced token acc 31.6%; free-running token-F1 12%; specificity drop 3.4%. All 4 greedy samples collapse to repetitive loops. Honest picture at 19% of one epoch.

## Interpretation

- **The architecture is sound.** Concepts are used, geometry is healthy (RankMe 59.8), ablation clears the Stage 1 floor (Δshuffle_beyond = 0.35 ≥ 0.3). The model was getting *better* right up until step 40k.
- **Divergence is an optimization failure, not an architectural dead-end.** The cosine schedule kept LR too hot for too long; once the loss landscape sharpened (memorization phase), the HF-default `max_grad_norm=1.0` let bad-direction updates dominate even though the *post-clip* norm was capped.
- **Best checkpoint is usable evidence, not a throwaway.** It clears the Stage 1 read floor on every gate except the beyond-window Δshuffle target (0.35 vs 0.5).
- Fair comparison caveat: this is a different dataset mix and seq-len (2048 vs 512) than E01–E04 — do not cross-compare absolute numbers. The within-experiment trajectory (improving → diverging) is the read.

## Decision

**Retune the optimizer and re-scope to 0.5 epoch**, then relaunch on Odra. Single-variable LR change is the primary lever; tighter gradient clipping is the secondary lever; larger batch smooths gradients and raises per-GPU power toward TDP.

| Knob | Attempt 2 (diverged) | **Attempt 3 (next)** | Why |
|---|---|---|---|
| LR (peak) | 1e-4 | **5e-5** | Halve — cosine was still ~8.5e-5 when instability hit |
| Warmup | 1500 | 2000 | Small bump — more time before peak LR |
| `max_grad_norm` | 1.0 (default) | **0.5** (explicit) | Caps bad-direction updates harder |
| Per-device batch | 8 | **12** | +50% throughput; smoother gradients |
| Effective batch | 48 | **72** | Bigger batch → flatter loss landscape |
| Epochs | 1 | **0.5** | ~7B tokens ≈ 110 GPU-h, starts the rank-rises-with-scale zone |
| Eval steps | 2000 | 4000 | Halves eval-idle GPU stalls |

Speculatively added an explicit **"3 consecutive rising eval_loss points"** early-kill rule to the spec — the 2026-06-28 signature would have tripped it at step ~48k, saving ~12 GPU-h.

## Notes

- The `e05_smollm-win128-launch` branch had a local unstaged change to `scripts/finish_goodwrite_symlink.sh` (unrelated); stashed before switching Odra to `dev` for the eval.
- Polonez had a local unstaged change to `scripts/train_perceiver_denoise_multigpu.sh`; stashed. Both stashes remain on the respective servers.
- The 2026-06-28 retune adds `MAX_GRAD_NORM` env-var wiring to `scripts/train_perceiver_denoise_multigpu.sh` (the launcher previously relied on HF Trainer's default 1.0).

*Related: `master_experiment_log.md`, `docs/experiments_specs/done_failed/E05_windowed_decoder_concept_memory.md`, `agenda.md`*
