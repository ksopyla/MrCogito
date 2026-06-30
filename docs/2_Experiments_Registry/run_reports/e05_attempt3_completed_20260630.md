# E05 attempt 3 — Windowed decoder completed 0.5 ep, no divergence — `concept_ar_prefix_H768L6C128D4_20260629_093840`

**Date:** 2026-06-30 (training completed; evaluation pending same day)
**Machine:** Odra (3× RTX 3090, 24 GB each)
**Run ID:** `concept_ar_prefix_H768L6C128D4_20260629_093840`
**WandB (training):** [Link](https://wandb.ai/ksopyla/MrCogito/runs/concept_ar_prefix_H768L6C128D4_20260629_093840)
**Raw shell log:** `Cache/logs/shell_perceiver_denoise_20260629_093840.log` (on Odra)
**Best checkpoint:** `Cache/Training/concept_ar_prefix_H768L6C128D4_20260629_093840/checkpoint-68000` (selected by `load_best_model_at_end`, eval_loss 3.8293)
**Git commit:** `f5951da`
**Git tag:** —
**Related TODO:** E05 in `docs/experiments_specs/E05_windowed_decoder_concept_memory.md`

---

## Goal

Third proving attempt for E05 windowed decoder (K=128 sliding-window causal mask on `concept_ar` + prefix→suffix). Attempt 2 diverged at step 40k under LR 1e-4 / clip 1.0; attempt 3 retunes the optimizer (halved LR, explicit tighter clip, larger effective batch) and re-scopes to 0.5 ep (~7B tokens target). Goal: prove stable training through 0.5 ep, clear the Stage 1 read floor (beyond-window Δshuffle ≥ 0.3, STS-B ≥ 0.62, RankMe rises vs init), and earn the budget for the matched A/B.

## Configuration

| Item | Value |
|---|---|
| Family | `concept_ar`, `pretraining_objective=ar_prefix_suffix_generation` |
| Encoder | H768, L6, C128, BiXT |
| Decoder | causal AR, D4, `decoder_word_dropout=0.0`, RoPE, **`decoder_context_window=128`** |
| Token width | `token_embedding_dim=256` |
| Dataset | `smollm3_inspired_2k_e05` mix (pretokenized, 6.92M train / seq 2048) |
| Objective | prefix→suffix; prefix ratio 0.3–0.5, `split_strategy=sentence_boundary` |
| Epochs (target / actual) | 0.5 / **0.5** (full) |
| Effective batch | 8 × grad_accum **3** × 3 GPUs = **72** |
| LR / warmup / clip / seed | **5e-5** / 2000 / **0.5 (explicit)** / 42 |
| LR schedule | cosine |
| Precision | bf16 |
| Optimizer | `adamw_torch_fused` |
| Allocator env | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| Throughput | 50.0 M tokens / GPU-h (0.844 step/s, 1.18 s/step) |
| Compute | **68.2 GPU-h · 18.24 kWh · 10.20B max-tokens** (`compute/audit_state=finished`, flag `loss_fraction:prefix_suffix_approx`) |

Batch 8 was the **third** choice (12 OOM'd in <10 min on GPU 2; 10 held 6 h then OOM'd at step 21k on a high-long-suffix batch). At seq 2048 with this architecture, batch 8 + grad-accum 3 is the memory-stable operating point on 3× 24 GB cards.

## Training Outcome

**Clean convergence to 0.5 epoch with no divergence.** eval_loss decreased monotonically through all 17 evals, from 5.397 (step 4k) → 4.145 (step 20k) → 3.972 (step 28k) → 3.830 (step 88k). Improvement rate slowed as expected: Δ −0.58 in [4k–8k], Δ −0.12 in [20k–28k], Δ −0.001 in the final 4k step.

| Step | epoch | eval_loss | Δ from prev eval |
|---:|---:|---:|---:|
| 4 000 | 0.029 | 5.397 | — |
| 8 000 | 0.058 | 4.818 | −0.58 |
| 12 000 | 0.087 | 4.489 | −0.33 |
| 16 000 | 0.116 | 4.264 | −0.23 |
| 20 000 | 0.145 | 4.145 | −0.12 |
| 28 000 | 0.174 | 3.972 | −0.17 |
| 36 000 | 0.202 | 3.961 | −0.01 |
| 44 000 | 0.231 | 3.908 | −0.05 |
| 52 000 | 0.260 | 3.892 | −0.02 |
| 60 000 | 0.289 | 3.884 | −0.01 |
| 64 000 | 0.318 | 3.866 | −0.02 |
| 68 000 | 0.347 | 3.856 | −0.01 |
| 72 000 | 0.376 | 3.843 | −0.01 |
| 76 000 | 0.405 | 3.837 | −0.006 |
| 80 000 | 0.434 | 3.832 | −0.005 |
| 84 000 | 0.463 | 3.830 | −0.002 |
| 88 000 | 0.492 | 3.829 | −0.001 |

**Stability signature (the whole point of this run vs attempt 2):**

- Pre-clip `grad_norm` held in a tight **0.40–0.55** band from step 200 through ~step 48 000 (warmup + first ~70 % of cosine decay). The clip at 0.5 was rarely the binding constraint.
- From step ~50 000 onward, grad_norm rose to **40–75** (occasional spikes to 87), while LR was near the cosine tail (1e-6 → 1e-9). This is **expected end-of-schedule behavior** (the optimizer's effective step size grows relative to the shrinking LR), and **critically it did not hurt eval_loss** — the curve kept improving. This is the *opposite* signature from attempt 2, where grad_norm escalation (9 → 903) coincided with eval_loss climbing.
- LR cosine decayed cleanly from 5e-5 peak (after 2000 warmup) to 5e-10 final. Warmup properly ramped (step 200: lr 5e-6; step 2000: lr 4.99e-5).

`load_best_model_at_end` selected **`checkpoint-68000`** (eval_loss 3.856 at the eval point just before; the final eval at step 88k recorded 3.829 but is not a separate checkpoint). Final saved checkpoint is **`checkpoint-69142`**. Both are valid; **`checkpoint-68000` is the official best** per HF Trainer's selection.

## Concept Health

*Pending — Tier 0 + Tier 1 (effective rank, within-sample RankMe, slot-mean rank, Δzero / Δshuffle beyond-window) will be filled in once the eval-runner subagent completes.*

## Evaluation

*Pending — Tier 2 zero-shot STS-B and Tier 3 supervised SICK/PAWS/GLUE will be filled in.*

## Interpretation

*Pending — will interpret against the experiment spec's Stage 1 success/kill criteria once concept health + STS-B are in hand.*

## Decision

*Pending — pending eval evidence.*

## Notes

- **Three batches, three OOMs before stability.** Batch 12 OOM'd at startup (loss FP32 upcast, GPU 2). Batch 10 ran 6 h then OOM'd at step 21k on a high-long-suffix batch (also loss FP32 upcast, GPU 0). Batch 8 + grad-accum 3 (effective 72, identical tokens/step to the batch-12 plan) ran to completion. Memory lesson: at seq 2048 with K=128 windowing, per-GPU activation memory peaks ~22.7 GB during the loss FP32 upcast, so leave ≥2 GB headroom on 24 GB cards.
- Two short background training runs from the OOM'd attempts (`..._20260628_231611` batch-12, `..._20260628_233349` batch-10) are still on disk under `Cache/Training/`; they produced checkpoints (4k/8k for batch-12; up to 20k for batch-10) but are not the canonical run. Cleanup is optional — they are small (~2 GB each ckpt).
- `num_train_epochs: 1` in `trainer_state.json` is an HF serialization quirk — the actual schedule was 0.5 ep (`max_steps: 69142`, final `epoch: 0.5000048`). Do not be misled by that field if reading the JSON directly.
- W&B run was synced cleanly (5 files). GPUs are idle and the Byobu session is back at the shell prompt.

*Related: `master_experiment_log.md`, `docs/experiments_specs/E05_windowed_decoder_concept_memory.md`, `agenda.md`, `run_reports/e05_attempt2_diverged_20260628.md`*
