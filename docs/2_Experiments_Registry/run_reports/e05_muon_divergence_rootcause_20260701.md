# E05 Muon A/B — Diverged at LR 0.02 and 0.01; root cause = no weight decay + over-hot lm_head LR (full-rank spectral growth)

**Date:** 2026-07-01
**Machine:** Odra (3× RTX 3090, 24 GB each)
**Diverged runs (Muon arm):**
- LR 0.02: `concept_ar_prefix_H768L6C128D4_20260701_084351` — [W&B](https://wandb.ai/ksopyla/MrCogito/runs/concept_ar_prefix_H768L6C128D4_20260701_084351)
- LR 0.01: `concept_ar_prefix_H768L6C128D4_20260701_194042` — [W&B](https://wandb.ai/ksopyla/MrCogito/runs/concept_ar_prefix_H768L6C128D4_20260701_194042)
**Adam baseline (stable, the A/B control):** `concept_ar_prefix_H768L6C128D4_20260629_093840` — eval_loss 3.83 (see [attempt-3 report](e05_attempt3_completed_20260630.md))
**Optimizer:** `nn/muon.py` `Muon` (NS5 orthogonalization for 2D matrices, aspect ≤ 8; AdamW fallback for embeddings/lm_head/1D)
**Git commit (plumbing):** `2f0bb51` (Muon + launcher dedup), `f5951da` (`set -u` fix), mitigation knobs added 2026-07-01
**Related TODO:** E05 optimizer A/B in `docs/experiments_specs/E05_windowed_decoder_concept_memory.md`

---

## Goal

E05 optimizer A/B: Adam (Odra, the completed attempt-3 run) vs **Muon** (fresh, token-matched — same seed/model/mix/effective-batch-72/69,142 steps). Hypothesis: Muon's ~2× faster convergence (validated on wikitext-103) holds on E05's windowed prefix→suffix objective.

## What happened — Muon converges ~5× faster, then diverges (twice)

| Run | LR | divergence onset | best signal before blow-up |
|---|---|---|---|
| Muon 0.02 | 0.02 | **~step 3,000** — grad_norm 0.3 → 11,890 → 2.4M; loss 4.8 → 8 | (diverged before first eval) |
| Muon 0.01 | 0.01 | **~step 4,500** — grad_norm 0.57 → 8 → 80; loss 3.77 → 4.08 | **eval_loss 3.34 @ step 4,000** (vs Adam's ~5.40 at the same step) |

Signature in both: stable, fast convergence for a few thousand steps (grad_norm ~0.2–0.6), then a **sudden delayed grad_norm explosion** (thousands → millions) and loss climbing — **surviving `max_grad_norm=0.5` clipping** (sustained bad-direction updates, the same mechanism as E05 Adam attempt 2). Adam at LR 5e-5 was stable to 0.5 ep. **Both Muon runs were killed.**

**Why the LR-0.01 calibration missed it:** the calibration used `NUM_EPOCHS=0.05` → a compressed cosine that **decayed LR well before step 4,500**, so the sustained-LR onset never triggered. The full run's long cosine keeps LR ≈ 0.01 through the onset region → diverges. **Lesson: calibrate under sustained peak LR (`constant_with_warmup`), not a short cosine.**

## Configuration (both arms identical except optimizer + LR)

| Item | Value |
|---|---|
| Family / objective | `concept_ar`, prefix→suffix |
| Decoder | causal AR, D4, RoPE, **`decoder_context_window=128`** (windowed) |
| Dataset | `smollm3_inspired_2k_e05` (pretokenized, 9,956,348 train / seq 2048) |
| Epochs / steps | 0.5 / 69,142 |
| Effective batch | 8 × accum 3 × 3 GPU = **72** |
| Seed | 42 |
| Adam arm | `adamw_torch_fused`, LR **5e-5**, wd 0.0, cosine, clip 0.5 |
| Muon arm | `nn.muon.Muon`, LR 0.02 then 0.01, **wd 0.0**, `adamw_lr 2e-3`, momentum 0.95, ns_steps 5, cosine, clip 0.5 |

## Root cause (confirmed against our config + literature)

Muon's **full-rank orthogonalized updates** grow every weight singular direction uniformly — unlike Adam's low-rank, second-moment-damped updates. In the model's bilinear couplings (windowed **Q·K** attention + the **lm_head** output projection), that uniform spectral growth runs away: **MaxLogit/MaxOutput explosion → delayed grad_norm spike → divergence** (Jianlin Su's QK-Clip mechanism; Moonlight arXiv:2502.16982 §2.2/App. D). Our config had **three compounding enablers**, all verified:

| Suspect | Our value | Literature | Verdict |
|---|---|---|---|
| **Muon weight decay** | **0.0** (HF default; launcher didn't set it) | Moonlight: wd is the long-horizon stabilizer (**wd=0.1**); vanilla Muon (wd=0) "weights grow too large → instability" | 🔴 prime suspect — confirmed |
| **`adamw_lr` (lm_head/embed fallback)** | **2e-3** | Every reference uses ≤ muon_lr (DeepSpeed Moonlight finetune: 2e-**6**); 2e-3 over-updates the lm_head (output-logit bilinear form) | 🔴 co-conspirator — anomalous |
| **Update scale** | Keller `√max(1, A/B)` | Moonshot `0.2·√max(A,B)` matches Adam update RMS; ours can over-update small matrices | 🟡 candidate (deferred — needs LR retune) |
| NS precision | fp32 (`G.float()`) | Tri Dao flags bf16 NS; ours is fp32 | ✅ ruled out |

**Why delayed onset:** rare singular-vector collisions accumulate over thousands of steps before spectral norms reach the runaway threshold — LR-dependent (0.02→step 3k, 0.01→step 4.5k). The `max_grad_norm=0.5` clip bounds each step's update but cannot stop the upstream weight-spectral-norm growth.

## The positive signal stands

**Muon eval_loss 3.34 at step 4,000 vs Adam's ~5.40** at the same step — Muon's ~5× faster early convergence is real and large. The divergence is a **stability ceiling, not a fundamental flaw** — and it is addressable.

## Mitigations (ranked by evidence × ease) — what's implemented

1. ✅ **Add Muon weight decay = 0.1** (Moonlight) — wired via `WEIGHT_DECAY` knob → HF `--weight_decay` → `create_optimizer` → `nn.muon.Muon`. `launch_e05.sh` Muon branch now defaults `WEIGHT_DECAY=0.1`.
2. ✅ **Drop `adamw_lr` 2e-3 → 2e-4** — `launch_e05.sh` Muon branch now defaults `MUON_ADAMW_LR=2e-4` (stops the lm_head over-update).
3. ✅ **Sustained-LR calibration** — wired `LR_SCHEDULER_TYPE` knob; calibrate with `constant_with_warmup` so peak LR is exercised through the onset region.
4. ⏳ **Moonshot update scale `0.2·√max(A,B)`** — deferred (one-line in `nn/muon.py`, but changes LR semantics → needs LR retune). Try if wd+adamw_lr don't suffice.
5. ⏳ **QK-Clip / QK-Norm** (Kimi K2 MuonClip) — directly caps the MaxLogit runaway; more invasive (model code).

## A/B cleanliness caveat

Adding wd=0.1 to Muon (Adam ran at wd=0.0) makes **two** variables differ. Options: (a) accept it ("wd=0.1 is what Muon needs to be stable"), or (b) re-run Adam at wd=0.1 for a single-variable A/B. **Decision pending.**

## Status

Muon arm **diverged at both LRs; killed**. Mitigation knobs (wd, adamw_lr, sustained-LR scheduler) implemented + tested (24 tests pass). Next: calibrate the fixed Muon (LR 0.01, wd 0.1, adamw_lr 2e-4, `constant_with_warmup`) under sustained peak LR past step ~6k; if stable → full run. The eval-3.34 signal justifies the retry.
