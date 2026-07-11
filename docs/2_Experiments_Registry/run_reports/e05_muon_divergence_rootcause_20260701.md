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

**Update (2026-07-02): the retry SUCCEEDED.** The fixed Muon run — `concept_ar_prefix_H768L6C128D4_20260702_031956` (LR 0.01, wd 0.1, adamw_lr 2e-4, sustained-LR-calibrated) — **completed 0.5 ep with eval_loss 2.606** (vs Adam's 3.83 — ~5× faster convergence + 1.22 nats lower), **stable end-to-end** (grad_norm ~0.5–1.6 through the whole run, no divergence; the sustained-LR calibration correctly predicted it). The fixes (wd=0.1 + adamw_lr=2e-4) tamed the Moonlight spectral-growth mechanism, exactly as diagnosed.

**Open:** concept geometry + STS-B/RankMe/Δshuffle semantics on the Muon checkpoint are **pending the eval suite** — eval_loss is an optimization signal, not a concept-semantics signal (Adam had eval_loss 3.83 but STS-B only 0.45). The A/B's research verdict waits for `experiment-evaluate`.

(Original status, 2026-07-01: Muon arm diverged at both LRs and was killed; mitigation knobs implemented + tested; retry pending.)

## Evaluation (2026-07-04) — head-to-head vs Adam (attempt 3)

> **⚠️ 2026-07-07 — Tier-1 rows below (RankMe, slot rank, anisotropy, recon-contract
> Δzero/Δshuffle incl. beyond-window) use the OLD data protocol** (streaming-train first-N →
> train-contaminated; seq 512 on a seq-2048-trained windowed model; unseeded). The Adam-vs-Muon
> *comparison* stays fair (same flawed protocol on both arms), but absolutes are suspect and the
> seq-512 truncation dilutes the beyond-window gate. Recompute planned with the pretokenized eval
> split at seq 2048. STS-B / SICK / PAWS / GLUE rows and W&B suffix-CE ablations are unaffected.

Checkpoint `concept_ar_prefix_H768L6C128D4_20260702_031956/checkpoint-69142` (best=last), `model_type concept_ar`, evaluated on Odra via the experiment-evaluate tiered suite (Tier 0 health → Tier 1 geometry+AR-ablation → Tier 2 zero-shot STS-B + floors → Tier 2.5 frozen mean-vs-attention probe → Tier 3 SICK/PAWS/GLUE). Health: no NaN/Inf (the `check_model_health` rc=1 is a benign `concept_ar` loader-recognition gap, not a defect). GLUE QQP/MNLI still running (downstream footnotes).

| metric | Adam (att 3) | **Muon** | Δ | direction |
|---|---|---|---|---|
| eval_loss (suffix CE) | 3.83 | **2.606** | −1.22 | Muon's headline optimization win |
| **STS-B zero-shot** (Pearson / Spearman) | 0.452 / 0.472 | **0.518 / 0.610** | **+0.066 / +0.138** | ✅ Muon now clears both floors |
| trivial floors (token-embed / teacher-mean) | 0.486 / 0.460 | (same) | — | Adam sat *below* both; Muon is *above* |
| SICK-R Pearson | 0.183 | **0.302** | **+0.119** | ✅ |
| SICK-E acc | 0.634 | **0.733** | **+0.099** | ✅ |
| PAWS acc / F1 | 0.550 / 0.253 | 0.567 / 0.202 | +0.017 / −0.051 | mixed |
| GLUE MRPC acc / F1 | 0.669 / 0.778 | **0.725 / 0.830** | +0.056 / +0.052 | ✅ |
| GLUE STSB Pearson | 0.354 | **0.532** | **+0.178** | ✅ |
| GLUE QQP / MNLI-m | 0.734 / 0.498 | *running* | — | — |
| cross-sample manifold RankMe | 113.9 | **218.4** | **+104.5** | ✅ more diverse pooled embeddings |
| mean→attention probe Δ (SICK-R, frozen) | +0.336 | **+0.052** | −0.284 | ❌ info NOT distributed across slots |
| **within-sample RankMe** (PRIMARY de-collapse) | 37.67 | **10.57** | **−27.1** | ❌ **much MORE collapsed** |
| slot-mean effective rank | 4.76 | 3.38 | −1.4 | ❌ (diagnostic) |
| anisotropy | 0.682 | 0.771 | +0.089 | ❌ narrower cone |
| **Δshuffle_beyond (K=128)** (E05 long-range gate) | 0.39 | **0.209** | **−0.181** | ❌ fails Stage-1 floor ≥0.3 |
| **Δzero_beyond** | 6.99 | **0.414** | **−6.58** | ❌ decoder barely depends on beyond-window concepts |
| (within-window Δzero / Δshuffle) | — | 1.098 / 0.894 | — | within-window concepts ARE used |
| greedy generations | repetition loops | repetition loops | — | degenerate either way |

**Generation samples (Muon, greedy, concept-conditioned):** *"I have no idea how much I have done, but I have no idea…"* / *"The children were not alone in the play, and the children were not alone in the play…"* / *"The virus is transmitted through the body through the body's immune system…"* — fluent-local, semantically-empty repetition loops.

Artifacts (Odra): concept-analysis JSON `Cache/Evaluation_reports/concept_ar_prefix_H768L6C128D4_20260702_031956_concept_analysis.json`; GLUE CSVs `glue-{mrpc,stsb,...}-checkpoint-69142-74M-20260704_*`; STS-B W&B `nbu3p0nk`; eval log `Cache/logs/eval_e05_muon_20260704_120122.log`.

## Tentative initial conclusions — ⚠️ NOT DECISIVE (pending discussion + literature)

*Flagged tentative — single run per arm, one confound open, and the authoritative prefix→suffix ablation not yet read. Do not treat as a verdict.*

1. **Lower loss ≠ better concepts (now shown, not just suspected).** Muon's 1.22-nat-lower eval_loss came with **harder concept collapse** (within-sample RankMe 10.6 vs 37.7) and **worse long-range concept-dependence** (Δzero_beyond 0.41 vs 6.99). The downstream-semantic gains (STS-B, SICK, GLUE) look driven by **better decoder within-window fluency (the bypass)**, not richer concept content.
2. **Optimizer pressure may amplify the bypass.** Faster/harder optimization (Muon ~5× faster) on a *bypass-able* windowed-AR objective tentatively made the concept bottleneck *worse*, not better — consistent with the project's decoder-bypass thesis.
3. **Tentative agenda read:** the fix for concept collapse is an *objective* change (bypass-free: E06 latent prediction, E04 parallel decoder, decoder-weakening), **not** a better optimizer (Adam→Muon barely moves, even regresses, the concept gates). Reinforces E06 next.

## Open questions / confounds for the deeper discussion (bring literature)
- **wd confound:** Muon ran wd=0.1, Adam wd=0.0. Could the concept collapse be driven by **wd=0.1** (weight decay shrinking the concept representations) rather than the optimizer? Needs Adam@wd=0.1 to isolate. Literature: does decoupled weight decay shrink representation rank / encourage collapse?
- **Authoritative ablation not read:** the Δshuffle/Δzero above use the *reconstruction* contract (`run_concept_analysis.py`); for a prefix→suffix run the authoritative suffix-CE `concept_ablation/*` is in the training W&B — not yet pulled. Reconstruction-contract numbers may understate prefix→suffix concept usage.
- **Single seed / single run** per arm — no error bars; RankMe 10.6 vs 37.7 is a large gap but un-replicated.
- **Is the RankMe regression a measurement sensitivity** (batch size, num_batches=20) or real? Adam's 37.7 used the same method, so the comparison is fair, but worth confirming.
- **Literature to bring in:** optimizer choice vs representation collapse / posterior collapse in VAEs & bottlenecks; does Muon's full-rank update specifically encourage low-rank *representations* (opposite of its weight updates)? decoder-bypass / informative-bottleneck theory.

## Compute audit + E02-long-matched 2-ep run (2026-07-04)

Authoritative post-hoc compute audit (`analysis/run_compute_audit.py`, dry-run):

| run | epochs | GPU-h | energy (kWh) | max-tokens | tok/GPU-h | GPU-h/Btok |
|---|---|---|---|---|---|---|
| **E02-long** (`...20260614_101305`, seq 512, 4 GPU) | 5 | **290.7** | 61.4 | **24.5 B** | 84.3 M | 11.86 |
| E05 Muon 0.5ep (`...20260702_031956`, seq 2048, 3 GPU) | 0.5 | **75.6** | 21.4 | 10.2 B | **134.9 M** | 7.41 |
| E05 Adam 0.5ep (`...20260629_093840`, seq 2048, 3 GPU) | 0.5 | 68.2 | 18.2 | 10.2 B | — | — |

**E05 is ~1.6× cheaper per token than E02-long** (134.9 vs 84.3 M tok/GPU-h): the windowed K=128 decoder replaces E02's full-context O(N²) decoder attention, and the 2K mix is pre-tokenized. This **decouples the two matching axes**:
- **Match compute-time (GPU-h) → ~1.9 ep → epoch=2** = 302 GPU-h ≈ E02-long's 290.7 (**+4%**), but **40.8 B tokens (1.66× E02-long)**.
- Match tokens → ~1.2 ep = 24.5 B tokens, but only 181 GPU-h.

**Decision: epoch=2** — matches E02-long *compute-time* (the stated goal); the token overshoot is benign (more exposure at matched compute supports the "is it under-trained?" hypothesis, not less).

**E05 Muon 2-ep run LAUNCHED 2026-07-04 22:50 CEST (Odra, fresh):** `OPTIMIZER=muon NUM_EPOCHS=2 bash scripts/launch_e05.sh`, byobu `E05-muon-2ep`, shell log `Cache/logs/shell_perceiver_denoise_20260704_225057.log`. Same stabilized recipe (LR 0.01, wd 0.1, `adamw_lr` 2e-4, eff batch 72, 9,956,348 interleaved mix). **Total optimization steps = 276,566** (~4.5 days wall-clock ≈ ~300 GPU-h ≈ E02-long). **Fresh** run, not resume — HF restores the cosine `scheduler.pt` anchored to the original 0.5-ep endpoint, so extending epochs on resume leaves LR≈0 on the added portion (gotcha); a fresh run gives a clean warmup+cosine and Muon-stability. First logged row: loss 12.09 / grad_norm 2.01 / lr 0.000995 (warmup); all 3 GPUs 97–100%. ⚠️ **Not a matched A/B** — the Adam arm is 0.5ep; this 2-ep Muon tests the "does more compute de-collapse it?" question *against E02-long*, not against Adam.
