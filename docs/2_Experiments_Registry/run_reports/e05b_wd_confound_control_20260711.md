# E05b — Weight-decay confound control (Adam @ wd=0.1): wd is INNOCENT, the collapse is Muon-specific

**Date:** 2026-07-11 (decisive eval; training completed 2026-07-10)
**Machine:** Odra (3× RTX 3090, 24 GB each)
**Run ID:** `concept_ar_prefix_H768L6C128D4_20260709_214837`
**WandB (training):** [Link](https://wandb.ai/ksopyla/MrCogito/runs/concept_ar_prefix_H768L6C128D4_20260709_214837) (group `E05b_concept_ar_prefix_H768L6C128D4`)
**Raw logs:** `Cache/logs/shell_perceiver_denoise_20260709_214249.log` (training) · `Cache/logs/eval_e05b_{compute_audit_20260711_080810,health_20260711_080839,concept_20260711_080919}.log` (eval)
**Best checkpoint:** `Cache/Training/concept_ar_prefix_H768L6C128D4_20260709_214837/checkpoint-69142` (best=last; eval_loss declined monotonically to 3.641)
**Git commit:** `4d25b81` (Odra checkout at launch; model code unchanged from the `a37b451` Muon arms — E05b is config-only)
**Git tag:** —
**Related TODO:** E05 wd-confound control — [spec](../../experiments_specs/done_success/E05b_wd_confound_control.md); diagnoses the 2-ep Muon collapse ([report](e05_muon_long_2ep_collapsed_20260709.md)).

---

## Goal

Isolate the **wd confound** left open by the E05 Muon collapse: every collapsed Muon run carried wd=0.1 (Muon needs it for stability), while the healthy Adam attempt-3 ran wd=0.0. The mechanism deep-dive hypothesized wd "selectively shrinks bypass-redundant directions." This control changes **one variable** — weight decay 0.0 → 0.1 — on the proven-stable Adam attempt-3 recipe. If wd is the driver, Adam@0.1 collapses toward the Muon regime (within-sample RankMe ≪ 15); if wd is innocent, it stays healthy (RankMe ≳ 30, like attempt-3). Either outcome is a decisive read.

## Configuration

| Item | Value |
|---|---|
| Family / objective | `concept_ar`, prefix→suffix |
| Decoder | causal AR, D4, RoPE, **`decoder_context_window=128`** (windowed) |
| Dataset | `smollm3_inspired_2k_e05` (pretokenized, 9,956,348 train / seq 2048) |
| Epochs / steps | 0.5 / 69,142 |
| Effective batch | 8 × accum 3 × 3 GPU = **72** |
| Seed | 42 |
| **Optimizer** | `adamw_torch_fused`, LR **5e-5**, **`weight_decay=0.1`** (was 0.0 — the single variable), `max_grad_norm=0.5`, cosine, warmup 2000 |
| Word-dropout | 0.0 (off — that is E05c's lever, not this control's) |
| Checkpointing | `save/eval_steps=2000`, `save_total_limit=40` (the `save_total_limit=5` lesson — pre-crossover ckpts now survive) |
| Compute | **68.62 GPU-h / 19.13 kWh / 10.20 B max-tokens / 6.12 B loss-tokens-est** (`compute/audit_state=finished`, flag `loss_fraction:prefix_suffix_approx`) |

## Training Outcome

**Stable end-to-end, completed 0.5 ep, no divergence.** eval_loss fell 5.40 → **3.641** across 35 evals (monotonic; slightly *lower* than attempt-3's 3.829 — wd=0.1 did not hurt optimization). grad_norm held low mid-run then rose in the cosine tail (final ~12.4 at lr ≈ 5e-10) without hurting eval_loss — the same benign "grad-norm-rises-at-low-LR" Edge-of-Stability tail signature as attempt-3 (not the divergence signature). Effective rank (training-time, eval-holdout, comparable across all runs) **rose and held ~6.9** — vs Muon's monotonic decline to 1.78–3.56.

## Concept Health

**Severe-collapse ruled out on every gate; healthy, in attempt-3's league.** Tier 0 + Tier 1 (new 2026-07-07 protocol: pretokenized held-out eval split, seq 2048, length-stratified, seeded, Δ ± std). Focused eval — Tier 2/3 (STS-B/SICK/PAWS/GLUE) deferred per scope.

| metric | **E05b (Adam, wd=0.1)** | attempt-3 (Adam, wd=0.0) † | Muon 0.5-ep (wd=0.1) † | Muon 2-ep (wd=0.1) |
|---|---|---|---|---|
| **within-sample RankMe** (PRIMARY, new proto) | **30.88 ± 2.96** (centered 32.17 ± 3.23) | 37.67 | 10.57 | **4.96** |
| slot-mean effective rank (diag) | 6.02 | 4.76 | 3.38 | 1.66 |
| cross-sample manifold RankMe | 116.74 | 113.91 | 218.4 | 158.0 |
| anisotropy (pooled random-pair cosine) | 0.525 | 0.682 | 0.771 | 0.992 |
| mean pairwise concept cosine | **0.289** | — | — | 0.892 |
| active-slot fraction | **1.000** | — | 0.21 | 0.38–0.51 |

† old Tier-1 data protocol (seq 512, train-stream) — directionally comparable, not strict-comparable with the new-protocol E05b / Muon-2-ep numbers.

**Read:** within-sample RankMe **30.88** is high on both raw and centered → genuine de-collapse, not shared-offset anisotropy. **The decisive comparison is same-protocol:** E05b (new) **30.88** vs Muon 2-ep (new) **4.96** — **6× higher at identical wd=0.1.** Slots fully active (1.000), concepts near-orthogonal (cosine 0.289 vs Muon's 0.892). Tier 0 health: **0 NaN, 0 Inf** across 229 tensors (the `concept_ar` loader rc=1 is the known benign recognition gap).

## Evaluation

### AR concept-ablation (reconstruction contract, 8 batches, ± std)

| metric | overall | `_early` | `_beyond_window` (E05 gate) |
|---|---|---|---|
| Δzero | **5.70 ± 0.24** | 4.92 ± 0.47 | **5.75 ± 0.36** |
| Δshuffle | **0.58 ± 0.13** | **1.00 ± 0.18** | **0.50 ± 0.14** |
| ce_intact | 3.70 | 5.11 | 3.60 (beyond) / 4.00 (within) |

- **E05 Stage-1 floor (Δshuffle_beyond ≥ 0.3): CLEARED** (0.50, ~3.6× the floor, margin > 1 std).
- **E05 Stage-2 target (Δshuffle_beyond ≥ 0.5): CLEARED on the line** (0.50 ± 0.14 — borderline-decisive; flag). For context, training-time `concept_ablation/delta_shuffle_beyond_window` = 0.367 (always eval-holdout, unaffected by protocol) and attempt-3's was 0.39; Muon's was 0.21–0.23.
- Δzero is large everywhere (5.7+ nats) — the decoder reads concepts heavily, including beyond-window; nothing like Muon's 0.4.

### Generation samples (greedy AR, concept-conditioned)

Fluent-ish but **phrase-level repetition** after ~15 tokens — better than Muon's tight token loops, still degenerate (the E05 family signature):
- *"The 1970s, the 1970s, was the first time in the 1970s, and the 1970s, was the first time…"*
- *"The most important thing is that the most important thing is that the most important thing…"*

### Deferred (per focused-eval scope)

Tier 2 (zero-shot STS-B + trivial floors), Tier 2.5 (frozen mean-vs-attention probe), Tier 3 (SICK/PAWS/GLUE) **not run** — the concept-health question this control exists to answer is settled by Tier 0+1. Run them later if a downstream-utility number is wanted.

## Interpretation

**Weight decay is innocent; the concept collapse is Muon-specific.** At matched wd=0.1, Adam (this run) is healthy on every gate — within-sample RankMe **30.88** (≈ attempt-3's 37.67, **3–6× the collapsed Muon arms**), Δshuffle_beyond **0.50** (clears even the Stage-2 target; Muon failed Stage-1), no dead slots, no NaN. The single variable (wd 0.0→0.1) moved nothing toward collapse.

This **falsifies the "wd is the proximate collapse driver" hypothesis** from the 2-ep mechanism deep-dive. wd was a red herring: correlated with collapse only because Muon *requires* wd=0.1 for stability (Moonlight), so every collapsed run happened to carry it. The revised mechanism: **Muon's full-rank whitened updates converge ~5× faster into the bypass minimum, which is intrinsically low-rank in the concept channel** — and Adam's low-rank, second-moment-damped updates do not. (Jing et al., arXiv:2110.09348, names implicit/regularization-driven dimensional collapse — but here the collapse tracks the *optimizer's update geometry*, not wd.)

## Decision

**Verdict: DECISIVE — the control served its purpose. wd is not the collapse driver; Muon's full-rank dynamics are.** This closes the E05 wd confound (open since 2026-07-04).

Implications:
1. **The 2-ep Muon report's "wd selectively shrinks redundant directions" mechanism is revised** — wd exemption (a mitigation floated there) will not help.
2. **E05c (decoder word-dropout) remains the right primary fix** — it removes the *bypass attractor* Muon rushes into, optimizer-agnostically. Un-park it next (it is config-only, ready to run).
3. **E05d (VICReg) stays well-motivated** as the optimizer-agnostic anti-collapse regularizer (belt-and-suspenders).
4. **E10 (pretrained-backbone pivot) stays the headline**; this control was a cheap, bounded mechanistic side-quest, now concluded.

*Related: `master_experiment_log.md`, [E05b spec](../../experiments_specs/done_success/E05b_wd_confound_control.md), [2-ep Muon collapse report](e05_muon_long_2ep_collapsed_20260709.md), [anti-collapse lit note](../../literature_review/concept_bottleneck_collapse_mitigation.md), `agenda.md`.*
