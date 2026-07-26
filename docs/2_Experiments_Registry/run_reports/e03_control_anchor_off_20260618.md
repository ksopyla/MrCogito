# E03 Control (Anchor-OFF, matched) — `concept_ar_H768L6C128D4_20260615_211458`

**Date:** 2026-06-18
**Machine:** Odra (3× RTX 3090, 24 GB each)
**Run ID:** `concept_ar_H768L6C128D4_20260615_211458`
**WandB (training):** [Link](https://wandb.ai/ksopyla/MrCogito/runs/concept_ar_H768L6C128D4_20260615_211458)
**Raw shell log:** `Cache/logs/shell_perceiver_denoise_20260615_211439.log`
**Best checkpoint:** `Cache/Training/concept_ar_H768L6C128D4_20260615_211458/checkpoint-10000`
**Git commit:** `0c14061`
**Git tag:** —
**Related TODO:** E03 in `docs/experiments_specs/done_success/E03_concept_anchor_decollapse.md`

---

## Goal

Run the **anchor-OFF arm** of the matched E03 pair: identical config to the E03A anchor-ON
warmup with `ANCHOR_LOSS=false` (= byte-for-byte E01 reconstruction config), same
seed/data/budget/batch. This control answers the one decisive E03 question — does reconstruction
at 0.3 epoch collapse *without* the anchor? It doubles as a fresh E01-recon baseline measured
with the upgraded metrics (manifold RankMe, early-position Δ, gap_clean_vs_wd).

---

## Configuration

| Item | Value |
|---|---|
| Family | `concept_ar` |
| Encoder | H768, L6, C128, BiXT |
| Decoder | causal AR, D4, `decoder_word_dropout=0.2`, RoPE |
| Token width | `token_embedding_dim=256` |
| Anchor | **OFF** (matched control) |
| Dataset | `HuggingFaceFW/fineweb-edu`, `sample-10BT` (9.57M / 100k holdout) |
| Objective | AR denoising reconstruction, `deletion_rate=0.6` |
| Epochs | 0.30 (matched warmup budget) |
| Global steps | 19,942 |
| Effective batch | 24 × grad_accum 2 × 3 GPUs = 144 |
| LR / sched / seed | 3e-4 / cosine / 42 |
| Throughput | 1.021 steps/s (~5.4 h) — 2.15× faster than anchor-ON (no teacher forward) |
| Precision | bf16 |

Matched to E03A on every knob except the anchor block (verified: per-device batch 24, grad-accum 2,
LR 3e-4, warmup 1500, seed 42, 19,942 steps, same git-commit code path).

---

## Training Outcome

Training completed cleanly to 0.30 epochs. `load_best_model_at_end` selected **checkpoint-10000**
(eval_loss **4.7915**, epoch 0.150). Eval CE then **rises monotonically** to 4.964 by epoch 0.27
even as train loss keeps falling — the same overfitting/collapse signature E01 showed over a full
epoch, here visible within 0.3 epoch.

| Epoch | eval_loss | slot eff. rank |
|---|---|---|
| 0.060 | 5.020 | **17.38** (peak) |
| 0.150 | **4.792** (best) | 6.39 |
| 0.271 | 4.964 | 5.08 |

Slot rank peaks early (17.4) then collapses to ~5 — concepts de-rank as the AR decoder learns to
bypass them via local context.

---

## Concept Health

All numbers from `analysis/run_concept_analysis.py` on **checkpoint-10000** (`--num_batches 20
--batch_size 16`). JSON: `Cache/Evaluation_reports/e03_ctrl_best_concept_analysis.json`.

| Metric | E03 control (ck-10000) | E03A anchor-ON (ck-19000) | Gate / note |
|---|---|---|---|
| Global eff. rank (slot SVD) | 5.93 / 128 | 10.34 / 128 | secondary; anchor +4.4 |
| **Manifold RankMe** (cross-sample) | 150.4 | **167.1** | anchor +16.7 ✓ (gate +8) |
| Manifold anisotropy | 0.57 (more collapsed) | 0.49 | lower = better |
| Dims for 95% var (slot) | 11.2 | 32.9 | anchor spreads ~3× wider |
| Mean concept cosine | 0.436 | — | high = slot redundancy |
| Δzero / Δshuffle (recon-contract) | 0.99 / 1.86 | 0.95 / 1.35 | both use concepts |
| Δzero / Δshuffle early-pos | **4.10 / 5.58** | 2.32 / 3.34 | control higher — see below |
| **gap_clean_vs_wd** | **1.677** ⚠ | **0.128** ✓ | 13× worse for control |
| STS-B zero-shot Pearson | **0.485** | **0.556** | anchor +0.071 |

Key observations:
- **Control collapses; anchor holds.** Slot rank 5.93 vs 10.34, RankMe 150 vs 167, anisotropy 0.57
  vs 0.49 — every geometry metric is worse without the anchor, confirming the anchor's de-collapse
  effect relative to the matched control.
- **`gap_clean_vs_wd = 1.677` is the cleanest signal.** The control decoder specializes to the
  word-dropout training distribution and bypasses the (collapsing) concepts via local context; the
  anchor arm's gap is 0.128 (decoder genuinely uses concepts). This 13× gap is the strongest
  evidence the anchor prevents posterior collapse / decoder bypass.
- **Early-Δ paradox:** the control shows *higher* early-position Δ (5.58 vs 3.34). This is **not**
  health — collapsed concepts occupy fewer directions that the decoder is then forced to lean on
  hard at early positions where bypass is impossible. Read alongside the collapsed rank and high
  gap_clean_vs_wd, the high early-Δ is a symptom of collapse, not de-collapse. (This is why the spec
  pairs early-Δ *with* RankMe rather than reading it alone.)

---

## Evaluation

### Tier 2 — Zero-shot STS-B (checkpoint-10000)

WandB: [bench-stsb_zero_shot-checkpoint-10000](https://wandb.ai/ksopyla/MrCogito/runs/l4krzij9)

| Checkpoint | Pearson | Spearman | Gate (≥ 0.62) |
|---|---|---|---|
| E03 control (ck-10000, best) | **0.485** | 0.533 | FAIL |
| E03A anchor-ON (ck-19000) | 0.556 | 0.572 | FAIL |
| E01 best (ck-4000, ref) | 0.556 | 0.575 | FAIL |

The control STS-B (0.485) is **below** E01-best (0.556) and the anchor arm (0.556). Without the
anchor, reconstruction at 0.3 epoch produces weaker semantics even at the lowest-AR-loss checkpoint.

---

## Interpretation

This control completes the matched E03 pair and answers the decisive question: **reconstruction at
0.3 epoch does collapse without the anchor** (slot rank 5.9, RankMe 150, anisotropy 0.57, STS-B
0.485, gap_clean_vs_wd 1.677). The anchor-ON arm beats this control on every relative criterion:
RankMe +16.7 (clears the spec's +8 gate), STS-B +0.071 (clears the +0.03 gate), gap_clean_vs_wd 13×
better, AR CE actually lower (4.45 vs 4.79).

But the anchor does **not** clear the spec's **absolute** gates at this budget: STS-B 0.556 < 0.62,
slot rank +4.4 < the +16 secondary gate, and the early-Δ legs are confounded by the control's
collapse-driven high early-Δ. So the de-collapse hypothesis is **directionally supported** but not
proven at the absolute-quality bar.

**Crucial caveat surfaced by the E02 5-epoch run (evaluated same day):** geometry evolves
strongly between 0.3 and 5 epochs and the *direction* depends on the objective. Prefix→suffix rank
*rises* 5.9 → 16.7 over 5 epochs; reconstruction rank *falls* over training. Both E03 arms ran the
reconstruction objective for only 0.3 epoch, so this pair cannot tell us whether the anchor prevents
the *full-run* reconstruction collapse — only that it helps at 0.3 epoch. The matched full-epoch
pair remains necessary for a definitive verdict.

---

## Decision

**Verdict: mixed / promising.** The matched control confirms reconstruction collapses without the
anchor and the anchor de-collapses relative to control on RankMe, STS-B, and (decisively)
gap_clean_vs_wd — but absolute gates (STS-B ≥ 0.62, slot rank +16) are unmet at 0.3 epoch. Next:
either (a) a matched **full-epoch** E03 pair, or (b) higher-value — combine the anchor with the
**prefix→suffix** objective, which already de-collapses on its own over long training (E02-long),
to test whether the anchor adds on top of the stronger objective rather than rescuing the weaker one.

*Related: `master_experiment_log.md`, `docs/experiments_specs/done_success/E03_concept_anchor_decollapse.md`, `agenda.md`*
