# E05d — Anti-collapse loss: VICReg variance+covariance on the concept matrix

- **Status:** **ON HOLD / design-only** — queued behind [E05b](E05b_wd_confound_control.md) (running) and [E05c](E05c_anticollapse_extension.md) (on-hold). Needs a reusable foundation component; implement via `research-implement` only after E05c shows whether killing the bypass alone suffices.
- **Serves:** the *other* diagnosed sub-mechanism — the wd-driven rank-1 collapse of `bixt.rv_lat`. Where E05c removes the *cause* (the bypass), this directly *regularizes the symptom* (rank collapse) with a proven, drop-in loss. The "additional loss component / contrastive loss" option from the user's ask.
- **Implementation plan:** TODO — author via `implementation-plan` before coding (a short plan: where the loss hooks in, tensor shapes, the variance/covariance ops, config knobs, unit test).
- **Owner / dates:** Krzysztof Sopyła · opened 2026-07-09
- **Literature:** [concept_bottleneck_collapse_mitigation.md](../literature_review/concept_bottleneck_collapse_mitigation.md) (Family C: VICReg arXiv:2105.04906; dimensional-collapse-by-implicit-regularization arXiv:2110.09348).

> One experiment = one hypothesis = one changed variable. The single variable is the **VICReg concept loss** (off → on). It is staged AFTER E05c so the non-bypassable objective and the anti-collapse regularizer are tested one at a time.

## Hypothesis
If wd (and Muon's full-rank dynamics) selectively collapse the concept channel's rank (Jing et al. arXiv:2110.09348: implicit regularization is a named cause of dimensional collapse), then a VICReg penalty on the concept batch — **variance hinge** `Σ_j max(0, γ − std(z_:j))` (keep every concept dimension active, forbidding the all-zero/constant solution) **+ covariance** `Σ_{i≠j} C(Z)²_{ij}` (decorrelate, forbidding rank-1 redundancy) — will hold within-sample RankMe up and keep `Δshuffle_beyond` healthy, even if a residual bypass survives. VICReg's ablation (Table 7) shows removing the variance term ⇒ immediate collapse; with it, no collapse.

## Builds-on
- **Foundation (NEW, reusable — not a fork):** a config-selectable `ConceptVICRegLoss` applied to the encoder's concept output `[B, C, H]` each step. New knobs: `CONCEPT_VICREG_WEIGHT` (default 0 = off), `CONCEPT_VICREG_GAMMA` (default 1.0), `CONCEPT_VICREG_COV_WEIGHT` (default 1.0). Plugged into the existing multi-loss aggregation in `training/train_perceiver_denoise.py` (alongside the AR CE) — the same place `concept_losses`/`contrastive_weight` already live. Reusable by every future concept-bottleneck experiment (E06/E08/E10), not E05-specific.
- **Init / checkpoint:** random init, seed 42.
- **Baseline to beat:** E05c (if run) — same recipe with VICReg off. Fallback baseline: the Muon 0.5-ep arm (`...20260702_031956`) RankMe 10.57.

## The single change
`CONCEPT_VICREG_WEIGHT` 0 → small positive (start ~1e-3 relative to AR CE; tune). Held fixed = the E05c recipe (Muon 0.5-ep + `DECODER_WORD_DROPOUT=0.3`) — so this isolates the VICReg contribution on top of the non-bypassable objective.

## Success criteria (set BEFORE running)
- **Within-sample RankMe ≥ 30** (clearly above E05c / Muon-arm levels) AND does not decline over training — the regularizer holds rank.
- Per-concept-dimension std stays ≥ γ/2 across the C·H dims (no dead dimensions — the variance hinge working).
- No STS-B / Δshuffle regression vs E05c (the decorrelation must not come at the cost of semantics).

## Kill criteria (set BEFORE running)
- RankMe still < 15 at 0.25 ep despite VICReg ⇒ the regularizer is insufficient alone (the bypass must be removed first — confirm E05c).
- Training destabilizes (grad_norm > 1e4) ⇒ VICReg weight too high; lower.
- STS-B drops > 0.1 vs E05c ⇒ the hard decorrelation is hurting semantics; lower cov weight or switch to softer Barlow-Twins.

## Plan
- **Data / Compute / Steps:** same as E05c (`smollm3_inspired_2k_e05`, Odra 3×3090, 0.5 ep, ~75 GPU-h).
- **Launch (after implementation):**
  ```bash
  OPTIMIZER=muon DECODER_WORD_DROPOUT=0.3 \
  CONCEPT_VICREG_WEIGHT=1e-3 CONCEPT_VICREG_GAMMA=1.0 \
  SAVE_TOTAL_LIMIT=40 EVAL_STEPS=2000 SAVE_STEPS=2000 \
  SKIP_PRETOKENIZE=1 bash scripts/launch_e05.sh
  ```
- **New foundation code:** YES — `ConceptVICRegLoss` (reusable, config-selectable) via `research-implement`. ~30 lines: flatten `[B,C,H]→[B·C, H]` (or compute per-slot), batch variance hinge + off-diagonal covariance, add to total loss. Unit test: zero/constant input ⇒ large penalty; identity-covariance input ⇒ ≈0.

## Why staged after E05c
E05c (kill the bypass) tests the *root cause*; E05d (VICReg) tests a *direct regularizer*. If E05c alone fixes the collapse (concepts become necessary → gradient pressure → wd can't shrink them), E05d may be unnecessary. If E05c kills the bypass but RankMe still drifts, E05d is the targeted add. Running them one at a time (per project discipline) isolates which lever does the work.

## Alternatives kept in reserve (Family C, from the lit note)
- **Barlow Twins** (cross-correlation → identity) — softer than VICReg, if cov hinge is too aggressive.
- **W-MSE whitening** (arXiv:2007.06346) — hard full-rank, but matrix-inversion instability.
- **Orthogonal regularization** on the `bixt` weight matrices (NeurIPS 2024, arXiv:2411.00392) — natural complement to Muon (which already whitens *updates*); could be the Muon-specific anti-collapse knob.

## Result
<Filled in AFTER, by experiment-track.>
- Run id: `<run_id>`
- WandB: <link>
- Run report: `docs/2_Experiments_Registry/run_reports/<...>.md`
- Verdict: promising | mixed | regression | killed — <one line>
