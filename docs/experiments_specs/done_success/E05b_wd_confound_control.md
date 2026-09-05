# E05b — Weight-decay confound control (Adam @ wd=0.1)

- **Status:** **done 2026-07-11** (trained 2026-07-09→10, evaluated 2026-07-11). Run id `concept_ar_prefix_H768L6C128D4_20260709_214837` · WandB [Link](https://wandb.ai/ksopyla/MrCogito/runs/concept_ar_prefix_H768L6C128D4_20260709_214837) (group `E05b_concept_ar_prefix_H768L6C128D4`) · shell log `Cache/logs/shell_perceiver_denoise_20260709_214249.log`.
- **Serves:** resolves the open `wd` confound from the E05 Muon collapse diagnosis ([run report](../../2_Experiments_Registry/run_reports/e05_muon_long_2ep_collapsed_20260709.md) §"Mechanism deep-dive") — a cheap, decisive, single-variable test of *whether weight decay is the proximate driver of the concept rank collapse*. Mechanistic follow-up, not the headline track (E10 remains the focus).
- **Implementation plan:** none needed — config only.
- **Owner / dates:** Krzysztof Sopyła · opened 2026-07-09

> One experiment = one hypothesis = one changed variable. The single research variable here is **weight decay** (0.0 → 0.1). The improved checkpointing/eval cadence (below) is *measurement infrastructure*, not a research variable — it does not change the model.

## Hypothesis
If decoupled weight decay is the proximate cause of the `bixt.rv_lat` rank-1 collapse (mechanism: wd selectively shrinks the bypass-redundant singular directions where `∇L≈0`, `muon.py:101`), then **Adam @ wd=0.1 will collapse toward the Muon arm's regime** (within-sample RankMe ≪ attempt-3's 37.67, toward Muon's 3–10); if wd is innocent, RankMe will stay ≈37 like attempt-3. Jing et al. (arXiv:2110.09348) names implicit/weight-decay-like regularization as a cause of dimensional collapse — this run isolates exactly that variable.

## Builds-on
- **Foundation:** the shared E05 launcher `scripts/launch_e05.sh` → `scripts/train_perceiver_denoise_multigpu.sh`; `WEIGHT_DECAY` is already wired (`train_perceiver_denoise_multigpu.sh:117,309`). No new code.
- **Init / checkpoint:** random init, seed 42 (identical to attempt-3 and the Muon arm).
- **Baseline to beat (the control reference):** `concept_ar_prefix_H768L6C128D4_20260629_093840` (**E05 Adam attempt-3**, wd=0.0, 0.5 ep) — within-sample RankMe **37.67** (old Tier-1 protocol), Δzero_beyond **6.99**, STS-B 0.452, eval_loss 3.83. Secondary reference: the collapsed Muon 0.5-ep arm `...20260702_031956` (wd=0.1) RankMe **10.57**.

## The single change
`WEIGHT_DECAY=0.1` (was 0.0). Everything else held fixed = attempt-3: `OPTIMIZER=adam`, LR 5e-5, `MAX_GRAD_NORM=0.5`, warmup 2000, cosine, effective batch 72, `NUM_EPOCHS=0.5`.

## Success criteria (set BEFORE running)
This is a **control** — success is a *decisive read*, not a directional pass:
- **Decisive isolation:** within-sample RankMe ends clearly in one regime — either **< 15** (collapses ⇒ wd is the driver, confirming the mechanism + explaining the Muon arm) **or** **> 30** (stays healthy ⇒ Muon's full-rank dynamics, not wd, drove the collapse). A result in 15–30 is inconclusive ⇒ re-run / inspect.
- Secondary: eval_loss completes ≤ attempt-3's 3.83 ± 0.2 (wd should not hurt optimization materially).

## Kill criteria (set BEFORE running)
- Divergence: grad_norm > 1e4, loss non-finite, or eval_loss rising monotonically over 3 consecutive evals (the known E05 divergence signature).
- If by 0.25 ep the run is healthy and clearly not collapsing, let it finish (the negative result is the point).

## Plan
- **Data:** `smollm3_inspired_2k_e05` (pretokenized, same manifest as attempt-3 / Muon arm).
- **Compute:** Odra (3× RTX 3090); ~**68 GPU-h** (≈ attempt-3's 68.2 GPU-h / 18.2 kWh / 10.2 B tokens).
- **Steps / epochs:** 0.5 ep / 69,142 steps.
- **Improved checkpointing (the `save_total_limit=5` lesson — pre-crossover ckpts must survive this time):** `SAVE_TOTAL_LIMIT=40` (keep all ~17 ckpts ≈ 11 GB), `EVAL_STEPS=2000`, `SAVE_STEPS=2000` (finer `concept_geometry/effective_rank` resolution so the collapse trajectory is visible live; doubles eval count but each eval is ~140 s).
- **Launch:**
  ```bash
  OPTIMIZER=adam WEIGHT_DECAY=0.1 \
  SAVE_TOTAL_LIMIT=40 EVAL_STEPS=2000 SAVE_STEPS=2000 \
  SKIP_PRETOKENIZE=1 bash scripts/launch_e05.sh
  ```
- **New foundation code:** none — config only.

## Reading the result
- **RankMe < 15** ⇒ wd is the proximate collapse driver. The fix is then *wd management* (e.g., exempt the `bixt`/concept-injection params from wd, or MuonClip for Muon stability without wd) + the non-bypassable objective ([E05c](../ahead/E05c_anticollapse_extension.md)).
- **RankMe > 30** ⇒ Muon's full-rank whitened updates (not wd) drove the collapse; wd is vindicated as merely "what Muon needs to be stable."
- **Either way:** also confirms whether the wd-vs-Muon confound (open since 2026-07-04) is wd or optimizer — long overdue, and cheap.

## Result
- Run id: `concept_ar_prefix_H768L6C128D4_20260709_214837`
- WandB: [link](https://wandb.ai/ksopyla/MrCogito/runs/concept_ar_prefix_H768L6C128D4_20260709_214837)
- Run report: `docs/2_Experiments_Registry/run_reports/e05b_wd_confound_control_20260711.md`
- Verdict: **DECISIVE — wd is innocent.** The control's success criterion was a clean read (RankMe < 15 ⇒ wd driver; > 30 ⇒ not). Result: within-sample RankMe **30.88** (centered 32.17), Δshuffle_beyond **0.50** (clears Stage-2), 0 NaN, active-slot 1.000 — healthy, in attempt-3's league, and **3–6× the collapsed Muon arms at identical wd=0.1**. ⇒ **weight decay is not the collapse driver; Muon's full-rank dynamics are.** Resolves the E05 wd confound (open since 2026-07-04).
