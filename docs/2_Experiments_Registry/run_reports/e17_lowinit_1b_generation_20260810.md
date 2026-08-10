# E17 low-init (`WRITE_GATE_INIT=0.01`) 1B — generation quality vs E16b

**Date:** 2026-08-10
**Machine:** Polonez (4× RTX 3090; eval on GPU 0)
**Run ID:** `backbone_concept_gemma_3_1b_pt_K512_concept_20260807_195730`
**(100M precursor):** `backbone_concept_gemma_3_1b_pt_K512_concept_20260801_211805`
**WandB (training):** [1B run](https://wandb.ai/ksopyla/[REDACTED]/runs/backbone_concept_gemma_3_1b_pt_K512_concept_20260807_195730) · [100M start](https://wandb.ai/ksopyla/[REDACTED]/runs/backbone_concept_gemma_3_1b_pt_K512_concept_20260801_211805)
**Raw log:** `Cache/logs/eval_E17_lowinit_7900_20260810_080313.log` · `Cache/logs/eval_E17_lowinit_7900_rest_20260810_080407.log`
**Best checkpoint:** `Cache/Training/backbone_concept_gemma_3_1b_pt_K512_concept_20260807_195730/checkpoint-7900`
**Last checkpoint:** `…/checkpoint-7905`
**Git commit:** `4833bf2`
**Git tag:** —
**Related TODO:** E17 open-gate fair test (per-layer + init 0.3 vs shared + init 0.3)

**Artifacts:**
- Matched assessment (vs base Gemma): `Cache/Evaluation_reports/…20260807_195730_ckpt7900_generation_assessment.json`
- Tier-1.5 quality runner: `Cache/Evaluation_reports/…20260807_195730_ckpt7900_generation_quality.json`
- Last-ckpt assessment (no base): `…_ckpt7905_generation_assessment.json`
- Compute audit: `Cache/Evaluation_reports/compute_audit/20260810_080334_*`

---

## Goal

Close the registered E17 init-0.01 / 1B arm: did per-layer banks (matched init 0.01 vs E16b) open the write path and recover free-run generation? Primary comparison is the matched E16b Tier-1.5 protocol (`analysis/run_e16b_generation_assessment.py`) against E16b `checkpoint-7900` numbers from 2026-08-01.

## Configuration

| Item | Value |
|---|---|
| Family | `backbone_concept` · `concept_io_mode=per_layer_banks` |
| Backbone | frozen Gemma-3-1B-pt + LoRA r=16 |
| Concepts | C=128 · K=512 · 4 private banks at global layers 5/11/17/23 |
| Gate init | `READ_GATE_INIT=0.01` · `WRITE_GATE_INIT=0.01` (matched to E16b) |
| Dataset | `e16b_long_4k_v1` · seq 4096 · causal LM |
| Optimizer | Muon · LR 0.01 · AdamW side 2e-4 |
| Budget | 1B non-padding tokens · eff. batch 72 · max_steps 7905 |
| Compute (1B run) | **230.7 GPU-h** · **62.8 kWh** · max_tokens **2.33B** (`compute/audit_state=finished`; flag `loss_fraction:unknown`) |
| Compute (100M start) | 52.7 GPU-h · 12.8 kWh · max_tokens 0.37B (`audit_state=flagged`) |

## Training Outcome

Training completed on Polonez (`train_runtime` ≈ 57.7 h wall on 4 GPUs). Best = last-eval = `checkpoint-7900` (`eval_loss` **2.264**). Write gates never opened across the full 1B:

| step | write_0..3 | max \|tanh(α)\| | RankMe | Δshuf/static/zero/one≥beyond |
|---|---|---|---|---|
| 790 (100M) | 0.001 / 0.011 / 0.007 / −0.003 | ≤0.011 | 122.6 | ≈0 / −0.0002 / −0.0002 / 0.0002 |
| 4740 | 0.008 / 0.024 / 0.009 / −0.010 | ≤0.024 | 104.1 | 0.0018 / 0.0017 / 0.0021 / 0.0007 |
| 7900 (1B) | 0.017 / 0.033 / 0.012 / −0.013 | ≤0.033 | **98.1** | **0.0043 / 0.0032 / 0.0041 / 0.0013** |

Read gates opened (0.37–0.84 at 1B). Geometry stayed healthy. Beyond-local causal-use deltas stayed well below the 0.01 gate (E16b cleared at **2.47 / 2.35**). Health check: no NaN/Inf (script exit 1 only from large-norm weight warnings).

## Concept Health

- **Mechanism (writes):** FAIL — all four write gates stay near init; max \|tanh\| **0.033 ≪ 0.1**.
- **Effective `a`:** FAIL — Δshuffle/static_beyond **0.004 / 0.003** (need ≥0.01).
- **Geometry:** PASS guard — within-sample RankMe **98** (need ≥38.4).
- Same cold-start confound as the 100M report: init 0.01 keeps writes inert on this topology through 1B, so the per-layer write structure never fairly engages.

## Evaluation — generation (Tier 1.5, matched protocol)

Short-prompt continuation, greedy, mean over 6 prompts (length cutoffs from one 256-token decode). E16b numbers from [2026-08-01 report](e16b_generation_quality_assessment_20260801.md). Shared init-0.3 @1B from existing Polonez artifact `…20260807_090248_ckpt7905_generation_assessment.json` (context, not this ID).

| Condition @256 | distinct-1 | REP-3 |
|---|---|---|
| **E17 low-init `real`** | **0.208** | **0.593** |
| E17 low-init `zero` | 0.208 | 0.534 |
| Base Gemma greedy | 0.163 | 0.706 |
| Base Gemma sample | 0.493 | 0.029 |
| **E16b `real`** | **0.04** | **0.94** |
| E16b `zero` | 0.15 | 0.73 |
| Shared init-0.3 @1B `real` | 0.056 | 0.904 |
| Shared init-0.3 @1B `zero` | 0.140 | 0.723 |

Length profile (E17 `real` greedy): @32 **0.76/0.01** → @64 **0.55/0.16** → @128 **0.39/0.33** → @256 **0.21/0.59**. Diversity falls with length (repetition signature) but far less catastrophically than E16b.

Context sweep (128 new tokens; longer prompt helps E17, hurts E16b historically):

| Prompt len | E17 d1/r3 | Base d1/r3 | E16b (Aug 1) d1/r3 |
|---|---|---|---|
| 128 | 0.32 / 0.46 | 0.21 / 0.63 | 0.16 / 0.74 |
| 512 | 0.27 / 0.54 | 0.34 / 0.42 | 0.48 / 0.38 |
| 1024 | 0.30 / 0.39 | 0.39 / 0.35 | 0.06 / 0.90 |
| 2048 | **0.43 / 0.30** | **0.50 / 0.20** | 0.07 / 0.91 |

Sampling: E17 `real` @256 **0.46 / 0.08** (near base sample). Chat probe still weak (echo / mixed-script junk), same as base PT.

**Snippet (E17 `real` greedy):** *“The future of renewable energy depends on the ability to store energy. The most promising way to store energy is through batteries…”* — stays in English prose (E16b often exited into `1.1.1.1…` / digit attractors).

Last ckpt (`7905`) matches best within noise (`real`@256 ≈ 0.24 / 0.50).

Tier-1.5 `run_generation_quality.py` on best (continuation|real @256): distinct-1 **0.18** / REP-3 **0.64** — same qualitative story; aggregates mix chat+continuation so prefer the matched assessment table above for E16b comparison.

## Interpretation

Against E17’s own success criteria this arm is a **failure**: writes never opened, beyond-local Δ stayed ~0.004, and free-run `real` greedy missed **d1≥0.5 / REP-3≤0.25** (got 0.21 / 0.59). `real ≥ zero` on distinct-1 holds as a tie, not a concept win.

Versus E16b free-run, the same budget + init with per-layer banks is a clear **relative** improvement: ~5× better distinct-1, no digit-attractor collapse, no `real ≪ zero` inversion, and long prompts no longer destroy free-run. That is not enough to claim the topology fixed the write path — gates were still dead — but it shows the cold-start shared-depth free-run pathology is not inevitable under per-layer banks.

The parallel shared + init-0.3 run at 1B (`…090248`) collapsing back to E16b-like `real`@256 **0.06/0.90** also matters: the 100M “init 0.3 opens gates / d1=0.29” sniff did **not** survive to 1B on shared topology. Opening gates early ≠ lasting free-run health. The fair topology test remains **per-layer + init 0.3 vs shared + init 0.3** with full generation assessment at matched budget.

## Decision

- Close E17 init-0.01 / 1B as **done_failed** on registered criteria (mechanism + absolute generation bar).
- Do **not** treat this as topology falsification — writes never engaged; confound persists.
- Next: open-gate fair A/B (new ID / E17 continuation with `WRITE_GATE_INIT=0.3`), with Tier-1.5 as a first-class gate at 100M **and** 1B (the 100M-only sniff is insufficient).
- E17a (untied writers) stays conditional on a live-gate per-layer result.

*Related: `master_experiment_log.md`, `docs/experiments_specs/done_failed/E17_four_bank_concept_memory.md`, `agenda.md`, [100M init report](e17_falsified_init_is_the_cause_20260802.md), [E16b gen report](e16b_generation_quality_assessment_20260801.md)*
