# E17b mid write-init (`WRITE_GATE_INIT=0.1`) 1B — generation + mechanism vs E17 / E16b / E16

**Date:** 2026-08-13
**Machine:** Polonez (4× RTX 3090; eval on GPU 0)
**Run ID:** `backbone_concept_gemma_3_1b_pt_K512_concept_20260810_135711`
**(Aborted smoke):** `backbone_concept_gemma_3_1b_pt_K512_concept_20260810_120432` (killed ~120 steps @ underfilled bs=3)
**WandB (training):** [1B run](https://wandb.ai/ksopyla/[REDACTED]/runs/backbone_concept_gemma_3_1b_pt_K512_concept_20260810_135711)
**Raw log:** `Cache/logs/shell_perceiver_denoise_20260810_135647.log` · eval `Cache/logs/eval_E17b_20260813_122251.log`
**Best checkpoint:** `Cache/Training/…20260810_135711/checkpoint-17780`
**Last checkpoint:** `…/checkpoint-17785` (= final)
**Git commit:** `dcab83d`
**Git tag:** —
**Related TODO:** fair per-layer + init 0.3 A/B (still open after mid-init failed to stick)

**Artifacts:**
- Matched assessment (vs base Gemma): `Cache/Evaluation_reports/…135711_ckpt17780_generation_assessment.json`
- Last assessment (no base): `…_ckpt17785_generation_assessment.json`
- Tier-1.5 quality: `…_ckpt17780_generation_quality.json`
- Concept analysis best/last: `…_best_concept_analysis.json` · `…_last_concept_analysis.json`
- Compute audit: `Cache/Evaluation_reports/compute_audit/20260813_122253_*`

---

## Goal

Close E17b: does raising only `WRITE_GATE_INIT` 0.01→0.1 on E17's per-layer banks open writes by 1B and clear the free-run bar (`real` greedy d1@256 ≥0.5 / REP-3 ≤0.25 with `real≥zero`), without collapsing free-run back to E16b?

## Configuration

| Item | Value |
|---|---|
| Family | `backbone_concept` · `concept_io_mode=per_layer_banks` |
| Backbone | frozen Gemma-3-1B-pt + LoRA r=16 |
| Concepts | C=128 · K=512 · 4 private banks at global layers 5/11/17/23 |
| Gate init | `READ_GATE_INIT=0.01` · **`WRITE_GATE_INIT=0.1`** (only change vs E17) |
| Dataset | `e16b_long_4k_v1` · seq 4096 · causal LM |
| Optimizer | Muon · LR 0.01 |
| Budget | 1B non-padding tokens · **eff. batch 32** (bs=8 × 4 GPUs × accum=1) · max_steps **17785** |
| Compute | **280.3 GPU-h** · **75.9 kWh** · max_tokens **2.33B** (`compute/audit_state=finished`; flag `loss_fraction:unknown`) |
| Throughput | train_runtime **252234 s** (~70.1 h wall) · **2.256** samples/s · train_loss **2.349** |

## Training Outcome

Stable finish on Polonez. Best = last-eval = `checkpoint-17780` (`eval_loss` **2.264**). Writes briefly crossed the open-gate floor near ~100M, then **closed again** through 1B:

| step (~tok) | write_0..3 | max \|tanh\| | RankMe | Δshuf/static/one_beyond |
|---|---|---|---|---|
| 1778 (~100M) | 0.095 / **0.140** / 0.116 / 0.055 | **0.140** | 50.1 | 0.0005 / 0.0004 / 0.0007 |
| 7112 | 0.049 / 0.037 / 0.081 / 0.029 | 0.081 | 69.1 | 0.0007 / 0.0007 / 0.0004 |
| 12446 | 0.034 / 0.033 / 0.051 / 0.036 | 0.051 | 58.4 | 0.0021 / 0.0017 / 0.0007 |
| **17780 (1B)** | 0.046 / 0.042 / 0.049 / 0.047 | **0.049** | **51.1** | **0.0039 / 0.0029 / 0.0012** |

Read gates opened strongly (0.22 / 0.91 / 0.88 / 0.98 at 1B). Health check: no NaN/Inf (exit 1 only from large-norm weight warnings, same as E17).

## Concept Health

Offline Tier-1 (`checkpoint-17780`, pretokenized holdout, seq 2048):

| Metric | Best | Last | Gate |
|---|---|---|---|
| within-sample RankMe | **67.8** (centered 92.4) | 68.0 | ≥38.4 **PASS** |
| Δshuffle ≥1024 | **0.0055** | 0.0090 | ≥0.01 **FAIL** |
| Δstatic ≥1024 | **0.0033** | 0.0057 | ≥0.01 **FAIL** |
| Δone-block ≥1024 | 0.0005 | 0.0005 | (diagnostic) |

- **Mechanism (writes @1B):** FAIL — all four \|tanh\| ≈0.04–0.05 ≪ 0.1; mid-init was not sticky.
- **Effective `a`:** FAIL — beyond-local Δ stays ~E17 (~0.004–0.009), nowhere near E16b (**2.47**).
- **Geometry:** PASS — RankMe healthy (below E17's ~98 / E16b's 101, above the 38.4 floor).

## Evaluation — generation (Tier 1.5, matched protocol)

Short-prompt continuation, greedy, mean over 6 prompts (matched `run_e16b_generation_assessment.py`). Prefer this table over the quality-runner mix (chat+continuation).

| Condition @256 | distinct-1 | REP-3 |
|---|---|---|
| **E17b mid-init `real`** | **0.196** | **0.601** |
| E17b mid-init `zero` | 0.141 | 0.709 |
| Base Gemma greedy | 0.163 | 0.706 |
| **E17 low-init `real`** | **0.208** | **0.593** |
| **E16b `real`** | **0.04** | **0.94** |
| Shared init-0.3 `real` | 0.056 | 0.904 |
| E16 (50M/2K) | — | mechanism null; no long free-run claim |

Length profile (E17b `real` greedy): @32 **0.69/0.04** → @64 **0.49/0.27** → @128 **0.35/0.38** → @256 **0.20/0.60** → @512 **0.10/0.79**. Same falling-diversity signature as E17; far from absolute bar (need ≥0.5/≤0.25).

Context sweep (128 new tokens): longer prompts do **not** recover free-run (d1 0.13→0.18 from 128→2048); unlike E17's helpful long-prompt lift to 0.43@2048, E17b stays weak.

Tier-1.5 `run_generation_quality.py` (continuation|real @256): d1 **0.179** / REP-3 **0.620**. `real≈shuffle≈static` ≫ `zero` on diversity — free-run is reading a near-static bank, consistent with dead writes.

**Snippet (`real` greedy):** *“the development of new technologies. The development of new technologies is a complex process…”* — English prose with local repetition loops (not E16b digit attractors).

Last ckpt (`17785`) matches best within noise (`real`@256 **0.205 / 0.594**).

## Cross-run comparison

| Run | Topology | init_w | Writes @1B | Δshuf_beyond | RankMe | free-run real@256 |
|---|---|---|---|---|---|---|
| **E16** (50M/2K) | shared | 0.01 | — | **0.0005** | 62 | n/a (short-ctx fail) |
| **E16b** | shared | 0.01 | ~0.05 (dead) | **2.47** | **101** | **0.04 / 0.94** |
| shared init-0.3 | shared | 0.3 | mixed open | **1.69** | ~50–60 | **0.06 / 0.90** |
| **E17** | per-layer | 0.01 | ≤0.033 | 0.004 | **98** | **0.21 / 0.59** |
| **E17b** | per-layer | **0.1** | ≤0.049 (closed after ~100M open) | 0.004–0.006 | **68** | **0.20 / 0.60** |

Reading: mid write-init on per-layer banks **does not** buy the E16b mechanism axis, and **does not** move free-run past E17. Teacher-forced causal use and free-run remain separate. Init 0.1 briefly satisfies the open-gate floor near 100M then regresses — sticky writes need more than a mid prior under plain CE.

## Interpretation

Against E17b's own success criteria this arm is a **failure**:
- writes not live at 1B,
- Δbeyond stays ≪0.01,
- absolute free-run bar missed (0.20/0.60 vs need 0.5/0.25),
- no-regression vs E17 is a near-tie (0.196 ≈ 0.208; last 0.205).

Relative to E16b free-run the per-layer family still looks healthier (~5× distinct-1, prose not digits, `real ≥ zero`). Relative to E16's short-ctx null mechanism, E17b adds nothing new on Δbeyond — same dead-write regime as E17 under long-ctx CE.

## Decision

- Close E17b as **done_failed** (criteria missed; useful negative on mid-init stickiness).
- Do **not** treat this as topology falsification — writes never stayed open.
- Next justified step: the still-missing fair cell **per-layer + `WRITE_GATE_INIT=0.3`** vs the already-run shared+0.3 control (agenda's open A/B). Soft mid-init alone is not enough.
- E17a (untied writers) stays conditional on a *sticky* open-gate per-layer result.

*Related: `master_experiment_log.md`, `docs/experiments_specs/done_failed/E17b_per_layer_mid_write_init.md`, `agenda.md`, [E17 report](e17_lowinit_1b_generation_20260810.md), [E16b gen](e16b_generation_quality_assessment_20260801.md), [shared init-0.3](e16b_shared_init030_1b_20260810.md)*
