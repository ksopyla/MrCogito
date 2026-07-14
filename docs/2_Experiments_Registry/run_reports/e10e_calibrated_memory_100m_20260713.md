# E10e — Calibrated concept memory at the original 100M budget — `backbone_concept_gemma_3_1b_pt_K512_concept_20260712_215506`

**Date:** 2026-07-13
**Machine:** Odra (3× RTX 3090, 24 GB each)
**Run ID:** `backbone_concept_gemma_3_1b_pt_K512_concept_20260712_215506`
**WandB (training):** [Link](https://wandb.ai/ksopyla/MrCogito/runs/backbone_concept_gemma_3_1b_pt_K512_concept_20260712_215506)
**Raw shell log:** `Cache/logs/shell_perceiver_denoise_20260712_215437.log`
**Best checkpoint:** `Cache/Training/backbone_concept_gemma_3_1b_pt_K512_concept_20260712_215506/checkpoint-1440`
**Git commit:** `6863e08`
**Git tag:** —
**Related TODO:** E10e calibrated-memory endurance diagnostic — [spec](../../experiments_specs/done_failed/E10e_calibrated_memory_100m.md)

---

## Goal

Test whether E10d's small, positive short-horizon recurrence signal compounds when the fully
calibrated memory path is trained for E10's original 100M-token / 1,449-step budget.

## Configuration

| Item | Value |
|---|---|
| Family / backbone | `backbone_concept`; frozen `google/gemma-3-1b-pt` + LoRA r=16 |
| Memory configuration | C=128; global-KV reads; recurrent BiXT write every K=512; read RMSNorm; read/write gate init 0.01 |
| Optimizer | AdamW; LoRA LR 1e-4; concept-memory LR 3e-4; weight decay 0; warmup 50 |
| Data / objective | Gemma-tokenized `smollm3_inspired_2k_e05`; seq 2048; causal next-token CE |
| Budget | 100M target tokens; 1,449 optimizer steps; effective batch 72; seed 42 |
| Compute | **19.74 GPU-h / 6.28 kWh / 0.214B max-token upper bound** (`compute/audit_state=finished`; flag `loss_fraction:unknown`) |

## Training Outcome

E10e completed all 1,449 steps in 6 h 35 m without NaN, Inf, OOM, NCCL failure, or eval-loss
divergence. Eval CE fell from 1.8374 at step 144 to **1.7972** at checkpoint 1440.

The run passed the Tier-0 checkpoint scan: 567 floating tensors, zero NaN, zero Inf, and zero
all-zero tensors. Its Tier-1 held-out pretokenized analysis also found all slots active.

## Concept Health

At the best checkpoint, held-out Tier-1 geometry was healthy: within-sample RankMe **93.62/128**
(centered **112.56**, minimum 62.33), slot-mean effective rank 13.93, and active-slot fraction
1.000. Training-time final telemetry likewise recorded RankMe **99.91** and centered RankMe
**117.78**.

This is a material improvement over initial E10's final training-time RankMe 77.11 (centered
123.14), but it is not evidence of persistent memory use by itself.

## Evaluation

### E10e's recurrence criteria

The authoritative training-time ablations at positions ≥1024 remained small:

| Checkpoint | Static − real CE | Shuffle − real CE | Interpretation |
|---|---:|---:|---|
| Step 720 (~50M) | +0.000334 | +0.001860 | Both below E10e's 0.002-nat midpoint kill threshold |
| Step 1440 (~100M) | +0.000962 | +0.001613 | Both below E10e's 0.01-nat final success threshold |

The run continued through its midpoint kill condition because it was explicitly launched as an
overnight endurance observation. The additional 50M tokens did not convert the sub-threshold
signal into a persistent-memory effect.

Held-out Tier-1 reconstruction ablations were similarly null within uncertainty: Δstatic
**+0.00026 ± 0.00055**, Δshuffle **+0.00031 ± 0.00044**, Δzero **+0.00078 ± 0.00078**, and
Δone-block **+0.000087 ± 0.00015**.

No matched `CONCEPT_NUM=0` control exists, so the E10 primary recovery-fraction criterion and
paired 2K/8K mechanism evaluation remain unresolved. Generic STS-B/SICK/PAWS/GLUE were not run:
they are not part of E10's fixed-memory mechanism gate.

### Same-budget comparison: initial E10 vs E10e

| Final training-time metric | Initial E10 | E10e | E10e − E10 |
|---|---:|---:|---:|
| Eval CE | 1.8150 | **1.7972** | **−0.0179** |
| Static − real, ≥1024 | −0.000324 | **+0.000962** | +0.001287 |
| Shuffle − real, ≥1024 | −0.000378 | **+0.001613** | +0.001990 |
| One-block − real, ≥1024 | −0.000825 | +0.000432 | +0.001257 |
| Within-sample RankMe | 77.11 | **99.91** | +22.80 |
| Centered RankMe | **123.14** | 117.78 | −5.36 |

Both runs use the same 100M target, 1,449 steps, data manifest, sequence length, C=128, K=512,
backbone, LoRA LR, seed, and effective batch. E10e is not a single-variable comparison: it
combines read RMSNorm, 0.01 gate initialization, and 3× concept-memory LR, and it uses later
code plus denser evaluation cadence.

## Interpretation

**Mixed optimization/geometry improvement, negative mechanism result.** The calibration stack
improved CE, raw concept geometry, and shifted the beyond-local ablations from slightly negative
to slightly positive. Its largest final effect, Δshuffle +0.001613 nats, is still about six times
below E10e's own 0.01 success target and far below E10's original 0.10 attribution target.

The large gates and retained RankMe therefore do not establish memory utility. The direct
information-carrying signal remains effectively null after two blocks. More exposure was not the
missing ingredient for this read/write interface.

## Decision

**Verdict: KILLED / NEGATIVE MECHANISM RESULT.** Do not extend this E10 global→concept
configuration further on plain CE. Preserve the checkpoints and evidence. Any next E10-family
attempt should change the memory interface or introduce a forced distant-information signal,
rather than further calibration or budget scaling.

*Related: `master_experiment_log.md`, [E10e spec](../../experiments_specs/done_failed/E10e_calibrated_memory_100m.md), `agenda.md`.*
