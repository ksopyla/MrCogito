# E10b — Normalized concept-read pilot

**Date:** 2026-07-12
**Machine:** Odra (3× RTX 3090, 24 GB each)
**Run ID:** `backbone_concept_gemma_3_1b_pt_K512_concept_20260712_133258`
**WandB (training):** [Link](https://wandb.ai/ksopyla/MrCogito/runs/backbone_concept_gemma_3_1b_pt_K512_concept_20260712_133258) (group `E10b_backbone_concept_gemma_3_1b_pt_K512`)
**Raw shell log:** `Cache/logs/shell_perceiver_denoise_20260712_132448.log`
**Best checkpoint:** `Cache/Training/backbone_concept_gemma_3_1b_pt_K512_concept_20260712_133258/checkpoint-360`
**Git commit:** `a7b2f62`
**Git tag:** —
**Related TODO:** E10b read-side RMSNorm — [spec](../../experiments_specs/done_failed/E10b_normalized_concept_read.md)

---

## Goal

Test whether E10's null recurrent-memory signal was primarily a scale-interface problem: normalize concept values before the four global-layer read K/V projections while holding the backbone, zero gate initialization, data, objective, and optimizer fixed.

## Configuration

| Item | Value |
|---|---|
| Family / backbone | `backbone_concept`; frozen `google/gemma-3-1b-pt` + LoRA r=16 |
| Single change | `read_concept_norm=true`; read/write gate initialization remained 0 |
| Concept mechanism | C=128, `global_kv` reads, recurrent BiXT write every K=512 tokens |
| Data / objective | Gemma-tokenized `smollm3_inspired_2k_e05`; seq 2048; causal next-token LM |
| Planned budget | 50M non-padding target tokens; 725 optimizer steps; 25M kill gate at step 360 |
| Actual stopping point | Step 400 after the step-360 decision checkpoint |
| Effective batch | 8/GPU × 3 GPUs × accumulation 3 = **72** |
| Optimization | AdamW, LR 1e-4 cosine, warmup 50, weight decay 0, clip 0.5, bf16 |
| Runtime | 6,939 s (1 h 56 m) |
| Compute | **5.78 GPU-h / 1.84 kWh / 0.059B max-token upper bound** (`compute/audit_state=finished`, flag `loss_fraction:unknown`; the decision checkpoint corresponds to approximately 25M non-padding target tokens) |

## Training Outcome

The run remained healthy through the pre-registered decision point. Eval CE improved monotonically from **1.8705** at step 72 to **1.8152** at step 360, with no NaN, OOM, NCCL failure, or loss divergence. It was stopped after step 400 once the step-360 evaluation met the kill criterion; checkpoint 360 is complete and preserved.

## Concept Health

At step 360, within-sample RankMe was **112.17/128** and centered RankMe was **125.15/128**, comfortably above the 38.4 success guard and 19.2 collapse kill threshold. Effective rank was 13.75, but the near-full centered RankMe shows shared-offset anisotropy rather than geometric collapse. The zero-initialized gates learned away from zero: reads were `[+0.0260, +0.0195, −0.0095, +0.0191]` and the write gate was `−0.0217`.

## Evaluation

At the decisive positions ≥1024 and step 360:

- static − recurrent CE: **+0.000371 nats**
- shuffled − recurrent CE: **+0.000179 nats**
- real recurrent CE: **1.180235**

Both signals were below the pre-registered **0.002-nat kill threshold** and roughly two orders of magnitude below the 0.01 success targets. Against E10 at the same step, static−real moved from −0.000143 to +0.000371, while Δshuffle moved from +0.000307 to +0.000179; this is not a coherent or meaningful recurrence improvement.

Local CE at positions <512 was **2.03264**, only +0.00101 nats versus matched E10, so normalization caused no local-language regression. The first carry region (positions 512–1024) did react to content perturbation (`Δshuffle=+0.02255`, `Δzero=+0.02096`), but that signal disappeared by positions ≥1024. Read normalization therefore did not establish persistent multi-block memory use.

## Interpretation

**Healthy geometry and optimization, but the normalized read interface did not repair recurrent usage.** E10b rules out low concept-value RMS as the primary explanation for E10's null beyond-local signal. The gates opened, local CE was preserved, and concepts remained diverse; nevertheless, changing concept content had effectively zero effect after two blocks.

The carry-only response suggests the path can influence the immediately following block. The missing beyond-local effect points instead toward the serial read/write bootstrap or rapid state-content loss, which is exactly the next E10c hypothesis.

## Decision

**Verdict: KILLED AT THE PRE-REGISTERED 25M GATE.** Do not spend the remaining E10b budget. Advance to E10c with the same normalized read and small 0.01 read/write gate initialization, holding all other factors fixed.

*Related: `master_experiment_log.md`, [E10b spec](../../experiments_specs/done_failed/E10b_normalized_concept_read.md), `agenda.md`.*
