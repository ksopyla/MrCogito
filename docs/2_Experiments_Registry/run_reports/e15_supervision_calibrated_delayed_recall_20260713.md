# E15 supervision-calibrated delayed recall — `backbone_concept_gemma_3_1b_pt_K512_concept_20260713_191759`

**Date:** 2026-07-13  
**Machine:** Odra (3× RTX 3090)  
**Run ID:** `backbone_concept_gemma_3_1b_pt_K512_concept_20260713_191759`  
**WandB:** https://wandb.ai/ksopyla/MrCogito/runs/backbone_concept_gemma_3_1b_pt_K512_concept_20260713_191759  
**Raw shell log:** `Cache/logs/shell_perceiver_denoise_20260713_191735.log`  
**Decision checkpoint:** `Cache/Training/backbone_concept_gemma_3_1b_pt_K512_concept_20260713_191759/checkpoint-2000`  
**Git commit:** `a304c0fd632b12bb262a279590e7ca256a51b2b5`  
**Git tag:** `odra-smoke-verified-20260712-7-ga304`  
**Related experiment:** `E15`

---

## Goal
Test whether E14's chance-level local-carry control was only an insufficient-label-budget issue:
with the same E10e checkpoint, data, architecture, and sparse answer-only objective, can 12,000
total supervised answers establish the block-2 copy/query task before judging block-4 memory?

## Configuration

| Item | Value |
|---|---|
| Family | Frozen Gemma-3-1B + LoRA r16; E10e C128/K512 `global_kv` recurrent memory |
| Init | Resumed E14 `checkpoint-164`, including optimizer/scheduler/RNG state |
| Data / objective | Immutable E14 manifest: 4,608 2,048-token rows; one 64-way answer label per row |
| Budget | 2,000 global steps; 24.576M input tokens; 12,000 total supervised answers |
| Effective batch | 6 (2/GPU × 3 GPUs × accumulation 1) |
| Compute audit | 2.690 GPU-h; 0.818 kWh; 24.576M tokens; `finished` with `loss_fraction:unknown` flag |

## Training Outcome

Training completed cleanly at global step 2,000 (2.604 epochs, 53.8 min recorded training
runtime). W&B logged final `train/loss=4.172` and final eval loss 4.1866; the raw Trainer log
identifies checkpoint 1,804 at 4.18225 as best, so the two sources disagree slightly on the
best eval scalar. No OOM, non-finite loss, or NCCL training failure occurred. The generic health CLI does not support the
`backbone_concept` family, but its fallback weight scan reported no NaN/Inf.

The planned step-1,000 pause was not executed: resumed Trainer state retained E14's 164-step
save cadence and checkpoint retention removed that intermediate checkpoint. This is a protocol
deviation, so the midpoint 20% gate was not observed; it does not invalidate the final frozen
12,000-label gate below.

## Evaluation

The final block-2 explicit-carry evaluation used the frozen 256 held-out counterfactual pairs.
In this view the fact is within the explicit K=512 token carry; success therefore does not require
recurrent concept memory.

| Mode | Answer CE | Top-1 accuracy |
|---|---:|---:|
| Real recurrent memory | 4.18369 | **0.98%** |
| Static initial memory | 4.18198 | — |
| Zero memory | 4.18171 | — |
| Conflicting twin donor | 4.18139 | — |

Paired CE margins relative to real were static **−0.00171**, zero **−0.00198**, and donor
**−0.00230** nats. Chance accuracy is 1.56% for the balanced 64-value vocabulary.

The preregistered final positive-control requirement was block-2 real-memory accuracy ≥80%.
It instead remained below chance at 0.98%. Per E15's frozen criteria, block-4 causal-memory
evaluation was not run and cannot be used to judge the E10e memory interface.

## Interpretation

Increasing exposure from E14's 984 to 12,000 answer labels did not establish even the
explicit-carry version of this sparse synthetic task. The result therefore rejects the narrower
claim that the E14 failure was explained by its 2M input-token/984-label stop alone. It remains
an **objective-protocol failure**, not evidence that recurrent writes or reads are the limiting
E10e component; E11 versus E12 is still not selected.

## Decision

Kill the current one-answer-per-2,048-token delayed-recall protocol. Do not spend another
continuation budget or launch block-4 attribution under this protocol. Any future forced-use
diagnostic must first demonstrate local-copy task competence with a materially different,
independently specified supervision design.

*Related: `master_experiment_log.md`, `docs/experiments_specs/done_failed/E15_supervision_calibrated_delayed_recall.md`, `agenda.md`*
