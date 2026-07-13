# E14 forced delayed-recall gate — `backbone_concept_gemma_3_1b_pt_K512_concept_20260713_172219`

**Date:** 2026-07-13
**Machine:** Odra (3× RTX 3090)
**Run ID:** `backbone_concept_gemma_3_1b_pt_K512_concept_20260713_172219`
**WandB:** https://wandb.ai/ksopyla/MrCogito/runs/backbone_concept_gemma_3_1b_pt_K512_concept_20260713_172219
**Raw shell log:** `Cache/logs/shell_perceiver_denoise_20260713_172156.log`
**Best checkpoint:** `Cache/Training/backbone_concept_gemma_3_1b_pt_K512_concept_20260713_172219/checkpoint-164`
**Git commit:** `389d6636ab67a7fee7b371a7a9a34604e8cf493f`
**Git tag:** —
**Related experiment:** `E14`

---

## Goal
Test whether E10e's unchanged global→concept recurrent mechanism can learn causal distant
memory when local tokens cannot identify a counterfactual answer.

## Configuration

| Item | Value |
|---|---|
| Family | Frozen Gemma-3-1B + LoRA r16, E10e C128/K512 global→concept recurrence |
| Data | 4,608 deterministic 2,048-token rows; one block-4 answer target per row |
| Objective | Answer-only next-token CE; 64 balanced single-token values |
| Effective batch | 6 (2/GPU × 3 GPUs × accumulation 1) |
| Decision point | Step 164 / 2,015,232 input tokens / 984 supervised answer tokens |
| Compute audit | 0.246 GPU-h, 0.069 kWh, 2.015M max tokens; `flagged` because the interrupted partial run has unknown loss-token fraction and small summary-vs-timeseries differences |

## Training Outcome
Training was stable and was paused immediately after the registered checkpoint. Logged train loss
fell from 13.406 to 4.723 by step 160; checkpoint eval CE was 4.595. No OOM, NCCL, non-finite
loss, or runtime failure occurred.

## Concept Health
- Within-sample RankMe: **91.38/128**; centered RankMe: **125.18/128**.
- Effective read gates: 0.0129–0.0170; write gate: 0.0145.
- The concept set remained geometrically healthy; geometry was not the stopping reason.

## Evaluation

The preregistered block-4 evaluation used 256 held-out counterfactual pairs (512 rows).

| Mode | Answer CE | Top-1 accuracy |
|---|---:|---:|
| Real recurrent memory | 4.6109 | 0.78% |
| Static initial memory | 4.6128 | 0.98% |
| Zero memory | 4.6144 | 0.98% |
| Conflicting twin donor | 4.6133 | 0.78% |

Paired CE margins versus real:
- static: **+0.00182** nats, 95% CI [−0.00079, +0.00444]
- zero: **+0.00350** nats, 95% CI [+0.00067, +0.00620]
- donor: **+0.00237** nats, 95% CI [−0.00012, +0.00487]

All three point estimates are below the frozen 0.01-nat kill threshold. The run therefore stopped
at checkpoint 164 and was not resumed.

Memory-age views reveal an important limitation of that verdict:
- **Block 2 (fact remains in explicit token carry):** real accuracy 1.95%, CE 4.5888.
- **Block 3 (fact beyond explicit carry):** real accuracy 1.37%, CE 4.5989.
- **Block 4:** real accuracy 0.78%, CE 4.6109.
- Chance accuracy with 64 balanced values is 1.56%.

## Interpretation
E14 met its preregistered kill condition, but it did **not** isolate a writer-versus-reader
failure. At the decision point the model had received only 984 supervised answer labels, and
even the block-2 positive-control view—where Gemma can directly attend to the fact through the
explicit token carry—remained at chance. The forced-use task itself was not yet learned.

Therefore this result is a **killed protocol / inconclusive architecture test**, not evidence
that E11 or E12 is already preferred. The input-token budget was a poor proxy for supervision
budget under one-label-per-2,048-token masking.

## Decision
Keep E14 stopped as required. Before changing the memory interface, freeze a follow-up that
first establishes local-carry task competence and budgets training by supervised answers or
uses denser delayed-recall queries. Only after that positive control passes can memory-age
failure distinguish writer retention (favoring E11/input-gated writes) from read integration
(favoring E12).

*Related: `master_experiment_log.md`, `docs/experiments_specs/E14_forced_delayed_recall_memory.md`, `agenda.md`*
