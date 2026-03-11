# Prefix Diffusion v2 Failure Analysis — `prefix_diffBiXT_T64_H512L6C128D2_20260308_065355`

**Date:** 2026-03-08  
**Machine:** Polonez (4x RTX 3090, 24 GB VRAM each)  
**Run ID:** `prefix_diffBiXT_T64_H512L6C128D2_20260308_065355`  
**WandB:** [Link](https://wandb.ai/ksopyla/MrCogito/runs/prefix_diffBiXT_T64_H512L6C128D2_20260308_065355)  
**Raw log:** `Cache/logs/shell_prefix_diffusion_20260308_065342.log`  
**Best checkpoint:** `Cache/Training/prefix_diffBiXT_T64_H512L6C128D2_20260308_065355/checkpoint-35000`  
**Git commit:** `031e0ad18f8511144d7f3665b33af2be235dcb4f`  
**Git tag:** `arch/prefix-diffusion-20260304-12-g031e`  
**Related TODO:** `TODO 13b` in `docs/1_Strategy_and_Plans/active_todos.md`

---

## Goal

Test whether the hardened prefix-diffusion stack fixes the failure mode of the first SODA-style text run.

Changes relative to the first prefix baseline:
- `BiXT` encoder is required
- `token_embedding_dim=64` blocks the easiest token-width shortcut
- `sentence_boundary` splitting gives cleaner semantic prefix/suffix boundaries
- evaluation routing metadata is saved directly in the checkpoint config

Hypothesis:
- If the first run failed because the architecture was too weak or the split too noisy, then this v2 clean baseline should raise concept rank materially above the old `6.19 / 128` result.

---

## Configuration

| Item | Value |
|---|---|
| Encoder | H512, L6, C128, `BiXT=True` |
| Decoder | Prefix diffusion, D2 |
| Token width | `token_embedding_dim=64` |
| Dataset | `JeanKaddour/minipile` |
| Prefix split | `sentence_boundary`, ratio 0.3-0.5 |
| Prefix / suffix minimums | 8 / 16 content tokens |
| Objective | Prefix -> suffix diffusion, ELBO-weighted CE |
| Concept losses | None |
| Epochs | 20 |
| Effective batch | 512 |
| Optimizer | `adamw_torch_fused`, cosine LR |
| Throughput | `0.751` train steps/s |

---

## Training Outcome

The run completed cleanly to 20 epochs. This is already an engineering improvement over the pre-fix launch, which crashed in DDP because the final BiXT layer had dead token-update parameters.

Key trainer metrics:

| Step | Epoch | Train loss | Eval loss |
|---|---|---|---|
| 5,000 | 2.57 | 15.239 | 7.6016 |
| 20,000 | 10.28 | 14.783 | 7.4425 |
| 25,000 | 12.85 | 14.736 | 7.4321 |
| 30,000 | 15.42 | 14.680 | 7.4280 |
| 35,000 | 18.00 | 14.669 | **7.4248** |
| 38,900 | 20.00 | final train average **14.9326** | no final eval beyond 35k |

Observations:
- Training was stable for the full run.
- Eval loss improved only modestly after the first checkpoint.
- The curve flattened very early; after 20k steps the run mostly plateaued.

---

## Concept Health

Final concept analysis was run on the best checkpoint (`checkpoint-35000`).

| Metric | Value | Interpretation |
|---|---|---|
| Effective rank | **5.74 / 128** | collapsed |
| Global effective rank | **4.94 / 128** | collapsed |
| Mean concept similarity | **0.5758** | concepts strongly correlated |
| Max concept similarity | **0.9994** | near-duplicate concept slots exist |
| Participation ratio (norm.) | **0.0525** | very poor dimension usage |
| Dimensions for 95% variance | **6.8** | almost all energy concentrated in a tiny subspace |
| Top-1 variance ratio | **0.4665** | one direction still dominates heavily |

Decision against project gates:
- Effective rank target from the research criteria is far above this result.
- Rank `< 10 / 128` is a hard collapse signal.
- This checkpoint therefore fails before any downstream benchmark could justify more evaluation spend.

---

## Comparison To Prefix Baseline v1

Comparison to `prefix_diff_H512L6C128D2_20260304_200437`:

| Metric | v1 (no BiXT, token_dim=512) | v2 (BiXT, token_dim=64) | Delta |
|---|---|---|---|
| Best eval loss | 7.42 | **7.425** | essentially unchanged |
| Effective rank | **6.19** | 5.74 | worse |
| Stability | completed | completed | unchanged |
| Evaluation routing | fragile legacy path | hardened metadata-driven path | improved engineering only |

Interpretation:
- The v2 architectural cleanup did not improve the learned concept geometry.
- The bottleneck is not "missing BiXT" or "too much token width" in isolation.
- The failure appears deeper: random-init prefix -> suffix generation still does not create enough semantic pressure to organize 128 useful concepts.

---

## Diagnosis

The run teaches two things:

1. Engineering fixes worked.
   The DDP crash was fixed, the checkpoint metadata is now correct, and the training path is stable/reproducible.

2. Research hypothesis still failed.
   Even with BiXT, thinner token states, and better sentence-aware splits, the model still compresses into roughly 5-6 active concept directions. That is basically the same collapse regime as the earlier prefix and diffusion runs.

Most likely interpretation:
- Prefix generation remains too difficult for a randomly initialized bottleneck to solve semantically.
- The model learns a weak continuation prior that lowers CE below random, but it still does not organize concepts into a rich semantic basis.
- More clean random-init scaling in the same family is unlikely to change this qualitatively.

---

## Decision

**Stop further random-init clean prefix-diffusion variants.**

Specifically:
- Do not run the `token_embedding_dim=32` clean ablation next.
- Do not spend benchmark evaluation budget on this checkpoint.
- If the prefix track continues, jump directly to warm-start / pretrained-backbone initialization.
- In parallel, keep TSDAE as the stronger active candidate for Track A.

---

## Notes

The concept-analysis script wrote the JSON result successfully and then hit a Python shutdown bug (`PyGILState_Release`) during interpreter finalization on Polonez. The metrics above are valid because the report and JSON were already emitted before shutdown.

*Related: `master_experiment_log.md`, `active_todos.md`, `docs/4_Research_Notes/diffusion_diagnosis_20260226.md`*
