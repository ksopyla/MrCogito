# Perceiver Denoise Reconstruction Baseline — `perceiver_denoise_H512L6C128D3_20260308_220324`

**Date:** 2026-03-11  
**Machine:** Odra (3x RTX 3090, 24 GB VRAM each)  
**Run ID:** `perceiver_denoise_H512L6C128D3_20260308_220324`  
**WandB (training):** [Link](https://wandb.ai/ksopyla/MrCogito/runs/perceiver_denoise_H512L6C128D3_20260308_220324)  
**Raw shell log:** `Cache/logs/shell_perceiver_denoise_20260308_220305.log`  
**Best checkpoint:** `Cache/Training/perceiver_denoise_H512L6C128D3_20260308_220324/checkpoint-202000`  
**Git commit:** `74e9f23`  
**Git tag:** `arch/perceiver-denoise-reset-6-g74e9f`  
**Related TODO:** `TODO 10` in `docs/1_Strategy_and_Plans/active_todos.md`

---

## Goal

Run the first canonical `perceiver_denoise` baseline after the perceiver reset and check whether dense denoising reconstruction plus `BiXT` produces materially better concept quality than the archived MLM and diffusion baselines.

Hypothesis:
- removing sparse MLM supervision,
- deleting tokens instead of feeding `[MASK]`,
- contextualizing token states with `BiXT`,
- and decoding with the shared stacked position-only decoder

should raise concept quality above the collapsed `~5 / 128` regime of the old L6 perceiver MLM baseline.

---

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_denoise` |
| Encoder | H512, L6, C128, `BiXT=True` |
| Decoder | stacked position-only decoder, D3 |
| Token width | `token_embedding_dim=512` |
| Dataset | `JeanKaddour/minipile` |
| Objective | dense denoising full reconstruction |
| Corruption | token deletion, `deletion_rate=0.6` |
| Concept losses | None |
| Epochs | 20 |
| Effective batch | 96 |
| Precision | bf16 |
| Flash Attention v2 | active |
| Throughput | `1.89` train steps/s |

---

## Training Outcome

The run completed cleanly to 20 epochs and selected `checkpoint-202000` as the best checkpoint.

Key metrics:

| Metric | Value |
|---|---|
| Final logged train loss | `1.9214` |
| Final run-average train loss | `2.1531` |
| Best eval loss | `1.8693` |
| Final eval loss | `1.8745` |
| Global steps | `208,340` |

What worked:
- training was stable end-to-end,
- logging, checkpoints, WandB, and checkpoint metadata were all recorded correctly,
- the run improved over the old L6 MLM baseline in raw concept rank (`5 -> 10.61`).

What did not work:
- the concept space is still clearly collapsed relative to project gates,
- the run did not yet translate its semantic signal into robust supervised pair-task fine-tuning.

---

## Concept Health

Final concept analysis was run on `checkpoint-202000`.

| Metric | Value | Interpretation |
|---|---|---|
| Effective rank | **10.61 / 128** | still collapsed |
| Normalized effective rank | **0.083** | poor |
| Participation ratio (norm.) | **0.229** | only partly healthy |
| Dimensions for 95% variance | **58.5** | better than MLM-style collapse |
| Mean pairwise similarity | **0.2006** | acceptable on average |
| Max pairwise similarity | **0.9991** | near-duplicate concepts remain |
| Mean dimension std | **0.6181** | usable but not strong |
| Top-1 dominance ratio | **0.094** | much healthier than classic one-direction collapse |

Interpretation:
- this is **not** a dead concept space,
- but it is still far from the target `> 64 / 128`,
- and the near-duplicate concept slots show that semantic specialization remains weak.

---

## Evaluation

### Zero-shot STS-B

WandB: [bench-stsb_zero_shot-checkpoint-202000-76M-20260311_2030](https://wandb.ai/ksopyla/MrCogito/runs/9p3cda65)

| Metric | Value |
|---|---|
| Pearson | **0.6066** |
| Spearman | **0.6225** |

This is the strongest frozen semantic signal seen so far for the maintained perceiver path and already clears the project zero-shot gate.

### Supervised spot-check on GLUE

Artifacts saved on Odra:
- `Cache/Evaluation_reports/glue-mrpc-checkpoint-202000-76M-20260311_2036-*`
- `Cache/Evaluation_reports/glue-stsb-checkpoint-202000-76M-20260311_2044-*`

WandB:
- [MRPC](https://wandb.ai/ksopyla/MrCogito/runs/glue-mrpc-checkpoint-202000-76M-20260311_2031)
- [STS-B](https://wandb.ai/ksopyla/MrCogito/runs/glue-stsb-checkpoint-202000-76M-20260311_2036)

| Task | Result | Reading |
|---|---|---|
| MRPC | **F1 78.68%**, accuracy `65.20%` | below the ViaDecoder baseline |
| STS-B | **Pearson 0.1015**, Spearman `0.1048` | failed badly under fine-tuning |

The broader QQP/MNLI sweep was intentionally stopped after MRPC and STS-B because the signal was already mixed and Odra should stay free for the next training run.

---

## Interpretation

This run teaches two things at once:

1. The denoising perceiver training path is operationally solid.
   Logging, WandB, checkpoint metadata, concept analysis, and evaluation routing all work on the new stack.

2. The research result is mixed, not yet a win.
   Dense denoising reconstruction clearly produces more semantic signal than the old collapsed MLM geometry would suggest, but it still does not give a concept space that is robust enough under supervised pair-task fine-tuning.

Most likely interpretation:
- dense denoising is moving the model in the right direction,
- but pure reconstruction is still not strong enough to fully organize the `128` concepts,
- and the next informative step is to add the contrastive pressure rather than spend more evaluation budget on this checkpoint.

---

## Decision

**Continue the denoising line, but do not call this baseline a Track A success.**

Next action:
- run the same architecture with `--objective_variant reconstruction+contrastive`,
- repeat concept analysis first,
- then repeat zero-shot STS-B and the pair-task spot-check before any full QQP/MNLI sweep.

Current status:
- stronger than diffusion self-reconstruction,
- more promising than the raw rank number alone suggests,
- still below the concept-quality gate.

---

*Related: `master_experiment_log.md`, `active_todos.md`*
