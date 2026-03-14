# Prefix Diffusion WikiText-103 Trainability Probe — `prefix_diffBiXT_T64_H512L6C128D2_20260311_194729`

**Date:** 2026-03-14  
**Machine:** Polonez (4x RTX 3090, 24 GB VRAM each)  
**Run ID:** `prefix_diffBiXT_T64_H512L6C128D2_20260311_194729`  
**WandB (training):** [Link](https://wandb.ai/ksopyla/MrCogito/runs/prefix_diffBiXT_T64_H512L6C128D2_20260311_194729)  
**Raw shell log:** `Cache/logs/shell_prefix_diffusion_20260311_192008.log`  
**Best checkpoint:** `Cache/Training/prefix_diffBiXT_T64_H512L6C128D2_20260311_194729/checkpoint-20000`  
**Git commit:** `—`  
**Git tag:** `—`  
**Related TODO:** `TODO 13c`

---

## Goal

Run one easier random-init prefix-diffusion probe on WikiText-103 to test whether a cleaner Wikipedia-derived corpus, a larger observed prefix, and longer training can rescue the collapsed concept geometry seen in the MiniPile prefix baselines.

The hoped-for signal was simple:
- materially lower suffix loss than the MiniPile runs,
- concept rank leaving the `~5-6 / 128` collapse regime,
- and enough semantic lift to justify continuing random-init prefix training.

---

## Configuration

| Item | Value |
|---|---|
| Family | `prefix_diffusion` |
| Encoder | H512, L6, C128, `BiXT=True` |
| Decoder | prefix-to-suffix diffusion, D2 |
| Token width | `token_embedding_dim=64` |
| Dataset | `Salesforce/wikitext`, subset `wikitext-103-v1` |
| Evaluation split | built-in `validation` split |
| Prefix split | `sentence_boundary`, ratio `0.7-0.8` |
| Objective | ELBO-weighted prefix -> suffix diffusion |
| Diffusion schedule | `t in [0.3, 1.0]` |
| Epochs | 40 |
| Effective batch | 512 |
| Throughput | `1.95` train steps/s |

---

## Training Outcome

The run completed cleanly to 40 epochs and selected `checkpoint-20000` as the best checkpoint.

Key metrics:

| Metric | Value |
|---|---|
| Train dataset size | `732,635` |
| Eval dataset size | `1,607` |
| Best eval loss | `7.0410` |
| Best checkpoint | `checkpoint-20000` |
| Final train loss | `13.9927` |
| Final checkpoint | `checkpoint-57240` |
| Train runtime | `29,398.7 s` (`8.17 h`) |

What improved:
- the longer WikiText-103 run was operationally stable from start to finish,
- throughput was much better than the earlier MiniPile prefix runs,
- and the cleaner split setup did produce a non-random zero-shot similarity signal.

What did not improve:
- the suffix loss barely moved after the early checkpoint window,
- the best checkpoint arrived at `20k` steps and later checkpoints drifted worse,
- and the concept bottleneck collapsed even harder than in the MiniPile v2 baseline.

---

## Concept Health

Concept analysis was run on the final exported checkpoint during the partial evaluation pass on 2026-03-14.

| Metric | Value | Interpretation |
|---|---|---|
| Effective rank | **3.91 / 128** | extreme collapse |
| Global effective rank | **3.35 / 128** | extreme collapse |
| Participation ratio (norm.) | **0.0453** | very poor dimension use |
| Dimensions for 95% variance | **4.0** | almost all energy in a tiny subspace |
| Mean concept similarity | **0.5940** | concepts strongly correlated |
| Max concept similarity | **0.9993** | near-duplicate slots remain |
| Uniformity loss | **0.3506** | clustered concepts |
| Top-1 variance ratio | **0.6132** | one direction dominates |

This is materially worse than the MiniPile v2 prefix baseline (`5.74 / 128`), so the easier data regime did not just fail to help, it pushed the run deeper into collapse.

---

## Evaluation

### Zero-shot STS-B

Artifacts:
- `Cache/Evaluation_reports/bench-stsb_zero_shot-prefix_diffBiXT_T64_H512L6C128D2_20260311_194729-34M-20260314_2001-results.csv`
- `Cache/logs/shell_stsb_zero_shot_prefix_diffBiXT_T64_H512L6C128D2_20260311_194729_20260314_200100.log`

| Metric | Value |
|---|---|
| Pearson | **0.5740** |
| Spearman | **0.5800** |

This is better than near-random and better than the older MiniPile prefix v1 STS-B fine-tune score, but it still misses the project zero-shot gate and stays below the denoising baseline (`0.607 / 0.622`).

### Partial GLUE observation

Artifact observed:
- `Cache/logs/shell_glue_eval_prefix_diffBiXT_T64_H512L6C128D2_20260311_194729_20260314_200100.log`

Only interim evidence was captured before the evaluation pass was stopped/documented:

| Observation | Value |
|---|---|
| Task reached | `mrpc` |
| Validation accuracy | `0.3775` |
| Validation F1 | `0.2865` |
| Status | partial only, no completed benchmark artifact captured |

### Skipped benchmarks

- `SICK` and `PAWS` were not started during the captured remote-evaluation window.
- The remote evaluator kept to the one-workload-at-a-time rule on Polonez while GLUE was still active/incomplete.

---

## Interpretation

This probe answers the key question behind TODO 13c:

No, the cleaner WikiText-103 setup does not rescue random-init prefix diffusion.

The run does learn something:
- suffix CE stays far below a random baseline,
- frozen STS-B is non-random,
- and the full training path is stable.

But the decisive signal is still the concept geometry. Rank `3.91 / 128` is worse than both MiniPile prefix baselines and far below the Track A gate. That means the easier corpus did not solve the underlying bottleneck-organization problem. The model is still compressing into a tiny semantic subspace instead of learning a broad concept basis.

---

## Decision

**Close the random-init WikiText-103 rescue probe as failed.**

Immediate next action:
- do not run more random-init prefix-diffusion clean variants on this line,
- if the prefix track continues, move directly to warm-start / pretrained-backbone initialization,
- keep `perceiver_denoise + contrastive` as the stronger active Track A follow-up.

This result is still useful:
- it removes the "MiniPile is just too hard" excuse,
- confirms that cleaner text alone is insufficient,
- and narrows the remaining plausible fixes to initialization and stronger semantic supervision.

---

## Notes

The remote evaluator also launched the maintained GLUE wrapper and confirmed it started normally, but only the early MRPC validation signal was captured before the report was returned. Documentation here therefore treats the run as **partially evaluated**, not fully benchmarked.

*Related: `master_experiment_log.md`, `active_todos.md`*
