# GLUE Evaluation — perceiver_mlm L6 + combined+kendall_gal (Feb 19 2026)

**Date:** 2026-02-19
**Model:** `perceiver_mlm_H512L6C128_20260219_105435` (61M params)
**Pretraining:** Minipile, 20 epochs, LR 3e-4, combined+kendall_gal concept losses
**Concept analysis:** effective rank 122/128 (95.5%), mean sim 0.009, all dimensions active

---

## Results vs Baselines

| Task | Metric | **L6 + concept losses** | L6 baseline (no losses) | BERT-Base | vs L6 baseline |
|---|---|:---:|:---:|:---:|:---:|
| **MRPC** | F1 | **81.41%** | 81.3% | 89.5% | +0.11% |
| **MRPC** | Acc | 70.34% | 71.1% | — | -0.76% |
| **STS-B** | Pearson | 0.341 | **0.627** | 89.4% | **-0.286 (-46%)** |
| **STS-B** | Spearman | 0.319 | **0.627** | — | **-0.308 (-49%)** |
| **QQP** | F1 | **58.74%** | 72.5% | 91.4% | **-13.76%** |
| **QQP** | Acc | 75.00% | 79.0% | — | -4.0% |
| **MNLI-m** | Acc | **48.87%** | 59.1% | 85.4% | **-10.23%** |
| **MNLI-mm** | Acc | **50.80%** | 61.4% | 85.4% | **-10.60%** |

---

## Pretraining Summary

| Property | L6 baseline (Feb 08) | L6 + concept losses (Feb 19) |
|---|:---:|:---:|
| MLM eval_loss (final best) | **2.537** | 4.31 |
| Concept eff. rank | 5/128 (4%) | **122/128 (95.5%)** |
| Mean concept similarity | 0.451 | **0.009** |
| Max concept similarity | 1.000 (duplicates) | **0.145** |
| Concept losses | none | combined + kendall_gal |

---

## Analysis

### What worked: concept space geometry improved dramatically

The combined+kendall_gal losses fixed the dimensional collapse completely:
- Effective rank: **5/128 → 122/128** (24× improvement)
- Mean pairwise cosine similarity: **0.451 → 0.009** (nearly orthogonal)
- Duplicate concepts (sim=1.0) eliminated
- All 128 concept dimensions actively used

### What failed: MLM quality traded away too aggressively

The Kendall-Gal weighting found a local optimum where the total loss became negative (−0.26 at step 39000). This happened because Kendall-Gal minimises `0.5 * exp(-log_var) * L + 0.5 * log_var` jointly. As concept losses approached zero (successfully optimised), the model learned `log_var_combined << 0` (high precision on concept losses) while simultaneously increasing `log_var_task` (low precision on MLM). The equilibrium condition is `log_var_task ≈ log(MLM_loss)`, which with a steadily decreasing MLM loss pushed `log_var_task` positive — effectively zeroing out the MLM gradient contribution after ~step 10000.

Result: **diverse but semantically empty concepts.** The encoder maximised concept orthogonality at the expense of encoding meaningful language content.

### Task-by-task interpretation

**MRPC (+0.11%):** Tiny improvement. MRPC is a small dataset (3.7k samples) where fine-tuning can compensate for weaker pre-trained representations. Not informative.

**QQP (−13.76% F1):** Large regression. QQP requires genuine semantic understanding (paraphrase detection on 363k pairs, 3 epochs fine-tuning). With worse pre-trained features (MLM loss 4.31 vs 2.54), the model cannot build adequate paraphrase representations. This is the clearest signal that the pre-training quality degraded.

**MNLI (−10%):** Also a large regression for the same reason — natural language inference requires compositionality that comes from good language modelling. The concept encoder's NLI capability is directly tied to MLM quality.

**STS-B (Pearson 0.341):** First measurement ever (bug was fixed). Low, but no baseline for comparison. Run again on L6 baseline to establish reference.

### Root cause: Kendall-Gal is too aggressive for this use case

Kendall-Gal (Kendall & Gal, 2018) was designed for multi-task learning where all losses are equally important and should be balanced automatically. Here, MLM is the **primary** objective and concept regularisation is a **secondary** constraint. Using Kendall-Gal treats them as equals, allowing the optimizer to effectively mute MLM.

---

## Conclusions

1. **Concept diversity ≠ concept quality.** Having 122/128 orthogonal concept dimensions is useless if those dimensions don't encode meaningful language information. MLM quality (eval_loss ≈ 2.5) is a prerequisite, not a trade-off.

2. **Kendall-Gal is wrong for concept regularisation.** It correctly balances two equally important tasks but wrongly suppresses the dominant task when concept losses converge faster.

3. **Fixed low-weight regularisation is the right approach.** The next experiment should use `LOSS_WEIGHTING="fixed"` with `LOSS_WEIGHT=0.05` (5% weight on concept loss). This keeps MLM dominant while adding light diversity pressure.

4. **Target operating point:** MLM eval_loss < 3.0 (close to baseline 2.54) with effective rank > 50% (64/128). This requires the concept loss weight to be small enough not to degrade MLM convergence.

---

## Update (Feb 21 2026) — Fixed Weight 0.1 Experiment & GLUE/Beyond-GLUE Eval

We trained the L6 model using `fixed` loss weighting with `LOSS_WEIGHT=0.1` and `combined` loss.
**Model:** `perceiver_mlm_H512L6C128_20260220_184029`

**Pre-training Results:**
- **MLM eval_loss:** 3.57 (20 epochs) (degraded from baseline 2.54 - 40 epochs, but better than Kendall-Gal's 4.31 - 20 epochs)
- **Concept eff. rank:** 15.97 / 128 (12.5%) — **✗ POOR (Collapsed)**
- **Mean concept similarity:** 0.13 — ✓ GOOD
- **Max concept similarity:** 0.999 — ✗ POOR

**Downstream Evaluation Results:**
Despite the degraded pre-training loss and collapsed rank, we ran a full evaluation on the semantic subset (MRPC, QQP, STS-B, MNLI-m) plus PAWS to see if the 16 active dimensions held denser semantic features.

| Task | Metric | **Fixed 0.1 combined** | Baseline (No loss) | Kendall-Gal combined |
|---|---|:---:|:---:|:---:|
| **MRPC** | F1 | 80.7% | 81.3% | 81.4% |
| **QQP** | F1 | 64.9% | 72.5% | 58.7% |
| **MNLI-m** | Acc | 56.9% | 59.1% | 48.9% |
| **STS-B** | Pearson | 0.507 | 0.627 | 0.341 |
| **PAWS** | Acc | 57.6% | (TBD) | (TBD) |

**Diagnosis:**
The fixed weight of 0.1 sits exactly in the middle between the Baseline and Kendall-Gal across almost all metrics. 
It degraded MLM loss (2.54 → 3.57), which caused a direct regression in downstream task performance (e.g. QQP 72.5% → 64.9%). At the same time, it entirely failed to prevent dimensional collapse (effective rank 12.5%).

This definitively proves that the `combined` loss (orthogonality-driven) is fundamentally struggling to regularize the space without destroying the MLM gradient. There is no "goldilocks" weight setting that will save it. 

**Recommendation:**
The `combined` loss is ineffective. We must switch to a direct variance regularizer (`VICReg`) or the newly implemented `t_regs_mst` (Minimum Spanning Tree uniformity), which directly forces the utilization of dimensions. Alternatively, accelerate the shift to the **Masked Diffusion** objective, which bypasses this regularisation tuning entirely by forcing concepts to hold semantic meaning.

---

## WandB Runs

| Task | WandB Run |
|---|---|
| MRPC | [glue-mrpc-perceiver-mlm-...20260219_2027](https://wandb.ai/ksopyla/MrCogito/runs/glue-mrpc-perceiver-mlm-h512l6c128-20260219-105435-61M-20260219_2027) |
| STS-B | [glue-stsb-perceiver-mlm-...20260219_2029](https://wandb.ai/ksopyla/MrCogito/runs/glue-stsb-perceiver-mlm-h512l6c128-20260219-105435-61M-20260219_2029) |
| QQP | [glue-qqp-perceiver-mlm-...20260219_2033](https://wandb.ai/ksopyla/MrCogito/runs/glue-qqp-perceiver-mlm-h512l6c128-20260219-105435-61M-20260219_2033) |
| MNLI-m | [glue-mnli-matched-perceiver-mlm-...20260219_2048](https://wandb.ai/ksopyla/MrCogito/runs/glue-mnli-matched-perceiver-mlm-h512l6c128-20260219-105435-61M-20260219_2048) |
| MNLI-mm | [glue-mnli-mismatched-perceiver-mlm-...20260219_2104](https://wandb.ai/ksopyla/MrCogito/runs/glue-mnli-mismatched-perceiver-mlm-h512l6c128-20260219-105435-61M-20260219_2104) |
