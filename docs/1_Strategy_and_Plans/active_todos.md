# Experiment TODO List v3

**Created: 2026-02-19** | **Updated: 2026-02-19**
**Status: Active**

## Summary of Feb 19 Results

| Experiment | Outcome | WandB Link |
|---|---|---|
| Training: combined+kendall_gal (20ep Minipile) | Completed. Concept eff. rank **95.5%** ✓ | [Link](https://wandb.ai/ksopyla/MrCogito/runs/perceiver_mlm_H512L6C128_20260219_105435) |
| GLUE eval: L6 + concept losses | QQP **−13.76%**, MNLI **−10%** vs baseline ✗ | [Link](https://wandb.ai/ksopyla/MrCogito/runs/glue-mrpc-perceiver-mlm-h512l6c128-20260219-105435-61M-20260219_2027) |
| Root cause | Kendall-Gal muted MLM gradient after step ~10k. MLM eval_loss 4.31 vs 2.54. | |
| Key lesson | **Concept diversity without semantic content = worse performance.** | |
| Next experiment | `fixed` loss weight (0.05) to keep MLM dominant, run baseline STS-B | |

**Full results:** [`concept_losses_20260219.md`](../2_Experiments_Registry/run_reports/concept_losses_20260219.md)

---

## Currently Running

- [x] ~~MLM training with `combined` + `kendall_gal`~~ — **COMPLETED & EVALUATED**
  - Result: concept diversity ✓ but GLUE regressed (QQP −13.76%, MNLI −10%)

## TODO 0: Run L6 baseline STS-B evaluation — DONE ✅

**Result (2026-02-20):** L6 baseline `perceiver_mlm_H512L6C128_20260208_211633` achieves STS-B **Pearson 0.627 / Spearman 0.627**.

The Kendall-Gal concept losses model scored only 0.341 — a **−46% regression** on STS-B. STS-B directly measures semantic similarity, which is exactly what concept encoders should be good at. This confirms the diagnosis: Kendall-Gal destroyed semantic content while maximising geometric diversity.

Reference targets for next training run (fixed 0.1 weighting):
- STS-B Pearson > 0.60 (within 5% of baseline) ← primary target
- Concept eff. rank > 30/128 ← secondary target
- QQP F1 > 70% ← no regression

**Status:** [x] Done

---

## TODO 0b: Re-train L6 with `fixed` concept loss weight — DONE ✅

**Result (2026-02-21):** The model was trained with `fixed` weighting (0.1 weight) for the `combined` loss. ([WandB Link](https://wandb.ai/ksopyla/MrCogito/runs/perceiver_mlm_H512L6C128_20260220_184029))
**Outcome:**
- MLM eval_loss degraded to **3.57** (vs baseline 2.54).
- Concept eff. rank collapsed to **15.97 / 128 (12.5%)**.

**Conclusion:** `combined` loss at 0.1 weight hurts the MLM objective while still failing to prevent dimensional collapse. This regularisation approach is fundamentally flawed for our setup. We should abandon `combined` loss and move to `t_regs_mst` or Masked Diffusion.

**Status:** [x] Done

---

## TODO 1: Fix STS-B Evaluation Bug (Priority: HIGH, Effort: 0.5h)

**Problem:** In `evaluate_model_on_glue.py`, the `compute_metrics` function passes 2D predictions `[N, 1]` to HuggingFace `evaluate` pearsonr/spearmanr metrics for STS-B, instead of the squeezed 1D `[N]` predictions. This causes either errors or wrong metric values.

**Fix location:** `training/evaluate_model_on_glue.py` → `compute_metrics()` function, lines ~510-515.

**Fix:**
```python
# Before (broken):
if task == "stsb":
    predictions_raw = predictions[:, 0]

# After (fixed):
if task == "stsb":
    predictions = predictions[:, 0]  # Squeeze to 1D for all metric computations
    predictions_raw = predictions
```

**Verify:** Run STS-B evaluation on existing L6 checkpoint. Should produce non-zero Pearson/Spearman.

**Status:** [x] Done — fixed in `training/evaluate_model_on_glue.py` and `evaluation/evaluate_model_on_glue.py`

---

## TODO 2: Update GLUE Evaluation Shell Script (Priority: HIGH, Effort: 1h)

**File:** `scripts/evaluate_concept_encoder_glue.sh`

**Changes:**
1. Change the default "all" task list to concept-relevant tasks only: `mrpc`, `qqp`, `stsb`, `mnli-matched`, `mnli-mismatched`
2. Add support for `perceiver_decoder_cls` model type auto-detection
3. Add L6 model paths as defaults/comments
4. Keep CoLA/RTE/SST-2 accessible via explicit `--task` but remove from "all" default

**Rationale:**
- CoLA (MCC ≈ 0.13) is an architectural ceiling — wasting GPU time
- RTE (2.5K samples) is too noisy at current performance level
- SST-2 is nearly saturated (~78%)
- MRPC, QQP, STS-B, MNLI directly test concept quality

**Status:** [x] Done — `all` now runs concept-relevant tasks; `all-glue` for the full set. Default model updated to L6 perceiver_mlm.

---

## TODO 3: Create Evaluation Folder + Beyond-GLUE Benchmarks (Priority: MEDIUM, Effort: 3-4h)

### 3a. Move evaluation script to `evaluation/` folder

Move `training/evaluate_model_on_glue.py` → `evaluation/evaluate_model_on_glue.py`.
Update all references:
- `scripts/evaluate_concept_encoder_glue.sh`
- `scripts/evaluate_concept_encoder_glue.ps1`
- `docs/` references

### 3b. Add two Beyond-GLUE evaluation datasets

**Selected datasets:**

| Dataset | Task | Why | HuggingFace ID | Size |
|---|---|---|---|---|
| **SICK** | Relatedness (regression) + Entailment (3-class) | Tests semantic similarity + entailment in one dataset; complementary to GLUE STS-B + MNLI | `sick` | 10K |
| **PAWS** | Adversarial Paraphrase (binary) | Tests if concepts understand *meaning* vs *word overlap*; concept encoder should excel here if concepts truly encode semantics | `google-research-datasets/paws`, `labeled_final` | 49K + 8K |

**Why these two:**

1. **SICK** gives us relatedness (Pearson/Spearman, like STS-B) + entailment (accuracy, like MNLI) in a single 10K dataset. Perfect for quick sanity checks. If concepts work, SICK relatedness should be better than GLUE STS-B (simpler sentences).

2. **PAWS** is adversarial: sentence pairs share almost all words but have different meanings ("Flights from NYC to LA" vs "Flights from LA to NYC"). Bag-of-words models fail hard. If the concept bottleneck truly captures semantic structure (not just word co-occurrence), it should do well here. If it fails, that's a strong signal the concepts are just doing bag-of-words.

### 3c. Create evaluation scripts

- `evaluation/evaluate_on_sick.py` — Eval on SICK (relatedness + entailment)
- `evaluation/evaluate_on_paws.py` — Eval on PAWS (adversarial paraphrase)
- `scripts/evaluate_concept_encoder_sick.sh` — Shell wrapper
- `scripts/evaluate_concept_encoder_paws.sh` — Shell wrapper

### 3d. Classification heads for new datasets

Reuse existing classification architecture (small heads):
- **SICK-Relatedness:** `ConceptEncoderForSequenceClassificationPerceiver` with `num_labels=1`, `problem_type="regression"` — CLS query → concepts → linear(H → 1). Same as STS-B.
- **SICK-Entailment:** `ConceptEncoderForSequenceClassificationPerceiver` with `num_labels=3` — same as MNLI.
- **PAWS:** `ConceptEncoderForSequenceClassificationPerceiver` with `num_labels=2` — same as MRPC.
- Also test `ConceptEncoderForSequenceClassificationViaDecoder` (reuses pretrained MLM decoder) for comparison.

**No new model classes needed.** The existing `ConceptEncoderForSequenceClassificationPerceiver` and `ConceptEncoderForSequenceClassificationViaDecoder` handle any num_labels via config. The heads are minimal: CLS query cross-attention + LayerNorm + Linear.

**Status:** [x] Done — `evaluation/` folder created with `evaluate_model_on_glue.py`, `evaluate_on_benchmark.py` (SICK + PAWS). Shell scripts: `scripts/evaluate_concept_encoder_sick.sh`, `scripts/evaluate_concept_encoder_paws.sh`.

---

## TODO 4: Re-run Concept Analysis After Training — DONE ✅

**Action (2026-02-21):** Ran `run_concept_analysis.py` on the `perceiver_mlm_H512L6C128_20260220_184029` checkpoint (fixed 0.1 weight for combined loss).

**Results:**
- Effective rank: **15.97 / 128 (12.5%)** — ✗ POOR (Target > 40)
- Mean pairwise concept similarity: **0.133** — ✓ GOOD (Target < 0.3)
- Max pairwise concept similarity: **0.999** — ✗ POOR (Target < 0.6)
- Top-1 singular value dominance: **83.1** — ✗ POOR (Target < 50)

**Decision gate reached:**
```
effective_rank < 20  → concept losses alone insufficient, implement Slot Attention or Masked Diffusion. 
Because the combined loss with weight 0.1 still collapsed, we abandon this regularisation track. 
```
**Status:** [x] Done

---

## TODO 5: Evaluate with ViaDecoder Classification (Priority: MEDIUM, Effort: 1h)

*Maps to roadmap [Phase 3](roadmap.md#phase-3-classification-via-decoder-2-3-days-coding-1-day-eval)*

**Depends on:** TODO 1 (STS-B fix)

**Action:** Run GLUE eval with `--model_type perceiver_decoder_cls` on existing L6 `perceiver_mlm` checkpoint.

**Why:** The `ConceptEncoderForSequenceClassificationViaDecoder` class loads BOTH encoder + decoder weights from the MLM checkpoint. The decoder already learned position→concept reconstruction. Pooling over the reconstructed sequence should give richer representations than a single CLS query.

**Tasks to evaluate:** mrpc, qqp, stsb, mnli-matched

**Status:** [ ] Not started

---

## TODO 6: Masked Diffusion Experiment (Priority: HIGH, Effort: 5 GPU-days)

*Maps to roadmap [Phase 9](roadmap.md#phase-9-masked-diffusion-decoder--replace-mlm-new--high-priority)*

**When:** After TODO 4 results are in, or in parallel on Odra.

**Action:** Run `bash scripts/train_diffusion_multigpu.sh` on Odra with warm-start from L6 MLM checkpoint + concept losses.

**Compare:** Concept analysis metrics vs TODO 4 results. If diffusion effective_rank > MLM effective_rank, switch to diffusion as primary objective.

**Status:** [ ] Not started

---

## TODO 7: Data Scaling (Priority: HIGH, Effort: 7 GPU-days)

*Maps to roadmap [Phase 4 & 5](roadmap.md#phase-4-scale-pretraining-data--add-contrastive-objective-5-7-days-training)*

**Depends on:** TODO 4 (concept losses validated)

**Action:** Train on OpenWebText + Wikipedia (15M samples, ~33GB) with:
- Best objective (MLM or diffusion, from TODO 4/6 comparison)
- Concept losses (combined + kendall_gal)
- Span masking (contiguous 3-10 tokens, 30% rate)
- 10-15 epochs (not 40 — more data, fewer epochs)

**Target:** MLM loss < 2.0, effective_rank > 60

**Status:** [ ] Not started

---

## TODO 8: Recursive Concept Encoder — Train & Evaluate (Priority: MEDIUM, Effort: 3 days code + 5 GPU-days)

**Goal:** Train and evaluate the TRM-style recursive concept encoder to test whether weight-tied iterations lower params while maintaining/improving concept quality.

**Implementation:** Fully separate from standard encoder. No changes to `concept_encoder.py`.

### Files

| File | Purpose | Status |
|---|---|---|
| `nn/concept_encoder_recursive.py` | `RecursiveConceptEncoderConfig`, `RecursiveConceptEncoder` (1 shared layer, K iterations) | [x] Done |
| `nn/concept_encoder_recursive_mlm.py` | `RecursiveConceptEncoderForMaskedLM` — Perceiver IO decoder on top of recursive encoder | [x] Done |
| `training/mlm_training.py` | Add `recursive_mlm` to `MODEL_REGISTRY` + `RecursiveConceptEncoderConfig` | [x] Done |
| `scripts/train_recursive_mlm.sh` | Multi-GPU training script for recursive variant | [x] Done |
| `tests/test_recursive_encoder.py` | Basic forward pass + param count verification | [x] Done |
| `tests/test_recursive_mlm.py` | MLM forward pass + loss + registry + test-time scaling | [x] Done |

### Architecture

```
Standard perceiver_mlm (L6):       Recursive (K=6 iterations):
  ConceptEncoder (6 layers)           RecursiveConceptEncoder (1 shared layer x6)
    + Perceiver IO decoder              + same Perceiver IO decoder
    + lm_head                           + lm_head
  Total: ~61M params                  Total: ~42M params (-31%)
```

The recursive MLM model (`RecursiveConceptEncoderForMaskedLM`) is structurally identical to
`ConceptEncoderForMaskedLMPerceiver` — same decoder, same lm_head, same loss_manager —
except `self.encoder` is a `RecursiveConceptEncoder` instead of a `ConceptEncoder`.

### Parameter comparison (H512, C128, intermediate=2048)

| Component | Standard L6 | Recursive K=6 | Savings |
|---|---|---|---|
| Encoder layers | 6 x 6.3M = **37.6M** | 1 x 6.3M = **6.3M** | **83%** |
| Token + position + concept embeddings | 26.9M | 26.9M | 0% |
| Perceiver decoder + lm_head | ~24M | ~24M | 0% |
| **Total model** | **~88M** | **~57M** | **35%** |

### Training plan

**Phase A — Minipile baseline (2 GPU-days on Odra):**
```bash
# model_type: recursive_mlm
# Architecture: RecursiveConceptEncoderForMaskedLM, H512, K=6, C128
# Data: Minipile, 20 epochs
# Losses: combined + fixed 0.1 (NOT kendall_gal — lesson from Feb 19)
# LR: 2e-4 (slightly lower than 3e-4 for gradient accumulation through K iters)
# Compare: MLM loss and concept analysis vs standard perceiver_mlm L6

bash scripts/train_recursive_mlm.sh
```

**Phase B — Iteration sweep (1 GPU-day on Odra):**

Train one model (K=6), then evaluate at different iteration counts:

| Eval K | Expected behavior |
|---|---|
| K=2 | Underfitting — concepts barely refined |
| K=4 | Faster inference, slightly worse quality |
| K=6 | Match training-time quality |
| K=8 | Mild improvement (extra refinement) |
| K=12 | Test-time compute scaling (should improve on hard tasks) |

Run GLUE (mrpc, stsb, qqp, mnli-matched) at each K value — no retraining needed,
just change `model.config.num_iterations` before evaluation.

**Phase C — GLUE + Beyond-GLUE comparison:**

| Benchmark | Standard L6 | Recursive K=6 | What it tells us |
|---|---|---|---|
| MRPC F1 | 81.3% | ? | Semantic similarity: do fewer params hurt? |
| STS-B Pearson | 0.627 | ? | Concept quality regression |
| QQP F1 | 72.5% | ? | Scale test (400K samples) |
| MNLI-m Acc | 59.1% | ? | Compositional reasoning |
| PAWS Acc | ? | ? | Meaning vs surface form |
| SICK Relatedness | ? | ? | Simpler similarity sanity check |

**Phase D — Warm-start from standard checkpoint:**

The recursive encoder can load layer-0 weights from an existing standard L6 checkpoint
via `encoder.load_from_standard_checkpoint()`. This gives the shared layer a head start:
```python
model.encoder.load_from_standard_checkpoint(
    "Cache/Training/perceiver_mlm_H512L6C128_20260208_211633/perceiver_mlm_H512L6C128_20260208_211633"
)
```

Then continue training — the shared layer starts from pretrained layer-0 weights while
the decoder initializes fresh (or also loaded from checkpoint).

### Key questions this experiment answers

1. **Does weight tying hurt concept quality?** Compare effective rank, concept similarity, MLM loss.
2. **Is 47% fewer encoder params "free"?** If GLUE scores are within 1-2pts, yes.
3. **Does test-time scaling work?** If K=12 beats K=6 on MNLI, concept refinement is real.
4. **Is this the path to the audio model?** 42M-param model that can scale compute at inference = ideal for real-time speech.

### Gradient flow considerations

1. **Residual connections** in cross-attn, self-attn, and FFN allow gradients to flow through K iterations.
2. **Pre-LN architecture** stabilizes gradients across iterations.
3. **Implicit gradient accumulation:** shared layer receives gradients from K positions — may need ~0.7x learning rate.
4. **ALBERT precedent:** 12M params, shared 12 layers, 90.6% MRPC F1 in our baselines. It works.

---

## Experiment Priority Order

```
Week 1 (NOW):
  [x] Training with concept losses (running on Polonez)
  [x] TODO 1: Fix STS-B bug
  [x] TODO 2: Update GLUE shell script
  [x] TODO 3: Evaluation folder + SICK + PAWS
  [ ] TODO 5: Eval ViaDecoder classification on existing L6
  [x] TODO 4: Concept analysis on new checkpoint (when training finishes)

Week 2:
  [x] TODO 8a: Build recursive MLM model + register in mlm_training.py — DONE
  [x] TODO 0b: Re-train L6 with fixed concept loss weight (Polonez) — DONE (Failed to fix collapse)
  [ ] TODO 8b: Train recursive_mlm on Minipile (Odra, 2 GPU-days)
  [ ] TODO 6:  Masked diffusion experiment (Polonez, 5 GPU-days)
  [ ] TODO 8c: Concept analysis on recursive checkpoint
  [ ] TODO 8d: GLUE + SICK + PAWS eval on recursive model

Week 3:
  [ ] TODO 8e: Iteration sweep (K=2,4,6,8,12) — eval only, no retraining
  [ ] TODO 7:  Data scaling (best model from TODO 4/6/8 comparison)

Week 4+:
  [ ] Full comparison table: standard vs recursive vs diffusion
  [ ] Pick winner, scale to OpenWebText + Wikipedia
```

---

*Plan created: 2026-02-19*
*Next review: after TODO 4 concept analysis*
