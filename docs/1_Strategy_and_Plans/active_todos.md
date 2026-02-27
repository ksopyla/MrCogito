# Experiment TODO List v3

**Created: 2026-02-19** | **Updated: 2026-02-27**
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
- [x] ~~Diffusion MLM L2 (TODO 6)~~ — **COMPLETED & EVALUATED (2026-02-25)**
  - Result: Stable training ✓, concept rank 2x better but still collapsed, STS-B near-random
- [ ] **L6 Diffusion + ELBO baseline (TODO 11)** — **RUNNING ON ODRA (2026-02-26)**
  - Config: H512 L6 C128 D2, ELBO=True, t_min=0.3, concept_losses=none, 3x RTX 3090
  - WandB: `diffusion_H512L6C128D2_20260226_155541`
  - At step 20k: train_loss 2.85, eval_loss 1.42, concept rank 5.45/128 (collapsed)
  - ETA: ~22h remaining from step 20k

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

**Done date: 2026-02-21**
**Result (2026-02-21):** The model was trained with `fixed` weighting (0.1 weight) for the `combined` loss. ([WandB Link](https://wandb.ai/ksopyla/MrCogito/runs/perceiver_mlm_H512L6C128_20260220_184029))
**Outcome:**
- MLM eval_loss degraded to **3.57** (vs baseline 2.54). *Note: Baseline was trained for 40 epochs, while this run was 20 epochs, so the degradation is less severe than it appears.*
- Concept eff. rank collapsed to **15.97 / 128 (12.5%)**.
- Downstream Evaluation:
  - **MRPC**: 80.7% (vs 81.3%)
  - **QQP**: 64.9% (vs 72.5%)
  - **MNLI-m**: 56.9% (vs 59.1%)
  - **STS-B**: 0.507 (vs 0.627)
  - **PAWS**: 57.6%
  - **SICK**: Failed due to `sick.py` HuggingFace dataset script deprecation.

**Conclusion:** Although the loss degradation is partially explained by fewer epochs, the dimensional collapse (rank ~16) remains fatal. `combined` loss at 0.1 weight fails to prevent this collapse. The compressed 16-rank space is not semantically dense enough to preserve performance on QQP, MNLI, and STS-B. We should abandon `combined` loss and move to `t_regs_mst` or Masked Diffusion.

**Status:** [x] Done

---

## TODO 1: Fix STS-B Evaluation Bug (Priority: HIGH, Effort: 0.5h) - DONE ✅

**Done date: 2026-02-20**

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

## TODO 2: Update GLUE Evaluation Shell Script (Priority: HIGH, Effort: 1h) DONE ✅

**Done date: needs to be checked**
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

## TODO 3: Create Evaluation Folder + Beyond-GLUE Benchmarks (Priority: MEDIUM, Effort: 3-4h) DONE ✅

**Done date: exact date needs to be checked**

### 3a. Move evaluation script to `evaluation/` folder

Move `training/evaluate_model_on_glue.py` → `evaluation/evaluate_model_on_glue.py`.
Update all references:
- `scripts/evaluate_concept_encoder_glue.sh`
- `scripts/evaluate_concept_encoder_glue.ps1`
- `docs/` references

### 3b. Add two Beyond-GLUE evaluation datasets DONE ✅

**Done date: exact date needs to be checked**
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

## TODO 4: Re-run Concept Analysis After Training — DONE v1, ✅

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
**Status:** [x] Done, but need to be validated on better trained concept model in the future.

---

## TODO 5: Evaluate with ViaDecoder Classification — DONE v1, need to be validated on better trained concept model in the future. ✅

**Done date: 2026-02-22**

**Action:** Ran GLUE eval with `--model_type perceiver_decoder_cls` on the L6 canonical checkpoint via HF Hub (`ksopyla/concept-encoder-perceiver_mlm_H512L6C128_20260208_211633`) on Odra.

**Results vs CLS-Query baseline:**

| Task | CLS-Query | ViaDecoder | Delta |
|---|---|---|---|
| MRPC F1 | 81.3% | **82.73%** | +1.4% ✓ |
| STS-B Pearson | 0.627 | **0.650** | +2.3% ✓ |
| QQP F1 | 72.5% | **73.35%** | +0.85% ✓ |
| MNLI-m Acc | 59.1% | **59.75%** | +0.65% ✓ |
| MNLI-mm Acc | 59.34% | **~61.0%** | ~+1.7% ✓ |

**Conclusion:** ViaDecoder consistently outperforms CLS-Query on all F1/Pearson metrics. Improvement is real but modest (+0.65–2.3%) mainly due to evaluation on not well trained concepts, bounded by concept collapse (eff. rank 5/128), should be evaluated one more time on better trained concept model. **ViaDecoder is now the default evaluation mode for all future GLUE runs.**

**Full analysis:** [via_decoder_eval_20260222.md](../2_Experiments_Registry/run_reports/via_decoder_eval_20260222.md)

**Status:** [x] Done

---

## TODO 6: Masked Diffusion Experiment (Priority: HIGH, Effort: 5 GPU-days) DONE ✅

**Done date: 2026-02-25 (evaluated)**

*Maps to roadmap [Phase 9](roadmap.md#phase-9-masked-diffusion-decoder--replace-mlm-new--high-priority)*

**Run 1 outcome (2026-02-21, `diffusion_H512L2C128D2_20260221_195554`):** FAILED — gradient explosion at epoch 12.

**Run 2 outcome (2026-02-23, `diffusion_H512L2C128D2_20260223_203349`):** COMPLETED — stable training with xattn-only decoder + AdaLN-Zero.
- Train loss: 14.19 → 2.894, Eval loss: 3.77 → 1.433
- Grad norm stable (peaked 6.81, ended 0.23)

**Evaluation results (2026-02-25):**

| Metric | Diffusion L2 | L2 Perceiver Baseline | L6 ViaDecoder Baseline |
|---|---|---|---|
| Concept eff. rank | **10.1/128 (7.9%)** | ~5/128 (est.) | 5/128 (4%) |
| MRPC F1 | **80.0%** | 80.6% | 82.73% |
| STS-B Pearson | **0.138** | N/A | 0.650 |
| PAWS Accuracy | **55.98%** | N/A | 57.6% |
| SICK Relatedness | **0.064** | N/A | N/A |
| SICK Entailment | **57.78%** | N/A | N/A |

**Concept analysis details:**
- Global effective rank: 10.1/128 (7.9%) — 2x better than L6 MLM baseline (5/128) but still collapsed
- Mean pairwise similarity: 0.187 — GOOD
- Max pairwise similarity: 1.000 — POOR (duplicate concepts exist)
- Top-1 dominance ratio: 0.099 — GOOD (much better singular value spread than MLM)
- Min dimension std: 0.631 — GOOD (no dead dimensions)

**Conclusion:** Diffusion training alone does NOT fix concept collapse. The architecture is validated (no gradient explosion, stable training), and shows 2x improvement in effective rank over MLM baseline. However, STS-B Pearson 0.138 is near-random, indicating concepts are geometrically better distributed but not semantically grounded. MRPC F1 80.0% matches L2 baseline — the encoder captures *some* information but the concept bottleneck is too leaky.

**Decision gate (per roadmap):** Diffusion L2 effective_rank = 10.1 (< 30/128 threshold). Diffusion alone insufficient. Wait for TSDAE comparison. If TSDAE also fails rank > 30, implement Slot Attention (C5).

**Status:** [x] Done — evaluated, results logged

---

## TODO 6b: Diffusion Slow Training — reworked, to verify on Polonez ✅

**Model:** `diffusion_H512L2C128D2_20260221_195554` (trained on Polonez)
**Context:** Training diffusion model on Polonez L2 took 26h39m — ~10× slower than perceiver MLM L2.
**Root cause (identified 2026-02-23):** Three compounding factors:
1. O(N²) self-attention in decoder (5.2× more FLOPs per sample than perceiver MLM)
2. Full lm_head over all 512 positions instead of sparse (6.6× wasted compute on vocab projection)
3. `GRADIENT_ACCUMULATION_STEPS=1` → 78,140 steps vs 39,070 for MLM (2× more steps)
   Total: 5.2× × 2× ≈ 10× slower — exactly matching the observation.
**Fix:** Decoder redesigned to cross-attention only + sparse lm_head + grad_accum=2. Expected ~6–8× speedup.
**Status:** [x] In progress — architecture fix in CHANGELOG `[2026-02-23]` need to be validated on Polonez.

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

## TODO 9: ~~Initialize Encoder from SoTA Pretrained Model~~ — Superseded by TODO 14 ✅

**Status:** [x] Superseded — see TODO 14 for updated rationale (Coconut-inspired strategy).

---

## Experiment Priority Order (Updated 2026-02-26)

Aligned with [roadmap.md v4](roadmap.md) Track IDs. See roadmap for full experiment details and decision gates.

**Strategic shift (v4):** Self-reconstruction alone is insufficient for a generative reasoning model. The training objective must evolve from reconstruction → generation → reasoning. New experiments (A9-A11, TODO 11-13) address this directly. Full analysis: [diffusion_diagnosis_20260226.md](../4_Research_Notes/diffusion_diagnosis_20260226.md)

```
Week 1 (2026-02-19 — DONE):
  [x] Training with concept losses (running on Polonez)
  [x] TODO 0:  Run L6 baseline STS-B evaluation
  [x] TODO 0b: Re-train L6 with fixed concept loss weight → FAILED (rank 12.5%)
  [x] TODO 1:  Fix STS-B bug
  [x] TODO 2:  Update GLUE shell script
  [x] TODO 3:  Evaluation folder + SICK + PAWS
  [x] TODO 4:  Concept analysis on new checkpoint

Week 2 (2026-02-21 — DONE — Architecture Overhaul):
  [x] Architecture review: identified 5 structural misalignments
  [x] Implemented TSDAE, BiXT, PosOnly, weighted pooling, pair_cls
  [x] TODO 5: ViaDecoder GLUE eval — DONE (+0.65–2.3% on all tasks)
  [x] Uploaded L6 canonical model to HF Hub

Week 3 (2026-02-23 — DONE — Track A: Diffusion L2):
  [x] TODO 6:  Masked diffusion L2 — DONE, rank 10/128, STS-B 0.138
  [x] TODO 6b: Diffusion slow training diagnosis — DONE, decoder redesigned

Week 4 (2026-02-26 — Diffusion diagnosis + fixes):
  [x] Diffusion L2 root cause analysis — DONE, 5 causes identified
  [x] TODO 12:  Fix ELBO loss weighting + t_min (A10, code change, 0.5 day) — DONE
  [~] TODO 11:  L6 Diffusion + ELBO baseline (A9+A10, Odra, 1.5 GPU-day) ← RUNNING
  [x] VICReg + t_regs_mst implementation (warmup, callback, tests) — DONE
  [ ] TODO 11b: L6 Diffusion + VICReg + t_regs_mst (A5+A9, Odra, 1 GPU-day) ← NEXT after TODO 11
  [ ] TODO 10:  Train TSDAE PosOnly on Minipile (A1, Odra, 5 GPU-days) ← AFTER TODO 11b
  [ ] TODO 10b: Train TSDAE PosOnly + BiXT on Minipile (A2, parallel on other server)

Week 5 (Prefix generation + evaluation):
  [ ] TODO 13:  Implement & train prefix generation (A11, 3 days code + 5 GPU-days)
  [ ] TODO 10c: Concept analysis on ALL Track A checkpoints (A7 REPEAT)
  [ ] TODO 10d: GLUE eval with ViaDecoder + perceiver_pair_cls (A6 REPEAT)
  [ ] TODO 10e: Zero-shot STS-B (A8, cosine similarity, no fine-tuning)

Week 6 (Decision gate + Track A winner):
  [ ] Compare: L6 diffusion vs TSDAE vs prefix generation
  [ ] Metrics: concept rank, STS-B, prefix generation loss (all three!)
  [ ] Decision gate: pick winner (see roadmap Gate 1)
  [ ] If all fail rank > 30: implement Slot Attention (C5) as fallback

Week 7-8 (Track A.4 + Track C start):
  [ ] Add contrastive loss (SimCSE) to Track A winner (A4)
  [ ] Recursive MLM baseline on Minipile (C1, TODO 8)
  [ ] Test-time compute scaling sweep K=2..12 (C3)
  [ ] TODO 14: Backbone init from SmolLM2-135M (B3, can start early if capacity allows)

Week 9-10 (Track B: Data Scaling):
  [ ] Scale to OpenWebText + Wikipedia with winner objective (B1, TODO 7)
  [ ] Span masking (B2)
  [ ] REPEAT: Full eval after scaling (B4/B5)

Later (see roadmap for full schedule):
  [ ] Dimension Inversion ablation (C4, token_dim=32, concept_dim=512)
  [ ] Progressive sequence length training (512 → 2K → 8K)
  [ ] Long-context eval SCROLLS/LongBench (D2/D3, after data scaling)
  [ ] Recursive encoder with Track A winner + variable-depth training (C2)
  [ ] Simple reasoning eval: ProntoQA with recursive encoder at variable K
```

---

## TODO 10: TSDAE Training Experiments (Priority: HIGHEST)

**Architecture changes completed (2026-02-21):**
- `DataCollatorForTSDAE`: token deletion (60%), dense labels, attention_mask zeroing
- `ConceptEncoderForMaskedLMPerceiverPosOnly`: dense reconstruction loss (all positions)
- `BiConceptEncoderLayer`: BiXT bidirectional cross-attention (O(C*N) complexity preserved)
- `ConceptEncoderForSentencePairClassification`: separate encoding, weighted concept pooling, cosine_only mode
- `ConceptEncoderForSequenceClassificationPerceiver`: CLS query → weighted concept pooling

**Training script:** `training/train_tsdae.py`
**Local test:** `scripts/test_tsdae_local.ps1`

**Experiment A — TSDAE PosOnly baseline (5 GPU-days on Polonez):**


**Experiment B — TSDAE PosOnly + BiXT (parallel on Odra):**
Same as A but with `--use_bixt`. Compare concept quality (effective rank, mean sim) and GLUE scores.

**Evaluation plan:**
1. Concept analysis: effective rank, mean pairwise similarity (target: rank > 64/128)
2. GLUE with perceiver_pair_cls: MRPC, QQP, STS-B, MNLI (separate encoding)
3. Zero-shot STS-B: cosine similarity of separately-encoded sentences (no fine-tuning)
4. Compare against MLM baseline (L6, eff rank 5/128) and diffusion

**Status:** [ ] Not started (implementation done, waiting for GPU time)

---

## TODO 11: L6 Diffusion + ELBO — Controlled Experiment (Priority: HIGHEST, Effort: 1 GPU-day)

*Maps to roadmap A9 + A10. Most informative experiment: tests both depth fix and ELBO simultaneously.*

**Goal:** Test whether L6 encoder depth + ELBO 1/t loss weighting + t_min=0.3 fix the semantic emptiness of diffusion-trained concepts. Combines the two most impactful fixes from the diagnosis in a single run rather than wasting GPU time on sequential ablations.

**Config:** `scripts/train_diffusion_multigpu.sh` (updated 2026-02-26): H512 L6 C128 D2, ELBO=True, t_min=0.3, LR 3e-4, 20 epochs, batch 64, grad_accum 2, bf16, no concept losses.

**Decision logic:**
- STS-B > 0.50 → depth + ELBO fix the bottleneck, diffusion objective is viable
- STS-B 0.30–0.50 → partial improvement, add contrastive loss or prefix generation
- STS-B < 0.30 → self-reconstruction is fundamentally insufficient, pivot to prefix generation
- Concept rank > 20/128 → geometry improvement scales with depth

**Machine:** Odra (3x RTX 3090), est. ~36h (Polonez crashed 2026-02-26, redirected to Odra)

**Full rationale:** [diffusion_diagnosis_20260226.md](../4_Research_Notes/diffusion_diagnosis_20260226.md) (Causes 1-3)

**Interim analysis (step 20k, 2026-02-27):** Concept rank 5.45/128 — collapse already visible. Loss plateau at eval_loss 1.42. Prognosis: high risk of repeating L2 STS-B ~0.15 without regularization. Let this run finish for a clean baseline, then immediately run TODO 11b with VICReg.

**Status:** [ ] Running on Odra — WandB: `diffusion_H512L6C128D2_20260226_155541`

---

## TODO 12: Fix Diffusion ELBO Loss Weighting + Raise t_min (Priority: HIGHEST, Effort: 0.5 day) DONE ✅

*Maps to roadmap A10. Small code fix with potentially significant impact.*

**Done date: 2026-02-26**

**Implementation:** Correct per-token 1/t ELBO weighting (not the batch-level approximation proposed in the diagnosis). Each masked token's CE loss is weighted by `1/t_sample.clamp(min=0.1)`, then normalized by `sum(weights)`. This properly reweights across noise levels — low-t samples get higher per-token weight, compensating for fewer masked positions.

**Changes:**
- `nn/concept_encoder_diffusion.py`: added `elbo_weight` parameter (default True), per-token 1/t weighting, `t_min` default 0.1→0.3
- `training/train_diffusion.py`: added `elbo_weight` arg, `t_min` default 0.1→0.3, logged to WandB
- `scripts/train_diffusion_multigpu.sh`: updated to L6 config + ELBO + t_min=0.3
- `tests/test_diffusion.py`: 10 tests (all pass)

**Validation:** Local 50-step test on wikitext-2: loss 9.5→2.3, grad_norm stable (peaked 46.4 step 1, then <7), eval_loss 2.78. No gradient explosion.

**Full rationale:** [diffusion_diagnosis_20260226.md](../4_Research_Notes/diffusion_diagnosis_20260226.md) (Causes 2-3: missing ELBO weighting, t_min too low)

**Status:** [x] Done — see CHANGELOG `[2026-02-26]`

---

## TODO 11b: L6 Diffusion + VICReg + t_regs_mst — Regularized Experiment (Priority: HIGHEST, Effort: 1 GPU-day)

*Maps to roadmap A5 + A9. Immediate follow-up to TODO 11 baseline.*

**Goal:** Test whether adding VICReg (cross-batch dimensional health) + t_regs_mst (within-sample concept diversity) regularization prevents the concept collapse observed in TODO 11 baseline (rank 5.45/128 at step 20k) while preserving MLM/diffusion quality.

**Config:** Same as TODO 11, except:
- `--concept_losses "vicreg t_regs_mst"` — VICReg (variance+covariance) + MST diversity
- `--loss_weighting fixed` — NOT Kendall-Gal (proven to suppress MLM, Feb 19)
- `--loss_weight 0.02` — small weight (Feb 21 showed 0.1 too aggressive, 0.02 = ~1.4% gradient contribution)
- `--concept_loss_warmup_steps 2000` — ~4% of steps, lets model establish gradients first
- Keep: `--elbo_weight True`, `--t_min 0.3`

**Why this will work (unlike Feb 19/21 `combined` loss):**
1. `t_regs_mst` operates **within-sample** (pairwise distances on `[B, C, C]` via `torch.cdist`), directly targeting the collapse mode we observe. The old `combined` loss operated across-batch and missed intra-sample collapse.
2. Fixed weight 0.02 is 5x smaller than the 0.1 that degraded MLM.
3. Warmup prevents early interference when concept representations are still random.
4. VICReg adds covariance decorrelation (missing from `combined`).

**Implementation:** Done — see CHANGELOG `[2026-02-27]`. Files: `nn/loss_manager.py`, `training/train_diffusion.py`, `scripts/train_diffusion_multigpu.sh`. 26 tests pass.

**Decision logic:**
- Rank > 30/128 AND task_loss < 3.0 → regularization works, proceed with this config
- Rank > 30/128 BUT task_loss > 3.5 → weight too aggressive, reduce to 0.01
- Rank < 20/128 → t_regs_mst insufficient, try Slot Attention or prefix generation

**Machine:** Odra or Polonez, immediately after TODO 11 finishes.

**Status:** [ ] Implementation done, waiting for TODO 11 baseline to finish

---

## TODO 13: Prefix Generation Training (Priority: HIGHEST, Effort: 3 days code + 5 GPU-days)

*Maps to roadmap A11. The most strategically important new experiment.*

**Motivation (SODA principle for text):** Current training asks the model to encode text X and reconstruct X (self-reconstruction). This permits surface-level hashing through the concept bottleneck. SODA (Hudson, CVPR 2024) shows that bottleneck diffusion models learn semantic representations only when the decoder generates DIFFERENT content than the encoder saw.

**Design:**
1. Split each training document at a random position (30-50% prefix, 50-70% suffix)
2. **Encoder:** receives clean prefix tokens → produces concepts
3. **Decoder:** generates suffix tokens via diffusion, conditioned on concepts
4. Concepts must capture the *semantic gist* of the prefix to enable coherent continuation
5. This directly trains for the inference-time use case: encode input → generate output

**Implementation plan:**
- New data collator `DataCollatorForPrefixGeneration` in `training/data_collators.py`
- New or extended model class `ConceptEncoderForPrefixDiffusion` in `nn/concept_encoder_diffusion.py`
- New training script `training/train_prefix_diffusion.py`
- New shell script `scripts/train_prefix_diffusion_multigpu.sh`

**Key decisions:** Encoder sees CLEAN prefix (matches inference), decoder uses diffusion on suffix tokens, position embeddings relative to suffix start, ELBO-weighted loss on suffix only.

**Evaluation:**
1. Suffix reconstruction loss (primary metric for generation capability)
2. Concept analysis: effective rank, pairwise similarity
3. STS-B Pearson (ViaDecoder fine-tuning)
4. Zero-shot STS-B (cosine similarity of prefix concepts)

**Full rationale:** [diffusion_diagnosis_20260226.md](../4_Research_Notes/diffusion_diagnosis_20260226.md) (Cause 4: self-reconstruction permits surface hashing; Fix 3: prefix generation)

**Status:** [ ] Not started

---

## TODO 14: Backbone Init from SmolLM2-135M (Priority: HIGH, Effort: 3 days code + 5 GPU-days)

*Replaces TODO 9 with updated rationale. Maps to roadmap B3.*

**Motivation (Coconut insight):** Coconut (Meta, 2024) shows that latent reasoning is much easier when the model already understands language. Training language understanding AND concept compression AND reasoning from scratch is too many things at once. SmolLM2-135M provides language understanding from 11T-token pretraining at zero cost.

**Action:**
1. Implement weight loading from SmolLM2-135M into ConceptEncoder's token embeddings and (optionally) as initialization for concept cross-attention layers
2. Verify dimension compatibility (SmolLM2-135M: H=576, 30 layers; may need projection)
3. Run baseline prefix generation + TSDAE with pretrained init vs random init
4. Compare concept quality, convergence speed, STS-B

**Status:** [ ] Not started

---

*Plan updated: 2026-02-27*
*Aligned with: [roadmap.md v4](roadmap.md) (2026-02-26)*
*Next review: after TODO 11 finishes → evaluate STS-B → launch TODO 11b (VICReg)*
*New (2026-02-27): added TODO 11b (VICReg + t_regs_mst regularization experiment)*
*New (2026-02-26): added TODO 11-14 based on diffusion diagnosis analysis ([diffusion_diagnosis_20260226.md](../4_Research_Notes/diffusion_diagnosis_20260226.md))*
*Related: [mlm_perceiver_diagnosis_20260221.md](../4_Research_Notes/mlm_perceiver_diagnosis_20260221.md), [diffusion_L2_eval_20260225.md](../2_Experiments_Registry/run_reports/diffusion_L2_eval_20260225.md)*
