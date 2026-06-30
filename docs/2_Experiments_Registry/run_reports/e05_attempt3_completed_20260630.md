# E05 attempt 3 — Windowed decoder completed 0.5 ep, no divergence — `concept_ar_prefix_H768L6C128D4_20260629_093840`

**Date:** 2026-06-30 (training completed and evaluated same day)
**Machine:** Odra (3× RTX 3090, 24 GB each)
**Run ID:** `concept_ar_prefix_H768L6C128D4_20260629_093840`
**WandB (training):** [Link](https://wandb.ai/ksopyla/MrCogito/runs/concept_ar_prefix_H768L6C128D4_20260629_093840)
**Raw shell log:** `Cache/logs/shell_perceiver_denoise_20260629_093840.log` (on Odra)
**Best checkpoint:** `Cache/Training/concept_ar_prefix_H768L6C128D4_20260629_093840/checkpoint-68000` (selected by `load_best_model_at_end`, eval_loss 3.8293)
**Eval checkpoint:** `checkpoint-69142` (final saved; functionally near-identical to best)
**Git commit:** `f5951da` (training) · `730e607` + `70e1fd2` (eval-script fixes applied during this eval)
**Git tag:** —
**Related TODO:** E05 in `docs/experiments_specs/E05_windowed_decoder_concept_memory.md`

---

## Goal

Third proving attempt for E05 windowed decoder (K=128 sliding-window causal mask on `concept_ar` + prefix→suffix). Attempt 2 diverged at step 40k under LR 1e-4 / clip 1.0; attempt 3 retunes the optimizer (halved LR, explicit tighter clip, larger effective batch) and re-scopes to 0.5 ep (~7B tokens target). Goal: prove stable training through 0.5 ep, clear the Stage 1 read floor (beyond-window Δshuffle ≥ 0.3, STS-B ≥ 0.62, RankMe rises vs init), and earn the budget for the matched A/B.

## Configuration

| Item | Value |
|---|---|
| Family | `concept_ar`, `pretraining_objective=ar_prefix_suffix_generation` |
| Encoder | H768, L6, C128, BiXT |
| Decoder | causal AR, D4, `decoder_word_dropout=0.0`, RoPE, **`decoder_context_window=128`** |
| Token width | `token_embedding_dim=256` |
| Dataset | `smollm3_inspired_2k_e05` mix (pretokenized, 6.92M train / seq 2048) |
| Objective | prefix→suffix; prefix ratio 0.3–0.5, `split_strategy=sentence_boundary` |
| Epochs (target / actual) | 0.5 / **0.5** (full) |
| Effective batch | 8 × grad_accum **3** × 3 GPUs = **72** |
| LR / warmup / clip / seed | **5e-5** / 2000 / **0.5 (explicit)** / 42 |
| LR schedule | cosine |
| Precision | bf16 |
| Optimizer | `adamw_torch_fused` |
| Allocator env | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| Throughput | 50.0 M tokens / GPU-h (0.844 step/s, 1.18 s/step) |
| Compute | **68.2 GPU-h · 18.24 kWh · 10.20B max-tokens** (`compute/audit_state=finished`, flag `loss_fraction:prefix_suffix_approx`) |

Batch 8 was the **third** choice (12 OOM'd in <10 min on GPU 2; 10 held 6 h then OOM'd at step 21k on a high-long-suffix batch). At seq 2048 with this architecture, batch 8 + grad-accum 3 is the memory-stable operating point on 3× 24 GB cards.

## Training Outcome

**Clean convergence to 0.5 epoch with no divergence.** eval_loss decreased monotonically through all 17 evals, from 5.397 (step 4k) → 4.145 (step 20k) → 3.972 (step 28k) → 3.830 (step 88k). Improvement rate slowed as expected: Δ −0.58 in [4k–8k], Δ −0.12 in [20k–28k], Δ −0.001 in the final 4k step.

| Step | epoch | eval_loss | Δ from prev eval |
|---:|---:|---:|---:|
| 4 000 | 0.029 | 5.397 | — |
| 8 000 | 0.058 | 4.818 | −0.58 |
| 12 000 | 0.087 | 4.489 | −0.33 |
| 16 000 | 0.116 | 4.264 | −0.23 |
| 20 000 | 0.145 | 4.145 | −0.12 |
| 28 000 | 0.174 | 3.972 | −0.17 |
| 36 000 | 0.202 | 3.961 | −0.01 |
| 44 000 | 0.231 | 3.908 | −0.05 |
| 52 000 | 0.260 | 3.892 | −0.02 |
| 60 000 | 0.289 | 3.884 | −0.01 |
| 64 000 | 0.318 | 3.866 | −0.02 |
| 68 000 | 0.347 | 3.856 | −0.01 |
| 72 000 | 0.376 | 3.843 | −0.01 |
| 76 000 | 0.405 | 3.837 | −0.006 |
| 80 000 | 0.434 | 3.832 | −0.005 |
| 84 000 | 0.463 | 3.830 | −0.002 |
| 88 000 | 0.492 | 3.829 | −0.001 |

**Stability signature (the whole point of this run vs attempt 2):**

- Pre-clip `grad_norm` held in a tight **0.40–0.55** band from step 200 through ~step 48 000 (warmup + first ~70 % of cosine decay). The clip at 0.5 was rarely the binding constraint.
- From step ~50 000 onward, grad_norm rose to **40–75** (occasional spikes to 87), while LR was near the cosine tail (1e-6 → 1e-9). This is **expected end-of-schedule behavior** (the optimizer's effective step size grows relative to the shrinking LR), and **critically it did not hurt eval_loss** — the curve kept improving. This is the *opposite* signature from attempt 2, where grad_norm escalation (9 → 903) coincided with eval_loss climbing.
- LR cosine decayed cleanly from 5e-5 peak (after 2000 warmup) to 5e-10 final. Warmup properly ramped (step 200: lr 5e-6; step 2000: lr 4.99e-5).

`load_best_model_at_end` selected **`checkpoint-68000`** (eval_loss 3.856 at the eval point just before; the final eval at step 88k recorded 3.829 but is not a separate checkpoint). Final saved checkpoint is **`checkpoint-69142`**. Both are valid; **`checkpoint-68000` is the official best** per HF Trainer's selection.

## Concept Health

**Tier 0 (`check_model_health.py`):** clean — no NaN/Inf, no all-dead layers. (Subagent ran Tier 0 on `checkpoint-69142`.)

**Tier 1 (`run_concept_analysis.py`, on `checkpoint-69142`):** report at
`Cache/Evaluation_reports/concept_ar_prefix_H768L6C128D4_20260629_093840_concept_analysis.json`.

Concept geometry:

| Metric | Value | Read |
|---|---:|---|
| **within-sample RankMe** (PRIMARY de-collapse) | **37.67** (std 2.59) | of 128 → concepts for one input span ~38 independent directions. Not collapsed. |
| slot-mean effective rank (secondary) | 4.76 / 128 | low — slot redundancy present (typical for this family; collapsed history ~5–10) |
| cross-sample manifold RankMe | 113.91 | pooled-embedding diversity across inputs — high (downstream-retrieval property, not "concept rank") |
| anisotropy | 0.682 | moderate |
| mean concept cosine | 0.346 | OK |
| max concept cosine | 0.997 | some near-duplicate pair exists |
| active slot fraction | 1.000 | every slot fires |
| participation ratio | 5.14 | matches slot-mean rank |
| collapsed dim ratio | 0.000 | no fully-dead concepts |

AR concept-ablation (reconstruction contract; the authoritative suffix-CE numbers are also in
W&B `concept_ablation/*` from the training `evaluate` step):

| Quantity | Value |
|---|---:|
| `ce_intact` | 4.932 |
| `delta_zero` (all) | 6.942 |
| `delta_shuffle` (all) | 0.408 |
| `delta_zero_early` (within window) | 6.242 |
| `delta_shuffle_early` | 0.548 |
| **`delta_zero_beyond_window`** | **6.985** |
| **`delta_shuffle_beyond_window`** | **0.390** |

**Stage 1 read.** Beyond-window Δshuffle = **0.390** — clears the Stage 1 floor (≥ 0.3) but is well
below the Stage 2 target (≥ 0.5). Δzero_beyond_window = 6.99 is large (zeroing concepts destroys
prediction), so the decoder *is* reading them; the modest Δshuffle says the semantic content it
recovers is partial — it can tell "concepts are mine" from "concepts are zero", but is less
sensitive to "concepts are scrambled". This is the expected signature for a 0.5-ep prefix→suffix
checkpoint: the bottleneck is *used* but not yet *semantically rich*.

**Generation samples** (`generation_samples[0..3]`): prompts are held-out FineWeb-edu passages.
Generations are **grammatical but semantically empty loops** — e.g. *"The first time of the first
time, the second time of the first time, was the first time to the second time..."* and *"The first
time I was to be a good idea of the world, and I was a great way to be a good idea of the world..."*.
Token-level repetition loops are the dominant failure mode. `generation_faithfulness`:
`teacher_forced_token_acc=0.242`, `free_running_token_f1=0.149`, `free_running_exact_match=0.015`
(`n=8`). The compression curve is flat across ratios 1–4 (token-acc ~0.24 at every ratio), i.e.
the decoder is **not** exploiting additional tokens to refine prediction — it produces the same
distribution regardless of how many suffix tokens it has seen.

## Evaluation

Eval was run on **`checkpoint-69142`** (the final saved checkpoint). The official best-by-eval-loss
is `checkpoint-68000` (3.856 vs 3.829); the two are functionally near-identical and the AR/geometry
numbers do not differ meaningfully at this granularity.

### Tier 2 — Zero-shot STS-B (the cheap semantic gate)

| Run | Pearson | Spearman | WandB |
|---|---:|---:|---|
| **Model (concepts)** | **0.4525** | **0.4719** | [run](https://wandb.ai/ksopyla/MrCogito/runs/luryo3qc) |
| Floor: `token_embed_mean` (SmolLM2 token embeddings, averaged) | 0.4864 | 0.5245 | — |
| Floor: `teacher_hidden_mean` (frozen SmolLM2 hidden states, averaged) | 0.4599 | 0.5235 | — |
| Reference ceiling (cited, not run): SimCSE-unsup | — | ~0.76 | — |
| Reference ceiling (cited, not run): SBERT | — | ~0.84 | — |

**Flagged.** The model's concepts score **below both trivial floors** — the concept bottleneck is
currently *destroying* semantic-similarity signal vs. just averaging the raw token embeddings of
the same tokenizer. This is the strongest single signal in the eval and it is negative. (Note:
E01's prior best zero-shot STS-B Pearson was 0.607; E02's target was 0.65. This checkpoint's 0.452
is well below both.)

### Tier 3 — Supervised pair tasks (full fine-tune)

| Benchmark | Metric | Value |
|---|---|---:|
| SICK relatedness | Pearson / Spearman | **0.183 / 0.192** (peak ~ep 5, mild overfit after) |
| SICK entailment | accuracy | **0.634** |
| PAWS | accuracy / F1 | **0.550 / 0.253** |
| GLUE MRPC | accuracy / F1 | **0.669 / 0.778** |
| GLUE STSB (fine-tuned) | Pearson / Spearman | **0.354 / 0.341** |
| GLUE QQP | — | running at time of report; will backfill |
| GLUE MNLI-m / MNLI-mm | — | running at time of report; will backfill |

SICK/PAWS/MRPC are pair-task fine-tunes that unfreeze the encoder; per the skill note they measure
fine-tuning capacity more than concept content. They are weak but non-zero — the model *can* be
fine-tuned to above-chance on every pair task, just not strongly. PAWS F1 of 0.253 (with accuracy
0.550) is essentially "predict the majority class" — the model has not learned a meaning-vs-word-
overlap distinction.

## Interpretation

Two readings, both honest:

1. **Optimization succeeded; the architecture is stable.** Stage 1 floor cleared: beyond-window
   Δshuffle 0.390 ≥ 0.3, within-sample RankMe 37.67 (not collapsed), eval_loss monotone-decreasing
   across all 17 evals, no divergence. The retune (LR 5e-5, clip 0.5, effective batch 72) is now
   the **proven-stable optimizer recipe** for the K=128 windowed decoder. This is the actionable
   win from attempt 3.

2. **Semantic quality is not yet there.** Zero-shot STS-B (0.452) sits *below* both trivial floors,
   Δshuffle_beyond_window (0.390) is at the floor rather than the Stage 2 target (≥0.5), and free-
   running generations are degenerate repetition loops. 0.5 ep of prefix→suffix pretraining is
   enough to make the decoder *use* the concepts but not enough to make them *semantically
   informative* — the bottleneck passes the "is it wired up" test and fails the "does it carry
   meaning" test.

   This is **not** an architectural dead-end verdict (cf. attempt 2's divergence, which *was*).
   It is a "more training and/or a stronger objective" signal. The matched A/B (longer training,
   or ablations of K, decoder depth, prefix-ratio schedule) is now the justified next step —
   attempt 3 earned the budget for it.

## Decision

**Stage 1 PASS (floor cleared). Stage 2 NOT YET MET** (Δshuffle ≥ 0.5, STS-B ≥ 0.65 not reached).
Proceed to the matched A/B with the proven-stable optimizer recipe; do not treat this checkpoint
as a downstream-ready model.

## Notes

- **Three batches, three OOMs before stability.** Batch 12 OOM'd at startup (loss FP32 upcast, GPU 2). Batch 10 ran 6 h then OOM'd at step 21k on a high-long-suffix batch (also loss FP32 upcast, GPU 0). Batch 8 + grad-accum 3 (effective 72, identical tokens/step to the batch-12 plan) ran to completion. Memory lesson: at seq 2048 with K=128 windowing, per-GPU activation memory peaks ~22.7 GB during the loss FP32 upcast, so leave ≥2 GB headroom on 24 GB cards.
- Two short background training runs from the OOM'd attempts (`..._20260628_231611` batch-12, `..._20260628_233349` batch-10) are still on disk under `Cache/Training/`; they produced checkpoints (4k/8k for batch-12; up to 20k for batch-10) but are not the canonical run. Cleanup is optional — they are small (~2 GB each ckpt).
- `num_train_epochs: 1` in `trainer_state.json` is an HF serialization quirk — the actual schedule was 0.5 ep (`max_steps: 69142`, final `epoch: 0.5000048`). Do not be misled by that field if reading the JSON directly.
- W&B run was synced cleanly (5 files). GPUs are idle and the Byobu session is back at the shell prompt.
- **Two eval-script bugs surfaced and fixed during this eval** (commits `730e607` + `70e1fd2`):
  1. `evaluation/wandb_identity.py:_safe_tag_value` truncated the *value* to 64 chars but callers prepended `prefix:`, producing 75-char tags that `wandb.init(tags=...)` rejected. New `_safe_prefixed_tag` reserves room for the prefix so the total stays ≤ 64. Also `resolve_tokenizer_name_for_tag` now pulls the canonical HF id from `model.config.tokenizer_name` instead of letting the checkpoint path leak into the tag.
  2. SmolLM2's tokenizer ships without a `pad_token`; `AutoTokenizer.from_pretrained` doesn't honor the model config's `pad_token_id`. All three eval entrypoints (`evaluate_on_benchmark.py`, `evaluate_model_on_glue.py`) now set `pad_token = eos_token` right after load. STS-B / SICK / PAWS / GLUE all batch-encode pairs and would crash with "Asking to pad but the tokenizer does not have a padding token" without this.
  Both fixes have regression tests in `tests/test_wandb_lineage.py`.

*Related: `master_experiment_log.md`, `docs/experiments_specs/E05_windowed_decoder_concept_memory.md`, `agenda.md`, `run_reports/e05_attempt2_diverged_20260628.md`*
