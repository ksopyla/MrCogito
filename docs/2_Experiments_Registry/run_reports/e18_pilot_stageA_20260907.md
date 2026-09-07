# E18 pilot — stage A (125M Perceiver AR v2, seq 8k, 1.0B tokens) — `perceiver_ar_perceiver_H768L1g1s12N2048_20260907_080943`

**Date:** 2026-09-07 (interim: dense control and stage B still running)
**Machine:** Polonez (4× RTX 3090; train 4 GPU; probes on GPU 0 next to the dense run)
**Run ID:** `perceiver_ar_perceiver_H768L1g1s12N2048_20260907_080943`
**WandB:** project `ksopyla/MrCogito`, job type `train_perceiver_ar_causal_lm`, same run name
**Raw log:** `Cache/logs/e18_stageA_20260907_080923.log` · `Cache/logs/training_20260907_080939.log`
**Last checkpoint (= stop point):** `Cache/Training/perceiver_ar_perceiver_H768L1g1s12N2048_20260907_080943/checkpoint-9030` (1.0B tokens; no `final/` — stopped by design, see below)
**Git commit:** `5c0e3dd` (Polonez tree at launch 08:09 UTC; later commits that day touched only pretokenization and docs)
**Spec:** [E18](../../experiments_specs/ahead/E18_perceiver_ar_v2_baseline.md) · [plan](../../experiments_specs/ahead/E18_perceiver_ar_v2_baseline_plan.md)

**Artifacts:**
- Position-bucket CE at 32k on PG-19 eval rows: `Cache/eval/e18/stageA_ckpt9030_buckets32k.json`
- Passkey retrieval 2k–32k: `Cache/eval/e18/stageA_ckpt9030_passkey.json`

---

## Goal

Stage A of the pilot: train the 125M Perceiver AR v2 (one global read, window-2048 stack) at seq 8k on
the long-document mix, to (a) provide the equal-token comparison for the matched dense control (P1/P4)
and (b) the warm start for stage B (32k), whose position buckets and passkey decide P3.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`, `par_mode=perceiver`: 1 pre-layer @ window 512 → 1 full-causal global layer → 12 stack layers @ window 2048 |
| Width | H=768, FF=2048 (SwiGLU), 6 query heads × 2 KV heads, head_dim 128, RoPE θ=5e5, NoPE every 4th stack layer |
| Input / head | tiny embedding e=256 + hashed 2/3-gram tables (65,536 buckets), value embeddings on layers 0/4/8, untied soft-capped head (cap 30), z-loss 1e-4, Liger fused CE |
| Attention | FlexAttention block masks, pad multiple 2048, gradient checkpointing |
| Data | `e18_pilot_longdoc_v1` (PG-19 / FinePDFs / FineWeb-Edu / stack-edu-python), SmolLM3 tokenizer (Llama-3 vocab 128,256), `BATCH_PACKING_MODE=length_group`, seq 8192, one document per row |
| Optimizer | Muon lr 0.01 (AdamW side 2e-4), wd 0.1, clip 0.5, constant-with-warmup (500 steps) |
| Budget | planned 2.0B tokens (18,064 steps of 8 × 4 GPU × accum 2 = 64 rows, ~110.7k real tokens/step); **stopped at step 9,030 = 1.0B tokens** |
| Throughput | **27.95k real tok/s** over 4 GPUs (mean of 515 log windows) ≈ 6.99k tok/s/GPU; pad ratio ≈ 0.05 |

## Training outcome

Stable. Loss 11.85 → 3.85 (train), no NaN, one early grad-norm spike (max 5.46 pre-clip, during warmup),
then grad norm ≈ 0.75. Eval loss (PG-19-heavy held-out, 256 rows):

| tokens | 0.20B | 0.40B | 0.60B | 0.80B | 1.00B |
|---|---|---|---|---|---|
| eval loss | 4.386 | 4.065 | 3.933 | 3.838 | **3.790** |

**Why stopped at 1.0B:** the schedule is constant-LR after warmup, so the 1.0B checkpoint is a valid
1.0B-token run; halving the stage freed the GPUs for the dense control (also 1.0B) and stage B the same
day instead of two days later. The dense control runs the identical 9,033-step budget.

## Probes on checkpoint-9030 (8k-trained model evaluated at 32k)

Position-bucket CE on the 8 PG-19 eval rows ≥ 32k tokens (small sample; stage B will use the same rows):

| bucket | [0, 2k) | [2k, 8k) | [8k, 16k) | [16k, 32k) |
|---|---|---|---|---|
| CE | 3.192 | 2.940 | 2.645 | 2.611 |

CE keeps falling beyond the 8k training length: the window layers plus the single global read
extrapolate without any 32k training. Whether stage B lowers the ≥ 8k buckets by the P3 margin (≥ 2%
relative) is the gate.

Passkey (5-digit key, teacher-forced argmax, 5 depths × 8 trials):

| context | 2k | 4k | 8k | 16k | 32k |
|---|---|---|---|---|---|
| accuracy | 0.40 | 0.275 | 0.15 | 0.075 | 0.00 |

Low, as expected for 125M at 1B tokens with no retrieval-style data; this is the baseline stage B must
beat (P3 asks for ≥ 90% at 32k after the 32k stage, which is a hard bar at this scale — read the delta,
not only the absolute).

## Gate status so far

| gate | status | evidence |
|---|---|---|
| P1 equivalence tests | ✅ | `tests/test_perceiver_ar_lm.py`, `tests/test_packed_dataset.py` (flex == sdpa == naive masks; packed == unpacked per-token loss) |
| P2 copy task | ⏸ deferred | two runs stuck at ln(256); checkpoint-500 kept; rerun when GPUs are free |
| P3 long-context use | ⏳ | needs stage B vs these baselines |
| P4 architecture tax | ✅ (interim) | 27.95k tok/s (perceiver) vs 27.0–27.4k tok/s (dense, first hour) at 8k → ≈ 1.02×, far above the 60% floor. At 8k with block 2048 the attention savings are small, so parity is the expected outcome; the real saving shows at 32k+ |

## Operational notes

- The queued waiters for dense / stage B grepped the log for an exit marker that was only echoed to the
  terminal; they would never have fired. Replaced by `Cache/jobs/e18_dense_then_stageB.sh` (dense 1B →
  stage B 32k warm-started from checkpoint-9030) with `Cache/logs/e18_{dense,stageB}.exit` markers.
- Flex `create_block_mask` in eager mode materialised ~8 GB of int64 index grids at 32k (probe OOM next
  to the training run); now compiled on CUDA for every mask (`b222723`). Passkey/copy probes project
  the head in chunks from `hidden_states()` instead of building S×V logits (`16a2653`).
- Dense control equal-token check-ins (train loss): 31M tokens 6.144 vs 6.151; 44M tokens 5.800 vs 5.816
  (perceiver vs dense) — parity, as P1 expects.

## Next

Dense control finishes ≈ 06:00 UTC 2026-09-08; stage B (0.5B tokens at 32k) ≈ 6 h after. Then: buckets +
passkey on stage B `final`, the P1 equal-token table, throughput at 32k vs dense, verdict per the spec's
kill criteria, and the AWS one-pager [E18_main_run_aws_plan](../../experiments_specs/ahead/E18_main_run_aws_plan.md).
