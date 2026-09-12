# E18 pilot — stage A (125M Perceiver AR v2, seq 8k, 1.0B tokens) — `perceiver_ar_perceiver_H768L1g1s12N2048_20260907_080943`

**Date:** 2026-09-07 (interim: dense control and stage B still running)
**Machine:** Polonez (4× RTX 3090; train 4 GPU; probes on GPU 0 next to the dense run)
**Run ID:** `perceiver_ar_perceiver_H768L1g1s12N2048_20260907_080943`
**WandB:** project `ksopyla/MrCogito`, job type `train_perceiver_ar_causal_lm`, same run name
**Raw log:** `Cache/logs/e18_stageA_20260907_080923.log` · `Cache/logs/training_20260907_080939.log`
**Last checkpoint (= stop point):** `Cache/Training/perceiver_ar_perceiver_H768L1g1s12N2048_20260907_080943/checkpoint-9030` (1.0B tokens; no `final/` — stopped by design, see below)
**Git commit:** `5c0e3dd` (Polonez tree at launch 08:09 UTC; later commits that day touched only pretokenization and docs)
**Spec:** [E18](../../experiments_specs/done_failed/E18_perceiver_ar_v2_baseline.md) · [plan](../../experiments_specs/done_failed/E18_perceiver_ar_v2_baseline_plan.md)

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

## Dense control (1.0B tokens, same budget) — P1 and the 32k baselines

Run `perceiver_ar_dense_H768L1g1s12N2048_20260907_193351` (`PAR_MODE=dense`, all 14 layers full-causal, otherwise
identical), 9,033 steps, finished 2026-09-08 06:00 UTC, ~27.4k tok/s.

| | Perceiver (stage A) | dense control |
|---|---|---|
| eval loss @ 1.0B tokens | 3.790 | **3.786** |
| train loss @ 31M / 44M / 91M tokens | 6.144 / 5.800 / 4.958 | 6.151 / 5.816 / 4.966 |
| throughput, 4×3090, seq 8k | 27.95k tok/s | ~27.4k tok/s |

**P1 (≤ 1% gap at equal tokens): ✅ 0.1%.** P4 (≥ 60% of dense throughput): ✅ ≈ 1.02×.

Same 32k probes on the dense `final` (8 PG-19 rows ≥ 32k; both models trained at 8k only):

| bucket | [0, 2k) | [2k, 8k) | [8k, 16k) | [16k, 32k) |
|---|---|---|---|---|
| Perceiver CE | 3.192 | 2.940 | 2.645 | **2.611** |
| dense CE | 3.207 | **2.843** | **2.567** | 2.664 |

Dense (full attention, RoPE extrapolated 4× past training) is ~3% better inside 16k but *turns up* in the
16k–32k bucket; the Perceiver keeps improving with distance. Passkey: dense 0.375 / 0.35 / 0.15 / 0.075 / 0.0
at 2k–32k, i.e. identical to the Perceiver within noise. Stage B (32k training) is measured against the
Perceiver's own 8k row above (P3: ≥ 2% relative on the ≥ 8k buckets).

## Gate status so far

| gate | status | evidence |
|---|---|---|
| P1 equivalence + parity | ✅ | tests (flex == sdpa == naive masks; packed == unpacked per-token loss); eval loss 3.790 vs dense 3.786 at 1.0B tokens (0.1%) |
| P2 copy task | ⏸ deferred | two runs stuck at ln(256); checkpoint-500 kept; rerun when GPUs are free |
| P3 long-context use | ⏳ | needs stage B vs these baselines |
| P4 architecture tax | ✅ | 27.95k tok/s (perceiver) vs ~27.4k tok/s (dense, full run) at 8k → ≈ 1.02×, far above the 60% floor. At 8k with block 2048 the attention savings are small, so parity is the expected outcome; the real saving shows at 32k+ |

## P2 copy task — what the tiny CPU study says (2026-09-07)

`verification/e18_copy_tiny.py` (0.97M-param models, context 130, batch 32, AdamW 3e-3, CPU; the same task
construction as the 32k dataset):

| variant | task | result @600 steps |
|---|---|---|
| perceiver (pre 1@16, global 1, stack 2@32) | mirror | floor (5.47) — also at 2,000 steps |
| dense (4 full layers) | mirror | floor — also at 1,500 steps |
| dense | plain copy | learns (CE 2.6, still falling) |
| perceiver pre@16 | plain copy | floor — also at 2,000 steps |
| perceiver pre@64 or pre@128 (pre-layer sees the offset) | plain copy | learns fast (CE 0.75–2.2) |
| perceiver pre@16, stack window 128 (stack full) | plain copy | floor |
| perceiver pre@16 + value embeddings on the global layer | plain copy | learns, slowly (CE 3.0 @600, 2.2 @1,500) |
| perceiver pre@16 + `swa_sink` | plain copy | floor |

Reading: (1) mirrored copy is the wrong gate for RoPE-only models at this budget (the dense control fails
too); (2) plain copy is solved instantly by whichever layer can *see* the offset **and has a value
embedding** (token identity in V); the global layer without a value embedding never learned it, with one
it learns but slowly. The pilot's `PAR_VALUE_EMBED_LAYERS=0,4,8` leaves the global layer (index 1) without
one — fix for the P2 rerun and worth carrying into the main run. Spec P2 amended accordingly (plain copy
at 32k, offset 16k). Builder: `scripts/build_copy_task_dataset.py --task copy`.

**Real-scale confirmation (accidental early run, 2026-09-08):** a waiter-script bug (checked the stage-B
exit *file's existence*, not its exit *code*) fired the corrected P2 config — plain copy at 32k, value
embeddings on layers 0/1/4 — right after stage B's first (failed) attempt, well before the intended
trigger. The training itself was valid: fresh 6-layer model, Muon, real 32k data.

| epoch | eval loss |
|---|---|
| 0.13 | 5.530 (still at the uniform floor) |
| 0.27 | 0.0047 |
| 0.40 | 0.00067 |
| 0.53 | 0.00018 |
| 0.66 | **0.000087** |

It crashed at epoch 0.66 on the single corrupt label (see below), before completing its 1.0-epoch budget.
This confirms the tiny-study diagnosis at pilot scale: **P2 (amended) converges to near-zero loss fast**
once the retrieving layer has a value embedding. The waiter bug is fixed (`Cache/jobs/e18_stageB_then_copy.sh`
gates correctly); the properly-triggered rerun after stage B will give a clean, complete accuracy number.

## Operational notes

- The queued waiters for dense / stage B grepped the log for an exit marker that was only echoed to the
  terminal; they would never have fired. Replaced by `Cache/jobs/e18_dense_then_stageB.sh` (dense 1B →
  stage B 32k warm-started from checkpoint-9030) with `Cache/logs/e18_{dense,stageB}.exit` markers.
- Flex `create_block_mask` in eager mode materialised ~8 GB of int64 index grids at 32k (probe OOM next
  to the training run); now compiled on CUDA for every mask (`b222723`). Passkey/copy probes project
  the head in chunks from `hidden_states()` instead of building S×V logits (`16a2653`).
- Dense control equal-token check-ins (train loss): 31M tokens 6.144 vs 6.151; 44M tokens 5.800 vs 5.816
  (perceiver vs dense) — parity, as P1 expects.
- **Stage B first attempt died (2026-09-08 06:35)**: NCCL watchdog timeout (30 min) while rank 0 computed the
  32k manifest's sequence-length cache single-process (~37 min on this host). Fix: `DDP_TIMEOUT=10800` in
  `launch_e18.sh`, cache precomputed out-of-band, chain relaunched (`Cache/jobs/e18_stageB_then_copy.sh`).
- **P2 plain-copy first attempt died**: one label value of 2^31−100 among 983M in the freshly built arrow
  dataset (single corrupt int32; input_ids clean). Collator now ignores out-of-vocab precomputed labels
  (−100) with a warning instead of aborting.

## Next

Dense control finishes ≈ 06:00 UTC 2026-09-08; stage B (0.5B tokens at 32k) ≈ 6 h after. Then: buckets +
passkey on stage B `final`, the P1 equal-token table, throughput at 32k vs dense, verdict per the spec's
kill criteria, and the AWS one-pager [E18_main_run_aws_plan](../../experiments_specs/ahead/E18_main_run_aws_plan.md).
