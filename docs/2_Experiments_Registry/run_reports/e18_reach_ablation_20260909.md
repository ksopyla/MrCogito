# E18 — reach ablation: does the single global read carry long-range information? (2026-09-09)

**Date:** 2026-09-09 · **Machine:** Polonez, GPU 0, no training · **Git:** `1a89736` (probe), `515820e` (global_positions knob)
**Spec:** [E18](../../experiments_specs/ahead/E18_perceiver_ar_v2_baseline.md) · stage A/B/dense: [e18_pilot_stageA_20260907.md](e18_pilot_stageA_20260907.md)
**Artifacts:** `Cache/eval/e18/reach/*.json` on Polonez (per-row bucket CE for every window) · job `Cache/jobs/e18_reach.sh` · log `Cache/logs/e18_reach.log`

---

## Why this measurement

P3 as written compares position-bucket CE across two *runs* (8k stage vs 32k stage). That comparison is
confounded twice: (1) by optimizer history — stage B warm-started weights only, at Muon lr 0.01 with a
500-step re-warmup (launcher bug, fixed in `ec6d99b`) and no decay, and ended **uniformly ~2% worse at
every position including [0,2k)**, an optimization regression, not a context result; (2) by the intrinsic
"later in a book is easier" gradient, which makes CE fall with position whether or not the model uses far
context (the dense 8k control shows the same fall).

`PerceiverARLM.reach_override(W)` removes both: **same tokens, same weights, only the global read's window
changes** (`full` → `swa(W)`). Δ(W) = CE(W) − CE(full), paired per row, with a standard error. Positions
p < W see identical keys and must give Δ = 0 exactly (unit-tested; observed 0.0000 on flex/bf16 too).

**Instrument checks (all passed):** 39 CPU tests + the CUDA flex ≡ sdpa test on Polonez; the P2 copy
model as positive control at the 16k copy offset:

| global read window | copy token accuracy (200 rows, 32k) |
|---|---|
| full | 0.99994 |
| 16384 (covers the offset) | 0.99994 |
| **16382** (two tokens short) | **0.0040** |
| 8192 | 0.0040 |

The window semantics hold to the token on real hardware, and the 6-layer × 2048 local stack cannot
bridge 16k on its own: **the global read is the retrieval channel** when the task is retrieval.

## Results — natural text

**8 PG-19 rows ≥ 32k, buckets by position (paired Δ vs full, mean ± se; + = worse without reach):**

| model / window | [0,2k) | [2k,8k) | [8k,16k) | [16k,32k) |
|---|---|---|---|---|
| **stage A** (8k-trained perceiver), global read → 512 | +0.0004 ±0.0007 | +0.0007 ±0.0009 | +0.0007 ±0.0010 | +0.0008 ±0.0007 |
| stage A, → 2048 | 0 | −0.0000 ±0.0003 | −0.0009 ±0.0003 | −0.0011 ±0.0005 |
| stage A, → 8192 | 0 | 0 | −0.0003 ±0.0001 | −0.0010 ±0.0004 |
| **stage B** (32k continuation), → 512 | +0.0008 ±0.0003 | −0.0000 ±0.0005 | −0.0002 ±0.0008 | −0.0009 ±0.0006 |
| stage B, → 2048 | 0 | −0.0004 ±0.0002 | −0.0008 ±0.0004 | **−0.0015 ±0.0005** |
| **dense control**, *all 14 layers* → 512 | +0.085 ±0.020 | +0.326 ±0.067 | +0.415 ±0.087 | +0.298 ±0.120 |
| dense, all → 2048 | 0 | **+0.099 ±0.017** | **+0.094 ±0.034** | −0.032 ±0.059 |
| dense, all → 8192 | 0 | 0 | −0.008 ±0.010 | **−0.093 ±0.027** |

**64 mixed rows ≥ 16k (higher power):**

| model / window | [0,2k) | [2k,8k) | [8k,16k) |
|---|---|---|---|
| stage A, global read → 512 | −0.0002 ±0.0001 | +0.0003 ±0.0001 | +0.0002 ±0.0001 |
| stage A, → 2048 | 0 | +0.0002 ±0.0000 | +0.0003 ±0.0000 |
| dense, all → 512 | +0.063 ±0.008 | +0.105 ±0.005 | +0.072 ±0.006 |
| dense, all → 2048 | 0 | **+0.035 ±0.003** | +0.006 ±0.004 |
| dense, all → 8192 | 0 | 0 | **−0.019 ±0.002** |

Absolute CE (stage A, full): 3.192 / 2.940 / 2.645 / 2.611 on the 32k rows; 3.801 / 4.035 / 4.040 on the 16k rows.

## Reading

1. **The bottom global read contributes ≈ 0.0003 nats (0.01%) to language-model loss at 8k–32k**, on
   both checkpoints and both row sets. The whole P1 parity with dense (3.790 vs 3.786) is carried by the
   12 × 2048 local stack, whose chained reach (~24.6k) covers the pilot context.
2. **The prize exists.** The dense model extracts **+0.035 nats (0.9%)** from direct access to keys 2k–8k
   back and **+0.10 nats (2.6%)** from reach 512→8k. That is the magnitude P3's ≥ 2% asks for, and the
   perceiver's single read captures none of it — it does not need to at this geometry, because the
   stack already has it.
3. **Beyond the training length the dense model's direct reach is harmful** (−0.019 ± 0.002 at
   [8k,16k) when capped at 8k; −0.093 at [16k,32k)): RoPE extrapolation failure. The perceiver has no
   such term — its windows never exceed the training range — which is what the stage-A "CE keeps falling
   past 8k" observation actually was.
4. **Stage B's far global read is slightly harmful** (−0.0015 ± 0.0005 at [16k,32k), 3σ, when capped at
   2k): softmax dilution over 16k+ keys carrying no useful signal. Small now; a warning for 1M.
5. P3 is **unmeasurable at this geometry**, not failed. Stage B's uniform degradation is an optimizer
   artifact (see stage A report); even a clean 32k stage could not have shown the global read's value
   with a 24.6k-reach stack inside a 32k context.

## What it does not settle

Two hypotheses are both consistent with (1)–(2): **H1** the bottom read is redundant here and becomes
load-bearing when the stack cannot reach (256k+); **H2** a single read at depth 1, with queries and keys
formed by one SWA-512 layer, can only do lexical matching (copy) and never semantic retrieval for LM.
The dense control cannot separate them. A geometry where the stack's reach ≪ context can.

## Next (iteration 2, needs a go)

Three matched 125M arms at seq 8k, **stack window N=256** (chained reach ≈ 3.1k, so 62% of the context is
beyond the stack), 0.5B tokens each, same data/optimizer as stage A: (A) global read at the bottom
(index 1, the E18 design); (B) global read at **mid-depth** (`PAR_GLOBAL_POSITIONS=7`, queries formed by
half the stack); (C) no global read (`PAR_GLOBAL_LAYERS=0 NUM_LAYERS=13`, same 14 layers/params).
Readouts: eval loss at equal tokens vs the dense 8k control and stage A; the reach probe on each; position
buckets on the 16k rows. Pre-registered reading: A ≈ C → H2 at this depth, B decides the main-run
placement; A < C by ≥ 1% with reach-Δ > 0 → H1, keep the design and harden the read for 1M.
