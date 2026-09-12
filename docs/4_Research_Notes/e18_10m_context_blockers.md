# What blocks E18 at 10M context

**Written:** 2026-09-09 · mutable research note (correct in place) · not on any current critical path

Parking note for later. E18's single global read makes **1M context comfortable at inference** and
512k trainable; **10M is a different regime**. This records which walls we hit and in what order, so
nobody re-derives it. Numbers use the E18 main config (`d=1280`, 10 q-heads × 128, `g=2` kv-heads,
23 layers, stack window `N=4096`, 594M dense) — see
[E18 spec](../experiments_specs/done_failed/E18_perceiver_ar_v2_baseline.md) and
[feasibility note](perceiver_ar_modern_reproduction_feasibility.md).

Per-token forward FLOPs decompose as `1.62 GFLOP (everything else) + 2560·M (the global layer)`:

| Context M | Global layer | Total fwd | Train (×3) | KV cache | Prefill (1×H100, 40% MFU) |
|---|---|---|---|---|---|
| 256k | 0.67 | 2.29 | 6.9 GFLOP/tok | 0.26 GB | ~2 s |
| 1M | 2.68 | 4.30 | 12.9 | 1.02 GB | ~15 s |
| **10M** | **26.8** | **28.4** | **85** | **10.2 GB** | **~17 min** |

## Not blockers (this is the architecture working as designed)

- **KV cache.** One global layer × 2 kv-heads = 1 KB/token. 10M = **10.2 GB**, fits one 80 GB card.
  A dense 23-layer model of the same width needs ~235 GB. The window layers stay at a fixed 86 MB.
- **Decode latency.** HBM-bound: ~11.4 GB read per token → **~3.4 ms/token (≈290 tok/s)** on H100.

## Blockers, in the order they bite

1. **The global layer is still O(M²).** At 10M it is 94% of forward FLOPs; training costs
   85 GFLOP/token, **17× the 8k cost**. The architecture makes long context *affordable*, not
   sub-quadratic. This alone rules out training at 10M on any budget we will have.
2. **Retained activations.** With per-layer gradient checkpointing every layer input `[B,S,d]` is
   kept for recompute: 23 × 2.56 GB = **59 GB at 1M**, 590 GB at 10M (bf16, batch 1, d=1280). The
   U-net skips add nothing on top — each pushed skip *is* the next layer's retained input, the same
   tensor — but they cap coarser checkpointing at ~L/2 retained tensors, so the floor with the best
   schedule is still **~28 GB at 1M, ~280 GB at 10M**. 1M training therefore needs context
   parallelism (`nn/sequence_parallel.py`) regardless; 512k (stage 3) is tight on 80 GB.
   *(Corrected 2026-09-09: an earlier version of this note called the skips an extra cost.)*
3. **Prefill wall.** ~284 PFLOP ≈ **17 minutes** on one H100. Fine for batch/offline, fatal for
   interactive. Needs chunked/streaming prefill with cache persistence to be usable at all.
4. **Positions.** RoPE θ=5e5 is nowhere near 10M; needs a YaRN/NoPE strategy validated at that
   range, and the global layer currently always carries RoPE (`nope_every` only applies at
   `i >= stack_start`).

## Paths worth trying when it matters

The useful asymmetry: **there is exactly one global layer**, so each fix is installed once.

- **Sparse/hierarchical global read** — block-sparse top-k retrieval over the prefix. One index, not 23.
- **Learned K/V compression of the prefix** — pool 10M token-keys into ~500k latent slots. This is
  where the project's original *concept vector* idea returns, in the one place it is cheap to test
  against an honest uncompressed baseline.
- **Checkpoint groups of layers + context parallelism** for the 512k/1M stages; dropping the U-net
  above some depth would lower the retained-activation floor from ~L/2 to ~L/k.
