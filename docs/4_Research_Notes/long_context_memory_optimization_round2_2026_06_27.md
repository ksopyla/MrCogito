# Long-Context Memory Optimization — Round 2 (output-head wall + sequence parallelism → 1M)

- **Status:** implemented + measured on Odra (3× RTX 3090, 24 GB) · 2026-06-27
- **Branch:** `e05-smollm-win128-launch` (commits `d0f48e0` → `6cae49d`)
- **Scope:** concept encoder (BiXT) + windowed AR decoder (`concept_ar`). Round-1 (F1–F4)
  reached 128K on one card; this round finds and removes the real ceiling and reaches
  **1,048,575-token (1M) context on 3× 3090**.
- **Goal alignment:** `docs/1_Strategy_and_Plans/vision_and_goals.md` — O(C·N) concept
  encoder + windowed decoder for 1M–10M context.

## Headline

| context (single seq) | hardware | peak/GPU | step | note |
|---|---|---|---|---|
| 128K | 1× 3090 | 8.7 GB | 4.3 s | round-1 ceiling was 128K @ 15.6 GB |
| 256K | 1× 3090 | 16.4 GB | 10 s | **round-1 OOM → now fits** |
| 196K | 3× 3090 (SP) | 5.0 GB | 2.3 s | |
| 393K | 3× 3090 (SP) | 8.1 GB | 5.6 s | |
| 786K | 3× 3090 (SP) | 17.2 GB | 11 s | |
| **1M (1,048,575)** | **3× 3090 (SP)** | **22.6 GB** | **16 s** | **fits (24 GB card)** |

## The real wall (not what round-1 assumed)

Round-1 (F1–F4) assumed the encoder/decoder attention was the long-context ceiling. A
phased forward profile (`scripts/profile_long_context.py`, reset+record peak around
encoder / decoder / lm_head+CE) showed the opposite — **the output head was the wall**:

| phase | peak @ 131K (round-1) |
|---|---|
| encoder (BiXT) | 2598 MB |
| decoder (windowed AR) | 4470 MB |
| **lm_head + CE** | **14647 MB** ← wall |

**Root cause (F2 was not actually capping memory on the training path):** the F2 chunked-CE
loop computed a correct loss, but under autograd it **retained every block's logits for
backward → the whole `[B,N,V]` lived in the graph**. `alloc_after` the CE phase was 7386 MB
@ 65K ≈ `[B,65536,49152]×2`. F2's `O(block·V)` claim held only for inference, not training.

### Fix F2′ — `ChunkedLMHeadCE` (custom autograd Function)
`nn/concept_encoder_perceiver.py`. Saves **only** `hidden` + `labels` and **recomputes**
`lm_head` per block in backward (`torch.autograd.grad`), so the only large tensors are the
`[B,N,H]` hidden gradient and the `[V,H]` weight gradient. True `O(block·V)` peak,
numerically equivalent to full CE (loss + `grad_hidden` + `grad_weight`; unit-tested).

Result: **single-GPU ceiling jumps 128K → 256K** (15.6 GB → 16.4 GB), and the encoder/decoder
become the wall as expected.

## Sequence parallelism (F6) → 1M

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments` aside, FSDP/DeepSpeed do **not** help here:
they shard params/optimizer, not the per-sequence activations (which dominate for this small
model at long N). The only lever past one card is **sharding the token axis N across GPUs**.
The windowed architecture makes this nearly free.

- **Encoder (BiXT):** concepts replicated; tokens sharded. `DistLatTokAttention` does a
  GLOBAL softmax over the sharded token axis (all-reduce max+sum of `[B,h,C,*]` — C=128, tiny)
  and all-reduces the lat output. `local_tok_lat`/`LocalTokLatAttention` handle the reverse
  direction locally (concepts replicated). Comm ≈ KB/layer.
- **Decoder:** runs locally per shard (windowed self-attn + cross-attn to the replicated
  concepts). The first `L·window` positions of each non-first shard are dropped from the loss
  (their local window can't see the predecessor shard; the global concepts still carry all
  cross-shard deps — architecturally aligned with E05's "concepts are the cross-window
  carrier"). Comm ≈ 0.
- **Global positions:** `position_offset = rank·shard_len` threaded into the encoder token-pos
  embeddings and the decoder RoPE, so a shard at offset 500K "sees" position 500K.
- **Loss:** sharded CE sum + count, all-reduced; each rank's loss has the global-mean VALUE
  and a per-shard GRADIENT, so the grad all-reduce reconstructs the exact global gradient.

### Gradient correctness for the replicated bottleneck (the subtle part)
A replicated activation (concepts) consumed by sharded decoders has a *partial* per-shard
gradient. Naïve handling makes either token-side params miss cross-shard terms or concept-side
params over-count by world_size. The correct recipe (`nn/sequence_parallel.py`):
1. **`AllReduceGrad` barrier** at the decoder→encoder seam sums the per-shard concepts grad.
2. **`DistLatTokAttention` / `LocalTokLatAttention`** all-reduce their concept-side grads
   (`d r_lat`, `d v_lat`) inside backward, so the residual chain stays full with one barrier.
3. **`sync_seq_parallel_grads`**: SUM token/suffix/decoder/lm_head params; AVG concept-side.

**Validated:** 2-rank gloo/CPU test matches single-GPU to **~1e-6** in loss and *every*
parameter gradient (token- and concept-side). On Odra (NCCL), 1M trains in 16 s/step at 22.6 GB/GPU.

## Optimizer (F7) — Muon

The user flagged that an optimizer change must consider **convergence**, not just memory.
Adafactor trades convergence for memory; **Muon** (Jordan 2024) is the better fit:
orthogonalized momentum (5th-order Newton-Schulz) for 2D weight matrices, AdamW fallback for
embeddings/lm_head/1D. `nn/muon.py`.

Convergence on **real wikitext-103** (SmolLM2, ctx 2048, 200 steps, identical init/data):

| step | AdamW (3e-4) | Muon (0.02) |
|---|---|---|
| 20 | 10.44 | 9.50 |
| 60 | 7.66 | 7.06 |
| 200 | 6.69 | **6.25** |

- **Muon reaches a given loss in ~½ the steps** (loss 7.0 at step ~140 vs ~60).
- **Wall-clock:** Muon 114 s vs AdamW 166 s for 200 steps (its batched Newton-Schulz GEMMs
  beat AdamW's per-element fp32 updates here).
- **Memory @ 1M:** Muon 22.4 GB vs AdamW 22.6 GB — small saving (activations dominate at
  this scale, not optimizer states); the win is convergence speed.
- LR differs from AdamW: Muon ~0.02 for matrix params, ~2e-3 for the AdamW fallback.

## Ceiling / what 10M would need

Per-GPU peak scales with the shard length (`≈ single-GPU memory at shard_len`). On 24 GB
3090s:
- **1M on 3 GPUs (Odra): 22.6 GB** — fits (1.8 GB headroom).
- **1M on 4 GPUs (Polonez): ~17 GB** — comfortable; ~1.5M reachable.
- **10M is the hardware ceiling for this card class**: 10M/8 = 1.25M shard ≈ 80 GB/GPU, or
  ~30× 3090. Reaching 10M needs 80 GB cards (A100/H100) or activation CPU-offload across many
  cards — beyond the 2–8 consumer-GPU target. This is the "theoretical computing limitation."

## Knobs added (all default-off; single-GPU path byte-unchanged)

- `chunked_ce_block_size` — F2′ (ChunkedLMHeadCE). Already existed; now actually caps memory.
- `sp_boundary_mask` — decoder positions dropped at non-first shard boundaries (default `L·window`).
- `model.set_sequence_parallel(pg)` + `sync_seq_parallel_grads(model, pg)` — F6 runtime hooks.
- `--optim {adamw, muon}` — F7 (adafactor also wired via the bench).

## Scripts

- `scripts/bench_memory.py` — single-GPU memory bench (+`--fwd_only` decomposition).
- `scripts/profile_long_context.py` — phased (encoder/decoder/CE) peak profiler.
- `scripts/bench_seq_parallel.py` — `torchrun` sequence-parallel memory/throughput bench.
- `scripts/bench_optimizer_convergence.py` — real-text (wikitext-103) loss-curve comparison.

## Tests

`tests/test_chunked_lm_head_ce.py` (F2′ numerical equivalence + scaling),
`tests/test_sequence_parallel.py` (DistLatTokAttention primitive + **full-model** SP ≡ single-GPU).
Full suite: 20 passed.

## Reproduce (Odra)

```bash
export PATH=$HOME/.local/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# single-GPU 256K (post-F2′)
uv run python scripts/bench_memory.py --seq_len 262144 --batch_size 1 \
  --decoder_context_window 128 --decoder_attn_impl chunked_window \
  --gradient_checkpointing --full_vocab --chunked_ce_block_size 2048
# 1M on 3 GPUs
uv run python -m torch.distributed.run --standalone --nproc_per_node=3 \
  scripts/bench_seq_parallel.py --seq_len 1048575 --steps 2 --optim muon
# convergence: AdamW vs Muon on wikitext-103
uv run python scripts/bench_optimizer_convergence.py --optim muon --lr 0.02 --steps 200
```

## Next (deferred)

- Cross-node SP (Odra+Polonez = 7 GPUs) over NCCL/TCP for ~2.5M context (PCIe-bound; latency-sensitive).
- Activation CPU-offload as a per-GPU safety net to push single-shard past 350K (Polonez has 256 GB RAM).
- Long real-data E05 run at 1M to measure concept quality / the beyond-window ΔCE gate at extreme context.
- 10M requires 80 GB cards or ~30× consumer GPUs (hardware ceiling for this class).
