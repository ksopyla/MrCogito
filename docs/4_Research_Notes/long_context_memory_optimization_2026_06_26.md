# Long-Context Memory Optimization — E02/E05 AR Decoder

- **Status:** implemented + measured on Odra (3× RTX 3090, 24 GB) · 2026-06-26
- **Branch:** `e05-smollm-win128-launch` (commits `7a888dd` → `0272d13`)
- **Scope:** E02 (full-causal AR) + E05 (windowed K=128 AR). Diffusion/recursive decoders postponed.
- **Goal alignment:** `docs/1_Strategy_and_Plans/vision_and_goals.md` — O(C·N) concept encoder + windowed decoder for 1M-10M context.

## The problem (baseline, measured)

The E05 windowed decoder passed its sliding-window mask as an explicit bool `attn_mask` to SDPA, forcing the math backend and full **O(N²)** attention materialization — the K=128 window capped *compute* but not *memory*. Baseline (single 3090, B=2, V=1024, H768/T256/L6/C128/D4, K=128):

| N | peak alloc | note |
|---|---|---|
| 2048 | 1948 MB | current run's regime |
| 8192 | 6501 MB | super-linear |
| 16384 | 14482 MB | |
| 32768 | **OOM** | ceiling at 24 GB |

Memory scaled ~quadratically, not linearly. `flex_attention` was prototyped but is **not a win on Ampere (3090, cc 8.6)** — its fused backend needs Hopper (cc 9.0+); on Ampere it materialized attention (9.4 GB at N=8192, OOM at 16384) — worse than SDPA-math. Rejected.

## The fixes (all default-preserving; E01-E04 byte-unchanged)

### F1 — Chunked windowed decoder attention (`decoder_attn_impl="chunked_window"`)
Hardware-agnostic O(N·K) memory: compute causal + last-K attention in query blocks; each chunk loads only its union key window `[s-K+1, e)` (width ≤ chunk+K-1), so the attention matrix is O(chunk·(chunk+K)) independent of N. Numerically equivalent to the SDPA-math bool-mask path within bf16 precision (max diff 0.0039, unit-tested). Also **skips the O(N²) full-mask build** (the int64 `[N,N]` diff in `build_sliding_window_causal_mask` was a 32 GB allocation at N=65536 — the actual OOM source before the fix).

### F2 — Chunked lm_head + cross-entropy (`chunked_ce_block_size>0`)
O(N·V) → O(block·V): compute lm_head+CE in N-blocks so `[B,N,V]` and the fp32 CE upcast are never materialised. Numerically equivalent to full CE (mean over non-ignored, unit-tested). Training-only; ablation/eval keep full logits for the window-slice Δ deltas. Returns `logits=None` in that path.

### F3 — Gradient checkpointing (`gradient_checkpointing_enable()`)
Manual per-layer `torch.utils.checkpoint` (use_reentrant=False) on the BiXT encoder + AR decoder layer loops. `supports_gradient_checkpointing=True` + `_set_gradient_checkpointing` propagate the HF Trainer flag. The encoder was the 18 GB hog at N=65536 (token FFN/BiXT intermediates ×6 layers kept for backward). Trades ~30% compute for a large activation cut.

### F4 — CUDA/PyTorch allocator
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` reduces reserved-but-unallocated fragmentation (~10% reserved win at 131K). `bf16` confirmed (not fp16). `adamw_torch_fused` kept (fp32 states, N-independent). RMSNorm bf16-input/float-weight fallback to non-fused kernel is benign. The allocator "97% full" reading in the real run was reserved-not-allocated, driven by the largest op — shrinking the ops (F1/F2) is what frees headroom.

## Measured results (single 3090, peak `max_memory_allocated`)

### E05 windowed, V=1024 (isolates attention mem), B=2

| N | baseline | F1 | F1+F3 | **F1+F2+F3** |
|---|---|---|---|---|
| 2048 | 1948 | 1963 | 973 | — |
| 8192 | 6501 | 5794 | 1593 | — |
| 16384 | 14482 | 10874 | 2578 | — |
| 32768 | OOM | 21063 | 4539 | — |
| 65536 | — | OOM | 8466 | — |
| 131072 | — | — | 16311 (B=2) | — |
| 262144 | — | — | OOM (B=2) / **16554 MB (B=1)** | — |

### E05 windowed, real V=49152 (the production vocab), B=2 unless noted

| N | full logits (no F2) | **F1+F2+F3** | step time |
|---|---|---|---|
| 2048 | 2559 | 2174 | 0.12s |
| 8192 | 7334 | 3497 | 0.41s |
| 16384 | 13715 | 5268 | 0.83s |
| 32768 | (OOM) | 8806 | 1.71s |
| 65536 | — | 15872 | 3.69s |
| 131072 | — | OOM (B=2) / **15664 MB (B=1)** | 4.19s |
| 262144 | — | OOM (B=1, ~26 GB needed) | — |

### E02 full-causal (no window), V=49152, F2+F3, B=2

| N | peak | step time |
|---|---|---|
| 2048 | 2176 | 0.12s |
| 8192 | 3496 | 0.83s |
| 16384 | 5267 | 2.92s |

E02's full-causal attention is O(N²) compute — at N=16384 it's 2.9s vs E05's 0.83s, and it would OOM at 32768 (16 GB attention matrix). **E05's windowed decoder is both more memory-efficient and faster at long N**, confirming the vision's architecture direction. F1 (windowed) is what unlocks long context for the decoder; F2/F3 help both E02 and E05.

## What the fixes changed in memory

- Decoder self-attn: O(N²) → **O(N·K)** (F1). ~30-40× at large N.
- lm_head + CE: O(N·V) spike → **O(block·V)** (F2). Cut 13715→5268 MB at N=16384/V=49152.
- Encoder/decoder activations: kept-for-backward → **recomputed** (F3). 8466→... enabled 65536→131072.
- Mask build: int64 O(N²) → **skipped** in chunked path (the hidden 32 GB OOM at 65536).

## Ceiling reached (single 24 GB 3090)

- **128K context at the real V=49152 fits** (15.7 GB, B=1).
- **256K needs ~26 GB at V=49152/B=1** — just over one 3090. Reaching 256K-1M requires:
  - distributing across the 3×3090 (72 GB total) via DDP/FSDP, or
  - a 40/80 GB card (A100/H100), or
  - a smaller vocab projection (tied/logits-chunked output head), or
  - streaming/chunked encoder cross-attention over N blocks (the O(C·N) similarity `[B,h,C,N]` is 536 MB at 256K and grows linearly — fine, but eventually needs streaming at 1M).

## Config knobs added (all default-off, byte-unchanged for E01-E04)

- `decoder_attn_impl`: `"sdpa"` (default) | `"chunked_window"` — env `DECODER_ATTN_IMPL`
- `decoder_attn_chunk_size`: 2048 — env `DECODER_ATTN_CHUNK_SIZE`
- `chunked_ce_block_size`: 0 (off) — env `CHUNKED_CE_BLOCK_SIZE`
- `gradient_checkpointing`: HF Trainer `--gradient_checkpointing True` (now supported)

## Reproduce

```bash
# on Odra, single GPU
uv run python scripts/bench_memory.py --seq_len 65536 --steps 2 --batch_size 1 \
  --decoder_context_window 128 --decoder_attn_impl chunked_window \
  --gradient_checkpointing --full_vocab --chunked_ce_block_size 2048
```

## Tests

`tests/test_e05_windowed_decoder.py`: +3 (chunked-window numerical equivalence vs SDPA; padding-mask correctness; chunked-CE equivalence + backward). Full suite 163 passed, 9 skipped.

## Next (not done — deferred per scope)

- Chunked/streamed encoder cross-attention (the O(C·N) similarity at 1M).
- DDP/FSDP memory split across the 3×3090 to reach 256K-1M.
- FlexAttention on Hopper (when available) for a fused windowed kernel.
- E02 long-context: capped at ~32K by its O(N²) attention — the windowed E05 path is the long-context answer.
