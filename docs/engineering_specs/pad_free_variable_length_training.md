# Pad-free / low-padding training for BackboneConceptLM

- **Type:** engineering foundation (data sampler + optional attention path), not an `E0NN` experiment.
- **Status:** design (2026-08-10) — triggered by E17b Polonez finding that `bs=8` was **~0.82×** E17 `bs=3` tokens/s because pad-to-batch-max inflated FLOPs.
- **Serves:** E16b/E17/E17b long-ctx Gemma backbone runs and any future variable-length 2k–4k mixes.
- **Owner:** Krzysztof Sopyła · opened 2026-08-10.

## Why now

E17b microbatch calibration filled the 3090s (~21 GiB at `bs=8`) but **slowed** wall-time vs E17 (`bs=3`). Root cause: `DataCollatorForCausalLM` pads each microbatch to that batch’s max length. On `e16b_long_4k_v1` (high length variance, ~20% rows already at 4096):

| per-device bs | mean padded L | pad fraction | compute/real tokens |
|---|---|---|---|
| 3 | ~2805 | ~44% | **1.86** |
| 8 | ~3794 | ~60% | **2.50** (~1.37× more FLOPs/real token) |

Attention/FFN still run on the full `[B,L]` rectangle; masks do not remove pad FLOPs. Filling VRAM with a fatter microbatch therefore **bought occupancy and paid padding**.

We will not keep raising batch size as the primary efficiency lever on this mix.

## Goals / non-goals

**Goals**
1. Cut pad FLOPs so larger microbatches (or equal microbatches) raise **real tokens/s**.
2. Preserve research invariants: **one document = one concept trajectory**; no fake cross-doc long-range signal (E05 policy).
3. Stay on the shared foundation (config/sampler/collator flags — no per-experiment trainer fork).
4. Work on Polonez/Odra Ampere 3090s + DDP + Muon + LoRA Gemma-3-1B.

**Non-goals (Phase 1–2)**
- Migrating off HF Trainer.
- Hopper-only FA3 / TMA tricks.
- Changing the 1B token budget or E17b hypothesis mid-run.

## Current foundation (reuse)

| Piece | Location | Today |
|---|---|---|
| Causal collator | `data/data_collators.py` `DataCollatorForCausalLM` | pad-to-batch-max, labels `-100` on pad |
| Pretok | `scripts/pretokenize_mix.py`, `data/dataset_preprocess.py` | variable-length rows; **explicitly does not pack** short docs |
| Backbone forward | `nn/backbone_concept_lm.py` `_forward_blocks` | K=512 blocks, **custom 4D** windowed masks, recurrent concept state `z` |
| Attn knobs | `DECODER_ATTN_IMPL` | perceiver AR only; backbone has **no** FA varlen path |
| Loss | chunked CE, DDP-normalized by global non-pad count | pad-safe already |

## Option catalogue (all possibilities)

### A. Data-side (no model change) — prefer first

| # | Technique | How it cuts pad waste | Expected gain | Concept Encoder fit |
|---|---|---|---|---|
| A1 | **Length bucketing / length-grouped sampling** | Batch similar lengths → pad only to local max | **1.3–1.7×** tok/s typical | **Best Phase 1** — preserves one-doc-per-row |
| A2 | **Token-budget batching** | Fix ~tokens/step; vary #rows | ~1.5–1.6× + steadier DDP | Phase 1b; good with unequal rank lengths |
| A3 | Dynamic pad-to-batch-max | Already current | baseline | Keep |
| A4 | Offline bin-pack into max_len packs | Concat docs offline to ~4k | up to 2–4× on short data | **Unsafe** without concept-boundary resets (see C) |
| A5 | Sort-within-shard / curriculum by length | Mild bucketing without custom sampler | small–medium | Easy smoke |

### B. Attention / kernel (pad tokens never computed)

| # | Technique | Mechanism | HF / lib status | Concept Encoder fit |
|---|---|---|---|---|
| B1 | **ModernBERT-style unpadding** | Strip pads, concat, FA2 once | Built into ModernBERT + FA2 ([HF ModernBERT](https://huggingface.co/docs/transformers/model_doc/modernbert), [blog](https://huggingface.co/blog/modernbert)) | Pattern only — encoder MLM, not our causal block loop |
| B2 | **`DataCollatorWithFlattening` + padding-free** | Flatten to `[1, Σℓ]`, `cu_seqlens` / flash kwargs | [HF padding-free guide](https://huggingface.co/docs/transformers/padding_free); Transformers ≥4.43; needs FlashAttention | **Blocked today:** backbone uses custom 4D masks; Gemma-3 packing tests skipped upstream (QK-norm) |
| B3 | FlashAttention-2 `flash_attn_varlen_func` | Attention cost `Σℓᵢ²` not `(Σℓᵢ)²` | Dao-AILab + HF flash utils | Requires rewiring Gemma load + block loop |
| B4 | PyTorch `torch.nn.attention.varlen` | First-class varlen, compile-friendly | PyTorch 2.5+ | Same integration cost as B3 |
| B5 | Flex Attention + `BlockMask` | `doc_id[q]==doc_id[kv]` fused | PyTorch flexattention; torchtune packing PR | Major refactor; research path |
| B6 | xFormers `BlockDiagonalCausalMask` | Block-diagonal FMHA | xFormers | Extra dep; still needs concept resets |
| B7 | SDPA flash (current) | Faster padded attn when shapes allow | Already probed for non-backbone | **Does not remove pad FLOPs** |
| B8 | Nested tensors (NJT) | Jagged storage | Immature w/ compile | Watch only |

### C. Packing + architecture (multi-doc in one row)

| # | Technique | Requirement for Concept Encoder | Status |
|---|---|---|---|
| C1 | Online/offline multi-doc pack | **`z ← concept_init` at every doc boundary**; seam-aware masks; label `-100` at seams; RoPE/pos reset | Phase 3 only after A1 measured |
| C2 | Pack only *within* already-long docs (no cross-doc) | N/A — already one row per doc | Nothing to do |
| C3 | TRL `packing` / `padding_free` SFT | Different trainer | Out of scope |

### D. Orthogonal efficiency (does not fix padding)

| # | Technique | Notes |
|---|---|---|
| D1 | Liger fused CE / kernels | Mem / LM-head; Linux only; already optional |
| D2 | Smaller microbatch (E17-style bs=3) | Best *immediate* wall-time on this mix — accepts underfilled VRAM |
| D3 | Activation checkpointing already on | Needed at 4k; interacts with large B |

## Recommended roadmap

```
Phase 0  Instrument  →  Phase 1  Bucket  →  Phase 2  Measure+tune  →  Phase 3  Varlen/pack (gated)
```

### Phase 0 — Instrument (½ day)

Log per step (rank0 + all-reduce):
- `pad_ratio = 1 - real_tokens / (B · L_max)`
- `mean_L`, `max_L`, `real_tokens`
- `tokens_per_sec_real` (already derivable from CE denom)

Hook: trainer callback or collator-side stats → W&B.  
**Success:** E17b-like `bs=8` shows pad_ratio ≳ 0.5; bucketed run shows pad_ratio ≲ 0.1.

### Phase 1 — Length-grouped sampler (primary recommendation)

**Implemented 2026-08-14:** cached bounded sortish sampling is available through
`BATCH_PACKING_MODE=length_group`; the historical default remains `none`. E17c enables a
20-window profile for its pending 4-GPU run. The sampler creates one deterministic global
index stream and lets Hugging Face Accelerate perform its normal disjoint rank sharding.

**Design**
- Add `LengthGroupedSampler` (HF-style) or in-house `TokenBudgetBatchSampler` selectable via env:
  - `BATCH_PACKING_MODE=none|length_group|token_budget` (default `none` for checkpoint comparability).
- Precompute lengths once from pretok Arrow (`len(input_ids)`) into a sidecar cache next to the manifest (same pattern as `*.token_stats.json`).
- Group indices into mega-batches of similar length, then shuffle groups (preserve SGD noise).
- Keep `DataCollatorForCausalLM` unchanged (still pad-to-batch-max — but batch max ≈ group max).

**Files**
- `data/length_grouped_sampler.py` (new)
- `training/concept_pretraining_factories.py` / trainer — wire `train_sampler`
- `scripts/train_concept_pretraining_multigpu.sh` — env knobs
- `tests/test_length_grouped_sampler.py`

**Falsifiable success**
- On Polonez, same model/config as E17b (`bs=8`, `accum=1`, seq 4k, same mix):  
  - pad_ratio median **≤ 0.10**  
  - real tokens/s **≥ 1.25×** the pad-to-batch-max `bs=8` baseline  
  - loss curve within noise of unbucketed over first ~1k steps (same seed ± sampler)

**Kill**
- pad_ratio not improved, or tokens/s gain &lt; 10%, or DDP hangs from uneven buckets → revert default to `none`.

### Phase 2 — Token-budget batching (optional, after A1)

If length groups still leave rank imbalance (one rank draws longer buckets):
- Cap each step at `TARGET_TOKENS_PER_DEVICE` (e.g. 8×2048).
- Allows mixing a few mid-length rows instead of one ultra-long + pads.

### Phase 3 — True pad-free / packing (gated; larger eng)

Only if Phase 1 gains saturate and VRAM still idle:

1. **Prove Gemma-3 FA2 packing** on a *vanilla* LoRA SFT smoke (no concepts) — upstream still marks packing broken for Gemma-3 QK-norm.
2. If yes, design `BackboneConceptLM` varlen path:
   - either drop custom 4D masks for FA varlen + sliding/global layer types, **or** Flex `BlockMask`;
   - **mandatory** `z` reset at doc boundaries for any multi-doc pack;
   - collator emits `cu_seqlens` / `doc_ids` / seam labels.
3. Spec as its own eng change + experiment (touches inductive bias).

**Do not** adopt ModernBERT / `DataCollatorWithFlattening` as a drop-in for E17-family runs until (1)+(2) clear.

## Attention library recommendation

| Priority | Choice | Why |
|---|---|---|
| Now | Keep SDPA / current 4D masks | Correct for block-recurrent + window |
| Phase 1 | No attn change | Sampler-only |
| Phase 3 candidate | FA2 varlen **or** PyTorch varlen | Real pad-free; Ampere OK |
| Watch | Flex Attention | Powerful packing masks; heavier migrate |
| Skip for now | FA3/Hopper, NJT production path | Hardware / maturity |

ModernBERT’s lesson we **do** take: **unpad once + pack for density**.  
ModernBERT’s lesson we **cannot** copy blindly: their model owns unpadding inside an encoder FA2 stack; ours owns a K=512 recurrent concept loop with custom masks.

## Risks

| Risk | Mitigation |
|---|---|
| Length grouping changes batch composition / LR noise | Seeded group shuffle; compare early loss; keep `BATCH_PACKING_MODE=none` as default until green |
| Long-doc under-sampling in short buckets | Stratify: never drop &gt;2k rows; oversample long buckets if needed |
| Concept leakage if someone enables packing early | Gate Phase 3 behind explicit `ALLOW_CROSS_DOC_PACK=1` + `z` reset tests |
| Gemma-3 FA packing broken upstream | Phase 3 smoke on vanilla Gemma before touching concepts |

## References

- HF padding-free training: https://huggingface.co/docs/transformers/padding_free  
- HF packing + FA2 blog: https://huggingface.co/blog/packing-with-FA2  
- ModernBERT blog (unpadding + packing): https://huggingface.co/blog/modernbert  
- ModernBERT docs: https://huggingface.co/docs/transformers/model_doc/modernbert  
- IBM / packing paper: https://arxiv.org/abs/2407.09105  
- Variable-length DDP throughput: https://duoan.github.io/posts/why-variable-sequence-length-breaks-ddp-throughput/  
- Internal: E17b batch calib (2026-08-10) — pad waste measurement on `e16b_long_4k_v1`

## Implementation checklist

- [x] Phase 0: pad_ratio W&B metric
- [x] Phase 1: length cache + `LengthGroupedSampler` + launcher flag
- [ ] Phase 1: Polonez smoke vs E17b `bs=8` baseline
- [ ] Phase 2 (optional): token-budget sampler
- [ ] Phase 3 (gated): Gemma FA packing smoke → concept `z`-reset packing design

**Phase 0–1 API / DDP / tests / W&B handoff:** see companion plan
[`pad_free_variable_length_training_plan.md`](./pad_free_variable_length_training_plan.md)
(exact sidecar contract, `PerceiverDenoiseTrainer._get_train_sampler` wiring, unit-test list,
interleaved-pretok vs IterableDataset gates). Pasteable checklist lives in §7 of that plan.

## Decision

**Recommend Phase 1 length bucketing as the default future path to utilise GPU without paying pad FLOPs.**  
Do **not** chase larger microbatches or higher accum for pad-heavy mixes.  
Treat ModernBERT / HF padding-free as the **Phase 3 north star**, not the next patch on a live E17b run.
