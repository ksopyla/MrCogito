# E22 — Implementation Plan

- **Spec:** [E22_perceiver_concept_lm.md](E22_perceiver_concept_lm.md) · **Status:** implemented and run 2026-09-12 · experiment killed 2026-09-12 (see spec Result); the family and this plan remain the foundation E23 builds on
- **Authored by:** `implementation-plan` · for → `research-implement`

> The HOW for: *a from-scratch Perceiver concept LM — SWA encoder → one pooled slot per r tokens →
> causal transformer over the slots → decoder confined to its segment that reads the slots.*
> New reusable family `perceiver_concept` in `nn/perceiver_concept_lm.py`, built from
> `nn/perceiver_ar_lm.py` primitives; no change to `perceiver_ar_lm.py` itself (E21 edits it on
> another branch — keep the two families conflict-free).

## 1. Reuse map

| Component | Action | Where |
|---|---|---|
| `TinyHashedEmbedding`, `Attention`, `Block`, `SwiGLU`, `attend`, `make_mask_pred`, `dense_bool_mask`, `_flex_block_mask`, `rope_cos_sin`, `apply_rope`, `chunked_softcap_ce`, `per_token_ce_chunked`, `_liger_flce`, `_get_flex` | **import and reuse** | `nn/perceiver_ar_lm.py` |
| `PerceiverConceptConfig` (`PretrainedConfig`, `model_type="perceiver_concept"`, `checkpoint_family="perceiver_concept"`, `pretraining_objective="causal_lm"`) | new | `nn/perceiver_concept_lm.py` |
| `ConceptPooler` (mean + zero-init learned-query attention per block, positional bias, QK-norm) | new | same |
| `ConceptCrossAttention` (token queries → concept K/V, RoPE both sides, flex/sdpa mask `pos(z) ≤ p ∧ doc match`) | new | same |
| `DecoderBlock` (segment-local self-attn via `Attention(pattern="swa")` fed segment-augmented doc ids → cross-attn → SwiGLU) | new | same |
| `PerceiverConceptLM` (`forward`, `hidden_states`, `concepts`, `concept_override`, `analytic_param_count`) | new | same |
| family routing `model_family == "perceiver_concept"` → `_build_perceiver_concept_model`; W&B identity | extend | `training/concept_pretraining_factories.py` |
| `ModelArguments.pcl_*`; `model_family` validation set; entrypoint `is_perceiver_ar`-style skips | extend | `training/concept_pretraining_args.py`, `training/train_concept_pretraining.py` |
| `MODEL_FAMILY=perceiver_concept` → `PCL_ARGS`; `DATASET_MIX_WEIGHT_OVERRIDE` → `--dataset_mix_weight_override` | extend | `scripts/train_concept_pretraining_multigpu.sh` |
| `scripts/launch_e22.sh` (thin wrapper; arms A / C / dense) | new | `scripts/` |
| `load_model` family-aware; `--probe concept` (paired `real/none/shuffled`) | extend | `evaluation/long_context_probes.py` |
| `--model_type perceiver_concept` | extend | `analysis/check_model_health.py`, `evaluation/lm_eval_perceiver_ar.py` (loader), `scripts/eval_perceiver_ar_suite.sh` (pass-through) |
| tests | new | `tests/test_perceiver_concept_lm.py` |

## 2. Config knobs (`PerceiverConceptConfig` ← `ModelArguments.pcl_*` ← launcher `PCL_*`)

| Field | Default (pilot) | Meaning |
|---|---|---|
| `enc_layers`, `enc_window` | 6, 512 | causal SWA encoder over tokens |
| `concept_ratio` r, `concept_slots` c | 16, 1 | slots per block; C = c·⌈S/r⌉ |
| `pool_pos_bias` | True | learnable [r, heads] bias on pooling logits |
| `latent_layers`, `latent_repeats` K | 4, 1 | causal transformer over the concept array; K>1 = weight-tied repeats |
| `dec_layers`, `dec_segment`, `dec_local` | 8, 1024, `"block"` | decoder self-attn confinement: `block` = segment-reset; `swa` = sliding window `dec_segment` |
| `concept_mode` | `"full"` | `"full"` (bet) · `"none"` (arm C: no cross-attention) |
| `xattn_kv_heads` | 2 | GQA kv-heads of the cross-attention |
| `enc_value_embed_layers`, `dec_value_embed_layers` | (0,3), (0,) | value embeddings (token identity in V) |
| `nope_every` | 0 | reserved; RoPE everywhere in the pilot |
| shared with E18 | — | `hidden_size 768`, `intermediate_size 2048`, `token_embedding_dim 256`, `num_kv_heads 2`, `head_dim 128`, `ngram_orders (2,3)`, `ngram_buckets 65536`, `value_embed_dim 64`, `rope_theta 5e5`, `logit_softcap 30`, `z_loss 1e-4`, `attn_backend flex`, `attn_pad_multiple 2048`, `use_liger` |

## 3. Forward pass (shapes; B batch, S tokens, d 768, r 16, C = ⌈S/r⌉·c)

```
input_ids [B,S], attention_mask [B,S] (right pad), labels [B,S], doc_ids [B,S] or None
pad to attn_pad_multiple (2048) → S; doc_ids default = zeros (pad → -1 via attention_mask)
pos      = _positions(S, doc_ids)                                        [B,S]  (reset at doc starts)
x0       = embed(input_ids, doc_ids)                                     [B,S,d]

ENCODER  h = Block_i(swa, enc_window)(…)  i < enc_layers                 [B,S,d]   (U-net skips off)

POOL     blocks j = 0..nb-1 over token index t ∈ [j·r, (j+1)·r)
         last(j) = (j+1)·r − 1 (clamped to S−1);  cdoc[b,j] = doc_ids[b,last];  cpos[b,j] = pos[b,last]
         ok[b,t] = attention_mask[b,t] ∧ doc_ids[b,t] == cdoc[b, t//r]   # tokens of an earlier doc inside the block are excluded
         mean_j  = Σ_t ok·h_t / Σ ok                                     [B,nb,d]
         attn_j  = softmax_t( q_c·k_t/√dh + bias[t−j·r, head] , mask ok ) · v_t   # q_c learned [c,h,dh], k/v from h_t
         z0[b, j·c + i] = mean_j + W_o(attn_j,i)   (W_o zero-init)       [B,C,d]
         cvalid[b,j] = any(ok[b, block j])  (all-pad block → invalid, never attended)

LATENT   z = Block_k(full, causal)(z0)  with doc_ids=cdoc, key_valid=cvalid, RoPE(cpos)  × latent_repeats
         z = concept_norm(z)                                              [B,C,d]

DECODER  seg_ids  = doc_ids·(S//dec_segment + 1) + (t // dec_segment)    [B,S]  (block mode)
         for each of dec_layers:
            y = y + SelfAttn(swa window=dec_segment, doc_ids=seg_ids)(norm(y))   # segment-confined
            y = y + XAttn(norm(y) → z)   mask: cpos[b,kv] ≤ pos[b,q] ∧ cdoc[b,kv]==doc_ids[b,q] ∧ cvalid[b,kv]
            y = y + SwiGLU(norm(y))
HEAD     logits/loss as in PerceiverARLM (softcap, z-loss, Liger or chunked CE), labels shifted inside
```
The cross-attention mask uses `cpos ≤ pos` (document-relative positions, both reset at doc
starts) — with packed docs, `cdoc == doc` already restricts to the same document, and within a
document positions are monotone, so `cpos ≤ pos` is the causal rule. A token in block j never
sees its own block's slot (last(j) ≥ t unless t is the block's last token — equality is allowed
for the last token: its slot pools tokens ≤ t only, so it is still causal).

`concept_override`: a context manager; `none` → cross-attention output replaced by 0 (concepts
invisible); `shuffled` → z rolled by one row in the batch (probe feeds two rows); `real` → no-op.

## 4. Masks and kernels
- Encoder / decoder self-attn: `attend(pattern="swa", window, doc_ids=…)` — the existing E18 path
  (flex block masks memoised per forward; sdpa reference).
- Latent: `attend(pattern="full", key_valid=cvalid, doc_ids=cdoc)` over C.
- Cross-attention: new `attend_cross(q[B,S,h,dh], k,v[B,C,g,dh], ok_fn)`; flex `create_block_mask(pred, B, None, S, C)` with the pred reading `cpos`, `cdoc`, `cvalid`, `pos`, `doc_ids` tensors; sdpa reference builds `[B,1,S,C]` bool. Memoised per forward like `block_masks`.
- Gradient checkpointing per block (encoder, latent, decoder) as in `_run_layers`.

## 5. Data and launch
- `DATASET_MIX_WEIGHT_OVERRIDE='{"pg19":…}'` on the E18b manifest (`dataset_mix_weight_override` exists
  in `DataArguments`; add the launcher plumbing). Verify the achieved token share with
  `scripts/manifest_token_stats.py` before launch.
- `BATCH_PACKING_MODE=pack` (model forward accepts `doc_ids`).
- `scripts/launch_e22.sh`: `E22_ARM ∈ {A (default), C, dense}`; pins model/optimizer/data; delegates.

## 6. Tests (`tests/test_perceiver_concept_lm.py`, CPU, sdpa)
1. shapes + finite loss; `concept_mode=none` runs and has no cross-attn params.
2. **causality:** perturbing token t changes no logit at positions < t (concept path included).
3. **closure:** with `concept_mode=none`, logits at position p depend on no token before p's segment start.
4. **pool masking:** a block straddling two documents pools only the last document's tokens.
5. flex == sdpa (skipped on CPU) for the cross mask; `attention_mask` right-padding identity.
6. `concept_override("none")` ≡ `concept_mode=none` logits; `analytic_param_count` == `sum(numel)`.
7. `hidden_states` / `return_per_token_loss` contract used by the probes.

## 7. Risks
- Flex `create_block_mask` with a batch-dependent pred over (S=32k, C=2k): build cost ~ E21's message mask (fine, compiled on CUDA).
- Segment-confined decoder makes the first tokens of every segment context-poor by design — S3's bucket [0,1k) check and the dense control measure the price; `dec_local=swa` is the registered fallback.
- Muon on a from-scratch encoder-decoder with a zero-init pooler: watch RankMe of z and the S1 probe at the first eval (K1/K2 are early).
