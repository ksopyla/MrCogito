# E05 — Implementation Plan

- **Spec:** [E05_windowed_decoder_concept_memory.md](E05_windowed_decoder_concept_memory.md) · **Status:** implemented (foundation), not yet launched
- **Authored by:** `implementation-plan` · for → `research-implement` (built 2026-06-18)
- **Data-loader extension plan (2026-06-21):** [../engineering_specs/long_context_data_mix_loader_architecture.md](../engineering_specs/long_context_data_mix_loader_architecture.md)

> The HOW for E05's single change (sliding-window causal decoder) + its enabling condition
> (seq-len 2K on a long-doc dataset mix). Repo-rooted; reuse-first; all new code is reusable
> and config-selectable. Default behaviour (window=None, single dataset) is byte-unchanged.

## 1. Source & fit
- **Origin:** the E04 grilling thread (2026-06-17/18) — a plain decoder-family swap (E04) is only *correlational* about the "AR-bypass causes collapse" mechanism. E05 is the **scalpel**: keep the AR decoder (a good generator, aligned with the generation north-star), remove only the *long-range* part of local context, and make the 128 concepts the **only** cross-window carrier. Long-context compression literature says this is the form that works: Gist [2304.08467], ICAE [2307.06945], AutoCompressor [2305.14788].
- **Synthesis verdict:** Adopt the windowed-AR + concepts-as-memory design; **2K** seq-len (user decision — 512 is too short for any K to bite) and a **dataset mix** (vs FinePDFs-only) so the long tail actually exercises the window without packing unrelated short docs.
- **Architecture mapping (ONE):** the **decoder** (a sliding-window causal mask). Everything else — encoder, concept bottleneck, loss, tokenizer — is held fixed; data (2K mix) is the shared enabling condition across both A/B arms.

## 2. Reuse map (modules read before building)
| Component | Action | Where |
|---|---|---|
| `ConceptCausalDecoderLayer._self_attention` | extend: accept `attn_mask`, use `is_causal=(mask is None)` | `nn/concept_encoder_perceiver.py` |
| `ConceptCausalDecoderStack` | extend: build/cache window mask per `(T, device)`, pass to layers | `nn/concept_encoder_perceiver.py` |
| `build_sliding_window_causal_mask` | new — reusable bool mask helper | `nn/concept_encoder_perceiver.py` |
| `ConceptEncoderForConditionalLM.concept_ablation_ce` | extend: `window_k` → beyond/within-window deltas | `nn/concept_encoder_perceiver.py` |
| `ConceptEncoderConfig` | extend: `decoder_context_window: Optional[int] = None` | `nn/concept_encoder.py` |
| `load_and_preprocess_text_dataset` | refactor tokenize fn to module-level `_make_tokenize_fn` (shared) | `data/dataset_preprocess.py` |
| `load_and_preprocess_dataset_mix` + `DATASET_MIXES` | new — reusable weighted-interleave loader | `data/dataset_preprocess.py` |
| entrypoint `main()` / `ModelArguments` / `DataTrainingArguments` / `build_perceiver_denoise_config` | extend: `--decoder_context_window`, `--dataset_mix` | `training/train_perceiver_denoise.py` |
| `PerceiverDenoiseTrainer._concept_ablation_metrics` | extend: pass `window_k=config.decoder_context_window` | `training/train_perceiver_denoise.py` |
| `run_concept_analysis.py` | extend: `--ablation_window_k`, beyond-window print | `analysis/run_concept_analysis.py` |
| launcher | extend: `DECODER_CONTEXT_WINDOW`, `DATASET_MIX` knobs (conditional args) | `scripts/train_perceiver_denoise_multigpu.sh` |

No forks; no new training script; no parked code touched.

## 3. Forward pass (tensor shapes)
Symbols: `B`=batch, `N`=tokens (=2048), `C`=128 concepts, `H`=768, `V`=49152, `K`=window, `L`=`decoder_num_layers`.
```
encoder input (B, N)      → BiXT encode (O(C·N))               → concepts (B, C, H)
decoder input (B, N)      = shift_right(target)                # teacher forcing
  per layer:  self-attn(Q,K,V over N) with WINDOW-CAUSAL mask  → (B, N, H)   # O(N·K) effective, masked SDPA
              cross-attn(Q=tokens, KV=concepts)                → (B, N, H)   # O(N·C), the ONLY cross-window path
              gated FFN
lm_head                                                        → (B, N, V)
```
- Window mask: bool `[N, N]`, `mask[i,j] = (j ≤ i) and (i − j < K)`; broadcasts over `[B, h, N, N]` in SDPA. Built once per `(N, device)` and cached on the stack.
- **Receptive field grows with depth:** L stacked window-K layers reach ≈ `L·(K−1)` back. With L=4, K=256 → ≈1020 at N=2048 (only the 2nd half is forced through concepts). Pick K against `L·(K−1)`.

## 4. Inputs & data
- **Dataset:** `DATASET_MIX=long_2k_base_v1` — FinePDFs-100BT 0.50 / FineWeb-Edu sample-10BT 0.30 / FineMath-3+ 0.20 (weights = interleave sampling probabilities; per-source `max_samples` caps). Rationale + % > 2K in the spec.
- **Collator:** reuse `DataCollatorForTSDAE` (reconstruction) — unchanged. The window lives entirely in the decoder; the collator/data contract is identical to E01/E03 (EOS-appended, variable-length, `labels=-100` on pad).
- **Preprocessing:** each source → normalise to a single `text` column (`_normalize_to_text_column`, supports multi-column join for future SFT sources) → EOS-append variable-length tokenize (`_make_tokenize_fn`, shared with the single-dataset path) → per-source train/eval split → `interleave_datasets(probabilities=…, stopping_strategy="all_exhausted")` for train, `concatenate_datasets` for a representative multi-source eval.
- **No packing** of unrelated short docs (avoids fake long-range signal); long docs are truncated to 2K (first-2K window keeps genuine within-doc long-range structure).

## 5. Loss & training objective
- **Loss:** unchanged — next-token CE via `encode_decode_loss` / `_teacher_forced_ce` (reconstruction). No new loss component. Concept losses off.
- **Objective:** `reconstruction` (`deletion_rate=0.6`, matched to E01/E03 for comparability). The windowed decoder cannot copy beyond-window tokens from local context, so reconstructing them forces routing through concepts.
- **Weighting:** n/a.

## 6. Config & launch
- **New config fields (backward-compatible):** `ConceptEncoderConfig.decoder_context_window: Optional[int] = None`. `DataTrainingArguments.dataset_mix: Optional[str] = None`. `ModelArguments.decoder_context_window: Optional[int] = None`.
- **Registry / routing:** unchanged — E05 reuses `concept_ar` (`ConceptEncoderForConditionalLM`); `checkpoint_family="concept_ar"`, canonical eval modes unchanged (encoder-only weighted_pool / sentence_pair), so existing eval routing + `run_concept_analysis.py --model_type concept_ar` work as-is.
- **Launch:** windowed = `EXPERIMENT_ID=E05 DECODER_TYPE=causal_ar DECODER_CONTEXT_WINDOW=256 DATASET_MIX=long_2k_base_v1 MAX_SEQ_LENGTH=2048 … bash scripts/train_perceiver_denoise_multigpu.sh`; control = same line without `DECODER_CONTEXT_WINDOW` (see spec for full command). `DECODER_CONTEXT_WINDOW` / `DATASET_MIX` are passed only when set, so all prior launches are unchanged.

## 7. Tests & smoke
- `tests/test_e05_windowed_decoder.py` (7, green): mask pattern; default = full-causal (no mask built); windowed forward shapes + finite loss; **single-layer reach = K**; **multi-layer reach ≈ L·(K−1)**; beyond-window ablation keys present (and absent without `window_k`); `long_2k_base_v1` mix registered + weights normalised + FinePDFs is the backbone.
- Local smoke (run 2026-06-18, green): SmolLM2 tokenizer → EOS-append → `DataCollatorForTSDAE` → windowed `ConceptEncoderForConditionalLM` forward/backward (loss 11.19→ finite, grads flow) + `concept_ablation_ce(window_k=32)` emits `delta_*_beyond_window`. Full suite: 140 passed (4 pre-existing `test_wandb_identity` failures from the E04 job_type rename — unrelated).

## 8. Risks & tradeoffs
- **Risk — depth dilutes the window:** effective field ≈ `L·(K−1)` may cover most of 2K. **Cheapest signal:** the offline `--ablation_window_k` beyond-window Δ on the windowed vs control checkpoint (spec primary gate). **Fallback:** lower K or `decoder_num_layers`; raise seq-len.
- **Risk — masked SDPA loses the flash kernel** (explicit `attn_mask`) → slower/more memory than `is_causal` at 2K. **Mitigation:** the control (window=None) keeps the flash path; budget the windowed arm accordingly; consider FlexAttention later if throughput bites.
- **Risk — 16:1 compression at 2K/C128** (vs 4:1 at 512) reshapes concept formation. **Mitigation:** held fixed across the A/B (both arms 2K), so it's not a confound *within* E05; it is a confound vs E01–E04 (don't cross-compare absolute numbers).
- **Risk — dataset-mix download/throughput** (FinePDFs parquet shards, large one-time tokenize). **Mitigation:** `max_samples` caps; `DDP_TIMEOUT` raised; calibrate batch first (stage 0).
- **Risk — short-doc dilution** (FineWeb-Edu 0.30 mostly < 2K) weakens the average long-range signal. **Mitigation:** the beyond-window metric is computed only where it applies; FinePDFs (0.50) carries the long tail; reweight toward FinePDFs if the gate is weak.

## 9. Code sketches (`# sketch` — decisions already implemented)
```python
# sketch: nn/concept_encoder_perceiver.py
def build_sliding_window_causal_mask(seq_len, window, device, dtype=torch.bool):
    idx = torch.arange(seq_len, device=device)
    causal = idx[:, None] >= idx[None, :]          # j <= i
    in_window = idx[:, None] - idx[None, :] < window  # i - j < window
    return causal & in_window                       # [T, T] bool, broadcasts in SDPA

# _self_attention(...): F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
#                                                      is_causal=attn_mask is None)

# concept_ablation_ce(..., window_k): adds delta_{zero,shuffle}_beyond_window (t >= K)
#   + ce_intact_{beyond,within}_window  — the E05 long-range memory gate.

# sketch: data/dataset_preprocess.py
DATASET_MIXES = {"long_2k_base_v1": [ {hf_id/ data_files, subset, text_columns, weight, max_samples}, ... ]}
def load_and_preprocess_dataset_mix(tokenizer, mix, ...):
    # per source: load(cap) -> "text" -> split -> tokenize; then
    # interleave_datasets(parts, probabilities=weights, stopping_strategy="all_exhausted")
    # + concatenate_datasets(eval_parts)
```
