# E05 — Implementation Plan

- **Spec:** [E05_windowed_decoder_concept_memory.md](E05_windowed_decoder_concept_memory.md) · **Status:** implemented and run; branch closed after failed semantic/mechanism gates
- **Authored by:** `implementation-plan` · for → `research-implement` (built 2026-06-18) · **reconciled 2026-06-25** to the spec (K=128 fixed, `prefix_suffix`, `smollm3_inspired_2k` recipe; removed raise-K fallback)
- **Data-loader extension plan (2026-06-21):** [../engineering_specs/long_context_data_mix_loader_architecture.md](../../engineering_specs/long_context_data_mix_loader_architecture.md)

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
Symbols: `B`=batch, `N`=tokens (=2048), `C`=128 concepts, `H`=768, `V`=49152, `K`=window (fixed=128), `L`=`decoder_num_layers` (=4).
```
encoder input (B, N)      → BiXT encode (O(C·N))               → concepts (B, C, H)
decoder input (B, N)      = shift_right(target)                # teacher forcing
  per layer:  self-attn(Q,K,V over N) with WINDOW-CAUSAL mask  → (B, N, H)   # O(N·K) effective, masked SDPA
              cross-attn(Q=tokens, KV=concepts)                → (B, N, H)   # O(N·C), the ONLY cross-window path
              gated FFN
lm_head                                                        → (B, N, V)
```
- Window mask: bool `[N, N]`, `mask[i,j] = (j ≤ i) and (i − j < K)`; broadcasts over `[B, h, N, N]` in SDPA. Built once per `(N, device)` and cached on the stack.
- **K is fixed (128), not scaled to N.** K is a coherence window for local fluency only; per the vision, the quantity that scales with N is the concept count C, never K (see spec §"K is a fixed constant"). The decoder's *true* local receptive field grows with depth: L stacked window-K layers reach ≈ `L·(K−1)` back ≈ 508 at N=2048. Most of a 2K sequence is therefore forced through concepts, as intended. **Gate caveat:** the primary ablation metric slices at `t ≥ K = 128`, but positions in `[128, 508)` still have partial-local reach; co-report a `--ablation_window_k 508` read for an unconfounded concept-only signal (see spec success/kill).

## 4. Inputs & data
- **Dataset:** `DATASET_MIX_RECIPE=smollm3_inspired_2k` (recipe file `data/mix_recipes/smollm3_inspired_2k.json`) — a SmolLM3-inspired open pretraining mix (FineWeb-Edu sample-10BT 0.30 / DCLM-baseline 0.20 / Stack-Edu-Python 0.15 / FineMath-3+ 0.15 / FinePDFs-100BT 0.10 / big-reasoning-traces 0.05 / OpenMathReasoning-CoT 0.05), with explicit long-tail boosters; projected ~21.3% docs >2K, ~8.8% >4K. Objective-compatible with both `reconstruction` and `prefix_suffix`. (The older hardcoded `DATASET_MIXES["long_2k_base_v1"]` remains in the loader for back-compat/tests but is not the launch mix.)
- **Collator:** reuse `DataCollatorForPrefixGeneration` (prefix→suffix) — unchanged. The window lives entirely in the decoder; the collator/data contract is identical to E02 (EOS-appended, variable-length, `labels=-100` on pad).
- **Preprocessing:** each source → normalise to a single `text` column (`_normalize_to_text_column`, supports multi-column join for future SFT sources) → EOS-append variable-length tokenize (`_make_tokenize_fn`, shared with the single-dataset path) → per-source train/eval split → `interleave_datasets(probabilities=…, stopping_strategy="all_exhausted")` for train, `concatenate_datasets` for a representative multi-source eval. Tokenization runs under `main_process_first` and caches to `HF_DATASETS_CACHE`, so the first DDP run warms the cache — no separate pretokenize step.
- **No packing** of unrelated short docs (avoids fake long-range signal); long docs are truncated to 2K (first-2K window keeps genuine within-doc long-range structure).

## 5. Loss & training objective
- **Loss:** unchanged — next-token CE via `encode_decode_loss` / `_teacher_forced_ce` (prefix→suffix reconstruction). No new loss component. Concept losses off.
- **Objective:** `prefix_suffix` (matched to E02-long, the semantic leader and de-collapse basis). The windowed decoder cannot copy beyond-window tokens from local context, so predicting the suffix from the prefix forces routing far-back dependencies through concepts. (E02-long showed prefix→suffix de-collapses concepts with scale; E05 builds directly on that.)
- **Weighting:** n/a.

## 6. Config & launch
- **New config fields (backward-compatible):** `ConceptEncoderConfig.decoder_context_window: Optional[int] = None`. `DataTrainingArguments.dataset_mix: Optional[str] = None` + `dataset_mix_recipe: Optional[str] = None` + `dataset_mix_weight_override: Optional[str] = None`. `ModelArguments.decoder_context_window: Optional[int] = None`. **K is a fixed per-experiment hyperparameter (128 for E05), never auto-tuned to N — see spec §"K is a fixed constant".**
- **Registry / routing:** unchanged — E05 reuses `concept_ar` (`ConceptEncoderForConditionalLM`); `checkpoint_family="concept_ar"`, canonical eval modes unchanged (encoder-only weighted_pool / sentence_pair), so existing eval routing + `run_concept_analysis.py --model_type concept_ar` work as-is.
- **Launch:** no dedicated E05 launcher script. Both arms are env-var invocations of the shared `scripts/train_perceiver_denoise_multigpu.sh` (which already wires `DECODER_CONTEXT_WINDOW` and `DATASET_MIX_RECIPE`, passed only when set). Windowed arm:
  `EXPERIMENT_ID=E05 DECODER_TYPE=causal_ar DECODER_CONTEXT_WINDOW=128 DATASET_MIX_RECIPE=smollm3_inspired_2k MAX_SEQ_LENGTH=2048 OBJECTIVE_VARIANT=prefix_suffix HIDDEN_SIZE=768 TOKEN_EMBEDDING_DIM=256 NUM_LAYERS=6 CONCEPT_NUM=128 DECODER_NUM_LAYERS=4 INTERMEDIATE_SIZE=2048 HIDDEN_ACT=silu NORM_TYPE=rmsnorm DECODER_POS_TYPE=rope TOKENIZER_NAME=HuggingFaceTB/SmolLM2-135M NUM_EPOCHS=0.3 PER_DEVICE_BATCH_SIZE=2 GRADIENT_ACCUMULATION_STEPS=8 DDP_TIMEOUT=14400 bash scripts/train_perceiver_denoise_multigpu.sh`
  Control arm: identical line, omit `DECODER_CONTEXT_WINDOW` (defaults to None = full causal). Full canonical commands are in the spec.

## 7. Tests & smoke
- `tests/test_e05_windowed_decoder.py` (8, green): mask pattern; default = full-causal (no mask built); windowed forward shapes + finite loss; **single-layer reach = K**; **multi-layer reach ≈ L·(K−1)**; beyond-window ablation keys present (and absent without `window_k`); `long_2k_base_v1` mix registered + weights normalised; **`smollm3_inspired_2k` recipe loads, weights sum to 1.0, and carries the projected >2K long-context tail** (the actual launch mix).
- Local smoke (run 2026-06-18, green): SmolLM2 tokenizer → EOS-append → `DataCollatorForTSDAE` → windowed `ConceptEncoderForConditionalLM` forward/backward (loss 11.19→ finite, grads flow) + `concept_ablation_ce(window_k=32)` emits `delta_*_beyond_window`. Full suite: 140 passed (4 pre-existing `test_wandb_identity` failures from the E04 job_type rename — unrelated).

## 8. Risks & tradeoffs
- **Risk — depth dilutes the window:** effective field ≈ `L·(K−1)` ≈ 508 may cover part of 2K, so the K-slice gate conflates partial-local and concept-only positions. **Cheapest signal:** the offline `--ablation_window_k` beyond-window Δ on the windowed vs control checkpoint at both K=128 (primary) and 508 (concept-only robustness). **Fallback (vision-consistent):** lower `decoder_num_layers` or raise seq-len — **do not raise K** (K is a fixed coherence window; raising it reintroduces O(N·K) local decoding and defeats the concept-bottleneck premise, see spec §"K is a fixed constant").
- **Risk — masked SDPA loses the flash kernel** (explicit `attn_mask`) → slower/more memory than `is_causal` at 2K. **Mitigation:** the control (window=None) keeps the flash path; budget the windowed arm accordingly; consider FlexAttention later if throughput bites.
- **Risk — 16:1 compression at 2K/C128** (vs 4:1 at 512) reshapes concept formation. **Mitigation:** held fixed across the A/B (both arms 2K), so it's not a confound *within* E05; it is a confound vs E01–E04 (don't cross-compare absolute numbers).
- **Risk — dataset-mix download/throughput** (FinePDFs/DCLM parquet shards, large one-time tokenize). **Mitigation:** `max_samples` caps; `DDP_TIMEOUT` raised; calibrate batch first (stage 0); first run warms `HF_DATASETS_CACHE` under `main_process_first`.
- **Risk — short-doc dilution** (FineWeb-Edu/DCLM mostly < 2K) weakens the average long-range signal. **Mitigation:** the beyond-window metric is computed only where it applies; FinePDFs (long-doc booster) + the recipe's explicit long-tail boosters carry the >2K tail (~21.3% projected); reweight via `--dataset_mix_weight_override` if the gate is weak.

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

# concept_ablation_ce(..., window_k): adds delta_{zero,shuffle}_beyond_window (t >= window_k)
#   + ce_intact_{beyond,within}_window  — the E05 long-range memory gate.
#   NB: window_k slices at K, but the true local reach is L*(K-1); co-report
#   --ablation_window_k L*(K-1) for an unconfounded concept-only read.

# sketch: data/dataset_preprocess.py
# Recipes live in data/mix_recipes/*.json (e.g. smollm3_inspired_2k); the legacy
# DATASET_MIXES dict (e.g. long_2k_base_v1) stays for back-compat. Either resolves
# through load_and_preprocess_dataset_mix(tokenizer, mix, ...):
#   per source: load(cap) -> "text" -> split -> tokenize; then
#   interleave_datasets(parts, probabilities=weights, stopping_strategy="all_exhausted")
#   + concatenate_datasets(eval_parts)
```

## 10. Optimizer A/B (Adam vs Muon) — divergence + root cause + mitigations (2026-07-01)

E05 ran an optimizer A/B: Adam (attempt 3, completed, eval_loss 3.83) vs **Muon** (`nn/muon.py`,
fresh, token-matched). Muon converges ~5× faster then **diverges** — a recorded finding, not a
dead end. Full detail: [run report](../../2_Experiments_Registry/run_reports/e05_muon_divergence_rootcause_20260701.md).

- **What:** Muon diverged at LR 0.02 (onset ~step 3k) and 0.01 (onset ~step 4.5k) — grad_norm
  0.3 → millions, loss climbing, surviving `max_grad_norm=0.5`. Adam (5e-5) stable to 0.5 ep.
  Positive signal: Muon eval_loss **3.34** at step 4k vs Adam's ~5.40.
- **Root cause (confirmed vs our config + Moonlight arXiv:2502.16982 / Kimi K2 MuonClip /
  Jianlin Su QK-Clip):** Muon's full-rank orthogonalized updates grow every weight singular
  direction uniformly; in the **Q·K + lm_head bilinear couplings** this runs away (MaxLogit/
  MaxOutput explosion → delayed grad spike). Three enablers in our config:
  1. **Muon weight decay = 0.0** (HF default; launcher didn't set it). Moonlight: wd is the
     long-horizon stabilizer (their wd=0.1); wd=0 → weight-RMS grows unbounded. **Prime suspect.**
  2. **`adamw_lr = 2e-3`** for the lm_head/embed fallback — anomalously high (refs use ≤ muon_lr,
     often 2e-6); over-updates the lm_head (the output-logit bilinear form). **Co-conspirator.**
  3. Update scale Keller `√max(1, A/B)` vs Moonshot `0.2·√max(A,B)` — candidate (deferred).
- **Calibration blind spot:** the LR-0.01 calibration used `NUM_EPOCHS=0.05` → a compressed cosine
  that decayed LR before step 4,500, so the sustained-LR onset never triggered. **Lesson: calibrate
  under sustained peak LR (`constant_with_warmup`), not a short cosine.**

### Mitigations implemented (2026-07-01)
- `scripts/train_perceiver_denoise_multigpu.sh`: wired `WEIGHT_DECAY` (→ HF `--weight_decay` →
  `create_optimizer` → `nn.muon.Muon`) + `LR_SCHEDULER_TYPE` (default cosine; `constant_with_warmup`
  for sustained-LR calibration) knobs.
- `scripts/launch_e05.sh` Muon branch now defaults: `LEARNING_RATE=0.01`, `MUON_ADAMW_LR=2e-4`
  (was 2e-3), `WEIGHT_DECAY=0.1` (was 0.0). Adam branch unchanged (LR 5e-5, wd 0.0).
- **Deferred (next-tier):** Moonshot update scale `0.2·√max(A,B)` in `nn/muon.py` (one-line, but
  changes LR semantics → needs LR retune); QK-Clip/QK-Norm (Kimi K2 MuonClip) — directly caps the
  MaxLogit runaway; more invasive (model attention code).
- **A/B caveat:** Muon wd=0.1 vs Adam wd=0.0 = two variables. Accept it ("wd=0.1 is what Muon needs")
  or re-run Adam at wd=0.1 for a single-variable A/B. **Decision pending.**

### Reuse map for the mitigation plumbing
| Component | Change | Where |
|---|---|---|
| `PerceiverDenoiseTrainer.create_optimizer` | already passes `self.args.weight_decay` → `Muon(weight_decay=...)` | `training/train_perceiver_denoise.py` |
| launcher weight-decay + scheduler knobs | new `WEIGHT_DECAY`, `LR_SCHEDULER_TYPE` env → `--weight_decay`, `--lr_scheduler_type` | `scripts/train_perceiver_denoise_multigpu.sh` |
| E05 Muon-arm defaults | wd 0.1, adamw_lr 2e-4, LR 0.01 | `scripts/launch_e05.sh` |
| `OptimizerArguments.muon_adamw_lr` | already wired (env `MUON_ADAMW_LR`) | `training/train_perceiver_denoise.py` |

