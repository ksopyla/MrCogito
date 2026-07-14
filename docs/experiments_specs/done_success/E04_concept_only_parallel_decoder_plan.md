# E04 — Implementation Plan (concept-only parallel decoder, linear Perceiver-IO)

- **Spec:** [E04_concept_only_parallel_decoder.md](E04_concept_only_parallel_decoder.md) · **Status:** implemented, run, and evaluated 2026-06-20; closed mixed
- **Authored by:** `implementation-plan` · for → `research-implement`

> The HOW for the spec's single change: swap the causal-AR decoder for a **bypass-free
> parallel decoder**, matched to the just-finished E03 anchor-OFF control. This plan reframes
> E04 (per the 2026-06-18 grill) as a **concept-formation A/B between two decoder families**
> (parallel Perceiver-IO vs causal-AR), judged on **within-sample RankMe + zero-shot STS-B**
> vs the control — NOT as proof that "the AR bypass causes collapse" (a swap changes 3 things
> at once: bypass, info-channel, prediction target → correlational only; the *mechanism* claim
> belongs to the decoder-weakening dose-response sibling and/or E05).

## 1. Source & fit
- **Origin:** agenda "attack concept collapse at the root"; collapse root-cause reframe (2026-06-14,
  `agenda.md`) — posterior-collapse via a teacher-forced AR bypass (Bowman 2015; Chen VLAE 2016; Alemi 2018).
- **Synthesis verdict:** *Adapt.* Take "remove the token bypass"; **drop** the causal claim from a single
  swap. Honest deliverable = first **matched** parallel-vs-AR concept-formation comparison under the new
  metrics (within-sample RankMe, the old perceiver runs predate this tooling).
- **Architecture mapping (ONE):** **decoder/head.** Encoder, concepts, tokenizer, data, objective fixed.

### Key facts established from code (read before building)
- `perceiver_posonly` is **Perceiver-IO**, not Perceiver-AR: `PerceiverDecoderStack` queries are pure
  position embeddings (`query_embeddings(position_ids)`, `nn/concept_encoder_perceiver.py:130-137`),
  cross-attend the C concepts, predict all N positions in parallel. No token content in the queries →
  the spec's "no input-token leak" verification already holds.
- **Deviation from canonical Perceiver-IO:** `PerceiverDecoderLayer` adds **full bidirectional
  self-attention over the N position queries** (`:93-100`) → **O(N²)**, violating the project invariant
  "O(C·N), no O(N² decoder self-attention)". This is the cost wall for longer context. → **Change 2.**
- **Data-contract bug for this path** (the launch blocker): EOS-append is gated on `is_causal_ar`
  (`training/train_perceiver_denoise.py:635`), so the perceiver path takes the default preprocessing
  (`data/dataset_preprocess.py:124-132`, `padding="max_length"`) → every row pre-padded to 512.
  `DataCollatorForTSDAE` ignores the tokenizer `attention_mask` and rebuilds it from *length*
  (`data/data_collators.py:81-84`), so all 512 positions are marked real: the encoder attends the
  eos/pad tail and the decoder is trained to predict `<|endoftext|>` on the padding. It also makes
  E04's data contract **differ from its own E03 baseline** (which used the EOS-append path). → **Change 1.**

## 2. Reuse map (read the modules first)
| Component | Action | Where |
|---|---|---|
| `ConceptEncoderForDenoisingPerceiver` | reuse as-is (forward, encode→decode→loss) | `nn/concept_encoder_perceiver.py:150` |
| `ConceptEncoder` (BiXT) | reuse as-is (identical in both arms → comparable) | `nn/concept_encoder.py:516` |
| `PerceiverDecoderLayer` | **edited**: output self-attn removed outright (no flag — N² declared wrong, not worth a compat knob) | `nn/concept_encoder_perceiver.py:63` |
| `PerceiverDecoderStack` | reuse (now linear) | `nn/concept_encoder_perceiver.py:116` |
| `AnchorDistillHead` (E03) | benefits — its docstring already claimed "O(C·N), no self-attn"; now consistent | `nn/concept_encoder_perceiver.py:1115` |
| `dataset_preprocess.load_and_preprocess_text_dataset` | reuse; caller passes EOS for recon path | `data/dataset_preprocess.py:65` |
| `train_perceiver_denoise.py` (entrypoint) | **edited**: `resolve_append_eos_token_id()` appends EOS for recon objectives too | `training/train_perceiver_denoise.py` |
| `DataCollatorForTSDAE` | reuse **after Change 1** (variable-length rows → mask correct) | `data/data_collators.py:24` |
| `utils_training.build_perceiver_wandb_identity` | **edited**: `decoder:`/`task:` tags + legible `job_type` + E04 default | `training/utils_training.py:484` |

## 3. Forward pass (tensor shapes)
Symbols: `B`=batch, `N`=tokens(≤512), `C`=128, `H`=768, `V`=SmolLM2 vocab.
```
(B, N)            → embed + token-pos (dim_tok=256) → proj         → (B, N, 256)   # ConceptEncoder
(B, N, 256)+C init→ BiXT cross/self ×L6  [S = r_lat@r_tok^T → B,h,C,N]  O(C·N)     → (B, C, 768)
(B, C, 768)       → decoder queries = pos-embed(0..N-1)            → (B, N, 768)
  per layer ×D4:  cross-attn(Q=pos-queries, KV=concepts)           → (B, N, 768)   O(N·C)  ✔
                  (NO self-attn over N queries — removed; linear Perceiver-IO)
(B, N, 768)       → lm_head                                        → (B, N, V)
loss = CE(logits, labels) over non-pad positions  (labels=-100 on pad, after Change 1)
```

## 4. Inputs & data
- **Dataset:** `HuggingFaceFW/fineweb-edu` `sample-10BT`, SmolLM2-135M tokenizer — **== E03 control.**
- **Collator:** `DataCollatorForTSDAE` (`deletion_rate=0.6`), reused **after Change 1**.
- **Change 1 (preprocessing/EOS):** append EOS for the reconstruction objective on the perceiver path,
  so rows are **variable-length** (not `padding="max_length"`); the collator then sets `encoder_mask=0`
  / `labels=-100` on padding correctly. Result: encoder ignores pad; decoder is not trained on pad;
  **data contract identical to the E03 control.** Rebuild the FineWeb-Edu tokenization cache for this path.
- **EOS/PAD:** `<|endoftext|>` = bos=eos=unk; `pad_token=eos`; `embedding_padding_idx → None` so the eos
  row stays trainable; EOS appended at the **end of content**, pad-fill beyond is masked to -100.

## 5. Loss & training objective
- **Loss:** dense token CE at every non-pad position (`reconstruction`), via the existing
  `reconstruction_loss` + `LossManager` (no concept losses, no anchor). **Objective:** denoise/reconstruct
  the full clean sequence from concepts. **Weighting:** none (task loss only).

## 6. Config & launch
- **No new config field** — output self-attn removed outright (N² declared wrong; no compat knob).
  The decoder is always linear Perceiver-IO now.
- **EOS-append** handled by `resolve_append_eos_token_id()` (no launcher knob needed).
- **No new MODEL_REGISTRY entry** (still `ConceptEncoderForDenoisingPerceiver`, family `perceiver_denoise`).
  Eval routing unchanged (geometry + STS-B both run on this family).
- **Launch (Odra 3× RTX 3090, after changes land + smoke passes; ≥1 epoch, 512 ctx):**
  ```bash
  EXPERIMENT_ID=E04 DECODER_TYPE=perceiver_posonly \
  HIDDEN_SIZE=768 TOKEN_EMBEDDING_DIM=256 NUM_LAYERS=6 DECODER_NUM_LAYERS=4 \
  CONCEPT_NUM=128 INTERMEDIATE_SIZE=2048 HIDDEN_ACT=silu NORM_TYPE=rmsnorm \
  OBJECTIVE_VARIANT=reconstruction DELETION_RATE=0.6 MAX_SEQ_LENGTH=512 \
  TOKENIZER_NAME=HuggingFaceTB/SmolLM2-135M \
  DATASET_NAME=HuggingFaceFW/fineweb-edu DATASET_SUBSET=sample-10BT \
  SEED=42 NUM_EPOCHS=1 PER_DEVICE_BATCH_SIZE=24 GRADIENT_ACCUMULATION_STEPS=2 \
  LEARNING_RATE=3e-4 WARMUP_STEPS=500 LOGGING_STEPS=100 EVAL_STEPS=2000 SAVE_STEPS=2000 \
  SAVE_TOTAL_LIMIT=3 DDP_TIMEOUT=3600 uv run bash scripts/train_perceiver_denoise_multigpu.sh
  ```
  (512 ctx, matched to the E03 control. Longer context = a later step: needs a data mix + a re-run
  control at the new length + the 16:1 compression-ratio caveat; deferred by decision.)
- **First-pass kill gate:** at the 0.3-epoch eval, confirm eval CE is finite/decreasing and within-sample
  RankMe is being logged; if NaN/diverging, stop. Otherwise let it run the full epoch.

## 7. Tests & smoke (DONE 2026-06-18)
- **Unit (linear decoder):** `tests/test_perceiver_denoise.py::test_perceiver_decoder_is_linear_no_output_self_attention`
  — asserts no `self_attn`/`pre_self_norm`, `cross_attn` present, no `self_attn` keys in state dict.
- **Unit (data contract):** `::test_perceiver_reconstruction_appends_eos_like_causal_ar` — `resolve_append_eos_token_id`
  returns EOS for the perceiver recon path (regression guard for the pad-mask bug). Existing
  `tests/test_tsdae_collator.py::TestTSDAECollatorPadAliasesEos` covers the variable-length mask/label correctness.
- **Unit (W&B):** `::test_wandb_identity_tags_parallel_reconstruction` / `::..._ar_generation` — decoder/task tags + job_type.
- **Local smoke (passed):** SmolLM2 tokenizer → EOS-append preprocessing → `DataCollatorForTSDAE` → model
  forward+backward: rows variable-length (8/18, not 512), pad→`-100`, encoder ignores pad, boundary EOS is a
  target, decoder has no `self_attn`, loss finite (11.39), gradients flow. Full suite: `tests/test_perceiver_denoise.py`,
  `test_tsdae_collator.py`, `test_concept_anchor.py`, `test_concept_ar_decoder.py` all green (45 tests).

## 8. Risks & tradeoffs
- **Risk:** removing output self-attn hurts reconstruction fluency. **Acceptable** — spec says recon CE may
  be worse; this is a representation probe. **Cheapest signal:** within-sample RankMe + STS-B vs control.
- **Risk:** parallel decoder *also* collapses (the old perceiver family did, at slot-rank ~5/128 but STS-B
  ~0.607). If so, E04 corroborates "collapse is largely a measurement artifact / not bypass-specific" —
  still a useful negative. **Kill:** by 25% budget, if RankMe **and** Δshuffle ≤ control → stop.
- **Risk:** cache rebuild for the EOS path costs one-off tokenization time on Polonez. Mitigate with
  `TRAIN_NUM_PROC` headroom.
- **Fallback:** if Change 1 is deemed too broad, the narrower alternative is to make `DataCollatorForTSDAE`
  honor the tokenizer `attention_mask` (fixes the bug but leaves the no-EOS contract differing from the
  baseline — worse for comparability). Preferred = Change 1.

## 9. Code sketches (`# sketch` — decisions, not demos)
```python
# sketch: nn/concept_encoder.py — ConceptEncoderConfig.__init__
decoder_output_self_attn: bool = True   # False → canonical linear Perceiver-IO (O(N·C)); default keeps old ckpts

# sketch: nn/concept_encoder_perceiver.py — PerceiverDecoderLayer
self.use_output_self_attn = getattr(config, "decoder_output_self_attn", True)
if self.use_output_self_attn:
    self.self_attn = nn.MultiheadAttention(...)   # only built when enabled
# forward(): cross-attn + FFN always; self-attn block guarded by self.use_output_self_attn

# sketch: training/train_perceiver_denoise.py — EOS for any full-reconstruction objective
appends_eos = (model_args.objective_variant == OBJECTIVE_RECONSTRUCTION) or is_causal_ar
append_eos_token_id = tokenizer.eos_token_id if (appends_eos and tokenizer.eos_token_id is not None) else None
```
