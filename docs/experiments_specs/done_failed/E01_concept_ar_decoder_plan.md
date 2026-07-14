# E01 — Implementation Plan

- **Spec:** [E01_concept_ar_decoder.md](E01_concept_ar_decoder.md) · **Status:** implemented and run; closed mixed / failed decisive quality gates
- **Authored by:** `implementation-plan` · for → `research-implement`

> The HOW for the spec's single change (the decoder). Repo-rooted; reuse-first; new code is a
> reusable, config-selectable foundation component, never a fork.

## 1. Source & fit
- **Origin:** Perceiver IO decode pattern (arXiv:2107.14795) is the *parallel* baseline we are
  leaving; the AR decoder follows the classic encoder→decoder Transformer (cross-attn to a fixed
  "memory" = our concepts) and the Perceiver-AR / Flamingo conditioning idea (concepts condition an
  AR decoder). See the comparison + decoding analysis in this chat; SmolLM2 (arXiv:2502.02737) is the
  modern small-LM reference (RoPE/SwiGLU/RMSNorm) and the later warm-start target.
- **Synthesis verdict:** *Adapt.* Take: per-position **cross-attention to concepts** (keep). Add:
  **causal self-attention over target tokens** + next-token CE (the missing AR ingredient). Drop:
  the conditional-independence parallel decode for generation; the Input+Position "hint" query (leaks).
- **Architecture mapping (ONE):** this touches the **decoder/head** (+ a small loss/objective change:
  next-token CE with decoder-input word-dropout).

## 2. Reuse map (read the modules first)
| Component | Action | Where |
|---|---|---|
| `ConceptEncoder`, `BiConceptEncoderLayer`, `BiXTCrossAttention` | reuse as-is (BiXT + token asymmetry) | `nn/concept_encoder.py` |
| `PerceiverDecoderStack`, `ConceptEncoderForSequenceClassificationViaDecoder`, `…ForSentencePairClassification` | reuse as-is (frozen STS-B / representation probe path) | `nn/concept_encoder_perceiver.py` |
| `ConceptEncoderForDenoisingPerceiver` | reuse as-is (the position-decoder baseline stays runnable) | `nn/concept_encoder_perceiver.py` |
| `ConceptCausalDecoderLayer` | **new** — causal self-attn (manual q/k/v + **RoPE**, SDPA `is_causal`) + cross-attn(concepts) + SwiGLU FFN | `nn/concept_encoder_perceiver.py` |
| `ConceptCausalDecoderStack` | **new** — token embeds (Ht→H proj) + N causal layers + final norm + lm_head | `nn/concept_encoder_perceiver.py` |
| `ConceptEncoderForConditionalLM` | **new** — encoder→concepts→AR decoder; next-token CE | `nn/concept_encoder_perceiver.py` |
| `ConceptEncoderConfig` | extend: `decoder_type`, `decoder_word_dropout`, `norm_type`, `decoder_pos_type`; honor `hidden_act` | `nn/concept_encoder.py` |
| FFN activation (`nn.GELU()` hardcoded) | extend: build from `ACT2FN[config.hidden_act]` (SwiGLU=silu) | `nn/concept_encoder.py`, `nn/concept_encoder_perceiver.py` |
| Norm (`nn.LayerNorm` hardcoded) | extend: `build_norm(norm_type, dim)` helper → `RMSNorm` when `rmsnorm` | `nn/concept_encoder.py`, `nn/concept_encoder_perceiver.py` |
| Tokenizer special tokens | handle: SmolLM2 has no `[MASK]`/`[CLS]`/`[SEP]`; set `pad=eos`; word-dropout fill = `unk`/reserved id | `training/train_perceiver_denoise.py` |
| `DataCollatorForTSDAE` | reuse as-is (objective A); word-dropout done in model forward | `data/data_collators.py` |
| `train_perceiver_denoise.py` | extend: `decoder_type`/`hidden_act`/`decoder_word_dropout` args; pick model class by `decoder_type`; build config | `training/` |
| `scripts/train_perceiver_denoise_multigpu.sh` | extend: new `"${VAR:-default}"` knobs (no copy) | `scripts/` |
| `MODEL_CLASSES`, eval routing | extend: register `concept_ar` family + checkpoint contract | `analysis/run_concept_analysis.py`, `evaluation/concept_eval_routing.py` |

## 3. Forward pass (tensor shapes)
Symbols: `B`=batch, `N`=input tokens (≤512), `T`=target tokens, `C`=128 concepts, `H`=768 hidden,
`Ht`=256 token-embedding dim (asymmetry), `V`=vocab (~50k).

```
encoder input_ids (B, N)            → embed(Ht)+learned pos, BiXT ×6         → concepts (B, C, H)     # reuse ConceptEncoder; tokens stay thin (Ht); RMSNorm
target  input_ids (B, T)            → shift_right → decoder_input (B, T)
decoder_input (B, T)                → tok_emb(Ht)→proj→H (NO abs pos: RoPE)  → h (B, T, H)            # word-dropout: replace p of embeddings with learned dropout-emb
for layer in causal_decoder ×4:                                                                       # L4 < encoder L6 (lean by design)
    h = h + causal_self_attn(RMSNorm(h), rope=on)                            → (B, T, H)             # RoPE on q,k; SDPA is_causal=True
    h = h + cross_attn(Q=RMSNorm(h), K=V=concepts)                          → (B, T, H)             # read concepts  O(T·C), no mask, no RoPE (concepts orderless)
    h = h + swiglu_ffn(RMSNorm(h))                                           → (B, T, H)
logits = lm_head(RMSNorm(h))                                                 → (B, T, V)             # untied (Ht≠H)
loss   = CE(logits[:, :-1].reshape(-1,V), labels[:, 1:].reshape(-1))         # next-token, ignore_index=-100
```
- **Objective A (recommended):** encoder `input_ids` = clean ids, `attention_mask` = TSDAE deletion
  (encoder can't see deleted tokens); decoder target = clean `input_ids`; `labels` = clean (−100 at pad).
- **Invariant:** encoder stays `O(C·N)`; the only `O(T²)` is the decoder's causal self-attention over
  the **output** length `T` — expected/required for AR and bounded by the answer length, not the context.

## 4. Inputs & data
- **Dataset:** `HuggingFaceFW/fineweb-edu` `sample-10BT`, subsampled (~1–2B tokens). `text` column →
  `load_and_preprocess_text_dataset` handles tokenization + holdout split (no code change).
- **Tokenizer:** `HuggingFaceTB/SmolLM2-135M` — BPE, vocab 49,152. **No `[MASK]`/`[CLS]`/`[SEP]`; base
  has `<|endoftext|>` as bos=eos=unk and no distinct pad.** Handling: set **`pad_token=eos_token`**
  to keep vocab/embeddings aligned with SmolLM2 warm-start. Padding is positional
  (`attention_mask`, labels `-100`), so real eos tokens remain trainable; model embedding layers skip
  `padding_idx` when pad aliases eos. `DataCollatorForTSDAE` already tolerates `None` CLS/SEP/BOS ids;
  TSDAE deletes via `attention_mask` (no `[MASK]`).
- **Preprocessing:** append `<|endoftext|>` (eos) to each document before/after tokenization (the current
  `tokenize_batch_function` does not add eos) so the decoder learns to stop.
- **Collator:** `DataCollatorForTSDAE` (reuse): returns `input_ids` (clean), `attention_mask`
  (deletion-visibility), `labels` (clean, −100 at pad). The model derives `decoder_input_ids` by
  shift-right of `input_ids` (prepend eos as start) and applies **word-dropout** by replacing `p` of
  non-special decoder-input embeddings with a **single learned "dropout" embedding** (`nn.Parameter`,
  applied post-embedding — *not* a token id, since SmolLM2 has no `[MASK]`) inside `forward`.

## 5. Loss & training objective
- **Loss:** plain next-token `CrossEntropyLoss(ignore_index=-100)` at the model level (mirror
  `reconstruction_loss` but causal/shifted). Optional concept regularizers via the existing
  `LossManager` stay available but **off** for E01 (keep the variable count down).
- **Objective:** AR denoising reconstruction. **Posterior-collapse guards (Bowman 2016):** lean
  decoder (**L4 < encoder L6**, decoder intermediate ≤ encoder), decoder-input word-dropout `p=0.4`
  (learned dropout embedding), no encoder-token skip path. **Eval ablations every eval step:**
  concept-zero ΔCE, concept-**shuffle** ΔCE (stronger), and the no-concept floor — all reported to W&B.

## 6. Config & launch
- **New config fields (backward-compatible defaults — old checkpoints load unchanged):**
  `decoder_type: str = "perceiver_posonly"` (existing) | `"causal_ar"`; `decoder_word_dropout: float = 0.0`;
  `norm_type: str = "layernorm"` | `"rmsnorm"`; `decoder_pos_type: str = "learned"` | `"rope"`. Honor
  existing `hidden_act` (default `"gelu"`) via `ACT2FN[config.hidden_act]` (`silu` ⇒ SwiGLU). Add a
  `build_norm(config.norm_type, dim)` helper used in place of hardcoded `nn.LayerNorm` (RMSNorm via
  `nn.RMSNorm` or a small module). E01 sets `silu` + `rmsnorm` + `rope`; defaults preserve all prior runs.
- **Registry / routing:** select model class in `train_perceiver_denoise.py` by `decoder_type`
  (`causal_ar` → `ConceptEncoderForConditionalLM`); add `concept_ar` to `analysis` `MODEL_CLASSES`;
  set `checkpoint_family="concept_ar"`, `evaluation_contract_version=1`,
  `canonical_single_eval_mode="via_decoder"` (repr probe reuses the position decoder),
  `canonical_pair_eval_mode="sentence_pair"`.
- **Launch:** see spec (env-var overrides; new knobs `DECODER_TYPE`, `HIDDEN_ACT`,
  `DECODER_WORD_DROPOUT`, `DATASET_SUBSET` added to the existing launcher).

## 7. Tests & smoke
- Unit test `tests/test_concept_ar_decoder.py`: tiny random config; assert (a) `forward` returns
  finite loss and `logits` shape `(B, T, V)`; (b) decoder self-attention is **causal** (changing a
  future target token does not change earlier-position logits); (c) zeroing **and** shuffling concepts
  changes the loss (concepts are wired in); (d) word-dropout=1.0 ⇒ decoder input is the learned
  dropout embedding everywhere (path exercised).
- Local MPS smoke: a few steps of `train_perceiver_denoise.py` with `--decoder_type causal_ar` on a
  tiny slice (`PYTORCH_ENABLE_MPS_FALLBACK=1`); assert it steps and loss is finite.
- Sanity: run `analysis/run_concept_analysis.py --model_type concept_ar` on the smoke checkpoint
  (effective rank computes; no crash).

## 8. Risks & tradeoffs
- **Risk — posterior collapse** (decoder LMs from its own left context, ignores concepts) — the
  central risk (Bowman 2016). **Guards:** lean decoder L4, word-dropout 0.4, no skip path. **Cheapest
  signal:** concept-zero/shuffle ΔCE + no-concept floor + effective rank (spec success #1). **Fallback:**
  raise word-dropout; thin the decoder further; the decisive structural fix is objective B
  (prefix→suffix, no copy path) — that is exactly E02.
- **Risk — `O(T²)` decoder cost** at long `T`. Mitigation: `T ≤ 512` for E01; long-context is a later
  experiment. Do **not** add token self-attention in the *encoder* (keep `O(C·N)`).
- **Risk — tying:** `token_embedding_dim(256) ≠ hidden_size(768)` ⇒ `lm_head` is **untied** (own
  768→V matrix), as in the existing dimension-inversion path. Keep that; don't force tying.
- **Risk — too many deltas vs old baseline** (data/act/scale). Accepted: E01 is a new reference line;
  metrics compared as anchors. Strict-control fallback documented in the spec.

## 9. Code sketches (`# sketch` — decisions, not demos)
```python
# sketch: nn/concept_encoder_perceiver.py
class ConceptCausalDecoderLayer(nn.Module):
    def forward(self, h, concepts, rope):                  # h:(B,T,H) concepts:(B,C,H)
        # causal self-attn with RoPE: manual q,k,v → rotate(q,k,rope) → SDPA(is_causal=True)
        h = h + self.self_attn_rope(self.n1(h), rope)                              # n* = build_norm(norm_type)
        h = h + self.cross_attn(self.n2(h), concepts, concepts, need_weights=False)[0]   # no RoPE: concepts orderless
        a, g = self.Wi(self.n3(h)).chunk(2, -1)
        return h + self.Wo(self.act(a) * g)                # act = ACT2FN[config.hidden_act] (silu ⇒ SwiGLU)

class ConceptEncoderForConditionalLM(PreTrainedModel):     # selected when config.decoder_type=="causal_ar"
    def forward(self, input_ids, attention_mask=None, labels=None, **_):
        concepts = self.encoder(input_ids, attention_mask).last_hidden_state          # (B,C,H)
        dec_in   = self._shift_right(input_ids)                                        # (B,T) prepend eos/bos
        dec_in   = self._word_dropout(dec_in, p=self.config.decoder_word_dropout)       # fill = unk/reserved (no [MASK])
        h        = self.decoder(dec_in, concepts)                                      # (B,T,H) causal + RoPE
        logits   = self.lm_head(h)                                                     # untied (Ht≠H)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                               labels[:, 1:].reshape(-1), ignore_index=-100) if labels is not None else None
        return MaskedLMOutput(loss=loss, logits=logits)

# sketch: nn/concept_encoder.py  (config — all defaults preserve existing checkpoints)
decoder_type: str = "perceiver_posonly"   # | "causal_ar"
decoder_word_dropout: float = 0.0
norm_type: str = "layernorm"              # | "rmsnorm"
decoder_pos_type: str = "learned"         # | "rope"
# replace nn.GELU() with ACT2FN[self.hidden_act]; replace nn.LayerNorm(...) with build_norm(norm_type, ...)
```
