# E03 — Implementation Plan

- **Spec:** [E03_concept_anchor_decollapse.md](E03_concept_anchor_decollapse.md) · **Status:** draft
- **Authored by:** `implementation-plan` · for → `research-implement`

> The HOW for the spec's single change: add an auxiliary per-token hidden-state distillation
> loss (concepts must reconstruct a frozen SmolLM2 teacher's per-token hidden states) on top of
> the E01 AR objective. Backward-compatibility is a hard requirement: `anchor_loss=False` (default)
> must leave the E01 model, checkpoints, and eval routes byte-for-byte unchanged.

## 1. Source & fit
- **Origin:** the chronic concept-collapse failure (rank 5–10/128 across ~60 runs); diffusion-LM
  scout ([CALM/ELF/Cosmos/LDLM](docs note)) verdict — *"validate the bottleneck first: anchor the
  latent to frozen pretrained hidden states (MSE-to-h) before any fancy decoder."* This is the shared
  Stage-A both the diffusion and recursion bets depend on.
- **Synthesis verdict:** **Adapt** — take the *frozen-encoder hidden-state anchor + standardized MSE*
  ingredient (Cosmos/LDLM/TEncDM); **drop** the diffusion decoder (AR stays the cleaner semantics
  probe), drop CFG/flow (out of scope), drop cross-tokenizer alignment (use a same-tokenizer teacher).
- **Architecture mapping (ONE):** this touches the **loss** (a new auxiliary regression objective)
  and adds a small **decoding head** that reads the existing concept bottleneck. Encoder, AR decoder,
  data, and tokenizer are all held at E01.
- **Scope boundary (deferred threads recorded in [agenda.md](../1_Strategy_and_Plans/agenda.md)):**
  E03 is the shared **Stage-A** de-collapse step. **Out of scope here**, tracked in the agenda roadmap:
  the unified eval interface + `lighteval` backend (engineering, staged), **Ouro-style recursion** (E04 —
  needs E03 + a depth-dependent bench), and **diffusion decode** (deferred ELF-style staged program; not
  CALM, not a bare AdaLN re-run). The **PCA/embedding warm-start module is NOT needed for E03** (frozen
  teacher, train-from-scratch); it is built when the first warm-start run needs it.

## 2. Reuse map (read the modules first)
| Component | Action | Where |
|---|---|---|
| `ConceptEncoderForConditionalLM` (E01 AR model) | extend: build anchor head when `config.anchor_loss`; add `anchor_predict()` | `nn/concept_encoder_perceiver.py:1078` |
| `ConceptEncoderForConditionalLM.encode_concepts/decode_logits/_teacher_forced_ce/_shift_right` | reuse as-is (AR loss + concepts) | `nn/concept_encoder_perceiver.py:1133/1163/1173/1152` |
| `PerceiverDecoderLayer` | reuse as-is — building block of the lean anchor head | `nn/concept_encoder_perceiver.py:59` |
| `PerceiverDecoderStack.build_queries` | reuse the position-query pattern (own query embeddings) | `nn/concept_encoder_perceiver.py:112/126` |
| `ConceptEncoderConfig` | extend: add `anchor_*` fields (backward-compatible defaults) | `nn/concept_encoder.py:43` |
| `PerceiverDenoiseTrainer.compute_loss` | extend: add an anchor manual-loss branch (mirrors the contrastive branch) | `training/train_perceiver_denoise.py:326` |
| `PerceiverDenoiseTrainer` (`__init__`, `_concept_ablation_metrics`) | extend: hold frozen teacher; log `anchor/mse` at eval | `training/train_perceiver_denoise.py:192/229` |
| `build_perceiver_denoise_config` / `ModelArguments` | extend: thread `anchor_*` args; infer teacher hidden via `AutoConfig` | `training/train_perceiver_denoise.py:379/72` |
| `DataCollatorForTSDAE` | reuse as-is (contract below) | `data/data_collators.py:24` |
| `scripts/train_perceiver_denoise_multigpu.sh` | extend: add `ANCHOR_*` knobs as a conditional arg block (RESUME_ARGS pattern) | `scripts/...sh:90` |
| `AnchorDistillHead` (lean: `anchor_head_layers` × `PerceiverDecoderLayer` + query emb + `Linear→teacher_hidden`) | **new — reusable, config-selectable** | `nn/concept_encoder_perceiver.py` |
| frozen teacher (`AutoModel.from_pretrained(anchor_model_name)`, eval/no-grad) | **new — held by the Trainer, NOT a saved submodule** | `training/train_perceiver_denoise.py` |

**Backward-compat invariant:** when `config.anchor_loss is False` the model builds **no** anchor
submodules → identical `state_dict` to E01; old checkpoints load unchanged. The frozen teacher is
never registered on the model, so it is never written to a checkpoint and is not needed at
eval/analysis. The anchor head **is** saved (extra `anchor_head.*`/`anchor_proj.*` keys); the
`concept_ar` eval route is `weighted_pool` (encoder-only) and already ignores the AR `decoder.*`/
`lm_head.*` keys, so extra anchor keys are tolerated the same way (HF logs unexpected keys, no error).
`run_concept_analysis.py` loads the full `ConceptEncoderForConditionalLM`, which rebuilds the anchor
head from `config.anchor_loss` (teacher not required).

## 3. Forward pass (tensor shapes)
Symbols: `B`=batch, `N`=tokens (≤512), `C`=128 concepts, `H`=768, `Ht`=teacher hidden (SmolLM2-135M = **576**, auto-inferred), `V`=vocab (49,152).

Training step (anchor objective), all in the Trainer's `compute_loss` manual branch:
```
# --- E01 AR loss (unchanged) ---
input_ids [B,N] (clean), attention_mask [B,N] (1=visible,0=deleted|pad), labels [B,N] (clean,-100@pad)
encode_concepts(input_ids, attention_mask)          -> concepts [B, C, H]
decoder_input = _shift_right(input_ids)             -> [B, N]
decode_logits(concepts, decoder_input, wd=cfg.decoder_word_dropout) -> logits [B, N, V]
ar_loss = _teacher_forced_ce(logits, labels)        -> scalar

# --- NEW anchor loss ---
with no_grad:
   teacher_mask = (labels != -100).long()           # clean, non-pad (NOT the TSDAE mask)
   teacher_h = teacher(input_ids, teacher_mask).last_hidden_state   -> [B, N, Ht]
   if standardize: teacher_h = F.layer_norm(teacher_h, (Ht,))       # per-token zero-mean/unit-var
pred_h = anchor_predict(concepts, N)                # AnchorDistillHead: pos-queries x-attn concepts -> [B,N,H] -> Linear -> [B,N,Ht]
anchor_mse = masked_mse(pred_h, teacher_h, teacher_mask)            # mean over non-pad positions
total = (loss_manager(ar_loss, concepts) if enabled else ar_loss) + cfg.anchor_loss_weight * anchor_mse
```
`anchor_predict` keeps O(C·N): position queries `[B,N,H]` cross-attend to `C` concepts (no token
self-attention). Eval uses the **unchanged** `model.forward` → pure AR CE (so `eval_loss` stays a
clean, comparable next-token CE for best-checkpoint selection); `anchor/mse` is logged separately.

## 4. Inputs & data
- **Dataset:** `HuggingFaceFW/fineweb-edu` `sample-10BT`, same subsample/preprocessing as E01; EOS
  appended per doc (`append_eos_token_id`, set because `causal_ar`). **Collator:** `DataCollatorForTSDAE`
  **reused unchanged** (train: fresh deletions; eval: seeded). **Tokenizer:** `HuggingFaceTB/SmolLM2-135M`
  — **mandatory** so teacher and model share token ids 1:1 (startup assert: tokenizer name ==
  `anchor_model_name` family). Teacher target = clean `input_ids` + `(labels!=-100)` mask, **not** the
  corrupted `attention_mask` (concepts must regenerate *uncorrupted* semantics — adds denoising pressure).

## 5. Loss & training objective
- **Objective:** unchanged `reconstruction` (AR denoising CE via `_teacher_forced_ce`). The anchor is an
  **orthogonal additive auxiliary** gated by `anchor_loss`, *not* a new `objective_variant` (so it can
  later compose with `prefix_suffix`; out of scope here).
- **New loss:** `anchor_mse = mean_{non-pad} || layer_norm(teacher_h) − Linear(head(concepts)) ||²`,
  weighted by `anchor_loss_weight` (λ, default 0.5; calibrate in smoke so λ·MSE ≈ AR CE magnitude).
  Computed in `compute_loss` (mirrors the existing contrastive manual path at `:338-376`), not in
  `loss_manager` (which is concept-geometry regularizers over `[B,C,H]`, a different signature).
- **Weighting:** `total = ar_loss + λ·anchor_mse`; concept `loss_manager` stays disabled
  (`CONCEPT_LOSSES=none`) as in E01.

## 6. Config & launch
- **New `ConceptEncoderConfig` fields** (`nn/concept_encoder.py`, backward-compatible defaults):
  `anchor_loss: bool = False`, `anchor_model_name: Optional[str] = None`,
  `anchor_loss_weight: float = 0.5`, `anchor_standardize: bool = True`,
  `anchor_head_layers: int = 2`, `anchor_target_layer: int = -1`,
  `anchor_teacher_hidden: Optional[int] = None` (set in `build_perceiver_denoise_config` from
  `AutoConfig.from_pretrained(anchor_model_name).hidden_size` so the head is rebuildable from config
  alone, no teacher needed at eval/analysis).
- **New `ModelArguments`** (`training/train_perceiver_denoise.py:72`): `anchor_loss`, `anchor_model_name`
  (default `HuggingFaceTB/SmolLM2-135M`), `anchor_loss_weight`, `anchor_standardize`, `anchor_head_layers`.
- **Registry / routing:** none. Same `concept_ar` family / `weighted_pool` single route /
  `sentence_pair` pair route; `MODEL_CLASSES["concept_ar"]` already maps to
  `ConceptEncoderForConditionalLM`. The contract version is unchanged.
- **Launch (anchor arm):** conditional `ANCHOR_ARGS` block in the existing launcher (RESUME_ARGS
  pattern); the **control arm is the same command with `ANCHOR_LOSS=false` == E01**:
  ```bash
  EXPERIMENT_ID=E03 DECODER_TYPE=causal_ar \
  HIDDEN_SIZE=768 TOKEN_EMBEDDING_DIM=256 NUM_LAYERS=6 DECODER_NUM_LAYERS=4 \
  CONCEPT_NUM=128 INTERMEDIATE_SIZE=2048 HIDDEN_ACT=silu NORM_TYPE=rmsnorm DECODER_POS_TYPE=rope \
  OBJECTIVE_VARIANT=reconstruction DELETION_RATE=0.6 DECODER_WORD_DROPOUT=0.2 \
  TOKENIZER_NAME=HuggingFaceTB/SmolLM2-135M \
  DATASET_NAME=HuggingFaceFW/fineweb-edu DATASET_SUBSET=sample-10BT \
  ANCHOR_LOSS=true ANCHOR_MODEL=HuggingFaceTB/SmolLM2-135M \
  ANCHOR_LOSS_WEIGHT=0.5 ANCHOR_STANDARDIZE=true ANCHOR_HEAD_LAYERS=2 \
  SEED=42 NUM_EPOCHS=0.3 SAVE_TOTAL_LIMIT=5 \
  bash scripts/train_perceiver_denoise_multigpu.sh
  ```
  Keep `EXPERIMENT_ID=E03` on the anchor-OFF control as well, so W&B groups the matched pair
  together while `ANCHOR_LOSS`/tags distinguish the arms.

- **Eval-time logging:** `eval_loss` stays **pure AR CE** (the anchor is absent from `model.forward`),
  so `metric_for_best_model="eval_loss"` is unchanged and directly comparable to the control. Extend
  `_concept_ablation_metrics` (`training/train_perceiver_denoise.py:229`, main process) to also log
  `anchor/mse_eval` over the same few held-out batches (the frozen teacher lives on the trainer) — so
  de-collapse is watchable on held-out data without polluting the selection metric.
- **Run order (the matched pair):** two jobs, identical `SEED`/data/budget — `ANCHOR_LOSS=true` and
  `ANCHOR_LOSS=false`. (1) 0.3-epoch warm-up gate on **both** → check the spec's 25% kills
  (`rank(anchor) − rank(control) ≥ +5`, AR CE not diverging vs control, `anchor/mse` decreasing).
  (2) If healthy, a matched fuller pair (~1 epoch) for the rank / STS-B / ΔCE verdict.

## 7. Tests & smoke
- **Unit** `tests/test_concept_anchor.py` (tiny random tensors, fake teacher `Ht=16`):
  - `anchor_loss=False` → model has **no** `anchor_head`/`anchor_proj`; `state_dict()` keys == E01 (backward-compat).
  - `anchor_loss=True` → `anchor_predict(concepts[B,C,H], N)` returns `[B, N, Ht]`; masked MSE finite; backward populates `anchor_head`/`anchor_proj` grads.
  - `model.forward(...)` (eval path) returns pure AR CE and **never** touches the teacher.
  - An E01 (`anchor_loss=False`) checkpoint loads into an `anchor_loss=True` config with only the anchor head randomly initialized (and vice-versa loads with unexpected-key warning, no error).
- **Smoke (MPS)**: `ANCHOR_LOSS=true ... MAX_STEPS≈20` tiny run — assert loss finite, `anchor/mse`
  logged and trending down, `concept_geometry/effective_rank` logged (already emitted by
  `_concept_effective_rank` at `:280`); then `run_concept_analysis.py --model_type concept_ar` on the
  smoke checkpoint loads and reports geometry.

## 8. Risks & tradeoffs
- **Powerful head hides collapse** (a deep head reconstructs `teacher_h` from a low-rank concept set →
  rank stays low but anchor MSE looks great). **Mitigation:** lean head (`anchor_head_layers=2`),
  and judge on `effective_rank` + STS-B, **never** anchor MSE alone. **Cheapest signal:** the spec's
  25% gate — `rank(anchor) − rank(control) < +5` → kill.
- **λ mis-set** (too high → AR CE regresses past the 0.2-nat guardrail; too low → no de-collapse).
  **Mitigation:** calibrate λ in the 0.3-epoch warmup; both failure modes are caught by the 25% gates.
- **Wrong teacher mask** (using the TSDAE-corrupted `attention_mask` would teach concepts to mirror a
  *corrupted* encoding). **Mitigation:** teacher mask = `(labels != -100)`; asserted in the unit test.
- **Standardization instability.** v1 = per-token `F.layer_norm` of targets (no learnable params);
  per-dim running standardization is a fallback if MSE scale drifts. Labeled a calibratable default.
- **Compute:** one extra frozen SmolLM2-135M forward/step (bf16, no-grad) — cheap on a 3090; teacher
  excluded from the optimizer/state_dict. DDP `find_unused_parameters=False` is safe: the anchor head
  is used every training step.
- **Fallback:** if de-collapse fails with a same-tokenizer *causal* teacher (SmolLM2), the spec's
  named follow-up is a one-variable swap to a frozen *bidirectional* teacher (ModernBERT) — which then
  needs the cross-tokenizer embedding work, deliberately deferred.

## 9. Code sketches (`# sketch` — decisions, not demos)
```python
# sketch — nn/concept_encoder_perceiver.py
class AnchorDistillHead(nn.Module):
    """Lean position-query head: reconstruct a frozen teacher's per-token hidden states from concepts.
    Reuses PerceiverDecoderLayer; O(C*N) (no token self-attention). Saved with the model."""
    def __init__(self, config, teacher_hidden: int):
        super().__init__()
        self.query_embeddings = nn.Embedding(config.max_sequence_length, config.hidden_size)
        self.layers = nn.ModuleList(PerceiverDecoderLayer(config) for _ in range(config.anchor_head_layers))
        self.output_norm = nn.LayerNorm(config.hidden_size)
        self.proj = nn.Linear(config.hidden_size, teacher_hidden)        # H -> Ht
    def forward(self, concepts, seq_length):                            # concepts [B,C,H]
        pos = torch.arange(seq_length, device=concepts.device).unsqueeze(0)
        h = self.query_embeddings(pos).expand(concepts.size(0), -1, -1)  # [B,N,H]
        for layer in self.layers:
            h = layer(h, concepts)                                       # x-attn to concepts
        return self.proj(self.output_norm(h))                            # [B,N,Ht]

# in ConceptEncoderForConditionalLM.__init__:  (backward-compat: only when enabled)
if getattr(config, "anchor_loss", False):
    self.anchor_head = AnchorDistillHead(config, config.anchor_teacher_hidden)
def anchor_predict(self, concepts, seq_length):
    return self.anchor_head(concepts, seq_length)

# sketch — training/train_perceiver_denoise.py : PerceiverDenoiseTrainer.compute_loss
# fall through to the manual path when anchor is on (mirrors the contrastive branch):
#   if not model.training or (objective in {RECON, PREFIX} and not self.anchor_loss): return model(**inputs).loss
base = model.module if hasattr(model, "module") else model
concepts = base.encode_concepts(input_ids, attention_mask, return_dict=True).last_hidden_state
wd = base.config.decoder_word_dropout if model.training else 0.0
logits = base.decode_logits(concepts, base._shift_right(input_ids), word_dropout_p=wd)
ar = base._teacher_forced_ce(logits, labels)
with torch.no_grad():
    tmask = (labels != -100).long()
    th = self.anchor_teacher(input_ids=input_ids, attention_mask=tmask).last_hidden_state
    if base.config.anchor_standardize: th = F.layer_norm(th, (th.size(-1),))
ph = base.anchor_predict(concepts, input_ids.size(1))
m = (tmask.unsqueeze(-1) > 0)
anchor_mse = (((ph - th) ** 2) * m).sum() / m.sum().clamp(min=1).float() / th.size(-1)
total = (base.loss_manager(task_loss=ar, concept_repr=concepts) if base.loss_manager.is_enabled else ar) \
        + base.config.anchor_loss_weight * anchor_mse
# self.log({"anchor/mse": anchor_mse.item(), "anchor/ar_ce": ar.item()})
```
```python
# sketch — frozen teacher set up once in PerceiverDenoiseTrainer.__init__ (NOT a model submodule)
# from transformers import AutoModel
# self.anchor_teacher = AutoModel.from_pretrained(cfg.anchor_model_name, torch_dtype=torch.bfloat16)
# self.anchor_teacher.eval().requires_grad_(False).to(self.args.device)
```
