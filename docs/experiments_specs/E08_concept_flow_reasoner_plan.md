# E08 — Concept-Flow reasoner — Implementation Plan

- **Spec:** [E08_concept_flow_reasoner.md](E08_concept_flow_reasoner.md) · **Status:** draft
- **Authored by:** `implementation-plan` · for → `research-implement`

> The HOW for the spec's single change (insert a Concept-Flow reasoner between the
> E02-long encoder and AR decoder, trained with a teacher-trace endpoint-flow target +
> crystallization CE). Repo-rooted: every class/path below was read before writing this.

## 1. Source & fit
- **Origin:** Paradigm A (encode→reason→decode, decoding-as-crystallization). Research basis:
  - Flow-matching over a compressed latent — **ELF** ([arXiv:2605.10938](https://arxiv.org/abs/2605.10938); bottleneck-dim-128 optimal, CFG first-class).
  - Anti-collapse machinery for weight-tied recurrence — **Huginn** sandwich-RMSNorm ([arXiv:2502.05171](https://arxiv.org/abs/2502.05171)) + **LoopFormer** timestep-conditioning / zero-init modulator ([arXiv:2602.11451](https://arxiv.org/abs/2602.11451)) + Dong-2021 rank-collapse theory (bare recursion *amplifies* collapse).
  - Endpoint/intermediate target crystallization — **RiM** JEPA-style ([arXiv:2605.30343](https://arxiv.org/abs/2605.30343)); Cosmos/LDLM standardized-MSE target practice.
- **Synthesis verdict:** **Adapt** — take flow-over-a-fixed-latent (ELF) + the anti-collapse recurrence machinery (Huginn/LoopFormer) + endpoint distillation (RiM); drop ELF's in-context conditioning (we use isolated cross-attention to preserve the bottleneck) and on-the-fly teacher generation (we use existing DeepSeek-class CoT datasets).
- **Architecture mapping (ONE):** this touches the **reasoning core** — a new module between the encoder and decoder. Encoder, decoder, data spine, and loss_manager are reused unchanged.

## 2. Reuse map (read the modules first)
| Component | Action | Where |
|---|---|---|
| `ConceptEncoderForConditionalLM` | **extend** — add `self.reasoner`; call it in `encode_decode_loss` | `nn/concept_encoder_perceiver.py:1395` |
| `encode_decode_loss(...)` | **extend** — 1 reasoner call + 1 flow-loss term (gated) | `nn/concept_encoder_perceiver.py:1676` |
| `compute_anchor_loss` / `AnchorDistillHead` / `masked_standardized_mse` | **reuse as the pattern** for a model-level auxiliary loss + standardized-MSE target | `nn/concept_encoder_perceiver.py:1279,1309,1499` |
| `ConceptCausalDecoderLayer` (sandwich/pre-norm + gated FFN + cross-attn) | **reuse the layer recipe** (norms, gated FFN) inside the new reasoner block | `nn/concept_encoder_perceiver.py:1044` |
| `build_norm(norm_type, dim)` | reuse as-is (RMSNorm) | `nn/concept_encoder.py:30` |
| `ConceptEncoderConfig` | **extend** — add reasoner + flow fields (backward-compatible defaults) | `nn/concept_encoder.py:43` |
| `DataCollatorForPrefixGeneration` | **extend** — add `split_strategy="boundary_token"` (problem⟨sep⟩solution) | `data/data_collators.py:163` |
| `LossManager` / `LossConfig` | reuse as-is (concept regularizers); flow loss is **not** a loss_manager component (needs the target) | `nn/loss_manager.py` |
| `scripts/launch_e05.sh` | **reuse the wrapper pattern** → `scripts/launch_e08.sh` | `scripts/launch_e05.sh` |
| `train_perceiver_denoise_multigpu.sh` + pretokenize→manifest spine | reuse as-is | `scripts/` |
| **`ConceptFlowReasoner`** | **NEW — reusable, config-selectable** | `nn/concept_flow_reasoner.py` |
| `compute_flow_loss(...)` model method | **NEW** (mirrors `compute_anchor_loss`) | `nn/concept_encoder_perceiver.py` |

**No new training script, no new model class** — the reasoner is a config-selectable component on the existing `ConceptEncoderForConditionalLM` (`reasoner_type="concept_flow"`), exactly as `anchor_loss` adds the anchor head.

## 3. Forward pass (tensor shapes)
Symbols: `B`=batch, `P`=prefix(question) tokens, `S`=suffix(answer) tokens, `C=128` concepts, `H=768` hidden, `V`=vocab.
```
prefix_ids [B,P] ──► ConceptEncoder (cross-attn) ──► z0  [B,C,H]            # encode_concepts()
z0 ──► ConceptFlowReasoner(K steps, t_k AdaLN) ──► zK [B,C,H]               # NEW
suffix_ids [B,S] ──► ConceptEncoder (no-grad)   ──► z* [B,C,H]  (target)    # 2nd encode, detached
zK ──► ConceptCausalDecoderStack (cross-attn) ──► hidden [B,S,H] ──► lm_head ──► logits [B,S,V]
                                                                              CE(logits, suffix_labels)   # crystallization (task)
flow_loss = masked_standardized_mse(zK, z*_answer)                            # endpoint flow (auxiliary)
```
**Invariant preserved:** reasoning is O(K·C²) (concept self-attn) + O(K·C·P) (token re-injection) — no O(N²). The decoder stays O(S·C) cross-attn + O(S²) causal (intrinsic to AR), unchanged.

## 4. Inputs & data
- **Dataset:** `concept_flow_reasoning_2k` mix (spec §Plan). Reasoning sources `nvidia/OpenMathReasoning` (cot: `problem`,`generated_solution`) and `allenai/big-reasoning-traces` (flat `text`); fluency replay fineweb-edu / dclm / stack-edu.
- **Collator:** `DataCollatorForPrefixGeneration` + new `split_strategy="boundary_token"`:
  - Reasoning rows (pretokenized `problem ⟨sep⟩ solution`): split at the first `sep` → prefix=problem, suffix=solution (deterministic, semantic boundary).
  - Web/code rows (no `sep`): fall back to `sentence_boundary` (E02-long behavior).
  - Output contract unchanged: `prefix_input_ids/mask`, `suffix_input_ids/mask`, `labels`. **The suffix doubles as the crystallization target AND the flow-target source** — no third sequence.
- **Preprocessing:** reuse `scripts/pretokenize_mix.py`; reasoning sources use `text_columns=["problem","generated_solution"]` joined by the sep token (the collator splits it back). Long traces (>~2K problem+solution) filtered by `max_seq_length=2048`.

## 5. Loss & training objective
- **Task loss (crystallization CE):** existing `_teacher_forced_ce` on the suffix, with the decoder cross-attending to **zK** (not z0). This is the load-bearing change — the decoder must crystallize the *reasoned* concepts.
- **Flow loss (auxiliary, NEW):** `compute_flow_loss(zK, z_target, mask)` → `masked_standardized_mse` (reuse `nn/concept_encoder_perceiver.py:1309`) toward `z* = encode_concepts(suffix_ids).last_hidden_state.detach()`. Endpoint matching (CF-0); intermediate trajectory targets (RiM-style z*_k per chunked CoT step) deferred to CF-1.
- **Total:** `loss = crystallization_CE + flow_loss_weight * flow_loss` (+ optional `loss_manager` concept regularizers on zK). Flow loss added in `encode_decode_loss` (like the anchor term), **not** via `loss_manager` (signature needs the target).
- **Weighting:** `flow_loss_weight` default 0.0 (off → pure reasoner+crystallization, no target); E08 sets ~1.0 (tune). Warmup the flow weight (reuse `FixedWeighting.warmup_steps` idea) so the reasoner isn't pulled toward an un-trained encoder's suffix-encoding at step 0.

## 6. Config & launch
- **New `ConceptEncoderConfig` fields** (all backward-compatible; defaults reproduce E02-long/E05 exactly):
  - `reasoner_type: Optional[str] = None`  (`None` | `"concept_flow"`; `None` → no reasoner submodule, old state_dicts load byte-identical)
  - `reasoner_num_iterations: int = 4`  (K, train)
  - `reasoner_inference_iterations: Optional[int] = None`  (override at inference → test-time-compute knob)
  - `reasoner_reinject_tokens: bool = True`  (Huginn input re-injection cross-attn)
  - `flow_loss_weight: float = 0.0`  (>0 enables the endpoint flow target + loss)
  - `flow_loss_warmup_steps: int = 0`
- **Routing:** wire `REASONER_TYPE`/`REASONER_K`/`FLOW_LOSS_WEIGHT` → config in `training/utils_training.py` (beside the `decoder_type == "causal_ar"` branch, ~L531) and `train_perceiver_denoise.py` config build. `OBJECTIVE_VARIANT=prefix_suffix` is reused (the flow loss is config-gated, not objective-gated); the collator's `boundary_token` split is selected by a `SPLIT_STRATEGY` env var.
- **Warm-start:** new `WARM_START_CHECKPOINT` env var → `from_pretrained(ckpt, ignore_mismatched=True)` on `ConceptEncoderForConditionalLM` with `reasoner_type="concept_flow"`: encoder+decoder+lm_head load from E02-long `checkpoint-296000`; the reasoner is random-init (zero-init modulator → step-0 ≈ identity, so the first forward reproduces E02-long's behavior). Pattern identical to how `anchor_loss=True` adds the anchor head on a pretrained checkpoint.
- **Launch (`scripts/launch_e08.sh`, mirrors `launch_e05.sh`):**
  ```bash
  EXPERIMENT_ID=E08 DECODER_TYPE=causal_ar OBJECTIVE_VARIANT=prefix_suffix \
  REASONER_TYPE=concept_flow REASONER_K=4 FLOW_LOSS_WEIGHT=1.0 \
  SPLIT_STRATEGY=boundary_token MAX_SEQ_LENGTH=2048 \
  HIDDEN_SIZE=768 TOKEN_EMBEDDING_DIM=256 NUM_LAYERS=6 CONCEPT_NUM=128 \
  DECODER_NUM_LAYERS=4 NORM_TYPE=rmsnorm TOKENIZER_NAME=HuggingFaceTB/SmolLM2-135M \
  PRETOKENIZE_MIX=concept_flow_reasoning_2k \
  WARM_START_CHECKPOINT=Cache/Training/concept_ar_prefix_H768L6C128D4_20260614_101305/checkpoint-296000 \
  LEARNING_RATE=5e-5 MAX_GRAD_NORM=0.5 NUM_EPOCHS=3 \
  bash scripts/launch_e08.sh
  # no-flow ablation: identical but REASONER_TYPE=none FLOW_LOSS_WEIGHT=0.0
  ```

## 7. Tests & smoke
- **Unit (`tests/test_concept_flow_reasoner.py`):** (a) shapes `[B,C,H]→[B,C,H]` for K∈{1,4,8}; (b) **zero-init modulator ⇒ `reasoner(z)≈z` at init** (the step-0 = E02-long property); (c) `compute_flow_loss` finite + zero when `zK==z*`; (d) `reasoner_type=None` model loads an E02-long state_dict with zero missing/unexpected keys (backward-compat).
- **Collator unit:** `boundary_token` split puts `problem` in prefix, `solution` in suffix; web row (no sep) falls back to sentence split.
- **Local MPS smoke:** tiny config (H=64, C=8, K=2), 5 steps on 8 toy rows → assert loss finite, `grad_norm` sane.
- **Concept-geometry sanity:** after smoke, `analysis/run_concept_analysis.py` on the tiny checkpoint — within-sample RankMe is computed (not collapsed at init).

## 8. Risks & tradeoffs (success/kill restated from the spec)
- **Risk — reasoner amplifies collapse** (Dong-2021/LoopFormer). **Cheapest signal:** within-sample RankMe at first eval (≥20 by step ~2k, else the anti-collapse machinery is insufficient — add sandwich post-norms / drop token re-injection). **Fallback:** flow_loss_weight=0 (pure crystallization through a random reasoner) isolates whether the flow *target* or the *recurrence* is the problem.
- **Risk — flow instability / Δt too large.** **Cheapest signal:** `grad_norm` band + eval_loss monotonic (the E05-attempt2 divergence signature). **Fallback:** Δt=1/K, lower `flow_loss_weight`, warmup.
- **Risk — warm-start mismatch** (E02-long encoder shaped for prefix→suffix). **Mitigation:** zero-init modulator (step-0 = E02-long); the endpoint target *re-shapes* the encoder — that is the intended de-collapse, not a bug.
- **Risk — metric comparability.** E02-long within-sample RankMe = **82.28** (the PRIMARY de-collapse metric; canonical record + caveats in the [E02-long run report](../2_Experiments_Registry/run_reports/e02_long_5epoch_20260618.md)). vs E05-attempt3's **37.67** → E02-long is 2.19× less collapsed. E08 warm-starts from 82.28, so the bar is **preserve ~82** (zero-init modulator ⇒ step-0 = E02-long) and beat E05's 37.67; spec threshold >60 sits between them. **The warm-start checkpoint lives on the NAS (`/nas/ml_data/mrcogito/checkpoints/concept_ar_prefix_H768L6C128D4_20260614_101305/checkpoint-296000`), readable from Odra via `from_pretrained` (one-time read; no `rsync` needed) — the run is not blocked on Polonez.**
- **Spec success/kill (restated):** PASS = within-sample RankMe **>60** ∧ STS-B **≥0.65** ∧ Δshuffle_beyond **≥0.5** ∧ token-F1 **>0.3** ∧ reasoning acc **> no-flow ablation by ≥5pp** (stretch: acc@K=8 > acc@K=4 by ≥2pp). KILL = STS-B **<0.45** ∨ RankMe **<20** ∨ divergence ∨ reasoning acc **≤ ablation**.

## 9. Code sketches (`# sketch` — interface only)
```python
# sketch — nn/concept_flow_reasoner.py
class ConceptFlowReasonerLayer(nn.Module):
    # sandwich-RMSNorm (4 norms) + concept self-attn (C×C) + optional token-reinject
    # cross-attn + AdaLN(t_k) modulation (zero-init) + gated FFN. Reuses build_norm + gated-FFN recipe.
    def forward(self, z, tok_emb, tok_kpm, t_emb): ...        # [B,C,H] -> [B,C,H]

class ConceptFlowReasoner(nn.Module):
    def __init__(self, config): ...                            # K layers (weight-tied: 1 layer, K iters)
    def forward(self, z0, token_embeddings, token_key_padding_mask=None,
                num_iterations=None) -> torch.Tensor: ...      # [B,C,H] -> [B,C,H]  (zK)

# sketch — nn/concept_encoder_perceiver.py (ConceptEncoderForConditionalLM)
# in __init__: self.reasoner = ConceptFlowReasoner(config) if config.reasoner_type=="concept_flow" else None
# in encode_decode_loss(), after concept_repr = encoder_outputs.last_hidden_state:
#     if self.reasoner is not None:
#         concept_repr = self.reasoner(concept_repr, token_embeddings=enc_token_emb,
#                                      token_key_padding_mask=enc_key_padding)   # z0 -> zK
# (decoder then cross-attends to concept_repr = zK, unchanged)
# flow target + loss (gated):
#     if self.training and self.config.flow_loss_weight > 0 and target_input_ids is not None:
#         with torch.no_grad():
#             z_star = self.encode_concepts(target_input_ids, target_attention_mask).last_hidden_state
#         flow = self.compute_flow_loss(concept_repr, z_star.detach(), target_mask)
#         loss = loss + self.config.flow_loss_weight * flow
```

*Handoff: spec + this plan → `research-implement`. No code is written here.*
