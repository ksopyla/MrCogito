# E09 — Implementation Plan (gated recurrent concept memory)

- **Spec:** [E09_recurrent_concept_memory.md](E09_recurrent_concept_memory.md) · **Status:** canceled — superseded by E10 before a standalone run
- **Authored by:** `implementation-plan` · for → `research-implement`

> The HOW for the spec's single change (concept memory `frozen` → `gated_recurrent`). Repo-rooted:
> every class/path below was read. Reuse-first; the only new module is a small config-selectable
> `ConceptWriteHead` that **reuses `BiXTCrossAttention`** as the concept write op.

## 1. Source & fit
- **Origin:** the E05/E02-long generation review — concepts are a *frozen prefix snapshot*
  (`ConceptCausalDecoderStack.forward` loops `h = layer(h, concepts, …)` with `concepts` never
  reassigned, `nn/concept_encoder_perceiver.py:1258`). Beyond the decoder's K=128 window the model
  has no evolving memory of its own output. User hypothesis: make concepts a writable running state.
- **Lit:** `docs/literature_review/recurrent_memory_transformers.md` — Block-Recurrent Transformer
  §B (gated recurrent state + BPTT across blocks), RMT §B (`[mem]` refreshed per segment, fine-tunes
  onto a pretrained backbone), Infini-attention §D (iterated small memory *degrades* — collapse
  warning), Coconut §B (curriculum). Agenda already earmarks Ouro's anti-collapse machinery for E08.
- **Synthesis verdict:** **Adapt** — make the C concepts a *gated recurrent state* refreshed from
  decoded windows, reusing BiXT's `lat←tok` direction as the write op. Take: block-causal recurrence
  + zero-init gate + sandwich-RMSNorm. Drop: single-vector depth-recursion (Huginn/Ouro) — we keep
  the C-concept **set**, the BiXT encoder, and the windowed AR decoder unchanged.
- **Architecture mapping (ONE):** touches the **concept bottleneck** (concepts become a recurrent
  state) and the **decoder training path** (block-causal decode in `encode_decode_loss`). Encoder
  module, decoder module, loss, and data are reused unchanged.

## 2. Reuse map (read first)
| Component | Action | Where |
|---|---|---|
| `BiXTCrossAttention` | **reuse as-is** = the write op (`update_tokens=False` ⇒ concepts only) | `nn/concept_encoder.py:335-444` |
| `ConceptEncoderForConditionalLM.encode_decode_loss` | extend — add block-recurrent branch, gated by config | `nn/concept_encoder_perceiver.py:1676-1787` |
| `ConceptCausalDecoderStack` / `decode_logits` | reuse as-is (called once per block) | `nn/concept_encoder_perceiver.py:1157, 1553` |
| `ConceptEncoder.forward` embed block (760-771) | refactor → new `ConceptEncoder.embed_tokens(ids)` | `nn/concept_encoder.py:736` |
| `_teacher_forced_ce_window` | reuse as-is — beyond-window CE metric | `nn/concept_encoder_perceiver.py:1638` |
| `ConceptEncoderConfig` | extend — new fields, safe defaults | `nn/concept_encoder.py:86` |
| `DataCollatorForPrefixGeneration` | reuse as-is (no data change) | `data/data_collators.py` |
| **NEW** `ConceptWriteHead` (BiXT + gate + sandwich-norm) | new — reusable, config-selectable | `nn/concept_encoder_perceiver.py` |
| `train_perceiver_denoise.py` model_args + `train_perceiver_denoise_multigpu.sh` + `launch_e05.sh` | extend — env-var pass-through (mirror `DECODER_CONTEXT_WINDOW` wiring) | `training/`, `scripts/` |

## 3. Forward pass (tensor shapes)
Symbols: `B`=8, `P`=prefix len, `S`=2048 suffix, `C`=128, `H`=768, `dim_tok`=256, `V`=49152, `K`=128
(block size = decoder window = concept-update cadence).

**`frozen` path (unchanged — default, byte-identical for all old checkpoints):**
```
prefix (B,P) ──encode_concepts──▶ concepts (B,C,H)                    # frozen
suffix (B,S) ──shift_right──▶ dec_in (B,S)
decode_logits(concepts, dec_in) ──▶ logits (B,S,V) ──▶ CE(logits, labels)
```

**`gated_recurrent` path (new — the single change):**
```
concepts = encode_concepts(prefix)                                     # (B,C,H)  initial state
carry = full((B,K), bos)                                               # block-0 window carry
blocks = []
for b in 0..ceil(S/K)-1:                                               # 16 blocks @ K=128
    s,e = b*K, min((b+1)*K, S)
    block_in = shift_right(suffix[:, s:e])                             # (B, K)
    dec_in   = cat([carry, block_in], dim=1)                           # (B, 2K)  window carry ⇒ K-window continuous
    logits_b = decode_logits(concepts, dec_in, position_offset=s-K)    # (B, 2K, V)
    blocks  += logits_b[:, -K:, :]                                     # (B, K, V)  this block's positions
    # ---- block-causal concept WRITE (gold block tokens, AFTER predicting block b) ----
    win_tok  = encoder.embed_tokens(suffix[:, s:e])                    # (B, K, dim_tok)  encoder embedding space
    concepts = ConceptWriteHead(concepts, win_tok)                     # (B,C,H)  α=0 ⇒ identity at init
    carry    = suffix[:, s:e]                                          # next block's window carry
logits = cat(blocks, dim=1)                                            # (B,S,V)
loss   = _teacher_forced_ce(logits, labels)                            # BPTT across blocks = one autograd graph
```
**Cost:** per block `O(K·K)` windowed self-attn over `2K` + `O(C·K)` write BiXT; total `O(S·K)` —
same asymptotic as the frozen windowed decode, plus `O(S·C)` for the writes (small). **~1.5–2× step
time** from 16 block-decodes vs 1 full forward (decoder is the cheap part; acceptable).

**Block-causality / no-leak:** concepts used to predict block *b* are written only from blocks
*<b* (the write happens *after* `logits_b`). BiXT is bidirectional *within* a window (fine — the
window is already produced); across windows it is strictly causal.

## 4. Inputs & data
- **Dataset:** `smollm3_inspired_2k_e05` pretokenized manifest (reuse; no re-tokenization).
- **Collator:** `data/data_collators.py:DataCollatorForPrefixGeneration` — **reuse as-is**. It
  yields `prefix_input_ids` / `suffix_input_ids` / `labels`. The K-block split is internal to
  `encode_decode_loss`, so no data/collator change.
- **Masking/split:** suffix `labels=-100` at pad (unchanged); block-decode handles variable last
  block via `min((b+1)*K, S)`.

## 5. Loss & training objective
- **Loss:** suffix next-token CE via `_teacher_forced_ce` (`nn/concept_encoder_perceiver.py:1602`),
  computed on the concatenated per-block logits — **unchanged**. No new loss component, no
  `loss_manager` change (E05 prefix_suffix runs with `loss_manager` off; same here — keeps one variable).
- **Objective:** prefix→suffix, block-causal. The recurrent concept state is the only addition; no
  auxiliary loss (a collapse regularizer would be a *follow-up* spec, not this one).
- **BPTT:** gradients flow through the `ConceptWriteHead` updates automatically (single graph across
  all 16 blocks); no manual truncation needed at S=2048. Activation memory bounded by gradient
  checkpointing (already supported, `_set_gradient_checkpointing`).

## 6. Config & launch
**New `ConceptEncoderConfig` fields** (`nn/concept_encoder.py:86`, all backward-compatible):
- `concept_memory_mode: str = "frozen"`        # `"frozen"` | `"gated_recurrent"`
- `concept_memory_block: Optional[int] = None` # K (suffix block = update cadence); None disables
- `concept_memory_write_layers: int = 1`       # BiXT write layers per update
- `concept_memory_share_weights: bool = False` # share write-BiXT with an encoder layer (default: fresh)
- `concept_memory_gate_init: float = 0.0`      # α init (0 ⇒ step-0 == frozen)

**Registry / routing:** `concept_ar` already maps to `ConceptEncoderForConditionalLM`
(`analysis/run_concept_analysis.py:MODEL_CLASSES`) — no new entry (the mode is a config field).
`run_concept_analysis.py` and eval routing work unchanged; the concept-write params are inert at eval
when `concept_memory_mode="frozen"`, and at eval for the treatment arm the block-recurrent path is
used for any forward that produces logits.

**Wiring chain (mirror `DECODER_CONTEXT_WINDOW`):**
1. `training/train_perceiver_denoise.py` model_args dataclass (near `:132`/`:681`): add
   `concept_memory_mode`, `concept_memory_block`; pass to `ConceptEncoderConfig(...)`.
2. `scripts/train_perceiver_denoise_multigpu.sh` (near `:176`): conditional
   `--concept_memory_mode` / `--concept_memory_block` from `CONCEPT_MEMORY_MODE` / `CONCEPT_MEMORY_BLOCK`.
3. `scripts/launch_e05.sh`: `export CONCEPT_MEMORY_MODE="${CONCEPT_MEMORY_MODE:-frozen}"` (default
   frozen ⇒ E05 byte-identical).

**Launch (the E09 arm):**
```bash
CONCEPT_MEMORY_MODE=gated_recurrent CONCEPT_MEMORY_BLOCK=128 \
  OPTIMIZER=adam SKIP_PRETOKENIZE=1 bash scripts/launch_e05.sh
```

## 7. Tests & smoke
- **Unit test** `tests/test_concept_memory.py`:
  1. `frozen` mode output byte-identical to the current path (regression guard).
  2. `gated_recurrent`: logits shape `[B,S,V]`; loss finite; **gradient reaches the write-BiXT
     params** (`assert p.grad is not None and p.grad.abs().sum() > 0`).
  3. **Zero-init property:** at `α=0`, `gated_recurrent` logits == `frozen` logits (numerically).
  4. **No-leak:** perturbing suffix block *b* gold tokens does not change logits for blocks *<b*.
- **Local MPS smoke** (tiny config H=64, C=8, D=2, K=8, S=64): run `encode_decode_loss` 3 steps on
  random data — loss finite + decreasing; then `uv run python analysis/run_concept_analysis.py
  --model_type concept_ar --model_path <tiny-ckpt>` for a geometry sanity (no collapse at init).
- **Key smoke assertion = the zero-init property (#3):** it is the cheapest proof that the change is
  clean (step-0 == E05) before any GPU run.

## 8. Risks & tradeoffs
- **Risk: collapse (Infini-attention).** *Cheapest signal:* within-sample RankMe via
  `run_concept_analysis.py` + the RankMe kill-gate (Stage 1b). *Fallback:* raise gate regularization
  / drop `concept_memory_write_layers` to 1 / abort to `frozen`.
- **Risk: window-carry correctness (fluency at block boundaries).** *Cheapest signal:* within-window
  suffix-CE matches the frozen path at `α=0` (unit test #3) and stays healthy as `α` learns.
  *Fallback:* widen carry to 2 blocks.
- **Risk: ~1.5–2× step time.** *Cheapest signal:* wall-clock/step vs E05 Adam. *Fallback:* KV-cache
  the window carry (follow-up, out of scope here).
- **Risk: 0.5 ep too short for the write op to learn.** *Cheapest signal:* beyond-window CE Δ by
  step 40k (Stage 1a kill-gate). *Fallback:* extend (follow-up spec).

## 9. Code sketches (`# sketch` — decisions, not demos)
```python
# sketch — ConceptEncoderConfig fields (nn/concept_encoder.py:86)
concept_memory_mode: str = "frozen"
concept_memory_block: Optional[int] = None     # K = block + update cadence
concept_memory_write_layers: int = 1
concept_memory_share_weights: bool = False
concept_memory_gate_init: float = 0.0          # 0 ⇒ step-0 reproduces frozen
```
```python
# sketch — refactor: ConceptEncoder.embed_tokens (extract from forward lines 760-771)
def embed_tokens(self, input_ids: torch.LongTensor) -> torch.Tensor:
    pos = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
    e = self.token_embeddings(input_ids) + self.token_position_embeddings(pos)
    if self.token_projection is not None:
        e = self.token_projection(e)
    return self.dropout(e)                     # (B, N, dim_tok)
```
```python
# sketch — NEW ConceptWriteHead: reuses BiXTCrossAttention as the concept write op
class ConceptWriteHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bixt = BiXTCrossAttention(
            dim_lat=config.hidden_size, dim_tok=config.token_embedding_dim,
            dim_attn=config.hidden_size, num_heads=config.num_attention_heads,
            attn_drop=config.attention_probs_dropout_prob, proj_drop=config.hidden_dropout_prob,
            update_tokens=False)               # lat<-tok only; concepts updated, tokens not
        self.norm_lat = build_norm(config.norm_type, config.hidden_size)
        self.norm_tok = build_norm(config.norm_type, config.token_embedding_dim)
        self.sandwich = build_norm(config.norm_type, config.hidden_size)   # Ouro anti-collapse
        self.alpha = nn.Parameter(torch.tensor(float(config.concept_memory_gate_init)))
    def forward(self, concepts, win_tok):       # (B,C,H),(B,K,dim_tok) -> (B,C,H)
        lat, _ = self.bixt(self.norm_lat(concepts), self.norm_tok(win_tok))
        return concepts + self.alpha * self.sandwich(lat)   # α=0 ⇒ identity at init
```
```python
# sketch — block-recurrent branch inside encode_decode_loss (gated by concept_memory_mode)
if getattr(self.config, "concept_memory_mode", "frozen") == "gated_recurrent":
    K = self.config.concept_memory_block
    B_, S = target_input_ids.shape
    bos = self.config.bos_token_id or self.config.eos_token_id or 0
    carry = torch.full((B_, K), bos, dtype=target_input_ids.dtype, device=target_input_ids.device)
    concept = concept_repr
    pieces = []
    for b in range((S + K - 1) // K):
        s, e = b * K, min((b + 1) * K, S)
        block_in = self._shift_right(target_input_ids[:, s:e])          # (B, K)
        dec_in = torch.cat([carry, block_in], dim=1)                    # (B, 2K) window carry
        logits_b = self.decode_logits(concept, dec_in,
                                      key_padding_mask=dec_key_padding,
                                      position_offset=s - K)            # (B, 2K, V)
        pieces.append(logits_b[:, -(e - s):, :])                        # this block's positions
        win_tok = self.encoder.embed_tokens(target_input_ids[:, s:e])   # gold block tokens
        concept = self.concept_write_head(concept, win_tok)             # block-causal write
        carry = target_input_ids[:, s:e]
    logits = torch.cat(pieces, dim=1)                                   # (B, S, V)
    loss = self._loss_from_logits(logits, labels, concept)             # CE; BPTT automatic
    return loss, logits, encoder_outputs
# else: existing single-forward path (unchanged)
```
```python
# sketch — beyond-window CE eval (reuse _teacher_forced_ce_window; log in trainer eval, ~train_perceiver_denoise.py:454)
K = base_model.config.concept_memory_block or base_model.config.decoder_context_window
metrics["suffix_ce_within_window"]  = self._teacher_forced_ce_window(logits, labels, K, beyond=False).item()
metrics["suffix_ce_beyond_window"]  = self._teacher_forced_ce_window(logits, labels, K, beyond=True).item()
```
```python
# sketch — free-generation inference extension (the notebook's generate())
# after each K generated tokens: concepts = model.concept_write_head(concepts, model.encoder.embed_tokens(cur[:, -K:]))
# (mirrors training; downstream deliverable, not part of the training spec)
```
