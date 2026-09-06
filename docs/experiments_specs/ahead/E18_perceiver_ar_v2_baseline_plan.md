# E18 — Implementation Plan

- **Spec:** [E18_perceiver_ar_v2_baseline.md](E18_perceiver_ar_v2_baseline.md) · **Status:** approved (pilot on Polonez authorized; AWS main run needs explicit go)
- **Authored by:** `implementation-plan` · for → `research-implement`

> The HOW for E18's bet: one global causal read over the whole context, deep local (window-N)
> stack, every token trained, tiny hashed input embeddings. Reuses the HF-Trainer spine, Muon,
> pretokenize/manifest flow and launcher; adds one reusable model family (`perceiver_ar`), one
> packing/mask utility, one long-context probe module, and one thin launcher.

## 1. Source & fit
- **Origin:** Perceiver AR (Hawthorne et al. 2022) + this repo's feasibility note
  [perceiver_ar_modern_reproduction_feasibility.md](../../4_Research_Notes/perceiver_ar_modern_reproduction_feasibility.md)
  (§4 defects, §5 v2 design); ECP/LLP 2024 (lossy history, latent-training dependency);
  Over-Tokenized Transformer 2025 (hashed n-gram input embeddings); modded-nanogpt records
  (value embeddings, U-net lambdas, zero-init); SmolLM3 (NoPE every 4th layer, doc masking).
- **Synthesis verdict:** Adapt. Take: single global read, latent = last-N, one-layer prefix cache.
  Drop: raw-embedding reader, loss on N of M tokens, 0.9 prefix dropout, absolute pos-emb.
- **Architecture mapping:** encoder (pre-encoder + global read) · decoder/head (window-N stack,
  untied head) · data (long-doc mix, doc-masked packing) · objective unchanged (next-token CE).
- **Boldness check:** the plan implements the single-bottom-global-read model exactly; the dense
  control is the only alternative arm, and it exists to attribute the win, not to hedge.

## 2. Reuse map (read first)
| Component | Action | Where |
|---|---|---|
| `train_concept_pretraining.py` / `PerceiverDenoiseTrainer` | reuse; `compute_loss` already returns `model(**inputs).loss` for `causal_lm` | `training/train_concept_pretraining.py`, `training/concept_pretraining_trainer.py:367` |
| `validate_training_configuration` | extend: `model_family` arg; `causal_lm` allowed when `model_family == "perceiver_ar"` | `training/concept_pretraining_args.py:298` |
| `build_pretraining_model` / `build_training_wandb_identity` | extend: third branch → `PerceiverARLM`; identity `model_family="perceiver_ar"`, `objective_family="causal_lm"` | `training/concept_pretraining_factories.py:216,295` |
| `DataCollatorForCausalLM` | reuse as-is for the pilot (per-document rows, right padding, labels = ids with −100 at pad) | `data/data_collators.py:163` |
| `PackedCausalLMCollator` | **new** (before AWS): concatenates rows to `max_seq_length`, emits `doc_ids [B,S]`; used by the flash-varlen and flex doc masks | `data/data_collators.py` |
| `Muon` | reuse; shape routing already sends embedding tables / lm_head / scalars to AdamW | `nn/muon.py:45` |
| `ChunkedLMHeadCE` | replace for this family by `chunked_softcap_ce` (checkpointed chunks, supports logit soft-cap) + Liger fused-linear-CE fast path when importable | new in `nn/perceiver_ar_lm.py` |
| `pretokenize_mix.py` + recipes | reuse; new recipe `e18_pilot_longdoc_v1.json` (SmolLM3 tokenizer, `--objective causal_lm`) | `scripts/pretokenize_mix.py`, `data/mix_recipes/` |
| `sequence_parallel.py` | not used by E18 (window-N stack keeps memory linear); revisit only for 1M on 24 GB cards | — |
| `PerceiverARLM`, `PerceiverARConfig` | **new reusable family** | `nn/perceiver_ar_lm.py` |
| `long_context_probes.py` | **new**: per-position CE buckets, passkey (teacher-forced argmax), copy task | `evaluation/long_context_probes.py` |
| `build_copy_task_dataset.py` | **new**: mirrored-copy arrow dataset for P2 (pattern of `build_delayed_recall_dataset.py`) | `scripts/` |
| `launch_e18.sh` | **new** thin wrapper pinning the pilot protocol; delegates to the generic launcher | `scripts/` |

## 3. Forward pass (tensor shapes)
Symbols: `B` batch, `S` sequence (padded/packed), `V`=128,256, `e`=256 tiny dim, `d` model width,
`h` q-heads, `g` kv-heads, `dh`=128, `N` stack window, `w` pre-encoder window.
```
ids [B,S]                       ─► tok_tab[V,e] + Σ_n ngram_tab_n[hash_n(ids)] [B,S,e]   # hash_n over (ids[i-n+1..i]); BOS-padded
                                ─► up: W0 e→d  +  W2(silu(W1 e→d)) d→d ─► RMSNorm ─► x0 [B,S,d]
layer ℓ (all layers share the block):
  h = RMSNorm(x); q,k,v = Wq h [B,S,h,dh], Wk h [B,S,g,dh], Wv h [B,S,g,dh]
  q,k = QKNorm(q),QKNorm(k); RoPE(q,k) unless ℓ ∈ NoPE set
  if ℓ ∈ value_embed_layers: v = v + λ_ℓ · Wve_ℓ(ve_tab_ℓ[ids]) [B,S,g,dh]
  o = Attention(q,k,v; pattern_ℓ, key_valid, doc_ids)      # pattern_ℓ ∈ {swa(w), full, swa(N)}
  x = x + Wo o                                             # Wo zero-init
  x = x + W_down( silu(W_gate RMSNorm(x)) ⊙ W_up RMSNorm(x) )   # SwiGLU, W_down zero-init
U-net skips: for ℓ ≥ L/2: x = x + σ_ℓ · skip[L-1-ℓ]; every layer: x = α_ℓ·x + β_ℓ·x0 (α=1, β=0 init)
head: h = RMSNorm(x) ─► chunked: logits_c = 30·tanh(W_head h_c / 30) ─► CE(labels shifted by 1)
```
Pattern schedule (config): `pre_layers` × `swa(w)` → `global_layers` × `full` → `stack_layers` × `swa(N)`.
Dense control = `PAR_MODE=dense`: 0 pre-layers, all layers `full`, same widths/params.
**Hooks:** `write_back_proj: Linear(d → g·dh·2)` zero-init, unused in E18 forward (E19 will map latent
states into the global layer's K/V space); `block_attention_mode` config field (`causal` | `bidirectional`),
E18 asserts `causal` (E20 switches the stack pattern to bidirectional-within-block).

**Attention backends (config `attn_backend`):**
- `sdpa` — reference: explicit boolean mask `[B,1,S,S]` = causal ∧ |i−j| < window ∧ key_valid ∧ same_doc.
  Tests only / tiny S.
- `flex` — `torch.nn.attention.flex_attention` with `mask_mod(b,h,q,kv)` = the same predicate;
  block masks built once per (S, pattern) and cached; doc_ids read from a per-batch tensor.
  Default on Ampere (Polonez/Odra).
- `flash` — `flash_attn_varlen_func` (FA2 Ampere / FA3 Hopper) with `cu_seqlens` from doc boundaries,
  `causal=True`, `window_size=(win−1, 0)`; requires packed inputs (no pad). Default on Hopper.

**Inference (`generate`, no-cache v1):** full recompute per emitted token, bounded to prompts ≤ 32k
for the pilot probes. KV-cache v2 (global layer: full cache; SWA layers: ring buffer of N) lands
with E19/E21 (they need it) — tracked, not blocking P1–P4.

## 4. Inputs & data
- **Pilot recipe** `data/mix_recipes/e18_pilot_longdoc_v1.json` (objective `causal_lm`, seq 32768 at
  tokenize time so one cache serves both stages; training truncates to `MAX_SEQ_LENGTH`):
  PG-19 `emozilla/pg19` 0.40 (max_samples 20k books) · FinePDFs 0.20 · FineWeb-Edu 0.30 ·
  stack-edu python 0.10; eval split deterministic (`test_size_percent` 0.5 → capped rows).
  Tokenizer `HuggingFaceTB/SmolLM3-3B` (verify `len(tok)==128256` and ids identical to
  `meta-llama/Llama-3.2-1B` on a 10k-sentence probe; recorded in the run report).
- **Collator:** pilot = `DataCollatorForCausalLM(max_length=MAX_SEQ_LENGTH)`; main = `PackedCausalLMCollator`
  (doc_ids → masks). Model reads `attention_mask` (key_valid) and optional `doc_ids`.
- **Probes** (`evaluation/long_context_probes.py`): (1) per-position CE on PG-19 eval rows ≥ 16k tokens,
  buckets `[0,8k)`, `[8k,32k)`; (2) passkey: filler from PG-19 eval, 5-digit key at depths
  {0.1…0.9}, question suffix, metric = argmax accuracy over the 5 answer tokens (teacher-forced);
  (3) copy task: `scripts/build_copy_task_dataset.py --context 32768 --n_train 20000 --n_eval 200`
  → mirrored random tokens in a reserved id range, loss only on second half; metric = token accuracy.

## 5. Loss & training objective
- Next-token CE over all non-pad positions (labels shifted inside the model), z-loss 1e-4 on the
  chunked log-partition (config `z_loss`, default 1e-4), logit soft-cap 30. Copy task: labels −100
  on the first half (dataset-provided labels, `PRESERVE_PRECOMPUTED_LABELS=true`).
- No auxiliary losses; `LossConfig.disabled()`.

## 6. Config & launch
- **New `ModelArguments` fields** (all default to values that keep every existing family byte-identical):
  `model_family="auto"` (`auto|concept|backbone|perceiver_ar`), `par_mode="perceiver"` (`perceiver|dense`),
  `par_pre_layers=2`, `par_pre_window=1024`, `par_global_layers=1`, `par_block=4096`, `num_attention_heads=None`
  (→ `hidden_size//128`), `num_kv_heads=2`, `par_ngram_orders="2,3"`, `par_ngram_buckets=131072`,
  `par_value_embed_layers="0,7,14"`, `par_nope_every=4`, `rope_theta=500000.0`, `attn_backend="flex"`,
  `logit_softcap=30.0`, `z_loss=1e-4`, `block_attention_mode="causal"`. Reused: `hidden_size`,
  `token_embedding_dim` (= e), `num_hidden_layers` (= stack layers), `intermediate_size`, `chunked_ce_block_size`.
- **Registry:** `build_pretraining_model` → `PerceiverARLM(PerceiverARConfig)`; `model_type="perceiver_ar"`
  (+`_dense` for the control); W&B identity `model_family="perceiver_ar"`, `checkpoint_family="perceiver_ar"`;
  `evaluation/concept_eval_routing.py` gets a `perceiver_ar` entry that routes to the probes (no concept metrics).
- **Launcher:** generic launcher gains `MODEL_FAMILY`, `PAR_*`, `NUM_KV_HEADS`, `ATTN_BACKEND`,
  `LOGIT_SOFTCAP`, `Z_LOSS` passed only when `MODEL_FAMILY=perceiver_ar` (`PAR_ARGS=()` block, same
  pattern as `BACKBONE_ARGS`). Thin wrapper `scripts/launch_e18.sh`:
  ```bash
  # pilot stage A (Polonez): 125M dense-equivalent, 8k
  bash scripts/launch_e18.sh                     # E18_STAGE=8k default
  E18_STAGE=32k RESUME_FROM_CHECKPOINT=<ckpt> bash scripts/launch_e18.sh
  PAR_MODE=dense bash scripts/launch_e18.sh      # matched control
  E18_TASK=copy bash scripts/launch_e18.sh       # P2 copy task (own manifest, 6 layers)
  ```
  Pins: `MODEL_FAMILY=perceiver_ar OBJECTIVE_VARIANT=causal_lm TOKENIZER_NAME=HuggingFaceTB/SmolLM3-3B
  HIDDEN_SIZE=768 NUM_LAYERS=12 INTERMEDIATE_SIZE=2048 TOKEN_EMBEDDING_DIM=256 NUM_KV_HEADS=4
  PAR_PRE_LAYERS=1 PAR_PRE_WINDOW=512 PAR_BLOCK=2048 PAR_NGRAM_BUCKETS=65536 PAR_VALUE_EMBED_LAYERS=0,4,8
  OPTIMIZER=muon LEARNING_RATE=0.01 MUON_ADAMW_LR=2e-4 WEIGHT_DECAY=0.1 MAX_GRAD_NORM=0.5
  LR_SCHEDULER_TYPE=constant_with_warmup WARMUP_STEPS=500 GRADIENT_CHECKPOINTING=True
  CHUNKED_CE_BLOCK_SIZE=2048 ATTN_BACKEND=flex DATASETS_TOK_DIR=…/datasets_tok_smollm3_32k
  PRETOKENIZE_MIX=e18_pilot_longdoc_v1 TARGET_TOKENS=2000000000 (8k) / 500000000 (32k)`.
  Batch: 8k → `PER_DEVICE_BATCH_SIZE=4 GRAD_ACCUM=4` (effective 64×8k = 0.5M tokens/step);
  32k → `1 × 8` (0.25M tokens/step + doc-length loss weighting by count, HF default).
- **Main run (AWS, H100, not launched by this plan):** `HIDDEN_SIZE=1280 NUM_LAYERS=20 INTERMEDIATE_SIZE=3456
  NUM_KV_HEADS=2 PAR_PRE_LAYERS=2 PAR_PRE_WINDOW=1024 PAR_BLOCK=4096 PAR_NGRAM_BUCKETS=131072 ATTN_BACKEND=flash`
  + packed collator + Nemotron recipe (separate `e18_main_stage1_v1.json`); Slurm/EC2 launcher is a
  follow-up engineering spec once P1–P4 pass.

## 7. Tests & smoke
- `tests/test_perceiver_ar_lm.py` (CPU, tiny dims, `attn_backend=sdpa`):
  1. mask semantics: `swa(w)` / `full` / `swa(N)` masks equal a naive loop reference, with padding and doc_ids;
  2. **window identity:** with S ≤ N every `swa(N)` layer equals a `full` layer, so the perceiver-mode
     forward equals the dense-mode forward under the same weights; and a `swa(w)` layer at position t equals
     a naive per-position attention over `[t−w+1, t]` (the training/inference identity for window caches);
  3. shapes + finite loss for perceiver and dense modes; parameter count matches the analytic formula;
  4. chunked soft-cap CE ≡ full CE (loss, grad wrt hidden, grad wrt head) at block sizes {1,16,64};
  5. hashed n-gram embedding: deterministic, BOS-padded, no cross-doc leakage across doc_ids boundaries;
  6. zero-init invariants: at init the model output equals the pre-encoder-only path (write-back, U-net
     skips, value-embed lambdas contribute 0).
- `tests/test_long_context_probes.py`: passkey builder places key at requested depth; copy-task labels
  mask first half; per-position buckets sum to the total.
- CUDA-only (skipped elsewhere): `flex` ≡ `sdpa` ≡ `flash` (if installed) to 1e-3 in bf16 at S=8192.
- Local smoke (macOS, CPU): `MODEL_FAMILY=perceiver_ar HIDDEN_SIZE=128 NUM_LAYERS=2 ... uv run python
  training/train_concept_pretraining.py --max_steps 5` on a 100-row minipile slice.
- Remote smoke (Polonez, 1 GPU): 100 steps at 8k, then a 20-step 32k forward/backward memory check.

## 8. Risks & tradeoffs
- **Tokenizer mismatch** (SmolLM3 ≠ Llama-3 ids): cheapest signal = the 10k-sentence id-equality probe;
  fallback = gated `meta-llama/Llama-3.2-1B` with the user's HF token, or drop ProLong reuse.
- **FlexAttention compile time / recompiles per length** on 3090s: cache block masks per (S, pattern),
  pad S to a multiple of 128; fallback `sdpa` at 8k only.
- **Single global read too weak (P3 fails):** one pre-registered fix (pre-layers 1→2, window 512→1024) then stop.
- **Muon on a new family:** reuse the E05/E16b-calibrated triple (lr 0.01, adamw 2e-4, wd 0.1, clip 0.5,
  constant-with-warmup); watch the divergence signature.
- **Hash-table params dominate AdamW state** at 128k vocab: state is 2× table size (fine at 125M pilot);
  at 594M main, 79M sparse params → ~0.6 GB, acceptable.
- **HF Trainer eval collects logits:** the family returns `logits=None` when labels are given and the
  launcher passes `--prediction_loss_only`; probes run outside the Trainer loop.

## 9. Code sketches
```python
# sketch — nn/perceiver_ar_lm.py
class PerceiverARConfig(PretrainedConfig):
    model_type = "perceiver_ar"
    # widths: hidden_size, intermediate_size, token_embedding_dim, vocab_size
    # pattern: par_mode, pre_layers, pre_window, global_layers, stack_layers, block(N)
    # heads: num_attention_heads, num_kv_heads, head_dim=128, rope_theta, nope_every
    # input: ngram_orders=(2,3), ngram_buckets, value_embed_layers, value_embed_dim=64
    # head/loss: logit_softcap, z_loss, chunked_ce_block_size
    # backend/hook: attn_backend, block_attention_mode, write_back_hook=True

def layer_patterns(cfg) -> list[tuple[str, int]]:
    if cfg.par_mode == "dense":
        return [("full", 0)] * cfg.total_layers
    return [("swa", cfg.pre_window)] * cfg.pre_layers + [("full", 0)] * cfg.global_layers \
         + [("swa", cfg.block)] * cfg.stack_layers

def mask_pred(pattern, window, doc_ids, key_valid):        # shared by sdpa / flex
    def f(b, h, q, kv):
        ok = kv <= q
        if pattern == "swa": ok &= (q - kv) < window
        ok &= key_valid[b, kv]
        if doc_ids is not None: ok &= doc_ids[b, q] == doc_ids[b, kv]
        return ok
    return f

class PerceiverARLM(PreTrainedModel):
    def forward(self, input_ids, attention_mask=None, labels=None, doc_ids=None, return_per_token_loss=False)
    # returns CausalLMOutput(loss=..., logits=None) when labels given; per-token CE [B,S] on request
    def prefix_kv(self, input_ids) -> (k, v)      # the one-layer message/prefix object (E19/E21 hook)
```
