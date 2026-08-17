# E17d — Depth-private concept layers as global-attention replacement — Implementation Plan

- **Spec:** [E17d_global_concept_assimilation.md](E17d_global_concept_assimilation.md) · **Status:** implemented
- **Authored by:** `implementation-plan` · for → `research-implement`
- **Token budget:** **300M** non-padding tokens (`TARGET_TOKENS=300000000`), same
  mechanism-verdict cadence as E17c. A 300B budget is not runnable on 4× RTX 3090
  (~1000× E17c) and is not this experiment.

> Implement the spec's job change: four depth-private banks stay; each former global
> layer mixes its bank **inside the attention residual**; previous windows exist only as
> those banks at train, eval, and `generate()`. Do not collapse to one bank, do not keep
> E17c's post-FFN sidecar, and do not leave the 512-token carry on at decode.
> ID remains **E17d**. `E18` is already claimed on an unmerged addressable-RAM branch;
> do not reuse it.

## 1. Source & fit
- **Origin:** E17c
  (`backbone_concept_gemma_3_1b_pt_K512_concept_20260814_133241`, `checkpoint-2370`)
  proved the cell can hold a block-start gist when the previous-window token carry is
  hidden, then ignore concepts once the new window has ~64 tokens. Bank 0 / layer 5
  took almost all of the signal because the read is a tanh-gated add *after* the Gemma
  layer (including FFN) while later layers already see that residual. Write schedule
  is already per-window; E17d does not move it. See
  [E17c report](../../2_Experiments_Registry/run_reports/e17c_depth_private_working_memory_20260815.md)
  and [five-whys](../../4_Research_Notes/e17c_failure_five_whys_20260815.md) (lands with
  the literature-confirm PR if not yet on `dev`).
- **Synthesis verdict:** **Adapt.** Keep Gemma's four global depths (friend: low→high
  abstraction). Give them the *global-attention* job (mix current tokens with compressed
  history, then FFN), not a memory sticky-note. Drop the optional 512-token cheat sheet
  at train *and* generate. Take Infini-style local+memory in one attention residual;
  drop Infini's full-sequence token path. Do not Adopt Block-Recurrent's "one recurrent
  layer" — that paper stacked identical cells; Gemma's global layers are interleaved.
- **Architecture mapping:** concept read adapter + inference carry policy + additive
  writes + late-bin permutation diagnostics. Frozen Gemma, LoRA, tokenizer, collator,
  mix, LM head, and the per-window write loop stay.
- **Boldness check:** this plan implements global-attention replacement at four depths
  with no token carry. It does not substitute `MEMORY_CARRY_DROPOUT=1.0` on the E17c
  sidecar, nor a one-bank ablation.

## 2. Reuse map (read the modules first)

| Component | Action | Where |
|---|---|---|
| `BackboneConceptConfig` | extend: `concept_read_placement="post_layer"` (default) and `inference_carry_policy="normal"` (default) | `nn/backbone_concept_lm.py` |
| `ConceptReadBranch` | reuse as-is (dedicated Q/K/V/O already exist) | `nn/backbone_concept_lm.py` |
| `GlobalLayerWithConceptRead` | extend: `attn_residual` wraps `self.layer.self_attn` so concept mix is added to the attention sublayer output **before** FFN; `post_layer` keeps today's sidecar | `nn/backbone_concept_lm.py` |
| `ConceptWriteHead` | reuse `update_mode="additive"`. **Must pass `gate_init=config.write_gate_init` into untied `write_heads`** — today's ModuleList omits it, so `alpha` stays 0.0 and additive E17d writes would be dead at init | `nn/backbone_concept_lm.py` |
| `concept_gate_metrics` | fix: else-branch currently uses `self.write_head.depth_alphas` and crashes when `write_head is None` (untied). Use `_writer_for_depth` + `alpha` or `depth_alphas` | `nn/backbone_concept_lm.py` |
| `_forward_per_layer_banks_block` | reuse write timing: read bank `g`, run layer `g`, write bank `g` from `hidden_states[:, -block_len:]`. Do **not** let layer 11 read layer 5's just-written bank | `nn/backbone_concept_lm.py` |
| `_forward_blocks` | extend: default `carry_policy` at eval/`generate` follows `inference_carry_policy`; training Bernoulli dropout unchanged | `nn/backbone_concept_lm.py` |
| `generate` / `next_token_logits` | extend: pass `carry_policy` (today hard-codes `normal`) | `nn/backbone_concept_lm.py` |
| `concept_ablation_ce` | extend: intra-block bins 0–64 / 64–128 / 128–256 / 256–512 under the default carry policy, plus per-bank late-bin Δ | `nn/backbone_concept_lm.py` |
| `ModelArguments` | extend with the two new fields; defaults preserve E17c CLIs | `training/concept_pretraining_args.py` |
| `build_pretraining_model` | pass the new fields into `BackboneConceptConfig` | `training/concept_pretraining_factories.py` |
| Generic launcher | env/CLI pass-through only | `scripts/train_concept_pretraining_multigpu.sh` |
| Thin wrapper | new `scripts/launch_e17d.sh` pinning E17d env, then `launch_e10.sh` | `scripts/launch_e17d.sh` |
| Unit tests | extend `tests/test_backbone_concept_lm.py` | same file |

**No new model class, training entrypoint, collator, or `LossManager` loss.** Additive
writes, dedicated reads, per-layer banks, and `carry_policy="drop_after_first"` already
exist. The new mechanism is read *placement* plus matched no-carry decode.

### Backward-compatibility boundary
Defaults must reconstruct E17c / E17b load paths:

```python
concept_read_placement = "post_layer"
inference_carry_policy = "normal"
# unchanged defaults:
concept_read_mode = "backbone_qkv"
tie_concept_writer = True
concept_write_mode = "additive"
memory_carry_dropout = 0.0
memory_pressure_tokens = 0
```

Old checkpoints omit the new keys; `__init__` supplies these defaults. Do not rename
`per_layer_banks`, `encode_concepts()`, or `gated_replace`. E17c safetensors must still
load.

## 3. Forward pass (tensor shapes)

Symbols: `B`=microbatch, `N=4096`, `K=512`, `G=4`, `C=128`, `H=1152`, `V`=Gemma vocab.
Global layers `g ∈ {5,11,17,23}` (tiny tests: one global at layer 5 of 6).

### 3.1 When banks update (do not change this)

```text
z = concept_init                                      [B, G, C, H]   # four notebooks
for window b = 0 .. ceil(N/K)-1:                      # NOT "once at the end"
    tokens = current K (E17d: previous K carry masked on every b>0)
    h = embed(tokens)                                 [B, Q, H], Q ≤ 2K layout kept
    for each Gemma layer in order:
        if layer is global g:
            read  z[:, g] as it was AFTER window b-1  [B, C, H]
            mix into this layer's attention residual  [B, Q, H]
            FFN on that mix
            write z[:, g] from h[:, -K:]              [B, C, H]  # for window b+1 only
        else:
            sliding-window token layer; no bank I/O
    score only current-window next-token targets
```

Layer 11 in window `b` reads **bank 11 from window `b-1`**, never bank 5's write from
window `b`. Local layers 6–10 have no bank; they see the hidden stream after layer 5's
mix+FFN. That *is* "how the next layer uses the concepts."

### 3.2 Attention-residual read (the new piece)

Gemma's global decoder layer already does (names from the wrapped module in
`GlobalLayerWithConceptRead.layer`: `input_layernorm`, `self_attn`, then the MLP block):

```text
# sketch: existing Gemma attention residual, then FFN
residual = h
x        = InputLN(h)                         [B, Q, H]
tok      = WindowedSelfAttn(x)                [B, Q, H]
h        = residual + tok
h        = h + FFN(LN(h))
```

E17c (`post_layer`) waits until after FFN, then adds `tanh(gate) * CrossAttn(h_post, z)`.

E17d (`attn_residual`) adds the concept mix to the **attention output**, so FFN sees it:

```text
# sketch: wrap self_attn; do not clone Gemma's MLP
x        = InputLN(h)                         [B, Q, H]   # already done by the layer
tok      = WindowedSelfAttn(x)                [B, Q, H]
global   = DedicatedCrossAttn(Q=x, KV=z_g)    [B, Q, H]   # ConceptReadBranch
h        = residual + tok + tanh(read_gate) * global
h        = h + FFN(LN(h))                     # unchanged Gemma MLP
```

Implementation: at wrap time, if `concept_read_placement=="attn_residual"`, replace
`layer.self_attn` with `_AttnWithConceptResidual` that calls the original attention,
adds the concept branch on the same normalized `x` the attention received, and
returns the same output signature (tensor or `(hidden, weights, ...)`). Proxy
`is_sliding` (Gemma3DecoderLayer reads it on `self.self_attn` before the call).
Do not register the parent as a submodule of the wrapper (cycle). Then
`GlobalLayerWithConceptRead.forward` runs `self.layer(...)` **without** a second
post-layer add.

`read_gate` stays (init 0.1) so a random dedicated read does not blow pretrained residuals
at step 0. Placement, not deleting the scale, is the bet. Query is the **pre-attn
normalized** stream (`x`), not E17c dedicated's post-FFN `outputs[0]`.

Do **not** reintroduce full `O(N²)` token attention. Token attn stays windowed; concepts
are `O(Q·C)` per global layer.

### 3.3 Write (reuse additive)

After that layer returns, `_forward_per_layer_banks_block` already does:

```text
z_g ← z_g + tanh(α_g) * sandwich(BiXT(norm(z_g), norm(h_block)))
```

E17d pins `CONCEPT_WRITE_MODE=additive`, `TIE_CONCEPT_WRITER=false`, `WRITE_GATE_INIT=0.1`.
No new writer class.

### 3.4 No token carry at train, eval, generate

Keep the existing K-position layout (pad + BOS sentinel on the carry slots) so RoPE and
prediction indices stay E17b-identical.

```text
# sketch
if b > 0:
    if training and memory_carry_dropout > 0:
        drop = Bernoulli(p)                 # E17d: p = 1.0 → always
    elif not training and inference_carry_policy == "drop_after_first":
        drop = True                         # E17d generate + Trainer eval
    else:
        drop = False                        # E17c eval / old checkpoints
```

Thread `carry_policy` through `generate()` → `next_token_logits()` → `_forward_blocks`.
Today those two public methods omit it, so decode always sees the cheat sheet.

## 4. Inputs & data
- **Dataset:** immutable `e16b_long_4k_v1` Gemma-tokenized mix, seq 4096.
  `data/mix_recipes/e16b_long_4k_v1.json`.
- **Collator:** `data/data_collators.py:DataCollatorForCausalLM`, unchanged. Carry masking
  is inside `_forward_blocks`, not the collator.
- **Split / packing:** existing length-grouped 4K protocol (`launch_e17c.sh` cadence).

## 5. Loss & training objective
- **Loss:** existing block-recurrent next-token CE via `ChunkedLMHeadCE` /
  `_lm_ce_sum`. No new `register_loss`.
- **Weighting:** `MEMORY_PRESSURE_TOKENS=0`, `MEMORY_PRESSURE_WEIGHT=1.0` — uniform CE.
  Do not ×4 the first 64 tokens; that trained the E17c gist.
- **Geometry:** VICReg stays unwired for backbone (out of scope). RankMe is a kill
  metric, not a training loss.

## 6. Config & launch
- **New config fields** (safe defaults):
  - `BackboneConceptConfig.concept_read_placement: str = "post_layer"`
    (`post_layer` | `attn_residual`)
  - `BackboneConceptConfig.inference_carry_policy: str = "normal"`
    (`normal` | `drop_after_first`)
- **CLI / env:** `--concept_read_placement`, `--inference_carry_policy`; launcher knobs
  `CONCEPT_READ_PLACEMENT`, `INFERENCE_CARRY_POLICY` on
  `scripts/train_concept_pretraining_multigpu.sh`.
- **Registry:** still `checkpoint_family="backbone_concept"`. No new model type.
- **Launch:** `SKIP_PRETOKENIZE=1 bash scripts/launch_e17d.sh` with pins:

```bash
EXPERIMENT_ID=E17d
CONCEPT_IO_MODE=per_layer_banks
CONCEPT_READ_MODE=dedicated
CONCEPT_READ_PLACEMENT=attn_residual
TIE_CONCEPT_WRITER=false
CONCEPT_WRITE_MODE=additive
WRITE_GATE_INIT=0.1
MEMORY_CARRY_DROPOUT=1.0
INFERENCE_CARRY_POLICY=drop_after_first
MEMORY_PRESSURE_TOKENS=0
MEMORY_PRESSURE_WEIGHT=1.0
READ_CONCEPT_NORM=true
READ_GATE_INIT=0.1
TARGET_TOKENS=300000000
# plus the E16b/E17 data/optim block copied from launch_e17c.sh
```

## 7. Tests & smoke
Extend `tests/test_backbone_concept_lm.py` (tiny random Gemma, K=8, H=64). Assert:

- Default config: `post_layer` + `inference_carry_policy=normal` matches pre-change
  CE on a fixed tiny batch (E17c path).
- `attn_residual`: permuting `z` changes the **attention-sublayer** output, not only
  the post-FFN residual (hook `self_attn` / wrapped attn).
- Write still happens once per window per bank; layer 11's read in window `b` equals
  bank 11 after window `b-1` (no same-window leak from bank 5's write).
- `generate(..., max_new_tokens=2)` with `inference_carry_policy=drop_after_first`
  actually masks carry (decode IDs in the carry slots are pad/BOS, not previous tokens).
- `concept_ablation_ce` emits late-bin keys, e.g. `delta_permutation_block_256_512`
  (name may map to `K/2:K` on the tiny K=8 model).
- Additive writer + `attn_residual` : finite 3-step loss, no NaN.

Local smoke: `uv run pytest tests/test_backbone_concept_lm.py tests/test_training_launcher_parameter_flow.py -q`
(tiny random Gemma, K=8). 300M Polonez launch is `SKIP_PRETOKENIZE=1 bash scripts/launch_e17d.sh`.

## 8. Risks & tradeoffs
- **Risk:** FinePDFs CE still does not need previous windows after ~64 local tokens, even
  with a true global mix. **Cheapest signal:** 100M late-bin (256–512) all-bank Δperm
  `< 0.03` or only bank 0 participates — kill, then the objective has to change (MQAR),
  not another read-placement tweak.
- **Risk:** wrapping `self_attn` breaks if Gemma's attention return signature changes,
  or if GradientCheckpointingLayer recomputes after `_read_z` is cleared.
  **Fallback:** branch on tuple vs tensor; pass `z` as a checkpoint tensor input and
  call `layer.forward` (skip nested Gemma checkpoint). Tested with
  `gradient_checkpointing_enable()` on the tiny Gemma3TextConfig.
- **Risk:** `tanh(0.1)` still lets the mix die. **Signal:** read-gate telemetry at 100M;
  if all `|tanh| < 0.02` and late-bin is dead, that is the same kill, not a silent
  raise-init A/B in this run.
- **Risk:** additive writes reopen E17b's "close the valve" attractor. Without the token
  carry they should stay needed. If write gates collapse and late-bin is dead, kill.
- **Do not fall back to:** one bank, `PRESSURE_TOKENS=512`, or keeping generate-time carry.

## 9. Code sketches (decisions, not demos)

```python
# sketch: config — defaults keep E17c loadable
concept_read_placement: str = "post_layer"       # E17d: "attn_residual"
inference_carry_policy: str = "normal"           # E17d: "drop_after_first"

# sketch: attn wrap inside GlobalLayerWithConceptRead
def attn_forward(normed_x, *args, **kwargs):
    out = original_self_attn(normed_x, *args, **kwargs)
    tok = out[0] if isinstance(out, tuple) else out
    mixed = tok + tanh(read_gate) * read_branch(normed_x, z_g, original_self_attn)
    return (mixed,) + tuple(out[1:]) if isinstance(out, tuple) else mixed

# sketch: generate must not omit carry_policy
logits = self.next_token_logits(
    cur, mask, concept_mode=fwd_mode, initial_concepts=frozen_z,
    carry_policy=self.config.inference_carry_policy,
)
```
