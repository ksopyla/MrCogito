# E17c — Depth-private gated working memory — Implementation Plan

- **Spec:** [E17c_depth_private_working_memory.md](E17c_depth_private_working_memory.md) · **Status:** done_failed mixed (eval 2026-08-15)
- **Authored by:** `implementation-plan` · for → `research-implement`

> Implement the full E17c claim: depth-private read/write spaces, selective replacement
> dynamics, and causal carry pressure. Do not reduce it to another E17b gate-init run.
> Preserve E17b by putting every new behavior behind backward-compatible config defaults.

## 1. Source & fit
- **Origin:** E17/E17b isolated four banks but left two important couplings: every
  `ConceptReadBranch` borrows its wrapped Gemma attention's Q/K/V/O, and one
  `ConceptWriteHead` / BiXT basis is tied across layers 5/11/17/23. E17b's writes opened
  near 100M and closed by 1B while read gates opened strongly; plain next-token CE could
  use local context without rewarding persistent content. See the
  [E17b report](../../2_Experiments_Registry/run_reports/e17b_per_layer_mid_write_init_20260813.md).
- **Synthesis verdict:** **Adapt.** Take depth-private state from E17, BiXT's dedicated
  latent↔token projection space, gated retention from recurrent memory/GRU-like systems,
  and conditioning dropout's anti-bypass principle. Drop E16's shared accumulator and
  any same-block reread.
- **Architecture mapping:** touches the concept read adapter, recurrent concept cell,
  block input policy, weighted causal CE, and concept-specific evaluation. Frozen Gemma,
  LoRA, tokenizer, collator, data mix, LM head, and generation algorithm remain unchanged.
- **Boldness check:** this plan implements a private dynamical working-memory system plus
  the causal pressure that trains it. It does not substitute untied additive BiXT alone
  (E17a) or per-layer + init 0.3.

## 2. Reuse map

| Component | Action | Where |
|---|---|---|
| `BackboneConceptConfig` | extend with safe-default E17c knobs | `nn/backbone_concept_lm.py` |
| `ConceptReadBranch` | extend: existing Gemma-projection path unchanged; optional dedicated Q/K/V/O path | `nn/backbone_concept_lm.py` |
| `GlobalLayerWithConceptRead` | extend locally: dedicated mode queries normalized post-layer state; legacy mode keeps pre-layer Gemma query exactly | `nn/backbone_concept_lm.py` |
| `ConceptWriteHead` | extend with `gated_replace`; additive path and parameter names unchanged | `nn/backbone_concept_lm.py` |
| `BiXTCrossAttention` | reuse as-is inside each writer (`update_tokens=False`) | `nn/concept_encoder.py` |
| `BackboneConceptLM.__init__` | extend: tied legacy `write_head` or untied `write_heads: ModuleList` selected by config | `nn/backbone_concept_lm.py` |
| `_forward_per_layer_banks_block` | extend: dispatch the matching writer; keep one read then one write per bank per block | `nn/backbone_concept_lm.py` |
| `_forward_blocks` / `_lm_ce_sum` | extend: train-only per-example carry pressure and weighted first-R CE; defaults execute old code | `nn/backbone_concept_lm.py` |
| `encode_concepts` | preserve `[B,C,H]` last-bank return contract | `nn/backbone_concept_lm.py` |
| `encode_concept_banks` | new public method, not a class: expose `[B,G,C,H]` for diagnostics | `nn/backbone_concept_lm.py` |
| `concept_ablation_ce` / `per_position_ce` | extend: deterministic carryless and single-bank permutation diagnostics | `nn/backbone_concept_lm.py` |
| `concept_gate_metrics` | extend: report dynamic update-gate mean and update/state RMS per bank | `nn/backbone_concept_lm.py` |
| `ModelArguments` | extend with config fields; defaults preserve prior CLIs | `training/concept_pretraining_args.py` |
| `build_pretraining_model` | pass the new fields into `BackboneConceptConfig` | `training/concept_pretraining_factories.py` |
| `PerceiverDenoiseTrainer` concept probes | extend per-bank geometry logging without changing other families | `training/concept_pretraining_trainer.py` |
| Generic launcher | add env/CLI pass-through only | `scripts/train_concept_pretraining_multigpu.sh` |
| Thin experiment wrapper | add pinned E17c env values, delegating to `launch_e10.sh` | `scripts/launch_e17c.sh` |
| Analysis | add per-bank geometry, carryless first-64 metric, and per-bank permutation output | `analysis/run_concept_analysis.py` |
| Existing generation assessment | reuse as-is | `analysis/run_e16b_generation_assessment.py` |
| Unit/regression tests | extend existing backbone suite; no parallel test hierarchy | `tests/test_backbone_concept_lm.py` |

**No new model class, training entrypoint, collator, loss-manager component, or duplicated
attention implementation.** The two existing deep modules own their concerns:
`ConceptReadBranch` owns projection policy; `ConceptWriteHead` owns state-transition policy.
`BackboneConceptLM` only selects and routes them.

### Backward-compatibility boundary
The defaults must reconstruct E17b:

```python
concept_read_mode = "backbone_qkv"
tie_concept_writer = True
concept_write_mode = "additive"
memory_carry_dropout = 0.0
memory_pressure_tokens = 0
memory_pressure_weight = 1.0
```

- Keep `write_head.*`, `depth_alphas`, and every existing state-dict key unchanged in
  legacy mode. Instantiate `write_heads.*` only when `tie_concept_writer=False`.
- Old configs lack the new fields; `BackboneConceptConfig.__init__` supplies these defaults.
- Do not rename `concept_io_mode="per_layer_banks"` or alter `encode_concepts()`.
- E17c keeps `checkpoint_family="backbone_concept"` and the existing evaluation routing.
- The implementation PR must compare a fixed tiny E17b input/state at the pre-change commit
  and post-change default path; valid-position CE and state dict keys must match exactly.

## 3. Forward pass

Symbols: `B`=microbatch, `N=4096`, `K=512`, `G=4`, `C=128`, `H=1152`,
`A=H`, `h=4` dedicated attention heads, `V`=Gemma vocabulary.

### 3.1 State initialization and block schedule

```text
input_ids, attention_mask                                      [B,N]
z = concept_init expanded                                      [B,G,C,H]

for block b = 0..ceil(N/K)-1:
    normal input: previous K carry + current block              [B,≤2K]
    pressured input (selected examples, b>0):
      mask prior K carry, retain one BOS sentinel at carry[-1] [B,≤2K]
    run Gemma layers in their existing order
    each global layer g reads z[:,g], then writes z_next[:,g]
    score only new current-block targets
```

Pressure selection is per example, sampled once outside checkpointed layer calls:
`pressure = training & b>0 & Bernoulli(0.5) & has_valid_current_tokens`. Replace carry
IDs for selected rows with `pad_token_id`, set carry mask to zero, then set the last
carry ID/mask to `bos_token_id`/one. Keeping the K-position layout preserves E17b's
prediction indices and RoPE positions. Padded carry queries are ignored by labels; the
existing diagonal mask escape keeps them finite.

### 3.2 Dedicated read at global layer g

```text
h_pre                                                   [B,Q,H]
h_post = GemmaDecoderLayer_g(h_pre)                     [B,Q,H]
q = DedicatedRead_g.q_proj(query_norm(h_post))          [B,h,Q,H/h]
k,v = DedicatedRead_g.kv_proj(concept_norm(z_g))        [B,h,C,H/h] each
read = o_proj(SDPA(q,k,v))                               [B,Q,H]
h = h_post + tanh(read_gate_g) * read                   [B,Q,H]
```

Implementation stays in `ConceptReadBranch`:
- `concept_read_mode="backbone_qkv"` constructs only the existing concept RMSNorm and
  calls the wrapped Gemma attention projections exactly as today.
- `"dedicated"` constructs `query_norm`, `q_proj`, combined `kv_proj`, per-head q/k
  RMSNorms, and `o_proj`. Use `F.scaled_dot_product_attention`; do not add a second
  attention class.
- Each `GlobalLayerWithConceptRead` already owns one branch, so dedicated projections are
  naturally untied across depths.
- In dedicated mode the branch receives `outputs[0]` (post-layer representation). The
  legacy path continues to receive `layer.input_layernorm(hidden_states)` to avoid changing
  E17b numerics.

### 3.3 Untied selective write at global layer g

Each entry of `write_heads[g]` is the existing `ConceptWriteHead` configured as
`gated_replace`:

```text
z_old = z[:,g]                                           [B,C,H]
x = current-block hidden states after layer g            [B,Kb,H]
lat_update = BiXT_g(norm_lat(z_old), norm_tok(x))         [B,C,H]
candidate = sandwich(lat_update)                         [B,C,H]
gate_logit = gate_proj(cat(norm_lat(z_old), candidate))  [B,C,1]
u = sigmoid(gate_logit)                                  [B,C,1]
z_new = (1-u) * z_old + u * candidate                    [B,C,H]
```

- `gate_proj: Linear(2H,1)` has zero weights and bias `logit(0.25)`.
- Fully padded rows return `z_old` unchanged.
- Store detached `mean(u)`, `RMS(z_new-z_old)`, and `RMS(z_old)` for telemetry; never
  use detached values in the forward computation.
- `gated_replace` creates no `alpha`/`depth_alphas`. The content-dependent gate is the
  only write valve; it cannot be globally starved behind a near-zero tanh scalar.
- `ConceptWriteHead(update_mode="additive")` executes its current lines unchanged.
- Add a private `BackboneConceptLM._writer_for_depth(g)` helper so the block loop does
  not know whether the model owns one tied head or a `ModuleList`.

The write occurs after the layer's read and is not read by bank g until the next block.
Other global layers read only their own banks. Complexity remains block-linear:
token attention `O(NK)` plus reads/writes `O(GNC)`; no full `O(N²)` path is added.

### 3.4 Weighted causal objective

For every block, retain the normal token-summed next-token CE. For selected pressured
examples, add `(memory_pressure_weight-1)=3` extra copies of CE over the first
`R=64` valid current-block targets:

```text
weighted_sum   = base_ce_sum + 3 * pressure_prefix_ce_sum
weighted_count = base_count  + 3 * pressure_prefix_valid_count
loss = DDP_true_global_mean(weighted_sum, weighted_count)
```

Reuse `ChunkedLMHeadCE` through `_lm_ce_sum`; call it once for the base target slice and
once for the selected `[B_pressure,R]` slice. Do not materialize `[B,N,V]` logits and do
not register a new `LossManager` loss. With all pressure knobs at defaults, take the
existing base-only branch and preserve its denominator exactly.

## 4. Inputs & data
- **Dataset:** `data/mix_recipes/e16b_long_4k_v1.json`, existing immutable
  Gemma-tokenized manifest, max length 4096.
- **Collator:** `data/data_collators.py:DataCollatorForCausalLM`, reused unchanged.
  It continues to right-pad and set padded labels to `-100`.
- **Preprocessing:** unchanged; no prefix/suffix split and no cross-document packing.
- **Masking:** pressure is an internal model intervention on the prior-block carry only.
  It never changes labels, current-block teacher-forced tokens, or concept states already
  written from earlier actual tokens.
- **Generation:** model evaluation mode disables stochastic pressure. Existing
  `generate()` continues to re-encode the growing prefix and uses normal explicit carry.

## 5. Loss and training objective
- **Base objective:** block-recurrent causal next-token CE in `BackboneConceptLM.forward`.
- **Memory pressure:** input intervention plus first-64 weighting described above; not an
  auxiliary decoder and not a representation regularizer.
- **Gradients:** flow through current target CE → dedicated read g → `z_g^b` → every prior
  gated replacement transition. BPTT remains across all eight 512-token blocks.
- **No detach:** never detach concept banks between blocks.
- **No extra objective:** no VICReg, reconstruction, delayed-recall labels, or synthetic
  supervision. The clean causal objective remains the sole target distribution.

## 6. Config and launch

Add these `BackboneConceptConfig` and matching `ModelArguments` fields:

| Field | Default | E17c |
|---|---:|---:|
| `concept_read_mode` | `"backbone_qkv"` | `"dedicated"` |
| `tie_concept_writer` | `True` | `False` |
| `concept_write_mode` | `"additive"` | `"gated_replace"` |
| `write_update_gate_init` | `0.25` (inactive unless gated) | `0.25` |
| `memory_carry_dropout` | `0.0` | `0.5` |
| `memory_pressure_tokens` | `0` | `64` |
| `memory_pressure_weight` | `1.0` | `4.0` |

Validation in `BackboneConceptLM.__init__`:
- dedicated/gated/untied modes require `concept_io_mode="per_layer_banks"` and
  `concept_num>0`;
- `0≤memory_carry_dropout≤1`, `0≤memory_pressure_tokens≤K`,
  `memory_pressure_weight≥1`;
- pressure tokens/weight must be inactive when dropout is zero;
- `0<write_update_gate_init<1`.

`training/concept_pretraining_factories.py` passes every field. The model registry remains
`MODEL_REGISTRY["backbone_concept"]`; no new family or eval route.

Add env defaults and CLI pass-through to
`scripts/train_concept_pretraining_multigpu.sh`:
`CONCEPT_READ_MODE`, `TIE_CONCEPT_WRITER`, `CONCEPT_WRITE_MODE`,
`WRITE_UPDATE_GATE_INIT`, `MEMORY_CARRY_DROPOUT`, `MEMORY_PRESSURE_TOKENS`,
`MEMORY_PRESSURE_WEIGHT`.

Create only a thin reproducibility wrapper:

```bash
# scripts/launch_e17c.sh
export EXPERIMENT_ID="${EXPERIMENT_ID:-E17c}"
export CONCEPT_IO_MODE=per_layer_banks
export CONCEPT_READ_MODE=dedicated
export TIE_CONCEPT_WRITER=false
export CONCEPT_WRITE_MODE=gated_replace
export WRITE_UPDATE_GATE_INIT=0.25
export MEMORY_CARRY_DROPOUT=0.5
export MEMORY_PRESSURE_TOKENS=64
export MEMORY_PRESSURE_WEIGHT=4.0
export READ_CONCEPT_NORM=true
export READ_GATE_INIT=0.1
export TARGET_TOKENS="${TARGET_TOKENS:-300000000}"
exec bash "${SCRIPT_DIR}/launch_e10.sh"
```

The wrapper's default token budget is the 300M mechanism verdict, not 1B. A later 1B
quality run is a separate cosine and is launched only if this gate passes. The full
equivalent command is frozen in the spec. `scripts/launch_e17b.sh` must remain
unchanged and runnable.

## 7. Evaluation contract

### Per-bank state exposure
Add:

```python
# sketch
@torch.no_grad()
def encode_concept_banks(self, input_ids, attention_mask=None) -> Tensor:
    """Return [B,G,C,H] for per-layer modes; raise for non-banked modes."""
```

Keep `encode_concepts(...).last_hidden_state == banks[:,-1]`. Update
`PerceiverDenoiseTrainer._concept_effective_rank` and
`analysis/run_concept_analysis.py` to use `encode_concept_banks` when present and log:
`bank_0..3/within_sample_rankme`, centered RankMe, pairwise cosine, and min/median/max.
Other model families continue through the existing single-state path.

### Carryless and per-bank necessity
Extend `per_position_ce` with internal keyword-only controls used by analysis:
`carry_policy="normal"|"drop_after_first"` and
`concept_bank_index: Optional[int]`. For `per_layer_banks`, apply shuffle/permutation
inside `_forward_per_layer_banks_block` to all banks or only the selected bank; do not
change shared/global modes.

Extend `concept_ablation_ce` to report:
- existing normal `real/shuffle/zero/static/one_block` keys unchanged;
- `pressure_ce_real_first64`, `pressure_ce_permutation_first64`,
  `pressure_delta_permutation_first64` over blocks 2–7;
- per-bank normal and pressure permutation deltas.

Use a seeded cyclic batch permutation and at least 24 held-out batches. The analysis JSON
must include per-batch values so the caller can bootstrap a 95% CI. A batch of one is invalid
for permutation and must be skipped explicitly.

### Generation
Reuse `analysis/run_e16b_generation_assessment.py` on best and last checkpoints. Report
`real`, `zero`, `shuffle`, `static`, and base Gemma with the same continuation prompt bank,
greedy lengths through 512, and sampling as diagnostic. Do not mix its headline with
`run_generation_quality.py`'s chat+continuation aggregate.

## 8. Tests and smoke

Extend `tests/test_backbone_concept_lm.py`:

1. **Legacy construction:** default `per_layer_banks` owns `write_head`, not
   `write_heads`; no dedicated read projections; state-dict keys match the pre-change
   tiny E17b capture.
2. **Legacy numerical regression:** fixed-seed tiny E17b valid-position CE, final banks,
   gate metrics, and save/load output match the pre-change capture.
3. **Dedicated construction:** four global layers own distinct read projection parameter
   identities; four distinct writer heads exist; no accidental parameter sharing.
4. **Gated cell equation:** zero gate-projection weights produce exactly `u=0.25`;
   output equals `(1-u)z + u*candidate`; padded rows are identity.
5. **Gradient reachability:** one 3-block backward gives finite nonzero gradients to each
   dedicated reader, each BiXT writer, each gate projection, concept init, read gates,
   and LoRA.
6. **Intra-block causality:** perturb future tokens inside block b; CE at earlier
   positions in b is unchanged with pressure both off and forced on.
7. **Cross-block causality:** perturb block b; predictions before/inside the unaffected
   prefix are unchanged, bank state after b changes, and later block predictions may change.
8. **No same-block reread:** instrument bank g; its write from block b is first observed
   by read g in block b+1, never by any read in b.
9. **Pressure mask:** selected examples cannot attend prior carry tokens; unselected
   examples match normal mode; BOS sentinel predicts the first current target.
10. **Weighted CE:** compare the chunked weighted implementation to dense-logit reference
    on a tiny vocabulary, including padding and mixed pressure rows.
11. **Determinism:** fixed seed reproduces pressure masks/loss with and without non-reentrant
    gradient checkpointing.
12. **Per-bank diagnostics:** `encode_concept_banks` is `[B,G,C,H]`;
    `encode_concepts` equals its last bank; single-bank permutation changes only the selected
    bank's read path.
13. **Checkpoint round-trip:** E17c save/load preserves config, all four cells, loss, banks,
    and generation; an old-style E17b checkpoint still loads with no unexpected/missing keys.
14. **Ablation modes:** `real/zero/shuffle/permutation/static/one_block/frozen` remain finite
    for E17c; all pre-existing E10/E16/E17 tests stay green.

Commands:

```bash
uv run pytest tests/test_backbone_concept_lm.py -q
uv run pytest tests/ -q
```

Tiny smoke (CPU in Cloud; MPS on the author's Mac) uses H=64, C=4, G=2, K=8,
three blocks, mixed pressure rows, three optimizer steps, then save/load and
`run_concept_analysis.py`. Assert finite/decreasing loss, nonzero cell gradients,
per-bank metrics present, and a generated continuation in each concept mode.

Before remote training:

```bash
EXPERIMENT_ID=E17c MAX_STEPS=50 REPORT_TO=none \
LOAD_BEST_MODEL_AT_END=False SAVE_STRATEGY=no EVAL_STRATEGY=no \
bash scripts/launch_e17c.sh
```

Record peak VRAM, samples/s, trainable parameter count by component, and select the largest
stable microbatch. Do not silently change C/K/pressure settings to fit memory.

## 9. Risks and tradeoffs
- **Risk — composition attribution:** E17c changes cell topology and training pressure
  together. That is intentional: the coherent claim is about a usable working-memory
  system. **Cheapest signal:** primary carryless Δpermutation. If positive, later specs
  can ablate dedicated reads, untied writers, gated replacement, and pressure.
- **Risk — pressure teaches a carryless dialect and harms normal LM CE.**
  **Cheapest signal:** normal held-out eval loss and `zero` generation at 100M.
  **Kill:** spec's +0.10 final guard / staged instability rules; do not lower pressure
  post hoc inside this run.
- **Risk — update gates saturate closed or open.** **Cheapest signal:** per-bank mean,
  p10/p50/p90 update gate and update/state RMS. The content gate's 0.25 live prior and
  pressure objective replace scalar-init escalation.
- **Risk — random dedicated readers initially perturb Gemma.** `READ_GATE_INIT=0.1`
  limits the residual while preserving gradient. If the run diverges, that is a kill;
  do not warm-start from Gemma QKV because that would negate the representation-space bet.
- **Risk — untied cells add tens of millions of trainable parameters and activation cost.**
  Report parameter/FLOP deltas and calibrate. The baseline comparison is mechanism- and
  token-budget-matched, not iso-parameter; an iso-parameter ablation follows only after
  a positive mechanism signal.
- **Risk — analysis still hides banks.** The new bank API and single-bank intervention
  are required before launch, not optional polish.
- **Risk — stochastic pressure plus checkpoint replay.** Sample masks outside checkpointed
  functions and test plain/checkpointed equality under a fixed mask.

## 10. Code sketches

```python
# sketch — config only; defaults are the E17b compatibility boundary
concept_read_mode: str = "backbone_qkv"
tie_concept_writer: bool = True
concept_write_mode: str = "additive"
write_update_gate_init: float = 0.25
memory_carry_dropout: float = 0.0
memory_pressure_tokens: int = 0
memory_pressure_weight: float = 1.0
```

```python
# sketch — deep-module selection, no new architecture class
if config.tie_concept_writer:
    self.write_head = ConceptWriteHead(..., update_mode=config.concept_write_mode)
    self.write_heads = None
else:
    self.write_head = None
    self.write_heads = nn.ModuleList(
        ConceptWriteHead(..., update_mode=config.concept_write_mode)
        for _ in self.global_layer_indices
    )

def _writer_for_depth(self, depth_index):
    return self.write_head if self.write_heads is None else self.write_heads[depth_index]
```

```python
# sketch — selective replacement inside ConceptWriteHead
candidate, _ = self.bixt(self.norm_lat(z), self.norm_tok(h_block), key_padding_mask=pad_mask)
candidate = self.sandwich(candidate)
u = torch.sigmoid(self.update_gate(torch.cat([self.norm_lat(z), candidate], dim=-1)))
next_z = torch.lerp(z, candidate, u)
return torch.where(valid_row[:, None, None], next_z, z)
```

```python
# sketch — pressure keeps dimensions/indexing identical to the normal 2K block input
pressure = sampled_pressure & valid_current
dec_ids[pressure, :carry_len] = pad_id
dec_mask[pressure, :carry_len] = 0
dec_ids[pressure, carry_len - 1] = bos_id
dec_mask[pressure, carry_len - 1] = 1
```
