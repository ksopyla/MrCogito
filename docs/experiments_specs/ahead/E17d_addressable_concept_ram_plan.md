# E17d — Addressable concept RAM — Implementation Plan

- **Spec:** [E17d_addressable_concept_ram.md](E17d_addressable_concept_ram.md) · **Status:** draft
- **Authored by:** `implementation-plan` · for → `research-implement`

> Implement the full E17d claim: concept banks are RAM with positions, sparse
> top-k erase-add writes, sparse top-k reads, and unused-slot invariance, under
> E17c's causal carry pressure. Do **not** keep BiXT/`gated_replace` "for a first
> version." Preserve E17b/E17c by putting every new behavior behind
> backward-compatible config defaults.

## 1. Source & fit
- **Origin:** E17c forced concept use (carryless Δperm 0.59) then collapsed
  RankMe to 6.7 because `gated_replace` interpolates **all 128 rows**. Literature:
  [addressable_memory.md](../../literature_review/addressable_memory.md)
  (NTM erase-add, DNC allocation, SAM hard top-k vs Hopfield/GSA dense mix).
  E17c plan:
  [E17c_depth_private_working_memory_plan.md](../done_failed/E17c_depth_private_working_memory_plan.md).
- **Synthesis verdict:** **Adapt.** Take NTM's unused-slot invariant and hybrid
  location/content scores, DNC's usage bias (not the temporal link matrix), SAM's
  hard top-k (not ANN). Drop GSA convex slot writes, Infini/Titans associative
  matrices, E11 mem-tokens, and BiXT-over-all-C.
- **Architecture mapping:** concept bottleneck read adapter + recurrent write
  cell + address book + diagnostics. Frozen Gemma, LoRA, tokenizer, collator,
  data mix, pressure CE, LM head, and generation stay E17c.
- **Boldness check:** the write is **not** BiXT then mask. A controller pools the
  block, addresses k rows, erase-adds only those rows. Reads are **not** dense
  SDPA over C. Do not implement "dedicated + gated_replace + entropy bonus" as a
  substitute.

## 2. Reuse map (read the modules first)

| Component | Action | Where |
|---|---|---|
| `BackboneConceptConfig` | extend with safe-default RAM knobs | `nn/backbone_concept_lm.py` |
| `topk_renorm` | **new** helper (STE hard top-k) | `nn/backbone_concept_lm.py` (module-level) |
| `ConceptWriteHead` | extend: `addressed_erase_add`; `additive`/`gated_replace` bit-identical | `nn/backbone_concept_lm.py` |
| `BiXTCrossAttention` | **do not call** on the addressed path | `nn/concept_encoder.py` |
| `ConceptReadBranch` | extend: `addressed_topk`; `backbone_qkv`/`dedicated` unchanged | `nn/backbone_concept_lm.py` |
| `GlobalLayerWithConceptRead` | extend: hold a reference to the depth's `address_emb`; pass it into the read branch | `nn/backbone_concept_lm.py` |
| `BackboneConceptLM.__init__` | validate new modes; zero-init `concept_init` when asked; wire address_emb into wrappers | `nn/backbone_concept_lm.py` |
| `_writer_for_depth` | reuse as-is | `nn/backbone_concept_lm.py` |
| `_forward_per_layer_banks_block` | extend: thread per-bank `usage [B,C]`; addressed writer returns `(z, usage)` | `nn/backbone_concept_lm.py` |
| `_forward_blocks` | extend: init `usage` zeros `[B,G,C]` at sequence start; carry across blocks | `nn/backbone_concept_lm.py` |
| `_forward_shared_depth_block` / `global_kv` | unchanged | `nn/backbone_concept_lm.py` |
| pressure / `_lm_ce_sum` | reuse E17c as-is | `nn/backbone_concept_lm.py` |
| `encode_concepts` / `encode_concept_banks` | reuse shapes; banks remain `[B,G,C,H]` | `nn/backbone_concept_lm.py` |
| `concept_ablation_ce` / `per_position_ce` | reuse E17c carryless + per-bank permutation | `nn/backbone_concept_lm.py` |
| `concept_gate_metrics` | extend: write entropy, occupancy, top-1 mass, unused `max|Δz|` | `nn/backbone_concept_lm.py` |
| `ModelArguments` | extend; defaults preserve prior CLIs | `training/concept_pretraining_args.py` |
| `build_pretraining_model` | pass the new fields | `training/concept_pretraining_factories.py` |
| `PerceiverDenoiseTrainer` | already logs `concept_gate_metrics()`; no new callback | `training/concept_pretraining_trainer.py` |
| Generic launcher | env/CLI pass-through only | `scripts/train_concept_pretraining_multigpu.sh` |
| Thin wrapper | pin E17d env, delegate to `launch_e10.sh` | `scripts/launch_e17d.sh` (new, like `launch_e17c.sh`) |
| Analysis | RankMe of written vs unwritten slots from usage/write-mass | `analysis/run_concept_analysis.py` |
| Generation assessment | reuse as-is | `analysis/run_e16b_generation_assessment.py` |
| Tests | extend existing backbone suite | `tests/test_backbone_concept_lm.py` |

**No new model class, training entrypoint, collator, LossManager loss, or
architecture fork.** `ConceptReadBranch` still owns read policy;
`ConceptWriteHead` still owns the state transition. Addressed mode **must not**
construct or call `self.bixt` for the update (BiXT is the dense mixer E17c used).

### Backward-compatibility boundary
Defaults reconstruct E17b (and E17c when its knobs are set):

```python
concept_read_mode = "backbone_qkv"
concept_write_mode = "additive"
concept_state_init = "gaussian"
address_write_topk = 0          # ignored unless addressed
address_read_topk = 0
address_allocation = False
address_allocation_scale = 1.0
address_usage_mu = 0.1
```

- Keep every existing state-dict key in legacy mode (`write_head.bixt.*`,
  `write_head.update_gate.*`, `concept_init`, dedicated `q_proj`/`kv_proj`).
- Addressed parameters live only under new names
  (`write_head.address_emb`, `write_head.write_queries`, `write_head.controller`,
  `read_branch.q_loc_proj`, …) so E17b/E17c checkpoints load with
  `strict=True` on the old keys.
- Do not rename `concept_io_mode="per_layer_banks"` or change
  `encode_concepts()` / `encode_concept_banks()`.
- `checkpoint_family` stays `"backbone_concept"`.
- Implementation PR must include a tiny E17c default-path numerical snapshot
  (valid-position CE + state-dict keys) matching the pre-change commit.

## 3. Forward pass (tensor shapes)

Symbols: `B`=microbatch, `N=4096`, `K=512`, `G=4`, `C=128`, `H=1152`,
`H_w=4` (`write_num_heads`), `K_w=4` write top-k, `K_r=8` read top-k,
`V`=Gemma vocab.

### 3.1 Outer loop (unchanged from E17c)

```text
input_ids, attention_mask                         [B, N]
z = concept_init expanded                         [B, G, C, H]   # zeros if concept_state_init=zeros
usage = 0                                         [B, G, C]

for block b = 0 .. ceil(N/K)-1:
    window = previous K carry + current block     [B, ≤2K]
    optional E17c pressure on carry (b>0)
    run Gemma layers; global layer g:
        read z[:,g] via addressed_topk
        write (z[:,g], usage[:,g]) via addressed_erase_add
    score only new current-block targets
```

Pressure sampling, BOS sentinel, weighted first-64 CE, and block causality stay
exactly as `_forward_blocks` / `_lm_ce_sum` already do
(`nn/backbone_concept_lm.py`). Generation still disables stochastic pressure.

### 3.2 Address book (per writer / bank)

Each `ConceptWriteHead` in `addressed_erase_add` owns:

```text
address_emb    Parameter [C, H]     # slot index → location; Xavier/H^-0.5
write_queries  Parameter [H_w, H]   # learned pooling queries
controller     Linear(H, 3H+3)      # per head: k_loc, k_cnt, add, erase_logit, g_logit, β_raw
```

`GlobalLayerWithConceptRead` at depth `g` stores a reference to
`_writer_for_depth(g).address_emb` (tied writer ⇒ all four layers share one
book; E17d is untied ⇒ four books). Address embeddings are parameters, not
sequence state.

### 3.3 Sparse write (replaces BiXT `gated_replace`)

Called from `_forward_per_layer_banks_block` after layer `g`, on
`hidden_states[:, -block_len:]` `[B, Kb, H]` and `z_g` `[B, C, H]`, matching
today's call at `nn/backbone_concept_lm.py` (~848).

```text
# 1. Pool the block — O(H_w · Kb · H), not O(C · Kb)
scores_pool = write_queries @ h_block^T / sqrt(H)          [B, H_w, Kb]
scores_pool.masked_fill_(block_pad, -inf)
pooled = softmax(scores_pool) @ h_block                    [B, H_w, H]

# 2. Sequential NTM heads (not a sum that re-densifies)
for h = 0 .. H_w-1:
    k_loc, k_cnt, add = split Linear(pooled[:,h])[:3H]     [B, H] each
    erase = sigmoid(erase_logit)                           [B, H]        # NTM erase ∈ (0,1)^H
    g_mix = sigmoid(g_logit)                               [B, 1]
    β     = 1 + softplus(β_raw)                            [B, 1]        # ≥ 1
    loc   = address_emb @ k_loc                            [B, C]
    cnt   = (z @ k_cnt)                                    [B, C]
    scores = g_mix * cnt + (1-g_mix) * loc
    if address_allocation:
        scores = scores - λ * usage                        # DNC-lite unused bias
    w = topk_renorm(softmax(β * scores), k=K_w)            [B, C]  # exactly C-K_w zeros
    z = z * (1 - w[...,None] * erase[:,None,:]) + w[...,None] * add[:,None,:]
    usage = (1-μ) * usage + μ * w

# fully padded rows: return (z_old, usage_old)
```

`topk_renorm` (module-level, tested alone):

```python
# sketch
def topk_renorm(weights: Tensor, k: int) -> Tensor:
    """weights already softmax-normalized on the last dim. Unselected → exact 0."""
    if k >= weights.size(-1):
        return weights
    _, idx = weights.topk(k, dim=-1)
    mask = torch.zeros_like(weights).scatter_(-1, idx, 1.0)
    sparse = weights * mask          # STE: backward through selected softmax values
    return sparse / sparse.sum(dim=-1, keepdim=True).clamp_min(1e-8)
```

Unaddressed invariance is algebraic: `w_i=0` ⇒ `Δz_i=0`. Do **not** implement
this as "run BiXT then multiply the residual by w" — BiXT would still mix every
row in the forward.

**Dtype:** keep `z` and the erase-add in FP32 (E17c's BF16-candidate vs FP32-state
bug). Autocast may wrap pooling SDPA; cast `pooled`/`z` to `z.dtype` before the
controller. Store detached telemetry: mean write entropy, occupancy, top-1 mass,
`max|Δz|` on the complement of the union of head masks, update RMS, state RMS.

**Writer signature change (addressed only):**

```python
# sketch
def forward(self, z, h_block, pad_mask, *, depth_index=None, usage=None):
    # additive / gated_replace: ignore usage, return z only (today's contract)
    # addressed_erase_add: usage is [B,C]; return (next_z, next_usage)
```

`_writer_for_depth` callers in the per-layer loop unpack only when
`writer.update_mode == "addressed_erase_add"`. Shared-depth / global_kv never
take this mode in E17d (validate in `__init__`: addressed modes require
`per_layer_banks`).

### 3.4 Sparse read (replaces dense SDPA over C)

`ConceptReadBranch.mode == "addressed_topk"` builds dedicated projections
(same `query_norm` / per-head q/k norms as `"dedicated"`) plus location/content
splits. It does **not** call `F.scaled_dot_product_attention` over all C.

```text
x_read = query_norm(h_post)                                [B, Q, H]
z_read = concept_norm(z)                                   [B, C, H]
q = q_proj(x_read) → [B, h, Q, Hd]
k_cnt = k_cnt_proj(z_read) → [B, h, C, Hd]
k_loc = k_loc_proj(address_emb) → [h, C, Hd]  (broadcast B)
v     = v_proj(z_read) → [B, h, C, Hd]

loc = (q · k_loc) / sqrt(Hd)                               [B, h, Q, C]
cnt = (q · k_cnt) / sqrt(Hd)                               [B, h, Q, C]
scores = g_r * cnt + (1-g_r) * loc                         # g_r sigmoid param, init 0.5
w = topk_renorm(softmax(β_r * scores), k=K_r)              [B, h, Q, C]
read = o_proj( (w @ v).reshape(B, Q, H) )                  [B, Q, H]
h = h_post + tanh(read_gate) * read                        # existing GlobalLayer residual
```

`address_emb` is the **same tensor** the writer owns (identity of locations).
`GlobalLayerWithConceptRead.forward` already chooses post-layer `outputs[0]`
for dedicated mode (`nn/backbone_concept_lm.py` ~285); addressed_topk uses that
same query source.

Dense `"dedicated"` SDPA path stays for E17c checkpoints. Do not add RoPE on
slots; location identity is `address_emb`, not token positions.

### 3.5 Complexity
Token self-attention remains windowed `O(N K)`. Addressed write is
`O(G · H_w · (K H + C H))`. Addressed read is `O(G · Q · C · H)` for scores
then `O(Q · K_r · H)` for the gather — still `O(C N)` class, no `O(N²)`.
Top-k does not change the score cost at C=128; it changes the **update
support**.

### 3.6 `concept_init`
`concept_state_init="gaussian"` (default): today's
`randn * H^-0.5` (`nn/backbone_concept_lm.py` ~529).
`"zeros"`: `nn.Parameter(torch.zeros(G, C, H))`, still trainable. E17d uses
zeros so blank rows are identical and **location + allocation** must choose
where to write (DNC/SAM empty-memory lesson). Do not freeze `concept_init`.

## 4. Inputs & data
- **Dataset:** `data/mix_recipes/e16b_long_4k_v1.json`, existing immutable
  Gemma-tokenized 4K manifest. Same as E17c.
- **Collator:** `data/data_collators.py:DataCollatorForCausalLM`, unchanged.
- **Preprocessing / masking / split:** unchanged. Pressure remains an internal
  model intervention on prior-block carry after collation
  (`_forward_blocks` ~986–1014).
- **Generation:** eval mode, no stochastic pressure. `generate()` re-encodes
  the growing prefix with normal carry.

## 5. Loss & training objective
- **Loss:** existing block-recurrent causal CE in `BackboneConceptLM.forward` +
  E17c first-64 pressure weight. **No** new `register_loss`, no VICReg, no
  occupancy regularizer, no copy/bAbI auxiliary.
- **Gradients:** CE → sparse read → `z_g^b` → prior erase-add writes →
  `address_emb` / controller / write_queries. BPTT across all eight 512-token
  blocks. Never detach banks or usage between blocks.
- **STE:** top-k mask is non-differentiable; gradient flows through the
  selected softmax values only (`weights * mask`). That is the intended
  unused-slot invariant, not a bug to "fix" with a soft extra mass.

## 6. Config & launch

New `BackboneConceptConfig` / `ModelArguments` fields (defaults = legacy):

| Field | Default | E17d |
|---|---|---|
| `concept_read_mode` | `"backbone_qkv"` | `"addressed_topk"` |
| `concept_write_mode` | `"additive"` | `"addressed_erase_add"` |
| `tie_concept_writer` | `True` | `False` |
| `concept_state_init` | `"gaussian"` | `"zeros"` |
| `address_write_topk` | `0` | `4` |
| `address_read_topk` | `0` | `8` |
| `address_allocation` | `False` | `True` |
| `address_allocation_scale` | `1.0` | `1.0` |
| `address_usage_mu` | `0.1` | `0.1` |
| E17c pressure knobs | off | same as E17c (0.5 / 64 / 4.0) |
| `read_gate_init` | `0.0` | `0.1` |
| `read_concept_norm` | `False` | `True` |
| `write_num_heads` | `4` | `4` (pooling heads = read heads) |

Validation in `BackboneConceptLM.__init__` (extend the existing block at ~423):
- `concept_read_mode ∈ {backbone_qkv, dedicated, addressed_topk}`
- `concept_write_mode ∈ {additive, gated_replace, addressed_erase_add}`
- addressed read or write requires `concept_io_mode="per_layer_banks"` and
  `concept_num>0`
- `addressed_topk` requires `address_read_topk ∈ [1, C]`
- `addressed_erase_add` requires `address_write_topk ∈ [1, C]`
- `concept_state_init ∈ {gaussian, zeros}`
- `0 < address_usage_mu ≤ 1`; `address_allocation_scale ≥ 0`
- E17d launcher sets **both** addressed read and write; the foundation may
  accept mixed modes for later ablations, but this run does not mix.

`training/concept_pretraining_factories.py` passes every new field next to the
existing `concept_write_mode=` kwargs (~238). Registry stays
`MODEL_REGISTRY["backbone_concept"]`.

Generic runner `scripts/train_concept_pretraining_multigpu.sh`: add
`CONCEPT_STATE_INIT`, `ADDRESS_WRITE_TOPK`, `ADDRESS_READ_TOPK`,
`ADDRESS_ALLOCATION`, `ADDRESS_ALLOCATION_SCALE`, `ADDRESS_USAGE_MU` with
legacy defaults and `--` pass-through. Do not copy the runner.

Thin wrapper `scripts/launch_e17d.sh` (clone `launch_e17c.sh` structure):

```bash
export EXPERIMENT_ID="${EXPERIMENT_ID:-E17d}"
export CONCEPT_IO_MODE=per_layer_banks
export CONCEPT_READ_MODE=addressed_topk
export CONCEPT_WRITE_MODE=addressed_erase_add
export TIE_CONCEPT_WRITER=false
export CONCEPT_STATE_INIT=zeros
export ADDRESS_WRITE_TOPK=4
export ADDRESS_READ_TOPK=8
export ADDRESS_ALLOCATION=1
export MEMORY_CARRY_DROPOUT=0.5
export MEMORY_PRESSURE_TOKENS=64
export MEMORY_PRESSURE_WEIGHT=4.0
export READ_CONCEPT_NORM=true
export READ_GATE_INIT=0.1
export TARGET_TOKENS="${TARGET_TOKENS:-300000000}"
exec bash "${SCRIPT_DIR}/launch_e10.sh"
```

`scripts/launch_e17c.sh` and `launch_e17b.sh` stay runnable. Full equivalent
command is frozen in the spec. Default budget is 300M, not 1B.

### Evaluation contract additions
Reuse E17c's `encode_concept_banks`, carryless first-64 permutation, and
`run_e16b_generation_assessment.py`. Extend `concept_gate_metrics()`:

```text
concept_address/write_entropy_{g}
concept_address/occupancy_{g}          # fraction of slots with write mass > 1/C
concept_address/top1_mass_{g}
concept_address/unused_max_abs_delta_{g}
concept_address/usage_mean_{g}
```

`analysis/run_concept_analysis.py`: when `concept_address/*` keys exist, also
write RankMe of slots with batch-mean usage `> 1/C` vs the complement. Do not
replace the existing all-slot RankMe (the spec's ≥38.4 gate is still all-slot).

## 7. Tests & smoke

Extend `tests/test_backbone_concept_lm.py` (no parallel test module):

1. **Legacy construction / numerical regression:** default `per_layer_banks`
   still matches the E17c-era capture (module tree, CE, keys). E17c knobs still
   construct dedicated + gated_replace + pressure.
2. **`topk_renorm`:** softmax input, k < C ⇒ exactly C−k zeros, remaining mass
   sums to 1; k ≥ C is identity; gradient w.r.t. unselected logits is 0;
   selected logits get nonzero grad.
3. **Unused-slot invariance:** force a write mask with known support; `Δz` on
   the complement is `< 1e-6` in FP32 (and under autocast BF16 after the FP32
   erase-add cast). Padded rows are identity on `(z, usage)`.
4. **Empty-memory location+allocation:** `z=0`, `usage=0`; first write's
   argmax is **not** uniform-random across seeds solely from content (content
   scores of zeros are identical); location and/or allocation break the tie.
5. **Sequential heads do not densify:** with K_w=1, H_w=2, at most 2 rows
   change (union of supports); not all C.
6. **Read top-k:** addressed_topk output ignores a perturbation to an
   unselected slot's value; perturbing a selected slot changes the residual.
7. **Shared address book:** layer-g read `address_emb` is the same Parameter
   identity as writer g (untied: four identities; tied: one).
8. **Gradient reachability:** 3-block backward, finite nonzero grads to
   `address_emb`, `write_queries`, controller, read location/content projs,
   `concept_init`, read gates, LoRA. No grad requirement on BiXT (it must not
   exist on this path).
9. **No BiXT on addressed path:** `getattr(writer, "bixt", None) is None` or
   unused; assert `write_head.bixt` is absent when `update_mode` is addressed.
10. **Causality:** reuse E17c intra-block / cross-block / no-same-block-reread
    tests with addressed modes + pressure on/off.
11. **Usage carry:** usage after block 0 is nonzero; block 1 write sees that
    usage (allocation scores differ vs a zeroed usage clone).
12. **Checkpoint round-trip:** E17d save/load preserves config, four address
    books, loss, banks, generation. An E17c checkpoint still loads (no
    unexpected missing *legacy* keys).
13. **Ablation modes:** `real/zero/shuffle/permutation/static/one_block/frozen`
    remain finite; `encode_concept_banks` is `[B,G,C,H]`.
14. **All pre-existing E10/E16/E17/E17c tests stay green.**

Commands:

```bash
uv run pytest tests/test_backbone_concept_lm.py -q
uv run pytest tests/ -q
```

Tiny smoke (CPU in Cloud; MPS on the author's Mac): H=64, C=8, G=2, K=8,
K_w=2, K_r=3, three blocks, mixed pressure, three optimizer steps, save/load,
`run_concept_analysis.py`. Assert finite decreasing loss, unused `max|Δz|` ~ 0,
nonzero address_emb grads, per-bank address metrics present.

Before remote training:

```bash
EXPERIMENT_ID=E17d MAX_STEPS=50 REPORT_TO=none \
LOAD_BEST_MODEL_AT_END=False SAVE_STRATEGY=no EVAL_STRATEGY=no \
bash scripts/launch_e17d.sh
```

Record peak VRAM, samples/s, trainable params by component (address books +
controllers vs E17c's BiXT). Do not silently change C/K/k/pressure to fit.

## 8. Risks & tradeoffs
- **Risk — STE top-k is a dead addressor.** If all mass collapses to slot 0,
  entropy < ln 2. **Cheapest signal:** 100M `top1_mass` / write entropy (spec
  kill). Do not relax to soft-k inside this run.
- **Risk — zero init + content addressing cannot pick a blank row.** That is
  why allocation and location embeddings are in the same bet, not a follow-up.
  **Cheapest signal:** first-step occupancy > 1 slot in the 50-step smoke.
- **Risk — four sequential heads with K_w=4 write 16 rows and look dense.**
  Occupancy still has to sit in [4, 48]; entropy cap 0.7 ln C. If heads
  duplicate the same argmax, occupancy stays small (WTA) — that is also a kill.
- **Risk — sparse reads starve the residual and Gemma ignores z.** Read gate
  init 0.1 and K_r=8 are the registered floor. **Cheapest signal:** carryless
  Δperm at 100M; if < 0.05 the RAM is unused. Do not fall back to dense SDPA.
- **Risk — extra params/VRAM vs E17c BiXT.** Report the delta. Token-budget
  match, not iso-parameter. Pooling is cheaper than BiXT-over-C; top-k scores
  are still O(QC).
- **Risk — autocast dtype on erase-add.** Reuse E17c's lesson: FP32 state, cast
  before `lerp`/multiply. Test under BF16 autocast.
- **Risk — RankMe of unused zero rows is ~1 and trips the <19.2 kill.** All-slot
  RankMe on a mostly-blank tape can look collapsed even when written rows are
  diverse. **Mitigation:** report written-vs-unwritten RankMe; the spec's
  all-slot ≥38.4 is still the registered gate — if zeros dominate, occupancy
  must rise or the bet fails honestly. Do not change the RankMe definition
  post hoc to exclude zeros without a spec amendment.
- **Fallback:** none inside this run. Failure → kill, then a follow-up spec
  (soft k, non-zero init, or dense read) — not a silent rewrite of E17d.

## 9. Code sketches (`# sketch` — decisions, not demos)

```python
# sketch — config only; defaults are the E17b/E17c compatibility boundary
concept_read_mode: str = "backbone_qkv"
concept_write_mode: str = "additive"
concept_state_init: str = "gaussian"
address_write_topk: int = 0
address_read_topk: int = 0
address_allocation: bool = False
```

```python
# sketch — ConceptWriteHead addressed path (no BiXT)
if update_mode == "addressed_erase_add":
    self.bixt = None
    self.address_emb = nn.Parameter(torch.randn(C, H) * H ** -0.5)
    self.write_queries = nn.Parameter(torch.randn(H_w, H) * H ** -0.5)
    self.controller = nn.Linear(H, 3 * H + 3)  # k_loc, k_cnt, add, erase, g, β
```

```python
# sketch — per-layer loop unpack
if writer.update_mode == "addressed_erase_add":
    banks[g], usage[g] = writer(write_base, h_blk, pad, usage=usage[g])
else:
    banks[g] = writer(write_base, h_blk, pad, depth_index=...)
```

```python
# sketch — wrap time: read sees the writer's address book
layers[i] = GlobalLayerWithConceptRead(
    ...,
    address_emb=(
        self._writer_for_depth(depth).address_emb
        if config.concept_read_mode == "addressed_topk"
        else None
    ),
)
```

## 10. Implementation order for `research-implement`
1. `topk_renorm` + tests.
2. `ConceptWriteHead` `addressed_erase_add` (no BiXT) + unused-slot tests.
3. Thread `usage` through `_forward_blocks` / `_forward_per_layer_banks_block`.
4. `ConceptReadBranch` `addressed_topk` + shared `address_emb` wiring.
5. `concept_state_init=zeros`, config/args/factory/launcher/`launch_e17d.sh`.
6. `concept_gate_metrics` + analysis RankMe split.
7. Causality / checkpoint / legacy regression tests.
8. Tiny local smoke, then 50-step Polonez guard. No 300M until tests are green.
