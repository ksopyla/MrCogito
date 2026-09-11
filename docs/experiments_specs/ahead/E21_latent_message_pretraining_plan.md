# E21 — Implementation Plan

- **Spec:** [E21_latent_message_pretraining.md](E21_latent_message_pretraining.md) · **Status:** approved for implementation (user: "plan carefully the e21 implementation, test it on Polonez", 2026-09-11)
- **Authored by:** `implementation-plan` · for → `research-implement`
- **Gate before the run:** the `perceiver_ar` evaluation layer (`docs/engineering_specs/long_context_reasoning_eval_layer.md`) has produced baseline rows for the E18 checkpoints that E21 warm-starts from.

> The HOW for: *a message boundary at P severs every local channel (SWA layers, n-gram hashes) so tokens
> ≥ P can see tokens < P only through the global read, and that read sees the prefix only as one
> compressed slot per r tokens. Plain CE then supervises the slots at every receiver position.*
> Two arms: **R** (r = 16, the claim) and **U** (r = 1, uncompressed ceiling). Everything else is E18b arm R.

## 1. Source & fit
- **Origin:** strategy synthesis 2026-09-11 — the ledger says a compressed channel dies whenever a raw path
  can satisfy the objective (E05, E10–E17, E18 arm C); the only de-collapsing objective (E02) closed the raw
  path. E21 closes it by construction on `q` of the rows.
- **Architecture mapping:** one new reusable module (`KVCompressor`, the E18c object), one new mask rule for
  the global read (query side ∈ {sender, receiver}; KV = raw keys ‖ slot keys), a boundary token that the
  collator inserts, and boundary-aware retrieval rows. No new layer types, no projector, weights shared.
- **Deviation from the spec's sketch (deliberate):** the spec's read formula inherits E18c's *always-on*
  compression (sender queries also read past blocks as slots). E21 does **not** need that bet and E18c has
  not run. Here compression applies **only to keys that cross the boundary**: within a side the read is the
  plain E18 read the warm-start checkpoint was trained with. Consequences: ordinary rows (P = S) are
  byte-identical to E18b-R (S5 is trivially close at step 0), arm U (r = 1) is exactly "raw K/V across the
  boundary", and the slot object is `prefix_kv()` compressed. E18c's always-on mode stays a follow-up knob
  (`global_kv_compress_ratio`) that can reuse the same `KVCompressor`.
- **Boldness check:** the bet (CE alone must route prefix information through 64 B/token slots) is kept
  whole; the deviation removes an unrelated second bet, it does not soften the gate.

## 2. Reuse map (read first)
| Component | Action | Where |
|---|---|---|
| `PerceiverARConfig` | add `message_boundary_token_id: int = -1` (−1 = off), `message_compress_ratio: int = 16`, `message_slot_rope: str = "end"`; `_validate`: ratio ≥ 1, token id < vocab | `nn/perceiver_ar_lm.py:49` |
| `make_mask_pred` / `dense_bool_mask` | **new siblings** `make_message_mask_pred` / `dense_message_mask` for the global read with `KV_LEN = S + S_slots` (raw block + slot block); existing functions untouched | `nn/perceiver_ar_lm.py:223` |
| `attend` | new branch when `slots is not None` (flex: `create_block_mask(..., Q_LEN=S, KV_LEN=S+n_slots)`; sdpa: dense mask; flash: raise) | `nn/perceiver_ar_lm.py:347` |
| `hashed_ngram_ids(ids, ..., doc_ids)` | reuse — pass the **local** doc ids (boundary = doc start) | `nn/perceiver_ar_lm.py:430` |
| `TinyHashedEmbedding.forward(ids, doc_ids)` | reuse with local doc ids | `nn/perceiver_ar_lm.py:470` |
| `Attention` | add `self.compressor = KVCompressor(cfg)` on the global layer(s) when the boundary is on; `forward` gains `message=None` kwarg (a `MessageCtx`) | `nn/perceiver_ar_lm.py:498` |
| `PerceiverARLM._run_layers` | derive `side`, `local_doc_ids`, slot layout once per forward; pass `doc_ids=local_doc_ids` to swa layers, `doc_ids=doc_ids` + `message=ctx` to full layers | `nn/perceiver_ar_lm.py:821` |
| `PerceiverARLM._positions` | reuse with the **original** doc ids (RoPE distances across the boundary must stay true; slots sit at the block-end position) | `nn/perceiver_ar_lm.py:777` |
| `sink_pos` (swa_sink) | compute from local doc ids (the receiver's anchor is P) | `nn/perceiver_ar_lm.py:839` |
| `prefix_kv` | return **slots** when the boundary is on: `(k_slots, v_slots, slot_pos)`; raw K/V when off (unchanged signature via a `compressed=True` flag) | `nn/perceiver_ar_lm.py:943` |
| `forward` / `hidden_states` | new kwargs `message_kv=None`, `position_offset=0` (receiver-only forward for the S6 round trip) | `nn/perceiver_ar_lm.py:879` |
| `reach_override` | pattern for the new `message_override(mode)` context manager | `nn/perceiver_ar_lm.py:751` |
| `DataCollatorForCausalLM` | add `message_boundary=(token_id, frac, min_len)`: replaces the token at P with the boundary id on `frac` of the documents (per document in packed rows), label at P → −100; skips rows already containing the id | `data/data_collators.py:186` |
| `DataArguments` | `message_boundary_token_id: int = -1`, `message_boundary_frac: float = 0.0`, `message_boundary_min: int = 4096` | `training/concept_pretraining_args.py:370` |
| `ModelArguments` | `par_message_boundary_token_id`, `par_message_compress_ratio` | `training/concept_pretraining_args.py:249` |
| `_build_perceiver_ar_model` | pass the two config fields; extend `_ALLOWED_FRESH` with `compressor` (warm start from E18b-R has none) | `training/concept_pretraining_factories.py:318` |
| collator construction | pass `message_boundary=` | `training/concept_pretraining_factories.py:508` |
| generic launcher | `PAR_MESSAGE_BOUNDARY_TOKEN`, `PAR_MESSAGE_COMPRESS_RATIO`, `MESSAGE_BOUNDARY_FRAC`, `MESSAGE_BOUNDARY_MIN` → flags | `scripts/train_concept_pretraining_multigpu.sh:77` |
| `scripts/launch_e18.sh` | pass-through, no default change | `scripts/launch_e18.sh:32` |
| `scripts/build_retrieval_mix_dataset.py::build_row` | `--boundary_id`: sources before the boundary, targets after; boundary written into the row | `scripts/build_retrieval_mix_dataset.py:107` |
| `evaluation/long_context_probes.py` | `--probe message` (paired CE deltas under `real/none/swapped/raw`), `--boundary_aware` for passkey/multikey (key < P, question ≥ P) | `evaluation/long_context_probes.py` |
| `tests/test_perceiver_ar_lm.py::tiny_cfg` | reuse for every new model test | `tests/` |

## 3. Forward pass (shapes)
Symbols: `B` batch, `S` row length (32768), `d` = 768, `g` = 2 kv-heads, `dh` = 128, `r` = ratio, `nb = ⌈S/r⌉`
slot blocks, `M = message_boundary_token_id`, `P_b` = index of `M` in row `b` (or none).

```
input_ids [B,S], doc_ids [B,S] (or None → zeros), labels [B,S]
is_boundary = input_ids == M                                   [B,S] bool
side        = cumsum(is_boundary within doc)                   [B,S] int  (0 sender, 1 receiver; ≥2 = later boundaries)
local_doc   = doc_ids * K + side  (K = 1 + side.max(), pad stays −1)     # boundary = doc start for local layers

embed:   x0 = TinyHashedEmbedding(input_ids, local_doc)        [B,S,d]   # n-grams never cross P
pos:     _positions(S, B, doc_ids)                              [B,S]     # NOT reset at P
swa layers:  attend(pattern=swa, doc_ids=local_doc)                       # window cannot cross P
global layer (per full layer):
   q,k,v as today                                                q [B,S,h,dh], k/v [B,S,g,dh]
   compressor (only if any side ≥ 1 in the batch, else skipped and 0·params added for DDP):
      blocks j ∈ [0,nb): tokens [j·r, (j+1)·r) ∩ [0,S)
      w = softmax_j( h·u_g ) over the block, per kv-head            u: [g,d] zero-init → uniform = mean pool
      k̄_j = k_norm( Σ w·wk(h) ) + Δk(h̄_j),  v̄_j = Σ w·v + Δv(h̄_j)   Δ: Linear(d, 2·g·dh) zero-init
      slot_pos_j = pos of the last token of block j;  RoPE(k̄_j, slot_pos_j)          k̄/v̄ [B,nb,g,dh]
      slot_valid[b,j] = block j lies entirely inside one doc AND entirely on side 0 of that doc
                        AND the doc has a boundary (a slot only exists as a message)
      slot_doc[b,j]   = that doc id (−1 if invalid)
   KV = concat(k ‖ k̄, v ‖ v̄) along the key axis                  [B, S+nb, g, dh]
   mask (query t, key κ):
      κ < S   (raw j=κ):   j ≤ t  ∧ doc(j)=doc(t) ∧ side(j)=side(t)            # no raw key crosses P
      κ ≥ S   (slot j):    side(t) ≥ 1 ∧ slot_valid[j] ∧ slot_doc[j]=doc(t) ∧ block_end(j) ≤ P(t)
   out = attention(q, KV)                                          [B,S,h,dh]
loss: CE over every token (label at P = −100, doc starts −100 as today)
```
Ordinary rows (no `M` anywhere): `side ≡ 0`, `local_doc ≡ doc_ids`, no slots → the mask reduces to
today's; the compressor is not evaluated. `message_compress_ratio = 1` makes every slot one token's
K/V (mean of one, Δ = 0 at init) → arm U.

Message object: `prefix_kv(x[0:P])` → `(k̄ [B,nb_P,g,dh], v̄, slot_pos)`; `nb_P·2·g·dh·2 B = P/16·1 KB ≈ 64 B/token`.

**Receiver-only forward (S6):** `forward(receiver_ids, message_kv=(k̄,v̄,slot_pos), position_offset=P)`
treats every token as side 1 of one document, uses the given slots as the only slot keys, RoPE with
positions `P + i`. Must reproduce the full forward's logits on `[P, S)` bit-exactly (same dtype/backend).

**Probe overrides** (`with model.message_override(mode)`): `none` → slot mask all False; `swapped` → slots
rolled by one along the batch (probe builds pairs of equal P); `raw` → raw keys may cross P (`side(j)=side(t)`
term dropped) and slots masked. `real` = no override.

## 4. Inputs & data
- **Manifest:** E18b's `e18b_lm_ret05_manifest.json` with the retrieval source rebuilt boundary-aware →
  `e21_lm95_ret05_boundary_manifest.json` (`$TOK/e21_retrieval_32k/{train,eval}`). LM shards unchanged
  (no re-pretokenization); the boundary on LM rows is drawn **in the collator** (`frac = 0.5`,
  `P ~ U[min, L − min]`, `min = 4096`, seeded per epoch by the trainer's generator), so data prep is the
  6 000 + 200 retrieval rows only (minutes on CPU).
- **Boundary token:** `128105` (`<|reserved_special_token_102|>`, next free after E18b's START/END).
- **Boundary-aware retrieval row:** `BOS filler [KEY v]… filler(≥min_gap) M filler [KEY START v END]… filler EOS`
  — every source before `M`, every target after it (the builder's chunk `n` is split around `M`).
- **Packing:** `BATCH_PACKING_MODE=length_group` (as E18b); the collator applies the boundary per document,
  so `pack` mode also works (tests cover both).

## 5. Loss & objective
Unchanged CE (chunked soft-capped / Liger). The only label change is −100 at `P` (the boundary token is
not predictable). Retrieval rows keep the START..END rule. No auxiliary loss in the pilot; the K3 fallback
(swapped-message negative term) is a follow-up, not implemented now.

## 6. Config & launch
```bash
TOK=/home/ksopyla/dev/hf_home/datasets_tok_smollm3_32k
# data prep (CPU, once):
uv run python scripts/build_retrieval_mix_dataset.py --base_manifest $TOK/e18_pilot_longdoc_v1_manifest.json \
  --fraction 0.05 --n_train 6000 --n_eval 200 --context 32768 --boundary_id 128105 \
  --out_dir $TOK/e21_retrieval_32k --out_manifest $TOK/e21_lm95_ret05_boundary_manifest.json --seed 0
# arm R
E18_STAGE=32k EXPERIMENT_ID=E21 SKIP_PRETOKENIZE=1 \
MANIFEST=$TOK/e21_lm95_ret05_boundary_manifest.json PRETOKENIZED_MANIFEST=$TOK/e21_lm95_ret05_boundary_manifest.json \
LOSS_SPAN_MARKERS=128103,128104 \
PAR_MESSAGE_BOUNDARY_TOKEN=128105 PAR_MESSAGE_COMPRESS_RATIO=16 MESSAGE_BOUNDARY_FRAC=0.5 MESSAGE_BOUNDARY_MIN=4096 \
MODEL_NAME_OR_PATH=Cache/Training/<E18b arm R>/final \
LEARNING_RATE=0.002 MUON_ADAMW_LR=4e-5 WARMUP_STEPS=100 LR_SCHEDULER_TYPE=cosine \
TARGET_TOKENS=500000000 PER_DEVICE_BATCH_SIZE=2 GRADIENT_ACCUMULATION_STEPS=4 bash scripts/launch_e18.sh
# arm U: PAR_MESSAGE_COMPRESS_RATIO=1
```
The length cache and token stats of the new manifest are precomputed in the prep job (E18b lesson).

## 7. Tests & smoke
- `tests/test_perceiver_ar_message.py` (new, tiny cfg, CPU/sdpa):
  1. boundary off (`token_id=-1`) ⇒ state dict and logits byte-identical to a model built from the same
     seed without the fields (no `compressor` params exist);
  2. boundary on but no `M` in the batch ⇒ logits identical to boundary-off (the compressor is inert);
  3. **severance:** with `M` at P, perturbing any input token < P changes no receiver logit when
     `message_override("none")`, and changes receiver logits only through the slots otherwise
     (gradient test: `d loss[≥P] / d x0[<P]` is zero under `none`, non-zero under `real`);
  4. no raw key crosses P: dense mask asserts for both directions; a slot straddling P is invalid;
  5. `r = 1` slots equal the raw RoPE'd K/V of the prefix (arm U identity at init);
  6. `swapped` uses the other row's slots (two rows with different prefixes, same P);
  7. **S6 round trip:** `prefix_kv(x[:P])` + receiver-only forward == full forward logits on `[P,S)`;
  8. packed rows with two documents each carrying its own boundary; sink anchor = P for receivers;
  9. flex vs sdpa parity on CUDA (skipped on CPU).
- `tests/test_data_collators.py`: boundary insertion frequency ≈ frac, `P ∈ [min, L−min]`, label −100 at P,
  rows already containing `M` untouched, per-document in packed rows, seeded determinism.
- `tests/test_retrieval_mix_dataset.py`: with `--boundary_id`, every source index < P < every target index,
  exactly one `M`, row length unchanged.
- `tests/test_long_context_probes.py`: `message` probe returns paired deltas with the expected keys;
  `--boundary_aware` passkey puts the key before P and the question after.
- Remote smoke (Polonez, 1 GPU, 30 steps, flex): loss finite, `train/data/...` unchanged, throughput within
  10% of E18b-R (the extra `nb = 2048` keys are 6% of the read); then 4-GPU DDP 30 steps (no unused-param
  error when a batch has no boundary rows).

## 8. Risks & tradeoffs
- **DDP unused parameters:** a micro-batch without any receiver would leave `compressor` params without
  gradient → `find_unused_parameters` error. Fix built in: when no receiver is present, add
  `0 · Σ compressor params` to the attention output (constant graph, zero cost).
- **Flex mask with KV_LEN ≠ Q_LEN:** supported by `create_block_mask`; the memo key must include
  `("message", r)` so it is not confused with the raw-read mask. Compile once per (B, S) pattern.
- **Slot RoPE extrapolation:** slot positions are block ends inside the trained span at 32k; at 64k/128k
  the same extrapolation question as E18b applies — the probe sweep answers it.
- **Boundary token in the LM stream:** the model sees `M` as input at P (receiver knows where it starts);
  predicting `M` is masked. A literal `<|reserved_special_token_102|>` in web text is negligible.
- **Throughput:** +`nb` keys on the read (6% at r = 16, +100% at r = 1 for arm U — arm U is ~10% slower
  overall); compressor cost is one softmax over r per block.
- **Memory:** slot K/V `[B, 2048, 2, 128]` bf16 ≈ 2 MB — negligible.
- **Warm start:** `compressor.*` are fresh (zero-init, tolerated in `_ALLOWED_FRESH`); everything else
  loads strictly.

## 9. Code sketches (`# sketch`)
```python
# sketch — nn/perceiver_ar_lm.py
@dataclass
class MessageCtx:            # per-forward, computed once in _run_layers
    side: Tensor             # [B,S] int  (0 sender / ≥1 receiver)
    boundary_pos: Tensor     # [B,S] int  P of the token's document (or S when none)
    slot_valid: Tensor       # [B,nb] bool
    slot_doc: Tensor         # [B,nb] int
    slot_end: Tensor         # [nb]   int  block end (exclusive)
    slot_pos: Tensor         # [B,nb] int  RoPE position of each slot
    override: str            # "real" | "none" | "swapped" | "raw"
    external: tuple | None   # (k̄, v̄, slot_pos) for the receiver-only forward

class KVCompressor(nn.Module):
    def __init__(self, cfg): self.ratio = cfg.message_compress_ratio; self.u = zeros(g, d); self.delta = Linear(d, 2*g*dh, bias=False) zero-init
    def forward(self, h, k_pre_norm, v, k_norm) -> (k̄, v̄):   # h [B,S,d], k/v [B,S,g,dh] → [B,nb,g,dh]
        pad S to nb·r; w = softmax over the block of (h @ u.T) [B,nb,r,g]; k̄ = k_norm(Σ w k) + Δk(mean h); v̄ = Σ w v + Δv(mean h)

def make_message_mask_pred(S, side, doc_ids, boundary_pos, slot_valid, slot_doc, slot_end, override):
    def pred(b, h, q, kv):
        is_raw = kv < S
        j = torch.where(is_raw, kv, kv - S)
        raw_ok = (j <= q) & same_doc(b, q, j) & ((side[b, j] == side[b, q]) | (override == "raw"))
        slot_ok = (side[b, q] >= 1) & slot_valid[b, j] & (slot_doc[b, j] == doc_ids[b, q]) & (slot_end[j] <= boundary_pos[b, q]) & (override != "none") & (override != "raw")
        return torch.where(is_raw, raw_ok, slot_ok)

class PerceiverARLM:
    @contextmanager
    def message_override(self, mode): ...
    def prefix_kv(self, input_ids, ..., compressed=True): ... → (k̄, v̄, slot_pos) when the boundary is on
    def forward(..., message_kv=None, position_offset=0): ...

# sketch — data/data_collators.py
class DataCollatorForCausalLM:
    def __init__(..., message_boundary: tuple[int, float, int] | None = None, seed: int = 0): ...
    # per document (row, or doc span in packed rows) of length L ≥ 2·min: with prob frac, P ~ U[min, L−min):
    #   input_ids[P] = M; labels[P] = -100   (skip documents that already contain M)

# sketch — scripts/build_retrieval_mix_dataset.py
def build_row(..., boundary=None):   # boundary id → chunk n split as [gap_a, M, gap_b]; assert all sources < P < all targets

# sketch — evaluation/long_context_probes.py
def probe_message(model, args, device):   # rows ≥ L from the manifest, P = L//2 (and P ∈ {L/4, 3L/4} optional)
    # per row: insert M at P; per_token CE via forward(return_per_token_loss=True) under each override;
    # report mean CE on [P+1, P+512) and [P+512, P+4096) and paired deltas real−none, swapped−none, raw−none
```
