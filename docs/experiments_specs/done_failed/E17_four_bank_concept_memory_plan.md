# E17 — Four-bank per-global-layer concept memory — Implementation Plan

- **Spec:** [E17_four_bank_concept_memory.md](E17_four_bank_concept_memory.md) · **Status:** done_failed with spec (2026-08-10)
- **Authored by:** `implementation-plan` · for → `research-implement`
- **Diagnosis backing:** [e16b_write_path_and_topology_diagnosis_20260801.md](../../2_Experiments_Registry/run_reports/e16b_write_path_and_topology_diagnosis_20260801.md)

> The HOW for E17's bet: privatize the concept memory so each of the 4 global layers reads &
> writes its own bank. One new config value on the existing `BackboneConceptLM` — no fork.

## 1. Source & fit
- **Origin:** the E16b write-path diagnosis — the shared 128-slot memory + tied writer make
  writes "altruistic" (a write at depth *d* only helps depths >*d*) and force four
  depth-specialized features through one accumulator, so the write gates get ~no gradient and
  stay dead (±0.05 at 1B tokens). Analogy: a standard transformer already gives every layer its
  own private KV cache; E16b's shared bank violates that. E17 restores it in compressed
  recurrent form at the 4 full-attention layers.
- **Synthesis verdict:** Adapt (not a paper port). Related to E13 (per-layer memory at all 26
  layers, gated on E12) but materially smaller — 4 banks at the 4 global layers, reusing E16b's
  read/write interface. BAPO framing: this is a **write-dynamics / effective-`a`** fix, not a
  nominal-capacity claim (Thm 10).
- **Architecture mapping:** touches the **concept memory topology** inside `BackboneConceptLM`
  only. Encoder (frozen Gemma), LoRA, objective (causal-LM CE), data, head — unchanged.
- **Boldness check:** the plan implements the spec's exact claim (private per-layer banks,
  tied writer). It does **not** substitute a safer shared-memory + higher-init retread.

## 2. Reuse map (read the modules first — `nn/backbone_concept_lm.py`)
| Component | Action | Where |
|---|---|---|
| `BackboneConceptConfig` | extend — add `"per_layer_banks"` to the valid `concept_io_mode` set | `nn/backbone_concept_lm.py:58-91` |
| `BackboneConceptLM.__init__` | extend — build **G** bank inits + per-layer `bank_index` on each `GlobalLayerWithConceptRead`; keep the **tied** `ConceptWriteHead` (`depth_alphas` reused as per-bank gates) | `nn/backbone_concept_lm.py:329-372` |
| `GlobalLayerWithConceptRead` | extend — add `bank_index`; read `concept_state[:, bank_index]` when state is banked `[B,G,C,H]`, else the shared `[B,C,H]` (unchanged path) | `nn/backbone_concept_lm.py:163-215` |
| `ConceptReadBranch` | reuse as-is (cross-attn into one bank `[B,C,H]` via the layer's own Q/K/V/O) | `nn/backbone_concept_lm.py:126-160` |
| `ConceptWriteHead` | reuse as-is (tied BiXT + `depth_alphas`); applied **per bank** | `nn/backbone_concept_lm.py:218-275` |
| `_forward_blocks` | extend — carry `z_banks [B,G,C,H]` across blocks; route per-layer read/write; ablation modes per-bank | `nn/backbone_concept_lm.py:570-747` |
| `_forward_shared_depth_block` | extend — a `per_layer_banks` branch: each global layer reads/writes its own bank (vs the shared accumulator) | `nn/backbone_concept_lm.py:496-568` |
| `encode_concepts` | extend — return **last bank** `[B,C,H]` as `last_hidden_state` (backward compat) + banks `[B,G,C,H]` available for per-bank probes | `nn/backbone_concept_lm.py:859-866` |
| `next_token_logits` / `generate` / `concept_mode="frozen"` | extend — `frozen` now freezes the **banks** (encode prompt → banks read-only) | `nn/backbone_concept_lm.py:868-975` |
| `concept_ablation_ce` | extend — headline ablation zeroes/shuffles **all banks** (E16b-comparable keys) + optional per-bank breakdown | `nn/backbone_concept_lm.py:977-1028` |
| `concept_gate_metrics` | extend — per-bank `read_g`/`write_g` (already keyed by layer; fits W&B) | `nn/backbone_concept_lm.py:1030-1056` |
| Trainer RankMe / ablation consumers | **no change needed** — they read `last_hidden_state` `[B,C,H]` = last bank (`concept_pretraining_trainer.py:278-338`, `analysis/run_concept_analysis.py:393`) |

**Parameter-accounting honesty (corrects the spec's "identical count"):** the *machinery* —
tied `ConceptWriteHead` (BiXT + norms) and the per-layer gate scalars — is identical to E16b.
Only the **learned bank initializations** scale: `concept_init [G,C,H]=[4,128,H]` vs E16b's
`[128,H]` → +3·C·H ≈ **+0.08% of trainable params**. Necessary (4 banks need 4 inits) and
negligible; it does not change the single-variable framing.

## 3. Forward pass (tensor shapes)
`B`=batch, `N`=tokens (≤4096), `K`=512 (block), `G`=4 (global layers = banks), `C`=128, `H`=backbone hidden.
```
input_ids (B, N) → embed_tokens → (B, N, H)
z_banks ← concept_init (G, C, H).unsqueeze(0).expand(B, -1, -1, -1)        # (B, G, C, H), carried block→block
for block b over K-token chunks (with one-block carry, as E16b):
    for layer ℓ = 0..25:
        if ℓ is sliding:  h ← Gemma sliding layer(h)                         # windowed self-attn only
        if ℓ is global g (g = bank_index ∈ {0,1,2,3}):
            z_g = z_banks[:, g]                                              # (B, C, H)
            read = ConceptReadBranch(input_layernorm(h), z_g, self_attn)     # (B, blk, H) via ℓ's Q/K/V/O
            h   = layer_out + tanh(read_gate_g) · read                        # READ bank g (selfish next block)
            z_banks[:, g] = ConceptWriteHead(z_g, h[:, -blk_len:],            # WRITE bank g (tied writer,
                                             block_pad_mask, depth_index=g)   #  gated by depth_alphas[g])
    last_hidden = norm(h); next-token CE on last-block positions (unchanged)
# encode_concepts → last_hidden_state = z_banks[:, G-1]  (B, C, H)  [backward compat]
```
Complexity per block is O((N/K)·(K + G·C)) — same class as E16b (G=4 constant); the **O(C·N)
preference holds** (no O(N²) decoder self-attn is introduced). The single-block DDP graph tie-in
becomes `loss + 0.0 · z_banks.float().sum()` (ties the shared `write_head` into the graph for
1-block batches), generalizing E16b's `0.0 · z.sum()`.

## 4. Inputs & data
- **Dataset:** `e16b_long_4k_v1` (Gemma-tokenized, 4K) — the same manifest as E16b; no change.
- **Collator:** `data/data_collators.py:DataCollatorForCausalLM` — reuse as-is (block-recurrent
  forward consumes `input_ids`/`attention_mask`/`labels` identically to E16b).
- **Preprocessing / masking:** none new; right-padding + `labels=-100` as E16b.

## 5. Loss & training objective
- **Loss:** unchanged causal-LM next-token CE via `_lm_ce_sum` / `ChunkedLMHeadCE`
  (`nn/backbone_concept_lm.py:448-456`). No new `register_loss`.
- **Objective:** raw causal-LM CE at seq 4096, identical to E16b.
- **Weighting:** none (single objective). The only graph change is the `0.0·z_banks.sum()` tie-in.

## 6. Config & launch
- **New config:** add `"per_layer_banks"` to the valid `concept_io_mode` set validated in
  `BackboneConceptLM.__init__` (`nn/backbone_concept_lm.py:289-294`). Default unchanged
  (`"global_kv"`), so old configs/checkpoints load unchanged. `concept_io_mode` is already a
  threaded field (`concept_pretraining_args.py:165`, `factories.py:232`,
  `train_concept_pretraining.py:340`).
- **Registry:** **no new entry** — `MODEL_REGISTRY["backbone_concept"]` already routes to
  `BackboneConceptLM`; the new mode is a config value on it. `run_concept_analysis.py` works
  unchanged (reads `last_hidden_state` = last bank).
- **Launcher:** `CONCEPT_IO_MODE` is already threaded by `scripts/launch_e10.sh` (E16b path).
  E17 launch (Polonez, 4 GPUs, eff batch 72 = per-device 3 × accum 6 × 4, 1B tokens, warmup 500,
  report @100M):
  ```bash
  EXPERIMENT_ID=E17 CONCEPT_IO_MODE=per_layer_banks \
  READ_CONCEPT_NORM=true READ_GATE_INIT=0.01 WRITE_GATE_INIT=0.01 \
  OPTIMIZER=muon LEARNING_RATE=0.01 MUON_ADAMW_LR=2e-4 MUON_MOMENTUM=0.95 \
  WEIGHT_DECAY=0.1 CONCEPT_MEMORY_LR= MAX_SEQ_LENGTH=4096 \
  PRETOKENIZE_MIX=e16b_long_4k_v1 TARGET_TOKENS=1000000000 WARMUP_STEPS=500 \
  AUTO_INTERVALS=1 SAVE_TOTAL_LIMIT=12 SKIP_PRETOKENIZE=1 \
  PER_DEVICE_BATCH_SIZE=3 GRADIENT_ACCUMULATION_STEPS=6 \
  bash scripts/launch_e10.sh
  ```

## 7. Tests & smoke
- **Extend `tests/test_backbone_concept_lm.py`:**
  - `test_per_layer_banks_param_count`: trainable params ≈ `shared_depth_recurrent` within
    +3·C·H (the extra bank inits); machinery (write_head, gates) identical.
  - `test_per_layer_banks_all_ablation_modes_finite`: parametrize `real/zero/shuffle/static/
    frozen` over `per_layer_banks` (mirror `test_shared_depth_preserves_all_ablation_modes`).
  - `test_per_layer_banks_write_is_per_layer`: with `read_gate` opened, `real` vs `shuffle`-per-
    bank CE diverges beyond-window (each bank is written by exactly one layer).
  - `test_per_layer_banks_checkpoint_roundtrip`: save → load → identical forward.
  - **Regression:** all existing `global_kv` / `shared_depth_recurrent` tests still pass.
- **Local MPS smoke:** build a tiny `per_layer_banks` model (two-global backbone, C=4), run 3
  train steps + `generate(concept_mode="frozen")`; assert finite loss, finite logits, 4 distinct
  write gates reported.
- **No-regression gate:** `uv run pytest tests/test_backbone_concept_lm.py -q` stays green.

## 8. Risks & tradeoffs
- **Risk — the "selfish gradient" hypothesis is wrong** (writes still don't open even per-bank).
  That *is* the experiment's falsifiable claim; **not** a reason to substitute a safer design.
  Cheapest signal: per-bank write-gate magnitudes at the 100M report checkpoint
  (`concept_gate_metrics`); the run continues to 1B regardless (per spec: no early kill).
- **Risk — per-bank state breaks the shared modes / DDP.** Mitigation: branch on
  `concept_io_mode`; reuse the non-reentrant gradient-checkpointing + `0.0·z_banks.sum()` tie-in
  that already handles E16b's shared-parameter recurrence under DDP.
- **Risk — RankMe / ablation comparability.** Mitigation: `last_hidden_state` = last bank
  `[B,C,H]` keeps the headline geometry comparable to E16b; per-bank Δshuffle/Δstatic added as
  extra metrics, not replacements.
- **Fallback (only if divergence kills fire):** reduce `per_device_batch` on Polonez; the
  architecture is unchanged.

## 9. Code sketches (`# sketch` — decisions, not demos)
```python
# sketch: config + banked state, pinned decisions
class BackboneConceptConfig(PretrainedConfig):
    # add to the valid set checked in BackboneConceptLM.__init__
    # concept_io_mode in {"global_kv", "shared_depth_recurrent", "per_layer_banks"}

# sketch: per-layer bank wiring in __init__
G = len(self.global_layer_indices)                       # 4 for Gemma-3-1b
if config.concept_io_mode == "per_layer_banks" and config.concept_num > 0:
    self.concept_init = nn.Parameter(randn(G, config.concept_num, self.hidden_size) * H**-0.5)
    self._banked = True
else:  # existing shared path
    self.concept_init = nn.Parameter(randn(config.concept_num, self.hidden_size) * H**-0.5)
    self._banked = False
for g, idx in enumerate(self.global_layer_indices):
    layers[idx] = GlobalLayerWithConceptRead(layer, self._concept_state, ...,
                                             bank_index=g, banked=self._banked)

# sketch: GlobalLayerWithConceptRead reads its own bank
z_all = concept_state                                # (B,G,C,H) if banked else (B,C,H)
z = z_all[:, self.bank_index] if self.banked else z_all   # (B,C,H)
read = self.read_branch(x_normed, z, self.layer.self_attn)

# sketch: per-bank write (tied head), inside the block loop
z_banks[:, g] = self.write_head(z_banks[:, g], h[:, -blk_len:], block_pad_mask,
                                depth_index=g)       # depth_alphas[g] gates bank g

# sketch: encode_concepts backward-compatible return
return BaseModelOutput(last_hidden_state=z_banks[:, -1])   # (B,C,H) = last bank
```
