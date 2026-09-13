# E25 — Implementation Plan

- **Spec:** [E25_e21_bapo_capability_ladder.md](E25_e21_bapo_capability_ladder.md) · **Status:** implemented
- **Authored by:** `implementation-plan` · for → `research-implement`

> The HOW for E21 on the E24 BAPO probe. First rung only: tiny packed `far_copy`. Port the
> strategy-branch E21 mechanism onto this branch's `perceiver_ar_lm.py` (keep
> `zero_init_residuals`). Do not substitute a raw global read.

## 1. Source & fit
- **Origin:** E21 latent-message spec (`cursor/strategy-sota-review-2026-09-e212`) + E24
  tiny `far_copy` ceiling (dense 99.4% / E18 63 bits).
- **Synthesis verdict:** Adapt E21's exclusive compressed read onto the DNA instrument;
  do not rerun the 32k LM. Adopt E24's 75% dense gate and `query` as the boundary token.
- **Architecture mapping:** decode-side exclusive route + concept bottleneck (`KVCompressor`
  in the global read's K/V). Loss is packed answer CE (unchanged).
- **Boldness check:** receivers cannot see raw prefix tokens. Incomplete last sender block
  is *not* pooled (faithful E21). If that kills INDEX, the next spec changes that mechanism
  — not this plan.

## 2. Reuse map (read the modules first)
| Component | Action | Where |
|---|---|---|
| DNA generator, ladder, probe loop, metrics, plots | reuse as-is | `data/symbolic_tasks.py`, `data/bapo_ladder.py`, `verification/bapo_capability_probe.py`, `evaluation/bapo_metrics.py`, `analysis/plot_bapo_capability.py` |
| `PerceiverARConfig` / `PerceiverARLM` / `Attention` | extend (E21 knobs default off) | `nn/perceiver_ar_lm.py` |
| `KVCompressor`, `MessageCtx`, `attend_message` | new — reusable, config-selectable | `nn/perceiver_ar_lm.py` (same home as strategy branch) |
| `build_model` | extend with `e21` | `evaluation/bapo_models.py` |
| `cache_profile` | `e21`: `a` = global KV / `r` | `evaluation/bapo_metrics.py` |
| Message unit tests | bring over | `tests/test_perceiver_ar_message.py` |

## 3. Forward pass (tensor shapes)
Symbols: `B`=batch, `S`=128, `r`=16, `nb=⌈S/r⌉`, `H`=128, `g`=KV heads, `dh`=32.
```
(B, S) ids
  → TinyHashedEmbedding with local_doc_ids (QUERY starts a new "doc" for n-grams)
  → SWA layers: mask uses local_doc_ids  →  no raw path across QUERY
  → global read:
       k,v raw [B,S,g,dh]
       k̄,v̄ = KVCompressor(h, k_raw, v)          [B,nb,g,dh]
       attend over KV = raw ‖ slots, mask:
         query t ≥ QUERY: raw keys of receiver side ∪ slots with slot_side < side(t)
         query t < QUERY: raw keys of sender side only
  → SWA stack (local_doc_ids) → RMSNorm → CE on packed y
```
At init `u=0`, `delta=0` → mean-pool per complete block. `ratio=1` is arm U (one slot = one token).

## 4. Inputs & data
- **Dataset:** on-the-fly `generate_row` `far_copy`, `--scale tiny`.
- **Collator:** none. Probe CE on packed answer tokens (existing).
- **Boundary:** `message_boundary_token_id = vocab.control("query")`. No new marker.

## 5. Loss & training objective
- Packed next-token CE on `y` only (probe). No auxiliary compressor loss.

## 6. Config & launch
- **New config fields** (defaults keep E18 checkpoints loadable):
  `PerceiverARConfig.message_boundary_token_id = -1`
  `PerceiverARConfig.message_compress_ratio = 16`
  `PerceiverARConfig.message_pool_remainder = False` (rung 1b; off = experiment 1)
- **Probe:** `ArchSpec.message_compress_ratio`; `--arch e21`.
- **Launch:**
  `uv run python verification/bapo_capability_probe.py --scale tiny --recipe far_copy --arch dense e18 e21 e18_local --out /opt/cursor/artifacts/e25_tiny_far_copy`

## 7. Tests & smoke
- `tests/test_perceiver_ar_message.py`: off-by-default byte-identical; local path severed;
  `none` / `raw` / `swapped` overrides; r=1 identity; `prefix_kv(as_message=True)` round-trip.
- `tests/test_bapo_ladder.py`: factory builds `e21` and loss is finite.
- Tiny probe as the first experiment (CPU).

## 8. Risks & tradeoffs
- **Risk:** QUERY not aligned to `r` drops the last incomplete sender block; a right-aligned
  span can sit in that remainder. **Cheapest signal:** E21 chance on tiny copy while E18 is
  99%. **Fallback:** next spec pools the remainder — not silently in this port.
- **Risk:** compressor params unused when a batch has no QUERY. **Mitigation:** strategy-branch
  dummy `0 * (u+delta)` term (already in Attention.forward).

## 9. Code sketches (optional, `# sketch` — decisions, not demos)
```python
# sketch
# e21 factory: same width/depth as e18, plus
#   message_boundary_token_id=vocab.control("query")
#   message_compress_ratio=16
# dense / e18 leave message_boundary_token_id=-1
```
