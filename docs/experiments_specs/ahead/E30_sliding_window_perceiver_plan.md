# E30 — Implementation Plan

- **Spec:** [E30_sliding_window_perceiver.md](E30_sliding_window_perceiver.md) · **Status:** active
- **Authored by:** `implementation-plan` · for → `research-implement`

> HOW for overlapping Perceiver banks on the exclusive E18/E21 platform.
> Do not substitute mean-pool, a latent mixer, TinyHashed, or extra SWA.

## 1. Source & fit
- **Origin:** [sliding_window_perceiver.md](../../experiment_ideas/sliding_window_perceiver.md);
  E21 MATCH death ([e25 report](../../2_Experiments_Registry/run_reports/e25_e21_dna_capability_report_20260916.md));
  [perceiver IO critique](../../literature_review/perceiver_io_latent_reasoning_critique.md);
  [slot vs latent](../../4_Research_Notes/slot_write_vs_latent_write_20260920.md);
  Tishby IB, ICAE, Fine-KV, HCA, LCLM, NSA sliding branch.
- **Synthesis verdict:** **Adapt** positional exclusive KV (E21 mask, HCA carrier)
  + **learned multi-query CA inside overlapping windows** (user banks). **Drop**
  uniform mean, fixed-`C` Perceiver, latent SA, TinyHashed/SWA as the bet,
  E26 AE / E27 anchors.
- **Architecture mapping:** encode write (compressor) + exclusive decode mask.
  Loss unchanged (packed answer CE).
- **Boldness check:** `K` CA filters per overlapping window, no mean residual.
  Not "one learned `u` on E21 blocks".

## 2. Reuse map (read the modules first)
| Component | Action | Where |
|---|---|---|
| `PerceiverARLM`, `Attention`, `MessageCtx`, `attend_message` | reuse exclusive mask / concat path | `nn/perceiver_ar_lm.py` |
| `KVCompressor` | keep as `message_write=block_mean` (E21 default) | `nn/perceiver_ar_lm.py` |
| `SlidingWindowPerceiverCompressor`, `swp_geometry` | **new** — config-selectable write | `nn/perceiver_ar_lm.py` |
| `PerceiverARConfig` | add `message_write`, `swp_*` (defaults keep E18/E21 loadable) | `nn/perceiver_ar_lm.py` |
| `ArchSpec` / `build_model` / `ARCHES` | add `e30` | `evaluation/bapo_models.py` |
| `cache_profile` | `e30` `a` = slot bytes `∝ C/N` | `evaluation/bapo_metrics.py` |
| BAPO probe | arch `e30`, write diagnostics, length-aware notes | `verification/bapo_capability_probe.py` |
| plots | `e30` color + optional entropy panel | `analysis/plot_bapo_capability.py` |
| DNA / ladder | reuse | `data/symbolic_tasks.py`, `data/bapo_ladder.py` |
| TinyHashed / decoder SWA | reuse as-is (not the bet) | `nn/perceiver_ar_lm.py` |

## 3. Forward pass (tensor shapes)
Symbols: `B` batch, `N` tokens, `K` bank size, `W` window, `st` stride,
`n_w` windows, `C = n_w·K`, `H` hidden, `g` KV heads, `dh` head dim,
`h_q` compressor heads, `d_q` query dim.
```
(B, N) ids
  → TinyHashedEmbedding                         (B, N, H)
  → SWA pre_layers (window = local_window)      (B, N, H)
  → global Attention, exclusive after QUERY:
       h, k_raw, v                              (B, N, H), (B, N, g, dh)
       starts = (0, st, 2st, …) + flush-right   n_w
       gather windows                           (B, n_w, W, ·)
       q_k learned                              (K, h_q, d_q / h_q)
       w = softmax_W(q · W_k(h) + pos_bias)     (B, n_w, K, W)
       k̄, v̄ = w · (k_raw, v)                   (B, C, g, dh)
       slot_pos[i,k] = start_i + (k+1)·(W/K) − 1
       attend_message: receiver raw ∪ earlier-side slots
  → SWA stack (QUERY = doc start)               (B, N, H)
  → RMSNorm → CE on packed y
```
No mean added. No `Z↔Z` mixer. Concat path only (`message_slots_inplace` is
illegal for `sw_perceiver` — `K` slots per window is not 1:1 with token
positions).

## 4. Inputs & data
- **Dataset:** on-the-fly `generate_row_for` DNA. First rung `far_copy` (INDEX
  smoke), claim rung `recall_single` (MATCH).
- **Collator:** none. Packed answer CE, existing probe.
- **Boundary:** `message_boundary_token_id = vocab.control("query")`.
- **Geometry:** `swp_auto_fit=True` so tiny seq=128 still has `n_windows≥2`
  (K shrinks 32→8). Spec table is the target, not a second architecture.

## 5. Loss & training objective
- Packed next-token CE on `y` only (`bapo_capability_probe.train_one`).
- No AE, no extra hop, no anchors.
- Channel checks: `message_override` `none` / `swapped` / `raw` (already on
  e21; run for e30 too).
- Write diagnostic: mean attention entropy over `W` vs `log W`; RankMe of
  `k̄` (existing `_slot_rankme`).

## 6. Config & launch
Backward-compatible defaults (E18/E21 checkpoints load):
```
PerceiverARConfig.message_write = "block_mean"   # or "sw_perceiver"
PerceiverARConfig.swp_bank_size = 32
PerceiverARConfig.swp_coverage = 8
PerceiverARConfig.swp_window = 0      # 0 → coverage * bank
PerceiverARConfig.swp_stride = 0      # 0 → round(0.75 * window)
PerceiverARConfig.swp_n_heads = 0     # 0 → max(4, num_attention_heads)
PerceiverARConfig.swp_query_dim = 0   # 0 → max(head_dim, 4 * token_embedding_dim)
PerceiverARConfig.swp_auto_fit = True
```
Validation: `sw_perceiver` forbids `message_slots_inplace`,
`message_identity_slots`, `message_prefix_ae`. Warm `q` / `wk` (`init_std`);
do not zero them in `post_init` (only `KVCompressor.delta` stays zeroed).

**Registry:** BAPO `ARCHES` += `e30`. Not a new `MODEL_REGISTRY` training
family — probe-only like `e21`.

**Launch:** see spec. Tiny CPU; MATCH wall may need `tiny_wide` / `bridge`.

## 7. Tests & smoke
- `tests/test_sw_perceiver.py`: geometry (seq=128 → K=8, 2 windows; seq=512 →
  K=32, W=256, stride=192); slots `[B,C,g,dh]` with `C=n_w·K`; queries not
  all equal (not a mean); exclusive `none` vs `real` logits differ when
  QUERY present; `block_mean` default byte-identical to today; inplace
  raises; param count < 10M at H=256; loss finite; participation in graph
  without QUERY.
- `tests/test_bapo_ladder.py`: factory builds `e30`, loss finite.
- Probe smoke: `--scale tiny --recipe far_copy --arch e30 --steps 20`.

## 8. Risks & tradeoffs
- **Risk:** concat extra KV dilutes softmax (E25 early concat was chance).
  **Mitigation:** exclusive mask unit test; `none` must kill the channel;
  optional `global_logit_scale=log` only if dilution shows at 512+.
- **Risk:** tiny seq never slides. **Mitigation:** auto-fit + protocol gate
  `n_windows≥2` before claiming S1.
- **Risk:** CA smears (entropy ≈ log W). **Kill K4.**
- **Risk:** late takeoff at longer seq (exclusive-slot law). **Mitigation:**
  length-aware LR in the small-model protocol; do not K1-kill a falling CE.
- **Fallback:** none inside this spec. A miss is a recorded gist-only result.

## 9. Code sketches (optional, `# sketch`)
```python
# sketch
@dataclass(frozen=True)
class SWPGeometry:
    bank_size: int
    window: int
    stride: int
    n_windows: int
    starts: tuple[int, ...]
    n_slots: int  # n_windows * bank_size

class SlidingWindowPerceiverCompressor(nn.Module):
    def forward(self, h, k_raw, v, k_norm, valid=None):
        # h [B,N,d], k_raw/v [B,N,g,dh] → k̄,v̄ [B,C,g,dh]
        ...
```
