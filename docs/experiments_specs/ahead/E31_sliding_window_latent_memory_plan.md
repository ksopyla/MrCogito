# E31 — Implementation Plan

- **Spec:** [E31_sliding_window_latent_memory.md](E31_sliding_window_latent_memory.md) · **Status:** draft
- **Authored by:** `implementation-plan` · for → `research-implement`
- **Branch:** `e31-latent-memory` (worktree; off `dev` at `9cb3719`)
- **Alignment:** [e31_architecture.html](../../3_Evaluations_and_Baselines/e31_architecture.html) ·
  [e31_wiring.html](../../3_Evaluations_and_Baselines/e31_wiring.html)

> The HOW for the bet: per-window **addressed latent vectors** written from **two-way
> context**, read by the unchanged exclusive global read. Open design decisions from the
> spec are implemented as **config flags** with the recommended defaults, so the
> alignment checklist can flip them without code changes.

## 1. Source & fit
- **Origin:** the E30 idea note (addressed latent arrays per window), the E30 review
  (built vs intended; salience wall), the design-space note (competition, per-head picks,
  reader width), the author's pre-encoding variants (two-way page encoder vs BiXT/Slot
  Attention).
- **Architecture mapping:** encoder (a new **writer branch** reading the token
  embeddings) → memory (C addressed latents) → the existing main path (decoder) reads
  them in its one global layer. Loss unchanged (packed answer CE).
- **Boldness check:** implements real latents (own state, residual + FFN, width 4×e),
  per-latent addresses, two-way context, per-head picks, competition. Not a reuse of the
  E30 pooling with more context.

## 2. Reuse map
| Component | Action | Where |
|---|---|---|
| `PerceiverARLM`, `Block`, `Attention`, `attend_message`, message masks | reuse; small hooks | `nn/perceiver_ar_lm.py` |
| `MessageCtx` | extend: `slots` (precomputed K/V), `window_valid` (per-window pool mask) | `nn/perceiver_ar_lm.py` |
| `_swp_slot_tensors` / `SlidingWindowPerceiverCompressor` | **fix leak**: per-window earliest-side + same-doc pick | `nn/perceiver_ar_lm.py` |
| `_swp_window_starts`, `rope_cos_sin`, `apply_rope`, `SwiGLU` | reuse | `nn/perceiver_ar_lm.py` |
| `LatentMemoryWriter`, `lm_geometry`, `window_pick` | **new**, config-selectable (`message_write="latent_memory"`) | `nn/latent_memory.py` |
| `ArchSpec`, `build_model`, `ARCHES` | extend: `e31_page`, `e31_bixt`, `e30_ctx`; platform knobs | `evaluation/bapo_models.py` |
| BAPO probe | extend: e31 flags, write diagnostics, rank/entropy/head diversity | `verification/bapo_capability_probe.py` |
| capability suite | run E31 through it (cells L2–L5), add `ARCH_FLAGS` for e31 | `evaluation/capability_suite.py` (lands with the concurrent suite work) |

## 3. Forward pass (shapes, claim scale)
`B` batch · `S` = 1024 · `e` = 128 token emb · `H` = 960 · `g` = 1 KV head · `dh` = 64 ·
`W` = 256, stride 192 → `n_w` = 5 windows · `K` = 32 latents (+1 null if on) ·
`D` = 512 latent width · `h_l` = 8 latent heads × 64 · `d_w` = 256 writer token width ·
`m` = 5 reader entries per latent → `C_r = n_w·K·m` = 800 reader slots.
```
ids [B,S] ── embed.tok ──► t [B,S,e]                      (writer input: raw token embeddings)
          └─ embed (tok + n-grams + up-proj) ──► x0 [B,S,H]   (main path, unchanged)

WRITER (nn/latent_memory.py)
  window_pick(side, doc, key_valid)         → tok_idx [n_w,W], pick [B,n_w,W]    (own earliest side, same doc)
  xw = in_proj(t[:, tok_idx])               → [B·n_w, W, d_w]
  A  page_bidir: 2 × (bidirectional MHA (4×64, RoPE in-window, key mask = pick) + SwiGLU)
  z0 = q[K'] + addr(window start)           → [B·n_w, K', D]
  round r (A: 2 rounds; B: 3 rounds):
     logits = (W_q z)(W_k xw)ᵀ/√64 + prior_k(t)       [B·n_w, 8, K', W]   (prior init: soft sub-page)
     competition: softmax over K' per (head, token), mask ¬pick, renormalise over W (+ε)
        else:     softmax over W
     z ← z + W_o(w · W_v xw);  z ← z + FFN(z)
     B only: xw ← xw + CA(xw → z);  xw ← xw + FFN(xw)          (tokens read latents back)
  drop null latent;  p̄_k = start + Σ_t w̄_k(t)·t  (mean over heads, detached)
  k̄ = RoPE(norm(W_K z) @ p̄), v̄ = W_V z                → [B, n_w·K·m, g, dh]
  slot_doc/side = the window's picked (doc, side), repeated K·m

MAIN PATH (unchanged): pre SWA → GLOBAL READ (raw ‖ slots, E21 exclusive mask) → SWA ×2 → CE
```
- The global layer takes `ctx.slots` as-is (already normalised + RoPE'd); `swapped` /
  `none` / `raw` overrides keep working (roll / mask).
- DDP / no-QUERY batches: `0 · writer.participation()` added to the residual.

## 4. Inputs & data
- Rungs 1–2: on-the-fly DNA / Glyph rows (`data/bapo_ladder.py`, `data/glyph_tasks.py`), no change.
- Rung 3 (after S1): extend `data/delayed_recall.py` with a **prose filler** option
  (paragraphs from an existing HF text set) and natural fact templates → a `text` family
  behind `generate_row_for`; LM smoke via `training/train_concept_pretraining.py`
  (`model_family=perceiver_ar`) with memory visibility by **closed windows** (new mask
  mode `lm_visibility="closed_windows"`). Planned, not in the first build.

## 5. Loss & objective
Packed next-token CE on the answer span (probe), unchanged. No auxiliary loss in the
first build. Teacher-KL on picks (DeepSeek-V3.2 indexer recipe) is a follow-up if the
write learns slowly.

## 6. Config & launch
`PerceiverARConfig` (inert unless `message_write="latent_memory"`; E18/E21/E30
checkpoints unchanged):
```
message_write      = "latent_memory"
lm_context         = "page_bidir" | "bixt"
lm_window = 256 · lm_stride = 192 · lm_latents = 32          (fixed geometry; S < W → one window)
lm_latent_dim = 512 · lm_heads = 8 · lm_writer_dim = 256
lm_enc_layers = 2   (A) · lm_rounds = 2 (A) / 3 (B)
lm_competition = True · lm_null_latent = False              (open decisions 1–2)
lm_reader_tokens = 5                                          (open decision 4: 320 dims/latent at g=1)
lm_pos_prior = True
```
**Decision 4, as built:** the spec proposed 5 KV heads. The capability suite and every
recorded reference run the platform at **1 KV head**; changing `g` changes every arm. So
the writer projects each latent to **m = 5 reader entries** of `g × 64` (320 dims per
latent) and the platform stays comparable. Cost: the reader sees 5× more slots (800 at
1024). If that dilutes the read, `lm_reader_tokens=1` + `--kv_heads 5` for all arms is
the alternative.

Arch names (BAPO factory): `e31_page` (A), `e31_bixt` (B), `e30_ctx` (E30 + 64-token
causal pre-encoder reach, the context-only control). Platform knobs applied to **every**
arm in an E31 job: `--token_embedding_dim 128 --ngram_orders none`.

Launch (after the suite lands on `dev`):
```bash
uv run python scripts/run_capability_suite.py --arch e31_page e31_bixt e30 e30_ctx \
  --sizes 30m --tier standard --mode scripts --gpus 0 1 2 --host odra \
  --out Cache/capability/e31_standard_30m
```
Until then, the probe directly (one cell):
```bash
uv run python verification/bapo_capability_probe.py --scale bridge_1k \
  --recipe recall_single select_1decoy chain_ordered \
  --arch dense e30 e30_ctx e31_page e31_bixt \
  --hidden 960 --head_dim 64 --stack_layers 2 --max_params 45000000 \
  --token_embedding_dim 128 --ngram_orders none \
  --lr 1e-4 --warm_residuals --steps 1200 --k1_mult 4 --eval_rows 256 --amp auto \
  --out Cache/e31_wave1
```
Coarse rung: add `--lm_window 512 --lm_stride 384 --lm_latents 8` (and the other rows of
the spec's geometry table).

## 7. Tests & smoke
`tests/test_latent_memory.py`:
- default config byte-identical (no writer built; `block_mean` path unchanged);
- writer shapes `[B, n_w·K·m, g, dh]`; slot_doc/side per window; null latent dropped;
- competition: per (head, token) the latent shares sum to 1 over latents (incl. null);
- per-head distributions are **not** averaged (distinct heads can peak on distinct tokens);
- expected position lies inside its window;
- **causality**: perturbing any receiver token never moves logits at earlier positions —
  for `sw_perceiver` (the leak fix), `e31_page`, `e31_bixt`, incl. receiver-only windows
  and two QUERYs in one row;
- `message_override=none` changes answer logits; `swapped` runs;
- loss finite, backward reaches writer params; participation without QUERY;
- factory: all new arches build; param report.
Smoke (CPU): `--scale tiny` with `--lm_window 64 --lm_stride 48 --lm_latents 8` (smoke only;
the claim geometry needs ≥ 1024 tokens), 300–800 steps, check loss falls and `none` →
chance.

## 8. Risks & tradeoffs
- **800 reader slots dilute the one read** (m = 5). Cheapest signal: E31 lookup worse than
  E30 at 1024 while write diagnostics look healthy → try m = 1 / g = 5.
- **Dead latents under competition** (a latent that wins no tokens renormalises noise):
  ε in the renorm, log per-latent usage; RankMe ≥ 8/window is S5.
- **Arm B instability** (iterative, tokens and latents co-adapt): spec K3; retry at 5e-5.
- **Param gap:** the writer adds ~3.5–5 M to a 31 M model (suite ±5 % rule will flag it);
  spec: report params, matched-depth control if the win is < 8 bits.
- **Two-way attention inside a window mixing sides:** prevented by the per-window pick
  mask (only the window's own earliest side, same document, is visible to the encoder).

## 9. Code sketch
```python
# sketch
class LatentMemoryWriter(nn.Module):
    def forward(self, tok_emb, side, doc, key_valid, pos, rope_theta, head_dim):
        """tok_emb [B,S,e] → dict(k [B,C_r,g,dh] (normed+RoPE), v [B,C_r,g,dh],
        slot_doc [B,C_r], slot_side [B,C_r], slot_pos [B,C_r], diag)."""
```
