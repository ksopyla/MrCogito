# E30 — Sliding-window Perceiver banks (exclusive compressed write)

- **Status:** active (foundation + tiny capability probe)
- **Serves:** Vision priorities 1–2 — a compressed prefix that carries *addressable* facts, with `C` scaling with `N`, after E21 mean-pool MATCH died and E26/E27 repairs missed. Live agenda pointer after this spec.
- **Implementation plan:** [E30_sliding_window_perceiver_plan.md](E30_sliding_window_perceiver_plan.md)
- **Owner / dates:** Krzysztof Sopyla · opened 2026-09-20 · closed —

> One coherent write-side bet on the E18/E21 exclusive-read platform. The decoder
> stack, QUERY cut, and DNA exam stay. The *write* changes: overlapping
> positional Perceiver banks that **cross-attend** a window, instead of one
> uniform mean per `r` tokens. Not a flavour of E21 (different encode object).
> Not TinyHashed, not extra encoder SWA, not a latent mixer, not prefix-AE.

## Hypothesis
If the exclusive prefix write becomes **overlapping Perceiver banks** — each
bank is `K` learned queries that multi-head cross-attend a window of `W`
tokens at coverage `W/K ≈ 8`, windows stride `≈ 0.75 W` so `C = K · n_windows`
grows with `N`, slots live in the global read's K/V, **no mean residual and no
slot–slot mixer** — then on a calibrated DNA MATCH rung whose dense control is
≥ 75% and whose E21 frozen-mean `r=16` recovers ~0 bits, E30 recovers
**≥ 0.75 × live E18 bits** **because** a lookup key is not a typical summary
(Tishby: keep `I(Z;Y)`, throw the rest of `I(X;Z)`), a uniform mean is the
wrong sufficient statistic, and `K` addressable filters per window plus
positional allocation raise BAPO count `a·b` inside the block without a
fixed-`C` Perceiver notebook.

## Builds-on
- **Foundation:** `nn/perceiver_ar_lm.py` (`PerceiverARLM`, exclusive
  `MessageCtx` / `attend_message`), `evaluation/bapo_models.py`,
  `verification/bapo_capability_probe.py`, `data/symbolic_tasks.py` +
  `data/bapo_ladder.py`. Shared causal-LM entrypoint unchanged. New write is a
  config-selectable compressor, not a `train_*.py` fork.
- **Init / checkpoint:** random init, models **< 10M**. Compressor queries
  warm (`init_std`); do **not** zero the write (cold-start law). E18
  checkpoints still load when `message_write=block_mean` (default).
- **Baseline to beat:** E25 DNA MATCH `recall_single` at the first length
  where **E21 frozen-mean `r=16` is dead and E18 is live** (ledger: seq=512
  E21 **0 bits**, E18 **47.94 bits**; identity `r=1` **43.08 bits**). INDEX
  copy at seq=1024 is the non-regression bar (E21 mean **53.82 bits**).
  Tiny seq=128 is a **training-flow smoke**, not the MATCH claim — one
  window cannot test sliding.
- **Materially new:** overlapping **multi-query** CA banks (`K=32` when
  `n_windows≥2` fits; auto-shrink `K` on short seq) vs E21 one-mean-per-block,
  vs E22 one learned query + mean + latent transformer, vs E26 AE, vs E27
  hybrid anchors. `C ∝ N`. Coverage 8× sits between ICAE's 4× rehearsal
  floor and Fine-KV's 16× collapse (E21's ratio).

## The architectural bet
```
tokens ─► TinyHashed embed (E18 factory, not the bet) ─► 1× SWA pre-encoder
  ─► global read, exclusive after QUERY:
       windows w_i = X[i·stride : i·stride+W], stride ≈ 0.75 W, W = 8K
       bank Z_i = {q_1..q_K}   # K learned queries, RoPE at page positions
       slot_{i,k} = Σ_t softmax_t(q_k · K_t + pos_bias) · V_t
       # no mean residual, no Z↔Z self-attn
       receiver attends earlier-side slots only (same E21 mask)
  ─► SWA stack (QUERY severs local) ─► CE on packed y
```
Canonical geometry (user notes, scaled to seq):

| seq | K | W | stride | n_windows | C | N/C |
|---|---|---|---|---|---|---|
| 128 (smoke) | 8 | 64 | 48 | 2 | 16 | 8× |
| 256 | 16 | 128 | 96 | 2 | 32 | 8× |
| 512+ (claim) | 32 | 256 | 192 | ≥3 | ≥96 | ~5–8× |

Auto-fit (`swp_auto_fit=True`) shrinks `K` by halves until `n_windows≥2` or
`K=1`. Sliding is **not claimed** unless `n_windows≥2`.

**In scope:** the write + exclusive read + <10M DNA protocol (dense, e18,
e18_local, e21 frozen-mean, e30) with plots and write diagnostics (attention
entropy vs `log W`, RankMe, `none`/`swapped`/`raw`).

**Out of scope (follow-ups after a positive MATCH signal):** TinyHashed
on/off, extra encoder SWA mixer, latent self-attn, prefix AE, hybrid
anchors, hops/SELECT walls, 32k LM, CogitoProbe.

## Idea assessment (notes × literature × ledger)

| Note | Verdict | Why |
|---|---|---|
| Do **not** average `r` tokens into a slot | **Adopt** | E25 MATCH 0 bits under frozen mean; learned `u`/`delta` wiped INDEX. Tishby 1999: a mean maximises a typical-summary statistic, not `I(Z; key)`. Fine-KV synthetic 94%→14% as 4×→16×. |
| Window latents need positions | **Adopt** | Hahn TACL 2020: softmax influence vanishes with `n` unless allocation is positional. HCA / Beacon / LCLM all pin slots to spans. Free `C=128` over the book is the old bandwidth story. |
| `C` must scale with `N` | **Adopt** | Perceiver IO critique; BAPO Thm 10 (layers at fixed bottleneck do not raise prefix bandwidth). E22's 1 slot / 16 tokens was right on count, wrong on the mean inside the block. |
| Bank `Z_i` of **32** queries, stride, 8–16× coverage | **Adopt (scaled)** | Flamingo 64 / Q-Former 32 is a local notebook, not one gist. Coverage 8× is the ICAE/Fine-KV middle (4× works, 16× dies). Overlap 25% is NSA's sliding branch: boundary tokens are not orphaned. `K=32` only when seq allows ≥2 windows (`seq ≳ 448`); shorter seq auto-shrinks `K`. "192+256 next tokens" is read as **stride then window**, not a growing window. |
| Latent dim ≥ 4× token embeddings | **Adapt, cheaply** | MATCH is **count-starved**, not width-starved (BAPO; MLA; 500xCompressor). Do not widen decoder `H` (confounds the <10M law). Scoring keys use `query_dim = max(head_dim, 4·token_emb)` and **≥4 heads**; stored slots stay in the global K/V (same carrier as E21). |
| Tiny token embeddings 128/256 | **Watch, not this bet** | DNA vocab is tiny; embedding width is almost free. Raising only E30 would confound vs E21. Factory stays `token_embedding_dim=min(32,H)` for the matched probe. |
| TinyHashed | **Reject as bet** | Author: no difference. Keep the E18 factory so the write is the only change. |
| Extra encoder SWA mixer | **Reject as bet** | Unproven here. LCLM wants *some* contextualise-before-pool; E18 already has `pre_layers=1` SWA. Do not add a second mixer. Decoder SWA stays because the exclusive cut needs it. |
| Slot–slot latent transformer | **Reject default** | LCLM: MLP adapter beat latent SA on LM loss. E22: mixer had 0.05 nats of far facts to mix. Gate on a bind exam later. |

## Why this is not a safe retread
E21 already is exclusive slots. E22 already is positional + one learned query
+ mean. E26/E27 already tried AE and hybrid keys. This bet changes the
**sufficient statistic and the tiling**: `K` CA filters per overlapping
window, `C ∝ N`, no mean, no mixer. If it works, lookup lives at ~8×
compression where means died. If it fails, 8× exclusive CA is recorded as
gist-only too — not "try wider `H`".

## Success criteria (set BEFORE running)
- **S0:** dense ≥ 75% on the same replica (task is solvable).
- **K2:** `e18_local` stays within 15 pp of chance (exam does not leak).
- **S1 (the claim):** on MATCH `recall_single` at the first calibrated
  length with `n_windows≥2` where E21 frozen-mean `r=16` recovers < 8 bits
  and E18 is live, E30 recovered bits ≥ **0.75 × live E18**.
- **S2 (the write moved):** mean attention entropy per query is
  **< 0.85 · log W** (not a smear), and `message_override=none` returns
  chance (channel is load-bearing).
- **S3 (INDEX non-regression):** tiny or bridge `far_copy` E30 ≥ **0.75 ×
  live E18** (copy must not die to buy lookup).

## Kill criteria (set BEFORE running)
- **K1:** dense < 75% after `steps × k1_mult` at that length — do not score
  E30; drop LR with length (3e-3 @128, 1e-3 @256, 3e-4 @512) before
  declaring the exam ill-posed (exclusive-slot law: late takeoff).
- **K3:** S1 miss — E30 MATCH bits < 0.75× E18 at the E21-mean wall, after
  S0/K2 pass. Record 8× CA banks as gist-only. Do not add a latent mixer or
  widen `H`.
- **K4:** S2 miss — entropy ≈ log W *and* `none` still beats chance — the
  write is a smear or a leak, not a pick.
- **K5:** S3 miss — INDEX collapses vs E21 mean. The bank destroyed the
  gist channel.

## Plan
- **Data:** on-the-fly DNA (`far_copy`, `recall_single`); Glyph/hops later.
- **Compute:** CPU Cloud for tiny + tiny_wide; Polonez/Odra if MATCH needs
  seq=512 (bridge). Claim protocol is models < 10M (`--max_params 10000000`).
  A separate width-matched 4-layer hunt at H=960 (~31M, `--max_params 40000000`)
  is allowed for longer seq / harder rungs; it does not replace K3.
- **Steps:** probe `--steps 800 --k1_mult 4`; MATCH may need more examples
  than INDEX (do not early-kill a still-climbing curve at 800 if CE is
  falling).
- **Launch:**
  ```bash
  uv run python verification/bapo_capability_probe.py \
    --scale tiny --recipe far_copy recall_single \
    --arch dense e18 e18_local e21 e30 \
    --message_identity_slots \
    --hidden 128 --max_params 10000000 --lr 3e-3 \
    --out Cache/e30_tiny
  uv run python analysis/plot_bapo_capability.py \
    --in_dir Cache/e30_tiny --out_dir Cache/e30_tiny/plots
  ```
  Tiny is smoke + protocol. Claim length: `--scale tiny_wide` (lr 1e-3) then
  `--scale bridge --recipe recall_single --evidence_align right --lr 3e-4`
  `--warm_residuals` if seq≥512.
- **New foundation code:** `SlidingWindowPerceiverCompressor` +
  `message_write=sw_perceiver` on `PerceiverARConfig`; arch `e30` in the
  BAPO factory; small-model protocol in
  `docs/engineering_specs/small_model_capability_protocol.md`.

## Result
- Tiny CPU smoke (2026-09-20, Cloud, H=128, <0.62M): identity+inplace e21 mixed concat
  e30 — unfair. INDEX E21 12.3→45.9 bits @8k; E30 10.0→15.7. MATCH E18 0-bit wall.
- Fair concat (2026-09-21): e21 frozen-mean **16.55 INDEX / 3.03 MATCH**; e30 **12.19 /
  0**. LR 3e-3 is right at H=128 (1e-3 chance @1600; 1e-2 faster gist).
- tiny_wide seq=256 lr 1e-3: e30 INDEX **7.75 bits > e18 6.26 > e21 4.47**; MATCH dense K1.
- 5.16M H=384 4L: 3e-3 RankMe-collapses e30 (0 bits). 1e-3 restores INDEX 12.0 / MATCH 3.8.
  E18 MATCH **25.35 bits** — 0-bit wall was capacity. Param-matched: dense 5.108M, e30 5.163M.
- 9.04M H=512 4L lr 1e-3: e30 INDEX **36.96 bits / 73%** beats e21 13.64; MATCH **12.30 vs
  e21 4.84**, still < 0.75× 5M e18 19.0. S1 at seq=128 miss. Claim length remains seq=512.
- Report: [e30_tiny_5m_9m_capability_20260921.md](../../2_Experiments_Registry/run_reports/e30_tiny_5m_9m_capability_20260921.md)
- Run id: `e30_tiny_fair_concat` / `e30_5m_tiny` / `e30_9m_tiny_index` / `e30_30m_gpu`
- WandB: —
- 31M GPU (2026-09-22, Odra+Polonez, H=960 4L): seq=512 MATCH e30 **47.6 bits** ≈ e18 48.0 > e21 44.0. E21-mean wall from E25 is **gone** at this width (warm + 3e-4). SELECT@128: e30 27.8 vs e21 8.8 vs e18 31.3. Report: [e30_30m_gpu_odra_polonez_20260922.md](../../2_Experiments_Registry/run_reports/e30_30m_gpu_odra_polonez_20260922.md).
- Verdict: **open** — do not kill. At 31M E30 tracks E18 on seq=512 MATCH. Length/hardness (2026-09-22): in-order 4-hop chain @1024, e30 **40 bits / 77%** vs e21 4 bits. Lookup dies between 512 and 1024 for the full read; e30 is the only one stably above chance (26 bits) and is at zero by 4096. Report: [e30_length_hardness_limits_20260922.md](../../2_Experiments_Registry/run_reports/e30_length_hardness_limits_20260922.md).
- Coverage and breadth (2026-09-22): a 128-token window (coverage 4, 352 notes) passes the 1024 chain in 4800 steps (**76%** tokens); a 512-token window scores **0 bits** even with 64 questions. That chain is **0 bits** at 2048 tokens for both notebooks while the full read still passes. Far copy at 1024 is unsolved by the full read. ~~50M full-read chain collapse~~ was too few steps: at 5e-5 the full read passes and the average passes after a doubled budget. Report: [e30_coverage_and_breadth_20260922.md](../../2_Experiments_Registry/run_reports/e30_coverage_and_breadth_20260922.md).
