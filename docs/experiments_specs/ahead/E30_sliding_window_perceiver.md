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
  seq=512 (bridge). Models < 10M (`--max_params 10000000`).
- **Steps:** probe `--steps 800 --k1_mult 4`; MATCH may need more examples
  than INDEX (do not early-kill a still-climbing curve at 800 if CE is
  falling).
- **Launch:**
  ```bash
  uv run python verification/bapo_capability_probe.py \
    --scale tiny --recipe far_copy recall_single \
    --arch dense e18 e18_local e21 e30 \
    --message_identity_slots --message_slots_inplace \
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
- Tiny CPU smoke (2026-09-20, Cloud, H=128, <0.62M, advertised 800 / K1×4): INDEX
  `far_copy` replica-matches E25 for dense **63.31 bits @2400** and E18 **63.05 @1400**;
  `e18_local` chance (K2). E21 identity-inplace **12.3 bits @2400** (still climbing;
  E25 concat-mean was 17.7 at this budget). E30 **10.0 bits @2400**, `n_windows=3`,
  entropy/log W **0.82**, `none` at chance (channel load-bearing). Extra INDEX budget
  to 8k (E25's extra-steps hunt): E21 **45.9 bits / 84%**; E30 **15.7 bits / 44%**
  still climbing (entropy/log W **0.79**, RankMe 3.9, `none` chance). MATCH
  `recall_single` replica-matches E25: dense **31.37**, E18 **0-bit wall**, E30
  chance / RankMe ~1 (not the MATCH claim — that needs the E21-mean wall at 512).
- Run id: *(after experiment-track on the claim length)*
- WandB: —
- Run report: —
- Verdict: —
