# E12 — Per-layer dynamic KV-prefix concept read (Design B; concepts at ALL layers)

- **Status:** ahead / design-only — E10's global→concept interface is closed; E12 remains
  an unimplemented candidate interface and is not currently scheduled
- **Serves:** same platform bet as [E10](../done_failed/E10_gemma_backbone_concept_memory.md). E12 is the
  **read-depth variant**: concepts are projected into **every layer's** attention KV as a dynamic
  prefix (prefix-tuning made dynamic — the concept state generates per-layer K/V through small
  learned projections), the most faithful realization of the "C·N attention across all layers"
  architecture assumption.
- **Implementation plan:** *(not yet written — authored only if triggered, see below)*
- **Owner / dates:** Krzysztof Sopyła · opened 2026-07-08 · closed —

> One experiment = one changed variable **vs E10**: read depth/mechanism (4 global layers' KV →
> all-L dynamic KV prefix). Everything else inherited from the E10 protocol.
>
> **Activation trigger (decided BEFORE E10 runs):** E12 is only worth running if E10's per-layer
> attention probe shows the concept read is **depth-starved** — i.e. concept-attention mass is
> concentrated in the first global layer(s) and beyond-local CE recovery plateaus below target while
> Δshuffle stays healthy (the state has content the deep layers can't reach). If E10 hits its
> primary criterion, E12 is a Watch, not a run.

## Hypothesis
If the concept state z is read at **every** layer ℓ via
`K_prefix^ℓ = P_k^ℓ(z), V_prefix^ℓ = P_v^ℓ(z)` concatenated to that layer's token KV (with a
zero-init per-layer gate on the concept contribution), then beyond-local CE recovery will exceed
E10's — because deep layers get direct access to long-range state instead of relying on residual
propagation from 4 injection points.

## The single change (vs E10)
Concept read path: `4 global layers, concepts as shared soft-token KV` → `all 26 layers, per-layer
learned K/V projections of the same concept state` (~2·L·H_c·(d_head·n_kv) new projection params;
tens of M — small vs the 1B backbone). Write op unchanged (E10's BiXT write head).

## Deferred design note — layer-specific concept memories (2026-07-12)
The current E12 hypothesis deliberately keeps **one shared recurrent state** `z_b [B,C,H]`, read
through distinct per-layer K/V projections. It does **not** give each layer its own concept bank.

Krzysztof wants to preserve a separate follow-up variant closer to a layer-wise transformer KV cache:

- retain the four Gemma positions that originally had global attention (layers 6/12/18/24), but
  replace each with its own concept-memory interface;
- maintain four distinct recurrent states
  `z_b^(6), z_b^(12), z_b^(18), z_b^(24)`, rather than broadcasting one `z_b` to all reads;
- give each state its own K/V projection and its own write/update path from that layer's
  current-block hidden states;
- after each block, update all four states independently, then use those four updated states on the
  next block.

This is **not part of E12 as currently scoped**: it changes both read depth and the memory-state /
write topology, so it needs its own frozen successor spec if E12 is reached. Its motivation is to
let early and late depth retain different long-range information while staying much cheaper than
26 layer-specific memory banks.

## Builds-on
- **Foundation:** `nn/backbone_concept_lm.py` grows `concept_io_mode="kv_prefix"` (third mode beside
  `global_kv` / `mem_tokens`); per-layer projections + gates are the only new params.
- **Init / checkpoint:** same frozen backbone + fresh LoRA (Q,O only — token K/V untouched by
  design); matched init with the arm it is compared against.
- **Baseline to beat:** the **E10 concept arm** (and E11 if run) under the identical protocol.

## Success criteria (finalized from E10's measured numbers)
- Beyond-local CE recovery > E10's by a margin that justifies the extra params (≥ +10% of the
  Stage-0 gap G), same eval protocol (2048 + 8192, Δshuffle, RankMe guard).

## Kill criteria
- E10 gates inherited; additionally kill if the per-layer gates stay ≈0 everywhere except the
  layers E10 already covers (evidence the depth hypothesis is false).

## Plan (sketch — frozen only when activated)
- Launch: `CONCEPT_IO_MODE=kv_prefix bash scripts/launch_e10.sh`; one arm, E10 budget.

## Result
<Filled in AFTER, by experiment-track.>

## References
- Prefix-Tuning (Li & Liang, arXiv:2101.00190) — pretrained attention reads learned per-layer KV.
- Flamingo (arXiv:2204.14198) zero-init tanh gating; LLaMA-Adapter zero-init attention.
- Design discussion (2026-07-07/08 chat): Design B.
