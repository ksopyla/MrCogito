# E12 — Per-layer dynamic KV-prefix concept read (Design B; concepts at ALL layers)

- **Status:** draft (design-only — queued behind E10/E11; no implementation plan yet)
- **Serves:** same platform bet as [E10](E10_gemma_backbone_concept_memory.md). E12 is the
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
