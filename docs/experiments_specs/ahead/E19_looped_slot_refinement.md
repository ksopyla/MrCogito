# E19 — Looped set refinement over load-bearing exclusive slots

- **Status:** draft 2026-09-16 (queue rank 5, **gated** — Wave B closed without S1) · not launched
- **Serves:** Vision latent reasoning — LOTUS-shaped loops over a *set*, only after
  the exclusive channel already carries content. Queue:
  [e21_improvement_queue.md](../../4_Research_Notes/e21_improvement_queue.md).
- **Implementation plan:** [E19_looped_slot_refinement_plan.md](E19_looped_slot_refinement_plan.md)
- **Owner / dates:** Krzysztof Sopyla · opened 2026-09-16 · closed —

> Genealogy reserved E19 for write-back / refinement. Literature lever 4: `R≥4`
> weight-tied self-attn over the slot set with per-slot targets; **reject** `R=1`
> extra exclusive attend (pause tokens / E25 extra hop). **Gate:** E26 or E27 MATCH
> S1, or E28 bits `fixed`@1024 S1, or E29 hop S1 — at least one load-bearing slot
> result. Arith is the **structure** exam, not semantic richness.

## Hypothesis
If exclusive slots that already pass a Wave A/B load-bearing gate are refined with
**R ≥ 4 weight-tied self-attention over the slot set** and a **per-slot** target
(prefix-block AE and/or packed `subexpr` node values), then on
**`ksopyla/cogito-probe-arith`** seq=1024 the **`subexpr`** recovered bits reach
**≥ 0.75 × dense** and **`match`** (Dyck-3) is strictly above chance **because**
LOTUS/Saunshi gains come from *tied depth over a set with step targets*, not from
one extra attend over frozen K/V — and **`eval` (root scalar) must not be the
success metric** (a 1-slot calculator; ~6.6 bits).

## Builds-on
- **Foundation:** `nn/perceiver_ar_lm.py` (`write_back_hook` already exists as a
  zero-init projection into global K/V; this spec adds **tied slot-set loops**,
  distinct from `message_extra_slot_attends`). CogitoProbe arith shards (PR 39).
  Probe/launcher shared.
- **Init / checkpoint:** warm-start from the gate checkpoint (E26/E27/E28/E29) so
  loops refine a live channel. Random-init loops-from-scratch is out (would retest
  the channel).
- **Baseline to beat:** the gate ckpt at **R=0** (single pass) on the same arith
  split; dense S0 ≥ 75% on `subexpr` @1024. E25 extra hop: hops-only, MATCH/SELECT
  still 0.
- **Materially new:** many tied loops + per-slot targets on exclusive *prefix* slots.
  Not `global_layers=2`. Not pause tokens. Not Coconut’s sequential latent chain.

## The architectural bet
```
slots z[0:C]  (exclusive compressed prefix)
  repeat r = 1..R:  z ← z + TiedSelfAttn+FFN(z)    # shared weights
  per-slot target:  PrefixAE and/or subexpr node values (not answer-only eval)
decode answers from refined z; extra_slot_attends stays 0
```
Eval mix (card): **`subexpr` ~50% primary**, **`match` ~25% stack**, **`eval` ~25%
shortcut control**. Bare `0-9 + - * ( ) [ ] { }` injected as 1 SmolLM3 token each;
do not BPE glued strings. **Out of scope:** bind/bits as this spec’s gates (optional
transfer). 32k. Unique extra global layer.

## Why this is not a safe retread
E25 already ran the one-hop and two-unique-global analogues. LOTUS `R=2→6` is the
existence proof that **set + loops + parallel step CE** is a different computation;
`R=1` on an `R=6` ckpt collapses. Pfau + BAPO Thm 8: hidden delay needs a re-readable
tape and dense supervision of the delay. Surprising if test-time `R` raises
`subexpr` flow on a rung single-pass slots cannot solve.

## Success criteria (set BEFORE running)
- **Gate (pre-flight):** at least one of E27 S1, E26 S1, E28 S1@1024, E29 S1.
- **S0:** dense ≥ 75% on seq=1024 `subexpr`.
- **S1 structure:** E21+loops `subexpr` bits ≥ **0.75 × dense** at 1024.
- **S2 stack:** `match` token-acc ≥ chance + 0.20 (Dyck-3 pointers, not arithmetic meaning).
- **S3 not a calculator:** `eval` may pass; it does **not** count toward S1. If `eval`
  ≥ 0.75 × dense **and** `subexpr` at chance → fail (shortcut).
- **S4 loops are the computation:** R=4 `subexpr` bits ≥ R=1 `subexpr` bits + 0.15 × dense
  (tied depth, not one extra attend). R=1 here is *one tied loop*, still
  `message_extra_slot_attends=0`.

## Kill criteria (set BEFORE running)
- **K0:** no load-bearing gate — do not launch.
- **K1:** dense `subexpr` < 75% — ill-posed arith rung.
- **K2:** only `eval` moves — calculator; reject arith as a reasoner (already the
  suite verdict for *semantics*; this kill is for *structure* too).
- **K3:** R=4 ≈ R=1 within 5 points on `subexpr` — extra hop, already falsified.
- **K4:** RankMe of slots collapses across loops (Ouro/Infini instability) — do not
  ship unstabilized recurrence.

## Plan
- **Data:** local `Cache/concept_probes/full_1k4k/arith` (`--families arith` on the
  same full 1k/4k build). Hub id `ksopyla/cogito-probe-arith` as name only. **No upload.**
  Primary metric `task=subexpr`; break out `match` / `eval`.
- **Compute:** Odra. **~4 GPU-h** (R=0 control is the gate ckpt; train R=4; probe R=1).
- **Launch:** `EXPERIMENT_ID=E19 PAR_MESSAGE_SLOT_LOOPS=4` on the gate checkpoint;
  `PAR_MESSAGE_EXTRA_SLOT_ATTENDS=0`. Seq=1024 first; 4k only if S1.
- **New foundation code:** reusable `message_slot_loops` (tied self-attn over slots)
  + per-slot target hook. Defaults 0; extra-hop flag untouched.

## Result
- Run id: `<run_id>`
- WandB: —
- Run report: `docs/2_Experiments_Registry/run_reports/<...>.md`
- Verdict: —
