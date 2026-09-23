# E24 bridge 512/1024 BAPO ladder — first GPU E18 vs dense numbers

**Date:** 2026-09-13
**Machine:** Polonez 4× RTX 3090 and Odra 3× RTX 3090
**Run ID:** `bapo_bridge512` (`e18_512_fc`, `b512_rs_fair`, `b512_sel_e18`, `b1k_fc`, `e18_1k_fc2`) plus 4k S0 hunts
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e24_bridge512/`
**Best checkpoint:** none (on-the-fly rows; models discarded after each rung)
**Git commit:** `4786ba8` (512 INDEX scoring) / `1919795` (named `bridge` scales + hunt wrapper)
**Git tag:** —
**Related:** tiny CPU report [`e24_tiny_bapo_ladder_20260913.md`](e24_tiny_bapo_ladder_20260913.md)

---

## Goal

Scale the DNA BAPO ladder past seq=128 and score E18 against a matched dense control, with
the 75% solvability gate, on far copy, keyed recall, select, and ordered chains. Advertised
`--scale medium` (4k, **spread** placement) is K1: dense never leaves ln(4). The first honest
GPU INDEX recipe is **right-align** (`row.gap == min_gap+1`) with `local_window < min_gap`.

## Configuration

| Item | 512 INDEX / recall | 512 select | 1024 INDEX |
|---|---|---|---|
| seq / gap / window | 512 / 64 / 16 | same | 1024 / 64 / 16 |
| `evidence_align` | `right` | `right` | `right` |
| Width | H=128 · 0.595M · 1 KV | H=256 · 2.73M · 8 KV | H=256 · 2.73M · 8 KV |
| AMP | CUDA bf16 (`--amp auto`) | same | same |
| Prize | copy 64 bits; recall/select 32 bits | 32 bits | 64 bits |

`e18_local` window 16 cannot see gap=65 evidence (K2). Packed answers stay 32 (copy) / 16 (recall, select).

## Training Outcome

| rung | dense | E18 | e18_local | calibrated? |
|---|---|---|---|---|
| `far_copy` seq=512, H=128 | **99.7%** @1950 · 63.5 bits · flow 0.992 | **100%** @1300 · **63.9 bits** · flow 0.999 | 26.7% · 0 bits | **yes** |
| `recall_single` seq=512, H=128, right-align | **100%** @900 · 31.7 bits | **24.2%** @2500 · **0 bits** | 24.2% · 0 bits | **yes** |
| `select_1decoy` seq=512, H=256 | **99.9%** @500 · 31.9 bits | **99.6%** @1500 · 31.8 bits | 29.2% | **yes** (H=128 was K1 @29%) |
| `far_copy` seq=1024, H=256 | **99.9%** @1200 · 63.6 bits | **26.7%** @4000 · **0 bits** | 26.7% · 0 bits | dense yes; **E18 fails S1** |
| `chain_ordered` seq=512, H=128 and H=256 | 25% @4000 | skipped | — | **no** (K1) |
| `far_copy` seq=2048, H=256 | 23% @5000 | skipped | — | **no** |
| `far_copy` 4k right, H=256, gap=65 or 257 | 26% @2000 | skipped | — | **no** |
| `far_copy` 4k right, H=512 16M, gap=1025 | **74.0%** @2000 · 47 bits (one seed); replica still floor @3950 | skipped | — | **not closed** (one run missed 75% by 1 pt) |

1024 dense S0 (`b1k_fc`) also hit **100%** @3450 on a longer budget; ignition time varies (1200–3450).

## What this changes

Tiny packed `far_copy` is a **fixed-offset INDEX** machine (span in a narrow gap band). Spread
placement at 512+ is a different task (find `spanmark` anywhere); a 0.6–16M 4-layer dense
control does not do that. Right-align isolates INDEX.

1. **S1 generalises to 512 and dies at 1024.** E18 copies 64 bits at seq=512 (even slightly
   faster than dense) and recovers **0 bits** at seq=1024 with the same right-align recipe
   whose dense control is 99.9%. One global read is not a 1024-key INDEX.
2. **S2 is not “cannot find the mark”.** `recall_single` right-align plants the only fact at
   gap=65 every row. Dense copies 32 bits in 900 steps. E18 stays at chance for 2500 steps
   (2.8× dense’s budget). The wall is **query–key binding / keymark skip**, not variable
   position. `select_1decoy` at H=256 is a type cue (`keymark` vs `decoy`) and E18 matches
   dense there, as at tiny.
3. **4k is still an S0 hunt.** Spread 4k is dead. Right-align 4k at 2.7M is dead. One 16M
   seed reached 74% (47 of 64 bits) at 2000 steps and missed the gate; a replica is still at
   chance past 3900. Do not score E18 at 4k.

## Gates vs this GPU evidence

| gate | result |
|---|---|
| **S0** | 512 copy/recall/select (select needs H=256). 1024 copy dense-only. 512 chain, 2048 copy, advertised 4k spread: K1. |
| **S1** positional | **PASS at 512. FAIL at 1024** (dense 63.6 bits, E18 0). |
| **S2** content | **PASS on `recall_single` at 512**, even with fixed offset. Select remains a type cue. |
| **S3** composition | 512 `chain_ordered` uncalibrated (hops sit at the left of a 512 body). |
| **K1** | Honoured: no E18 numbers on 4k spread, 512 chain, 2048 copy. |
| **K2** | `e18_local` stayed near chance on every scored retrieval rung. |

## In flight (do not treat as results)

- 4k right-align H=512 16M, 8000-step replica (`r4k_g1024_long`)
- 4k right-align H=768 ~48M (`r4k_h768`)
- 1024 E18 with SSMax `log` scale and with H=512 (dilution / width hunts after the 0-bit fail)

## Concept Health

Not a language-model run. Effective `a` is recovered bits. At seq=512 copy, E18’s 128 B/tok
cache recovers 64 bits; on 512 recall and 1024 copy the same cache recovers 0.

## Immediate next action

Keep 4k as an S0 hunt until a dense seed is ≥75% **and** a replica confirms. Do not score E18
there. The load-bearing follow-up is whether 1024 INDEX is an E18 width/dilution failure
(SSMax / H=512 hunts) or a hard one-global-read limit.
