# E25 bridge 512 far_copy raw override — E21 vs E18 vs dense (rung 5d)

**Date:** 2026-09-13
**Machine:** Odra 1× RTX 3090 (GPU 0). Polonez left empty. E22 Byobu sessions untouched.
**Run ID:** `e25_512_raw`
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_bridge512_raw/`
**Raw log:** `/opt/cursor/artifacts/e25_bridge512_raw_probe.log`
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `bab201b` (Odra E24 worktree; E22 main checkout untouched)
**Git tag:** —
**Related:** r=1 wall [`e25_bridge512_r1_far_copy_20260913.md`](e25_bridge512_r1_far_copy_20260913.md)

---

## Goal

One change vs identity r=1 (E21 **0 bits** at seq=512): `--message_override raw`. Local SWA /
n-grams still treat QUERY as a document start. The global read sees **uncompressed prefix
K/V across QUERY** (`raw_cross`), not concatenated slots. Remainder **off**. Default r=16
(compressor unused under raw). Dense S0 in this JSON.

Splits exclusive *slot routing* from QUERY-as-document-start.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense / e18 / e21 / e18_local |
| Width | H=128 · 0.595M (e21 0.603M) |
| E21 | query boundary id 10, r=16 (ignored by raw), remainder **off**, **`msg_override=raw`** |
| Data | `--scale bridge` seq=512, gap=64, window=16, packed span 32 / **64-bit** prize |
| Placement | `--evidence_align right` (row gap 65/65/65) |
| Init | `--warm_residuals` |
| Device | CUDA bf16 (`--amp auto`) |

```
bash scripts/e24_bapo_hunt.sh e25_512_raw 0 \
  --scale bridge --recipe far_copy --arch dense e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 128 --amp auto \
  --message_override raw --steps 800 --k1_mult 4
```

Byobu session `E25_raw`. Log line: `msg_boundary=10  msg_r=16  msg_remainder=False  msg_override=raw`.

## Training Outcome

| arch | acc | step | recovered bits | flow |
|---|---|---|---|---|
| dense | **100.0%** | 350 | 63.94 | 0.999 |
| e18 | **99.6%** | 300 | 62.23 | 0.972 |
| **e21 raw** | **100.0%** | **450** | **63.96** | **0.999** |
| e18_local | 25.1% | 800 | 0 | 0 |

E21 raw was still chance at step 350 (acc 0.24–0.27, CE ≈ ln 4), **77.8% at 400**, **100% at
450**. Do not relabel this 64-bit score as default E21 slots (r=16/64/1 real still **0 bits**).

S1 vs 0.75× E18: need **46.67 bits / flow 0.729**. E21 raw has **63.96 / 0.999**.

## Gates vs this rung

| gate | result |
|---|---|
| **S0** | **PASS.** Dense 100% / 63.94 bits. |
| **S1 vs 0.75× E18** | **PASS.** 63.96 ≥ 46.67 bits. |
| **S2** plots | **PASS.** |
| **K1** | not triggered |
| **K2** | **PASS.** `e18_local` 25.1% / 0 bits. Local cut still on. |
| **K3** | not triggered (climbing, then solved). |

Do **not** extra-step. Do not 1024. Do not Glyph.

## Interpretation

**Slot routing (concatenated extra KV) is the 512 killer**, not QUERY-as-document-start and
not pooling. Raw in-stream prefix K/V with `same_doc` crossing sides copies 64 bits while
local SWA stays severed. Identity concat slots (r=1 real) scored 0 bits on the same recipe,
so RoPE-at-block-end is not the r=1 story (end index = the token).

## Decision

Keep the spec in `ahead/`. Next ONE experiment: **in-place sender-prefix KV replacement**
(`message_slots_inplace`, default off) — write slot K/V into prefix token positions
(KV_LEN stays S, no extra concat) with QUERY local-doc cut still on and receivers still
blocked from *uncompressed* sender tokens. Score **real** E21 at **r=1** first on this
recipe. Not another override. Not remainder. Not 1024. Not Glyph.

*Related: `master_experiment_log.md`, `docs/experiments_specs/done_success/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
