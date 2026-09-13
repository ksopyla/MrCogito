# E25 bridge 512 far_copy r=1 in-place — E21 vs E18 vs dense (rung 5e)

**Date:** 2026-09-13
**Machine:** Odra 1× RTX 3090 (GPU 0). Polonez left empty. E22 Byobu sessions untouched.
**Run ID:** `e25_512_ip_r1`
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_bridge512_ip_r1/`
**Raw log:** `/opt/cursor/artifacts/e25_bridge512_ip_r1_probe.log`
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `1989d8d` (Odra E24 worktree; E22 main checkout untouched)
**Git tag:** —
**Related:** raw PASS [`e25_bridge512_raw_far_copy_20260913.md`](e25_bridge512_raw_far_copy_20260913.md) · concat r=1 wall [`e25_bridge512_r1_far_copy_20260913.md`](e25_bridge512_r1_far_copy_20260913.md)

---

## Goal

One change vs identity concat r=1 (E21 **0 bits**) and vs raw override (E21 **63.96 bits**):
`--message_slots_inplace` so slot K/V is written into **sender prefix positions**
(`KV_LEN` stays S, no extra concat). Real E21 (`msg_override=real`), **r=1** first.
QUERY local-doc cut stays on. Receivers must not see *uncompressed* sender tokens.
Dense S0 in this JSON.

Splits in-place length-S geometry from concat extra KV.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense / e18 / e21 / e18_local |
| Width | H=128 · 0.595M (e21 0.603M) |
| E21 | query boundary id 10, **r=1**, remainder **off**, **`msg_override=real`**, **`msg_inplace=True`** |
| Data | `--scale bridge` seq=512, gap=64, window=16, packed span 32 / **64-bit** prize |
| Placement | `--evidence_align right` (row gap 65/65/65) |
| Init | `--warm_residuals` |
| Device | CUDA bf16 (`--amp auto`), `backend=sdpa` (same as the raw PASS) |

```
bash scripts/e24_bapo_hunt.sh e25_512_ip_r1 0 \
  --scale bridge --recipe far_copy --arch dense e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 128 --amp auto \
  --message_ratio 1 --message_slots_inplace --steps 800 --k1_mult 4
```

Byobu session `E25_ip_r1`. Log line:
`msg_boundary=10  msg_r=1  msg_remainder=False  msg_override=real  msg_inplace=True`.

Dense first (underscore `--no-dense_first` unused). Hunt JSON: `message_slots_inplace: true`,
`message_override: real`, `attn_backend: sdpa`.

## Training Outcome

| arch | acc | step | recovered bits | flow |
|---|---|---|---|---|
| dense | **100.0%** | 350 | 63.95 | 0.999 |
| e18 | **100.0%** | 300 | 63.93 | 0.999 |
| **e21 r=1 in-place** | **25.1%** | 800 | **0.00** | **0.000** |
| e18_local | 25.1% | 800 | 0 | 0 |

Locked prior rungs (do not re-run): E21 concat r=16/64/1 **0 bits**; E21 raw **63.96 bits**
@450. This JSON recalibrates S0: dense **63.95**, E18 **63.93**.

E21 CE stuck at ln(4) for all 16 evals (acc 0.24–0.27). Best acc 26.7%. Tiny E21 at 800
steps on seq=128 was already 31.6% / CE 1.358. Length-S in-place identity slots did not
move the 512 floor. No OOM.

S1 vs 0.75× E18: need **47.95 bits / flow 0.749**. E21 has **0.00**.

## Gates vs this rung

| gate | result |
|---|---|
| **S0** | **PASS.** Dense 100% / 63.95 bits. |
| **S1 vs 0.75× E18** | **FAIL.** 0.00 vs 47.95 bits. |
| **S2** plots | **PASS.** |
| **K1** | not triggered |
| **K2** | **PASS.** `e18_local` 25.1% / 0 bits. Local cut stays on. |
| **K3** | **triggered.** flow 0.000; chance; not climbing. Extra 8k not run. |

Do **not** relabel E18's 64 bits as E21. Do **not** extra-step. Do **not** stack r=16
in-place. Do not 1024. Do not Glyph. Do not another override.

## Interpretation

Matching raw's **KV_LEN = S** is **not** enough. Concat extra KV was sufficient to kill
512 INDEX (raw PASS vs concat 0 bits), but writing claimed-identity slots into prefix
positions under the exclusive in-place mask still scores **0 bits**.

Both this JSON and the raw PASS used `backend=sdpa`, so a flex-vs-sdpa swap is a
weaker first split. The remaining fork on this recipe is **exclusive mask vs in-place
values**: either the compressor's r=1 K/V (un-RoPE / scatter / re-RoPE) is not the raw
token stream the global read can use, or `dense_inplace_mask`'s `~replace` leak term
still hides the channel that `raw_cross` opens.

## Decision

Keep the spec in `ahead/`. Do not run r=16 in-place. Next ONE experiment: **mask vs
values** on this same 512 `far_copy` recipe — write **raw token K/V** into the existing
in-place `replace` positions (skip compressor values; keep exclusive `~replace` so
receivers still cannot see uncompressed remainder). Flag default **off**. Recalibrate
dense S0 in that JSON. If that PASSES, compressor values (even r=1) are the remaining
512 killer under length-S geometry. If it stays at chance, the exclusive leak mask is
the killer vs `raw_cross`. Not flex-vs-sdpa first. Not remainder. Not 1024. Not Glyph.

*Related: `master_experiment_log.md`, `docs/experiments_specs/ahead/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
