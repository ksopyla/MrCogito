# E25 bridge 512 select_1decoy r=1 in-place hard identity — E21 vs E18 vs dense (rung 5m)

**Date:** 2026-09-14
**Machine:** Odra 1× RTX 3090 (GPU 0). Polonez left empty. E22 Byobu sessions untouched.
**Run ID:** `e25_512_ip_id_select`
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_bridge512_ip_id_select/`
**Raw log:** `/opt/cursor/artifacts/e25_bridge512_ip_id_select_probe.log`
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `a691d36` (Odra E24 worktree; E22 main checkout untouched)
**Git tag:** —
**Related:** MATCH identity PASS [`e25_bridge512_ip_id_recall_20260914.md`](e25_bridge512_ip_id_recall_20260914.md) · tiny select wall [`e25_tiny_select_1decoy_20260913.md`](e25_tiny_select_1decoy_20260913.md)

---

## Goal

One **task** change vs 512 `recall_single` r=1 identity (E21 **43.08 bits** MATCH live):
same inplace identity r=1 recipe (H=128, no raw_kv) on packed `select_1decoy`.
Does exclusive uncompressed KV do **type-cue select**, or only MATCH/INDEX?

Tiny select was a type-cue wall (E21 1.47 bits vs E18 23). E24 GPU 512 select had
E18 99.6%. Recalibrate dense S0. Score vs 0.75× E18; also vs 0.75× dense.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense / e18 / e21 / e18_local |
| Width | **H=128** · 0.595M (e21 0.603M) · kv=1 · **`logit_scale=none`** |
| E21 | query boundary id 10, **r=1**, remainder **off**, **`msg_override=real`**, **`msg_inplace=True`**, **`msg_rawkv=False`**, **`msg_idslots=True`** |
| Data | `--scale bridge` seq=512, gap=64, window=16, packed answer 24 / **48-bit** prize, 1 decoy |
| Placement | `--evidence_align right` (row gap 92/92/92) |
| Init | `--warm_residuals` |
| Device | CUDA bf16 (`--amp auto`), `backend=sdpa` · ~1GB, no OOM |

```
bash scripts/e24_bapo_hunt.sh e25_512_ip_id_select 0 \
  --scale bridge --recipe select_1decoy --arch dense e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 128 --amp auto \
  --message_ratio 1 --message_slots_inplace --message_identity_slots \
  --steps 800 --k1_mult 4
```

Byobu `E25_512_id_select`. Log:
`msg_r=1  msg_inplace=True  msg_rawkv=False  msg_idslots=True`.

Dense first. Hunt JSON: `message_ratio: 1`, `message_identity_slots: true`,
`message_inplace_raw_kv: false`.

## Training Outcome

| arch | acc | step | recovered bits | flow |
|---|---|---|---|---|
| dense | **99.9%** | 400 | 47.80 | 0.996 |
| e18 | **100.0%** | 400 | **47.92** | 0.998 |
| **e21 identity r=1** | **100.0%** | **700** | **47.98** | **1.000** |
| e18_local | 23.4% | 800 | 0 | 0 |

E18 is **live**. E21 chance through ~500, then 52.0% @550 → 73.3% @600 → **100% @700**
(early stop). Tiny select was 1.47 bits; this is the prize ceiling.

S1 vs 0.75× E18: need **35.94 bits**. E21 has **47.98**.
Content vs 0.75× dense: need **35.85 bits**. E21 has **47.98**.

Do **not** relabel E18's 47.92 bits as E21. E21 recovered **47.98** (same 48-bit prize).

## Gates vs this rung

| gate | result |
|---|---|
| **S0** | **PASS.** Dense 99.9% / 47.80 bits. |
| **S1 vs 0.75× E18** | **PASS.** 47.98 ≥ 35.94 bits. |
| **content vs 0.75× dense** | **PASS.** 47.98 ≥ 35.85 bits. |
| **S2** plots | **PASS.** |
| **K1** | not triggered |
| **K2** | **PASS.** `e18_local` 23.4% / 0 bits. |
| **K3** | not triggered (flow 1.00). |

Do **not** extra-step (early-stop 100%). Do not 4k. Do not Glyph. Do not remainder.
Do not unfreeze. Do not 1024.

## Interpretation

**Exclusive identity does type-cue select, not only MATCH/INDEX.** Same r=1 inplace
recipe that copies INDEX and MATCH copies the 48-bit `select_1decoy` prize. Tiny
select's 1.47-bit wall does not hold at 512 with uncompressed exclusive KV. r=16
pooling remains the MATCH killer; it was not re-run here.

## Decision

Keep the spec in `ahead/`. SELECT-capable recipe: inplace + `--message_identity_slots`
+ **r=1**. Next ONE experiment: seq=512 packed **`chain_ordered`**, inplace r=1
identity (hops vs type-cue). Not 4k. Not Glyph. Not learned pool. Not 1024.

*Related: `master_experiment_log.md`, `docs/experiments_specs/done_success/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
