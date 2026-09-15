# E25 bridge 512 far_copy r=16 in-place frozen mean-pool — E21 vs E18 vs dense (rung 5h)

**Date:** 2026-09-14
**Machine:** Odra 1× RTX 3090 (GPU 0). Polonez left empty. E22 Byobu sessions untouched.
**Run ID:** `e25_512_ip_r16_mean` (800) · `e25_512_ip_r16_mean_s3k` (3200) · `e25_512_ip_r16_mean_s8k` (8000)
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_bridge512_ip_r16_mean/`
**Raw log:** `/opt/cursor/artifacts/e25_bridge512_ip_r16_mean_probe.log` (8k; also `_800.log`, `_s3k.log`)
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `cd9c481` (Odra E24 worktree; E22 main checkout untouched)
**Git tag:** —
**Related:** r=1 identity PASS [`e25_bridge512_ip_id_far_copy_20260913.md`](e25_bridge512_ip_id_far_copy_20260913.md)

---

## Goal

One change vs r=1 inplace hard identity (E21 **62.64 bits**): keep the trusted inplace
scatter + `--message_identity_slots` (no learned `u`/`delta`, no raw_kv) and set
`--message_ratio 16`. Compression question: do **frozen 16-token means** carry INDEX
at seq=512? Tests lock this path as `k_norm(mean of 16)` / `mean(v)`, not last-token
copy and not a no-op. Dense S0 in each JSON. Climbing at 800 → extra-step 3200 then 8000.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense / e18 / e21 / e18_local |
| Width | H=128 · 0.595M (e21 0.603M) |
| E21 | query boundary id 10, **r=16**, remainder **off**, **`msg_override=real`**, **`msg_inplace=True`**, **`msg_rawkv=False`**, **`msg_idslots=True`** |
| Data | `--scale bridge` seq=512, gap=64, window=16, packed span 32 / **64-bit** prize |
| Placement | `--evidence_align right` (row gap 65/65/65) |
| Init | `--warm_residuals` |
| Device | CUDA bf16 (`--amp auto`), `backend=sdpa` |

```
bash scripts/e24_bapo_hunt.sh e25_512_ip_r16_mean 0 \
  --scale bridge --recipe far_copy --arch dense e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 128 --amp auto \
  --message_ratio 16 --message_slots_inplace --message_identity_slots \
  --steps 800 --k1_mult 4
# climbing: same flags --steps 3200 (s3k) then --steps 8000 (s8k)
```

Byobu sessions `E25_ip_r16_mean`, `E25_ip_r16_s3k`, `E25_ip_r16_s8k`. Log line:
`msg_boundary=10  msg_r=16  msg_remainder=False  msg_override=real  msg_inplace=True  msg_rawkv=False  msg_idslots=True`.

Dense first (underscore `--no-dense_first` unused). Hunt JSON:
`message_identity_slots: true`, `message_inplace_raw_kv: false`.

## Training Outcome

| arch | acc | step | recovered bits | flow |
|---|---|---|---|---|
| dense (8k JSON) | **100.0%** | 350 | 63.91 | 0.999 |
| e18 (8k JSON) | **100.0%** | 350 | 63.94 | 0.999 |
| **e21 @800** | 42.4% | 800 | 8.18 | 0.128 |
| **e21 @3200** | 48.0% | 3200 | 22.98 | 0.359 |
| **e21 @8000** | **84.3%** | **8000** | **47.36** | **0.740** |
| e21 best | **88.7%** | 7900 | (final metric is @8000) | — |
| e18_local (8k JSON) | 23.9% | 8000 | 0 | 0 |

800 JSON recalibrates S0: dense **63.94**, E18 **63.46**. 8k JSON: dense **63.91**, E18 **63.94**.

E21 at 800 was already above chance and still climbing (not a floor). 3200 reached
half-span (~50% / 23 bits). 8000 is a live copy channel (84.3%, best 88.7% @7900).

S1 vs 0.75× E18 in the 8k JSON: need **47.95 bits / flow 0.749**. E21 has **47.36 / 0.740**.

## Gates vs this rung

| gate | result |
|---|---|
| **S0** | **PASS.** Dense 100% / 63.91 bits (8k JSON; 63.94 at 800). |
| **S1 vs 0.75× E18** | **NEAR-PASS.** 47.36 vs 47.95 bits; flow 0.740 vs 0.749. |
| **S2** plots | **PASS.** |
| **K1** | not triggered |
| **K2** | **PASS.** `e18_local` 23.9% / 0 bits. Local cut stays on. |
| **K3** | not triggered (flow 0.128 at 800, 0.74 at 8k). |

Do **not** extra-step to 16k (same near-pass shape as tiny INDEX @8k). Do not 1024.
Do not Glyph. Do not remainder. Do not flex-vs-sdpa.

## Interpretation

**16-token frozen means carry INDEX at 512.** This is not chance and not a lost marked
span (alignment / remainder / pooling would have stayed at the 25% floor). It is the
same slow INDEX copy as tiny r=16 @8k (47.01 vs 47.29 bits). Concat exclusive slots
at this recipe were **0 bits**; inplace frozen mean on the trusted mask is a live
channel. Scatter was already fine at r=1 identity; compression by mean-pool is lossy
and slow, not dead.

Do **not** relabel E18's 64 bits as E21. E21 recovered **47.36 bits**, not 63.94.

## Decision

Keep the spec in `ahead/`. Next ONE experiment: **inplace r=16 learned pool**
(`--message_slots_inplace --message_ratio 16`, **no** `--message_identity_slots`, no
raw_kv) on this recipe. One change: unfreeze `u`/`delta`. If that PASSES faster than
mean, learning helps. If chance, learned pool wrecks r=16 the way `delta` wrecked r=1.
Not 1024. Not Glyph. Not remainder.

*Related: `master_experiment_log.md`, `docs/experiments_specs/done_success/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
