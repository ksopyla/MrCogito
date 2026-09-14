# E25 bridge 512 chain_ordered r=1 in-place hard identity — dense K1 (rung 5n)

**Date:** 2026-09-14
**Machine:** Odra 1× RTX 3090 (GPU 0). Polonez left empty. E22 Byobu sessions untouched.
**Run ID:** `e25_512_ip_id_chain`
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_bridge512_ip_id_chain/`
**Raw log:** `/opt/cursor/artifacts/e25_bridge512_ip_id_chain_probe.log`
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `b46affe` (Odra E24 worktree; E22 main checkout untouched)
**Git tag:** —
**Related:** SELECT identity PASS [`e25_bridge512_ip_id_select_20260914.md`](e25_bridge512_ip_id_select_20260914.md) · tiny hops wall [`e25_tiny_chain_ordered_e21_steps_20260913.md`](e25_tiny_chain_ordered_e21_steps_20260913.md)

---

## Goal

One **task** change vs 512 `select_1decoy` r=1 identity (E21 **47.98 bits**): same
inplace identity r=1 recipe (H=128, no raw_kv) on packed `chain_ordered`. Does
exclusive uncompressed KV compose **hops**, or only INDEX/MATCH/SELECT?

Tiny chain was a hops wall (E21 ~10 bits / 49% @8k). Recalibrate dense S0. If
dense misses 75% (K1), do **not** score E21.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense first; e18 / e21 / e18_local skipped on K1 |
| Width | **H=128** · 0.595M · kv=1 · **`logit_scale=none`** |
| E21 (requested, not scored) | r=1 inplace identity, no raw_kv |
| Data | `--scale bridge` seq=512, gap=64, window=16, packed answer 24 / **48-bit** prize, hops=2, key_len=24 |
| Placement | `--evidence_align right` (row gap 281/324/381) |
| Init | `--warm_residuals` |
| Device | CUDA bf16 (`--amp auto`), `backend=sdpa` · ~0.9GB, no OOM |

```
bash scripts/e24_bapo_hunt.sh e25_512_ip_id_chain 0 \
  --scale bridge --recipe chain_ordered --arch dense e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 128 --amp auto \
  --message_ratio 1 --message_slots_inplace --message_identity_slots \
  --steps 800 --k1_mult 4
```

Byobu `E25_512_id_chain`. Dense first. Hunt JSON: `hidden: 128`,
`global_logit_scale: none`, `message_identity_slots: true`.

## Training Outcome

| arch | acc | step | recovered bits | flow |
|---|---|---|---|---|
| dense | **24.2%** | 3200 | **0.00** | **0** |
| e18 | skipped (K1) | — | — | — |
| e21 | skipped (K1) | — | — | — |
| e18_local | skipped (K1) | — | — | — |

Dense sat at chance every eval through the full K1 budget (800×4=3200). CE stuck
at ln(4) ≈ 1.386. JSON `calibrated: false`. Hunt exit 2.

Do **not** score S1 vs E18 or vs dense. There is no E21 number. Do not relabel
anything as E21.

## Gates vs this rung

| gate | result |
|---|---|
| **S0** | **FAIL / K1.** Dense 24.2% / 0 bits < 75%. |
| **S1 vs 0.75× E18** | **not scored.** |
| **content vs 0.75× dense** | **not scored** (dense itself 0 bits). |
| **S2** plots | **PASS** (dense-only uncalibrated curves). |
| **K1** | **triggered.** Stop. Do not score E21. |
| **K2** | **not scored** (`e18_local` skipped). |
| **K3** | **not scored.** |

Do **not** extra-step E21 (never trained). Do not extra-step dense to 8k: 3200
steps at chance is not a ~50% hops plateau. Do not 4k. Do not Glyph. Do not
unfreeze. Do not 1024.

## Interpretation

**Packed 512 `chain_ordered` is not dense-solvable at H=128.** This is an
instrument/S0 failure, not a hops verdict for exclusive identity. Tiny chain at
seq=128 was dense-solvable (93%) with an E21 hops wall (~49% / 10 bits). Cannot
ask whether r=1 identity composes hops until a dense control hits 75% on this
scale.

## Decision

Keep the spec in `ahead/`. Next ONE experiment: seq=512 packed **`chain_ordered`**
**dense S0 hunt** — H=256 `--global_logit_scale log` (the width that kept E18
alive at 1024 INDEX), still dense-first. Skip E21 until S0 PASSes. Not 4k. Not
Glyph. Not learned pool.

*Related: `master_experiment_log.md`, `docs/experiments_specs/ahead/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
