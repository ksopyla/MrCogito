# E25 bridge seq=288 chain_ordered --key_len 13 `--global_layers 2 --message_keep_local_swa` — dense K1 (rung 5bn)

**Date:** 2026-09-15
**Machine:** Odra 1× RTX 3090 (GPU 0). Polonez left empty. E22 Byobu sessions untouched.
**Run ID:** `e25_288_chain_k13_glob2_keepswa` (800 advertised; dense K1 budget 3200; e18 / e21 / `e18_local` skipped; 8k not run)
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_bridge288_chain_k13_glob2_keepswa/`
**Raw log:** `/opt/cursor/artifacts/e25_bridge288_chain_k13_glob2_keepswa/probe.log`
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `0ca748d` (Odra E24 worktree; E22 main checkout untouched)
**Git tag:** —
**Related:** seq=272 glob2 keep_local_swa S1 PASS [`e25_bridge272_chain_k13_glob2_keepswa_20260915.md`](e25_bridge272_chain_k13_glob2_keepswa_20260915.md) · seq=288 glob2 without keep_local_swa dense S0 PASS / E21 FAIL [`e25_bridge288_chain_k13_glob2_20260914.md`](e25_bridge288_chain_k13_glob2_20260914.md) · seq=320 glob2 K1 [`e25_bridge320_chain_k13_glob2_20260914.md`](e25_bridge320_chain_k13_glob2_20260914.md)

---

## Goal

Keep-SWA hops length map after 272 glob=2 `--message_keep_local_swa` **S1 PASS
25.55 bits**. Same identity compressor + two global Blocks + `--message_keep_local_swa`
at seq=**288** (bridge default is 512). Window 16 < gap 64. Remainder off.
`--hops 1` is illegal. Recalibrate dense S0 in this JSON even though 288
without the flag was already S0 PASS 22.20 (the flag is E21-only).

If dense misses 75% (K1), skip E18/E21. Chance at 800 → no 8k. Climbing short
of S1 → 8k only then. Do not 16k. Do not relabel E18 as E21. Code default
`--message_keep_local_swa` stays **false**.

## Configuration

| Item | Value |
|---|---|
| Family | dense first; e18 / e21 / e18_local skip on K1 |
| Width | **H=256** · **2.802M** · kv=1 · **`logit_scale=log`** · stack=2 |
| E21 (not scored) | query boundary id 10, **r=1**, remainder **off**, **`msg_inplace=True`**, **`msg_idslots=True`**, **`msg_keepswa=True`**, **`msg_extrahops=0`**, **`msg_updatekv=False`**, **`msg_anchors=none`**, **`glob_layers=2`** |
| Data | `--scale bridge --seq_len 288`, **`--key_len 13`**, hops=2, **26-bit** prize |
| Geometry | min_gap **64** > local_window **16**. Right-align gap 169/188/206 |
| Init | `--warm_residuals` |
| Device | CUDA bf16 (`--amp auto`), `backend=sdpa` · 0.04 s/step, no OOM |

```
bash scripts/e24_bapo_hunt.sh e25_288_chain_k13_glob2_keepswa 0 \
  --scale bridge --seq_len 288 --recipe chain_ordered --arch dense e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 256 --global_logit_scale log \
  --key_len 13 --amp auto --message_ratio 1 --message_slots_inplace \
  --message_identity_slots --global_layers 2 --message_keep_local_swa \
  --steps 800 --k1_mult 4
# --seq_len 288 required (bridge default is 512)
# --global_layers 2, --message_keep_local_swa required
# no --hops (recipe hops=2; --hops 1 is illegal)
# no --message_pool_remainder
# no --message_extra_slot_attends (default 0)
# no --message_update_slot_kv
# no --message_global_anchors (default none)
# dense first (underscore --no-dense_first unused)
# dense K1: do not extra-step; do not score E21; do not 16k
```

Byobu `E25_288_chain_keepswa`. Log:
`seq=288  gap=64  window=16  prize=26.00 bits  answer_len=13
row_gap[min/med/max]=169/188/206`
dense `patterns=[('full', 0)×5]  glob_layers=2` · 2.802M.

Hunt JSON: `seq_len: 288`, `min_gap: 64`, `local_window: 16`, `hidden: 256`,
`global_logit_scale: log`, **`global_layers: 2`**, `message_ratio: 1`,
`message_pool_remainder: false`, `message_slots_inplace: true`,
`message_identity_slots: true`, `message_inplace_raw_kv: false`,
**`message_keep_local_swa: true`**, `message_extra_slot_attends: 0`,
`message_update_slot_kv: false`, `message_global_anchors: none`,
`hops: 2`, `prize_bits: 26.0`, `calibrated: false`, `dense_steps_used: 3200`.
Hunt exit **2**. Params <100M. Dense S0 recalibrated at two full layers
(2.802M). Code default `--message_keep_local_swa` stays **false**.

## Training Outcome

| arch | acc | step | recovered bits | flow |
|---|---|---|---|---|
| dense | **47.0%** | 3200 | **8.80** | **0.338** |
| e18 | skipped (K1) | — | — | — |
| e21 | skipped (K1) | — | — | — |
| e18_local | skipped (K1) | — | — | — |

Dense @800: 28.5% / CE 1.375 (near chance 25% / ln(4)). Climb started after
~1800. Best **50.7% @3000**. Final **47.0% / 8.80 bits** @3200 (CE 0.917,
flow 0.338). Never crossed 75%. JSON `calibrated: false`. Hunt exit 2.

`--message_keep_local_swa` is wired E21-only (`bapo_models.py`). Same DNA +
seed 0 + 2.802M dense as the prior 288 hunt without the flag, which was
**93.1% / 22.20 bits** @3200 (crossed 75% at 2200). No `nn/` /
`verification/` / `evaluation/` / `data/` diff since that commit
(`835d15d`). This JSON is a missed dense trajectory at the edge of
solvability, not a keep_local_swa leak into dense. Do not treat the prior
288 S0 PASS as this-JSON S0.

Do **not** score S1 vs E18 or vs dense. There is no E21 number. Do not
relabel the skipped E18 arm as E21.

**8k not run** (dense K1; extra-step 8k is for E21 climbing short of S1, not
for dense S0 — dense already used `k1_mult=4`). Do not 16k.

## Gates vs this rung

| gate | result |
|---|---|
| **S0** | **FAIL / K1.** Dense 47.0% / 8.80 bits @3200 < 75%. Recalibrated at `global_layers=2`. |
| **S1 vs 0.75× live E18** | **not scored.** |
| **content vs 0.75× dense** | **not scored.** |
| **S2** plots | **PASS** (dense-only uncalibrated curves). |
| **K1** | **triggered.** Stop. Do not score E21. |
| **K2** | **not scored** (`e18_local` skipped). |
| **K3** | **not scored.** |

## Interpretation

**This-JSON packed 288 hops with SWA kept is uncalibrated.** Dense recovered
**8.80 of 26 prize bits** (not the chance floor of seq=320 glob=2 K1, which
was 26.2% / 0 bits), then plateaued ~47–51%. Protocol skip: E18/E21 not
scored, so this is **not** a keep-SWA S1 FAIL and **not** a keep-SWA S1
PASS.

Live keep-SWA S1 remains seq=272 **25.55 bits**. Keep-SWA hops wall after
this 16-token step: **(272 keepswa S1 PASS, 288 K1]**. Prior 288 without
`--message_keep_local_swa` stays dense **S0 PASS 22.20** / E21 **0 bits**
(severed SWA). Do not jump to 320 (that length was dense K1 without the
flag, chance 0 bits).

## Decision

Keep the spec in `ahead/`. **8k not run.** Do **not** 16k. Do not hops 268
without the flag. Do not glob=3 / extra hops / window 32. Do not Glyph. Do
not unfreeze `u`/`delta`. Do not restore full raw prefix KV (that is E18).
Do not reopen SELECT **(692, 696]**. Do not INDEX extra-steps. Do not MATCH
remainder/pooling A/Bs. Code default `--message_keep_local_swa` stays
**false**. Code default `--global_layers` stays 1. Next ONE (do not run):
**STOP keep-SWA hops length extra-steps at this 16-token step.** Parent may
later try 320 keepswa only if dense S0 is plausible (320 without the flag
was dense K1 26.2% / 0 bits). `--hops 1` is illegal.

*Related: `master_experiment_log.md`, `docs/experiments_specs/ahead/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
