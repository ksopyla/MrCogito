# E25 bridge seq=288 chain_ordered --key_len 13 `--global_layers 2 --message_extra_slot_attends 1` — E21 hops S1 FAIL (rung 5br)

**Date:** 2026-09-15
**Machine:** Odra 1× RTX 3090 (GPU 0). Polonez left empty. E22 Byobu sessions untouched.
**Run ID:** `e25_288_chain_k13_glob2_extrahop` (800 advertised; dense-matched 1850; e18 / e21 / e18_local scored; 8k not run)
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_bridge288_chain_k13_glob2_extrahop/`
**Raw log:** `/opt/cursor/artifacts/e25_bridge288_chain_k13_glob2_extrahop/probe.log`
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `2fdf3f3` (Odra E24 worktree; E22 main checkout untouched)
**Git tag:** —
**Related:** seq=272 glob2 extra exclusive hop S1 PASS [`e25_bridge272_chain_k13_glob2_extrahop_20260915.md`](e25_bridge272_chain_k13_glob2_extrahop_20260915.md) · seq=288 glob2 without extra hop dense S0 PASS / E21 FAIL [`e25_bridge288_chain_k13_glob2_20260914.md`](e25_bridge288_chain_k13_glob2_20260914.md) · seq=288 glob2 keep_local_swa dense K1 [`e25_bridge288_chain_k13_glob2_keepswa_20260915.md`](e25_bridge288_chain_k13_glob2_keepswa_20260915.md)

---

## Goal

Hops length extra-hop map after 272 glob=2 `--message_extra_slot_attends 1`
**S1 PASS 25.48 bits**. Same identity compressor + two global Blocks + extra
exclusive hop at seq=**288** (bridge default is 512). Window 16 < gap 64.
Remainder off. keep_local_swa **off**. update_slot_kv **off**. `--hops 1` is
illegal. Recalibrate dense S0 in this JSON even though 288 without the flag
was already S0 PASS 22.20 (the flag is E21-only).

If dense misses 75% (K1), skip E18/E21. Chance at 800 → no 8k. Climbing short
of S1 → 8k only then. Do not 16k. Do not relabel E18 as E21. Code default
`--message_extra_slot_attends` stays **0**. Code default keep_local_swa stays
**false**.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense / e18 / e21 / e18_local |
| Width | **H=256** · **2.802M** (e21 **2.835M**) · kv=1 · **`logit_scale=log`** · stack=2 |
| E21 | query boundary id 10, **r=1**, remainder **off**, **`msg_override=real`**, **`msg_inplace=True`**, **`msg_rawkv=False`**, **`msg_idslots=True`**, **`msg_keepswa=False`**, **`msg_extrahops=1`**, **`msg_updatekv=False`**, **`msg_anchors=none`**, **`glob_layers=2`** |
| Data | `--scale bridge --seq_len 288`, **`--key_len 13`**, hops=2, **26-bit** prize |
| Geometry | min_gap **64** > local_window **16**. Right-align gap 169/188/206 |
| Init | `--warm_residuals` |
| Device | CUDA bf16 (`--amp auto`), `backend=sdpa` · ~1.3GB, 0.04–0.05 s/step, no OOM |

```
bash scripts/e24_bapo_hunt.sh e25_288_chain_k13_glob2_extrahop 0 \
  --scale bridge --seq_len 288 --recipe chain_ordered --arch dense e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 256 --global_logit_scale log \
  --key_len 13 --amp auto --message_ratio 1 --message_slots_inplace \
  --message_identity_slots --global_layers 2 --message_extra_slot_attends 1 \
  --steps 800 --k1_mult 4
# --seq_len 288, --global_layers 2, --message_extra_slot_attends 1 required
# no --hops (recipe hops=2; --hops 1 is illegal)
# no --message_pool_remainder
# no --message_keep_local_swa (default false; SWA still severed)
# no --message_update_slot_kv
# no --message_global_anchors (default none)
# dense first (underscore --no-dense_first unused)
# chance at 800 and at dense-matched 1850: do not extra-step; do not 16k
```

Byobu `E25_288_chain_extrahop`. Log:
`seq=288  gap=64  window=16  prize=26.00 bits  answer_len=13
row_gap[min/med/max]=169/188/206`
dense `patterns=[('full', 0)×5]  glob_layers=2` · 2.802M.
e18/e21 `patterns=[('swa', 16), ('full', 0), ('full', 0), ('swa', 16), ('swa', 16)]`
e21 `msg_r=1  msg_remainder=False  msg_inplace=True  msg_rawkv=False
msg_idslots=True  msg_keepswa=False  glob_layers=2  msg_extrahops=1
msg_updatekv=False  msg_anchors=none`.

Hunt JSON: `seq_len: 288`, `min_gap: 64`, `local_window: 16`, `hidden: 256`,
`global_logit_scale: log`, **`global_layers: 2`**, `message_ratio: 1`,
`message_pool_remainder: false`, `message_slots_inplace: true`,
`message_identity_slots: true`, `message_inplace_raw_kv: false`,
`message_keep_local_swa: false`, **`message_extra_slot_attends: 1`**,
`message_update_slot_kv: false`, `message_global_anchors: none`,
`hops: 2`, `prize_bits: 26.0`, `calibrated: true`, `dense_steps_used: 1850`.
Hunt exit **0**. Params <100M. Dense S0 recalibrated at two full layers
(2.802M). Code default `--message_extra_slot_attends` stays **0**. Code
default `--message_keep_local_swa` stays **false**.

## Training Outcome

| arch | acc | step | recovered bits | flow |
|---|---|---|---|---|
| dense | **99.5%** | 1850 | **25.77** | **0.991** |
| e18 | **26.8%** | 1850 | **0.00** | **~0** |
| **e21 @800** | **26.8%** | 800 | **0.00** | **0** |
| **e21 final** | **23.8%** | 1850 | **0.00** | **0** |
| e18_local | **24.5%** | 1850 | **0.00** | **~0** |

Dense crossed 75% at 1550 (76.1%), early-stop **99.5% / 25.77 bits** @1850
(CE 0.012). JSON `calibrated: true`. Hunt exit 0. **S0 PASS.** Recalibrated
at `global_layers=2`. Prior 288 without extra hop was dense **93.1% / 22.20
bits** @3200; keepswa 288 was dense **K1 47.0%**. This JSON is a live dense
trajectory, not the keepswa miss.

E18 sat at chance every eval (CE at ln(4) ≈ 1.386; best 26.8% is noise
around 25%). Final **26.8% / 0.00 bits** @1850. Live **~0**, same as 288
without extra hop. Do **not** pass S1 via 0.75×0. Score S1 vs **0.75× this-JSON
dense 19.32 bits**. Also report vs 0.75× live E18 (0.00) — unused as a pass
bar.

E21 sat at chance through the advertised 800 floor (26.8% / CE 1.386) **and**
through the dense-matched 1850 budget (final 23.8% / **0.00 bits**; CE at
ln(4); best 26.8% @350). Not climbing.

**8k not run** (chance floor at 800; same floor at 1850). Do not 16k. Do not
relabel the live-looking E18 arm as E21 — E18 was scored and is 0 bits.

## Gates vs this rung

| gate | result |
|---|---|
| **S0** | **PASS.** Dense 99.5% / 25.77 bits @1850 ≥ 75%. Recalibrated at `global_layers=2`. |
| **S1 vs 0.75× live E18** | **not a pass.** E18 is ~0 bits; do not use 0.75×0. |
| **content vs 0.75× dense** | **FAIL.** 0.00 vs 19.32 bits. |
| **S2** plots | **PASS** (learning-curves / heatmap / recovered-bits / flow / bytes-per-token). |
| **K1** | not triggered. |
| **K2** | **PASS.** `e18_local` 24.5% / 0 bits. Extra hop did not leak a full raw prefix. |
| **K3** | **triggered.** E21 chance floor @800 and @1850. |

## Interpretation

**Packed 288 hops with one extra exclusive hop is dense-solvable at two
global Blocks, but neither E18 nor E21 composes the DFA.** The same
compressor that early-stopped E21 at **99.0% / 25.48 bits @5450** on seq=272
with `msg_extrahops=1` is chance at seq=288 for both global-read arms, while
dense recovers **25.77 of 26 prize bits**. This is not an E21-only
compression failure: uncompressed E18 is also 0 bits (same as 288 without
the extra hop). Extra exclusive hop substitutes for local SWA at 272 and
does not at 288. SWA stays severed (`msg_keepswa=False`). Window 16 still
cannot span gap 169–206.

Extra-hop hops wall after this 16-token step:
**(272 extrahop S1 PASS, 288 FAIL]**. Live extra-hop S1 remains seq=272
**25.48 bits**. Keep-SWA wall stays **(272 S1 PASS, 288 K1]**. Severed-SWA
hops wall stays **(264 S1 PASS, 272 FAIL]**. Dense hops wall at glob=2 stays
**(288 S0 PASS, 320 K1]**. Do not jump to 320 (that length was dense K1
without the flag).

## Decision

Keep the spec in `ahead/`. **8k not run.** Do **not** 16k. Do not hops 320.
Do not MATCH glob=2. Do not SELECT 694. Do not glob=3 / window 32 /
`--message_update_slot_kv` / keepswa on the same hunt. Do not Glyph. Do not
unfreeze `u`/`delta`. Do not restore full raw prefix KV (that is E18). Code
default `--message_extra_slot_attends` stays **0**. Code default
`--message_keep_local_swa` stays **false**. Code default `--global_layers`
stays 1. Next ONE (do not run): **STOP extra-hop hops length extra-steps at
this 16-token step.** Parent may later try 320 extra hop only if dense S0 is
plausible (320 without the flag was dense K1 26.2% / 0 bits). `--hops 1` is
illegal.

*Related: `master_experiment_log.md`, `docs/experiments_specs/done_success/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
