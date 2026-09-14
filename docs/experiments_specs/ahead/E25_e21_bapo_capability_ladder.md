# E25 — E21 capability limits on a calibrated BAPO DNA ladder (one rung at a time)

- **Status:** active (tiny DNA limits; 512 concat 0 bits; raw/identity/raw-KV PASS;
  r=16 frozen mean NEAR-PASS @512; learned pool 0 bits; **1024 frozen mean S1 PASS**
  53.82 vs 47.34 bits; **512 recall frozen mean 0 bits**; **512 recall r=1 identity
  S1 PASS 43.08 vs 35.00**; **512 recall r=4 frozen mean S1 PASS 47.04 vs 35.97**;
  **512 recall r=8 frozen mean S1 PASS 47.16 vs 35.97**;
  **512 recall r=12 frozen mean S1 FAIL 34.33 vs 35.97** (climbing, remainder off);
  **512 recall r=10 frozen mean 0 bits** (chance floor @800 remainder off);
  **512 recall r=10 remainder-on S1 PASS 44.21 vs 35.85**;
  **512 recall r=12 remainder-on S1 PASS 46.06 vs 35.97** (leftover 8 was the
  1.64-bit miss, not the r=12 grid);
  **512 select r=1 identity S1 PASS 47.98 vs 35.94**; **512 chain H=128 K1** dense
  24.2%; **512 chain S0 hunt K1** H=256 log 32.4% / H=512 MHA 24.1%; **512 chain
  key_len=13 K1** dense 22.5%; **256 chain hops FAIL** dense 96.4% / E18 0 / E21
  0). Keep in `ahead/`. Do not score Glyph yet.
- **Serves:** the Vision's "does the compressed channel carry *addressable* content, and how
  many bits?" question, now on **E21** (exclusive compressed read) rather than E18's raw
  one-global-read. Observation base for later gating / compression. Reuses E24's DNA instrument.
- **Implementation plan:** [E25_e21_bapo_capability_ladder_plan.md](E25_e21_bapo_capability_ladder_plan.md)
- **Owner / dates:** Krzysztof Sopyla · opened 2026-09-13
- **Engineering foundation:** [bapo_capability_ladder.md](../../engineering_specs/bapo_capability_ladder.md)
  · E21 spec on `cursor/strategy-sota-review-2026-09-e212`

> One experiment = one coherent claim about *what E21 can and cannot do* on a calibrated
> ladder, scored **one rung at a time**. Architecture changes only after a measured wall.
> Not a new training fork. Not a relabel of E24's E18 tables.

## Hypothesis
If the E18 platform is run as E21 — a **message boundary** at the DNA `query` token that severs
every sliding-window / n-gram path, and the global read sees the prefix **only as compressed
slots** (one learned slot per `r = 16` tokens, `KVCompressor`) — then on tiny packed
`far_copy` (seq=128, 64-bit prize), whose dense control already hit **99.4% / 63.3 bits** and
whose uncompressed E18 control already hit **99.2% / 63.0 bits**, E21 recovers
**≥ 0.75 × E18's recovered bits** (≥ 47 bits, information_flow ≥ 0.74)
**because** INDEX is a positional copy; mean-pool slots at init plus a trained pooling query
are enough to carry a marked span that already sits behind `min_gap`, and the E21 cut only
removes a local path `e18_local` already proved cannot use. **Kill the rung** (and change the
architecture before scoring recall or 512) if E21 stays at chance after 4× the dense step
budget while `e18_local` also stays at chance — then the slots are not a copy channel at this
scale, not a leak.

## Builds-on
- **Foundation:** `nn/perceiver_ar_lm.py` (E18 + E21 config: `message_boundary_token_id`,
  `message_compress_ratio`, `KVCompressor`), `evaluation/bapo_models.py` (`e21` arch),
  `data/symbolic_tasks.py` + `data/bapo_ladder.py`, `verification/bapo_capability_probe.py`.
  Shared causal-LM entrypoint unchanged.
- **Init / checkpoint:** random init, <100M, same tiny recipe as E24 (H=128, packed span 32).
  No pretrained warm-start. Compressor at init is mean-pool (`u = 0`, `delta = 0`).
- **Baseline to beat:** E24 tiny `far_copy` E18 **99.2% / 63.0 bits** (uncompressed one-read
  ceiling) and dense **99.4% / 63.3 bits** (solvability). `e18_local` **~25% / 0 bits** (K2).
- **Materially new:** the E21 *exclusive compressed read* on the same rows E18 already copies.
  E24 measured raw keys. This measures slots. Do not reuse E18 accuracy as an E21 score.

## The architectural bet
```
DNA row:  [BOS] filler [spanmark span] filler [QUERY] [ANSWER] y [END] [EOS]
                                                     ▲
                                          message_boundary_token_id = query

sender (t < QUERY):  E18 forward; slots = Pool_1(block of r=16 tokens), complete homogeneous blocks only
receiver (t ≥ QUERY):
   SWA / n-grams:  QUERY is a document start  →  local path cannot see the span
   global read:    raw keys of the receiver side  ∪  slots of earlier sides
                   (nothing from the prefix except slots)
```
`e18` on the same probe is arm U (`ratio` unused, no boundary). `e21` is the compressed exclusive
channel. Dense stays the solvability gate. `e18_local` stays the leak check.

**Out of scope for this first rung:** recall, select, chains, 512/1024/4k, Glyph, ratio
sweeps, SSMax, remainder-block pooling. Those are the *next* small experiments, each frozen
only after this result.

## Why this is not a safe retread
E18 already copies 64 bits at seq=128 with **raw** keys. E21 asks whether **~5–8 slots**
carry the same prize when the local bypass is structurally cut. That is the BAPO `(a,b)`
question E18 never posed: `b=0` on the prefix, `a` = slot KV. The E21 LM spec was a 32k
pretraining run; this is the cheap controlled exam that spec's K1/K2 needed and did not run.

## Success criteria (set BEFORE running)
- **S0 (instrument, already measured):** tiny packed `far_copy` dense ≥ 75%. Cite E24:
  99.4%. Re-run dense in the same probe so the JSON is self-contained; skip later arches if
  this replica misses 75% (K1).
- **S1 (this rung):** E21 `far_copy` information_flow ≥ 0.75 × E18 in the same run, and
  `e18_local` stays near chance (K2).
- **S2 (plots):** learning-curve / heatmap / recovered-bits plots exist for this rung.

## Kill criteria (set BEFORE running)
- **K1:** dense < 75% in this replica — instrument broken; do not score E21.
- **K2:** `e18_local` > chance + 0.15 — QUERY/gap leak; stop and fix the generator.
- **K3:** E21 at chance (flow < 0.05) after `k1_mult ×` dense's finish step. Then slots are
  not an INDEX channel at r=16 / seq=128. **Do not** step to recall. Next experiment is one
  architecture change (remainder-block slot, or r that divides the prefix), not a knob sweep.

## Plan
- **Data:** on-the-fly DNA `far_copy`, `--scale tiny`, packed `span_len=32`. Boundary = existing
  `query` control id. No new generator.
- **Compute:** Cloud CPU for tiny (same machine as E24 tiny). GPU later rungs on Odra/Polonez.
- **Steps:** probe defaults (800, `k1_mult=4`, dense first, early-stop 99%).
- **Launch:** `uv run python verification/bapo_capability_probe.py --scale tiny --recipe far_copy --arch dense e18 e21 e18_local --out …`
- **New foundation code:** E21 `KVCompressor` + message mask on `nn/perceiver_ar_lm.py`
  (off by default; byte-identical to E18). Probe arch `e21`. No new `train_*.py`.

## Result
- Run id: `e25_tiny_far_copy` · `e25_tiny_far_copy_remainder` · `e25_tiny_far_copy_query_align` · `e25_tiny_far_copy_e21_steps` · `e25_tiny_recall_single` · `e25_tiny_recall_single_e21_steps` · `e25_tiny_select_1decoy` · `e25_tiny_chain_ordered` · `e25_tiny_chain_ordered_e21_steps` · `e25_512_fc` · `e25_512_r64` · `e25_512_r1` · `e25_512_raw` · `e25_512_ip_r1` · `e25_512_ip_rawkv` · `e25_512_ip_id` · `e25_512_ip_r16_mean` · `e25_512_ip_r16_mean_s3k` · `e25_512_ip_r16_mean_s8k` · `e25_512_ip_r16_learned` · `e25_1k_ip_r16_mean` · `e25_1k_ip_r16_mean_s8k` · `e25_512_ip_r16_mean_recall` · `e25_512_ip_id_recall` · `e25_512_ip_id_select` · `e25_512_ip_id_chain` · `e25_512_chain_s0` · `e25_512_chain_s0_h512` · `e25_512_chain_k13` · `e25_256_chain_k13` · `e25_512_ip_r4_mean_recall` · `e25_512_ip_r4_mean_recall_s8k` · `e25_512_ip_r8_mean_recall` · `e25_512_ip_r8_mean_recall_s8k` · `e25_512_ip_r12_mean_recall` · `e25_512_ip_r12_mean_recall_s8k` · `e25_512_ip_r10_mean_recall` · `e25_512_ip_r10_rem_recall` · `e25_512_ip_r10_rem_recall_s8k` · `e25_512_ip_r12_rem_recall` · `e25_512_ip_r12_rem_recall_s8k`
- WandB: n/a (probe; no `compute/*`)
- Run reports: [`tiny`](../../2_Experiments_Registry/run_reports/e25_tiny_far_copy_20260913.md) · [`remainder`](../../2_Experiments_Registry/run_reports/e25_tiny_far_copy_remainder_20260913.md) · [`QUERY-align`](../../2_Experiments_Registry/run_reports/e25_tiny_far_copy_query_align_20260913.md) · [`INDEX extra steps`](../../2_Experiments_Registry/run_reports/e25_tiny_far_copy_e21_steps_20260913.md) · [`recall`](../../2_Experiments_Registry/run_reports/e25_tiny_recall_single_20260913.md) · [`recall extra steps`](../../2_Experiments_Registry/run_reports/e25_tiny_recall_single_e21_steps_20260913.md) · [`select`](../../2_Experiments_Registry/run_reports/e25_tiny_select_1decoy_20260913.md) · [`chain`](../../2_Experiments_Registry/run_reports/e25_tiny_chain_ordered_20260913.md) · [`chain extra steps`](../../2_Experiments_Registry/run_reports/e25_tiny_chain_ordered_e21_steps_20260913.md) · [`512 r=16`](../../2_Experiments_Registry/run_reports/e25_bridge512_far_copy_20260913.md) · [`512 r=64`](../../2_Experiments_Registry/run_reports/e25_bridge512_r64_far_copy_20260913.md) · [`512 r=1`](../../2_Experiments_Registry/run_reports/e25_bridge512_r1_far_copy_20260913.md) · [`512 raw`](../../2_Experiments_Registry/run_reports/e25_bridge512_raw_far_copy_20260913.md) · [`512 in-place r=1`](../../2_Experiments_Registry/run_reports/e25_bridge512_ip_r1_far_copy_20260913.md) · [`512 in-place raw KV`](../../2_Experiments_Registry/run_reports/e25_bridge512_ip_rawkv_far_copy_20260913.md) · [`512 in-place identity`](../../2_Experiments_Registry/run_reports/e25_bridge512_ip_id_far_copy_20260913.md) · [`512 in-place r=16 mean`](../../2_Experiments_Registry/run_reports/e25_bridge512_ip_r16_mean_far_copy_20260914.md) · [`512 in-place r=16 learned`](../../2_Experiments_Registry/run_reports/e25_bridge512_ip_r16_learned_far_copy_20260914.md) · [`1024 in-place r=16 mean`](../../2_Experiments_Registry/run_reports/e25_bridge1k_ip_r16_mean_far_copy_20260914.md) · [`512 recall in-place r=16 mean`](../../2_Experiments_Registry/run_reports/e25_bridge512_ip_r16_mean_recall_20260914.md) · [`512 recall in-place r=1 identity`](../../2_Experiments_Registry/run_reports/e25_bridge512_ip_id_recall_20260914.md) · [`512 select in-place r=1 identity`](../../2_Experiments_Registry/run_reports/e25_bridge512_ip_id_select_20260914.md) · [`512 chain H=128 K1`](../../2_Experiments_Registry/run_reports/e25_bridge512_ip_id_chain_20260914.md) · [`512 chain S0 hunt`](../../2_Experiments_Registry/run_reports/e25_bridge512_chain_s0_20260914.md) · [`512 chain key_len=13`](../../2_Experiments_Registry/run_reports/e25_bridge512_chain_k13_20260914.md) · [`256 chain hops`](../../2_Experiments_Registry/run_reports/e25_bridge256_chain_k13_20260914.md) · [`512 recall r=4 mean`](../../2_Experiments_Registry/run_reports/e25_bridge512_ip_r4_mean_recall_20260914.md) · [`512 recall r=8 mean`](../../2_Experiments_Registry/run_reports/e25_bridge512_ip_r8_mean_recall_20260914.md) · [`512 recall r=12 mean`](../../2_Experiments_Registry/run_reports/e25_bridge512_ip_r12_mean_recall_20260914.md) · [`512 recall r=10 mean`](../../2_Experiments_Registry/run_reports/e25_bridge512_ip_r10_mean_recall_20260914.md) · [`512 recall r=10 remainder`](../../2_Experiments_Registry/run_reports/e25_bridge512_ip_r10_rem_recall_20260914.md) · [`512 recall r=12 remainder`](../../2_Experiments_Registry/run_reports/e25_bridge512_ip_r12_rem_recall_20260914.md)
- Verdict: **mixed.** Tiny INDEX near-pass at 8k. Tiny MATCH/select/chain walls. GPU seq=512
  exclusive concat slots **0 bits** (r=16/64/1). `--message_override raw` **PASSES** (100% /
  63.96 bits). r=1 in-place learned compressor **0 bits**. In-place raw KV **PASSES** (63.29
  bits). In-place **hard identity PASSES** (99.8% / 62.64 bits @750). In-place **r=16
  frozen mean-pool NEAR-PASS** (84.3% / 47.36 bits @8k vs 0.75× E18 47.95). Scatter is
  fine; 16-token means carry INDEX (slow, same shape as tiny). In-place **r=16 learned
  pool 0 bits** @800 (chance floor). Learning wrecks; keep frozen mean. Seq=1024 inplace
  r=16 frozen mean **S1 PASS** (91.2% / 53.82 bits @8k vs 0.75× E18 47.34; H=256 SSMax
  log). Seq=512 packed `recall_single` inplace r=16 frozen mean **0 bits** @800 (E18
  live 47.94; vs 0.75× E18 35.95 and vs 0.75× dense 35.23). Frozen means carry INDEX,
  not MATCH. Seq=512 packed `recall_single` inplace r=1 identity **S1 PASS** (91.6% /
  43.08 bits @800 vs 0.75× E18 35.00 and vs 0.75× dense 35.77). Exclusive identity
  binds a key; r=16 pooling is the MATCH killer. Seq=512 packed `select_1decoy`
  inplace r=1 identity **S1 PASS** (100% / 47.98 bits @700 vs 0.75× E18 35.94 and vs
  0.75× dense 35.85). Exclusive identity does type-cue select. Seq=512 packed `chain_ordered` H=128
  **K1** (dense 24.2% / 0 bits @3200; e18/e21 skipped). Seq=512 packed `chain_ordered`
  dense S0 hunt **K1** at H=256 SSMax log (32.4% / 1.10 bits) and H=512 full MHA
  (24.1% / 0 bits). Seq=512 `--key_len 13` (tiny 26-bit keys, hops=2) **K1**
  (dense 22.5% / 0 bits @3200). Seq=256 `--key_len 13` hops=2 **S0 PASS** (dense
  96.4% / 24.12 bits); E18 **0 bits**; E21 **0 bits** (floor). Do not pass S1 via
  0.75×0; vs 0.75× dense 18.09 **FAIL**. Dense hops wall is between 256 and 512.
  Seq=512 packed `recall_single` inplace r=4 frozen mean **S1 PASS** (99.3% /
  47.04 bits @1900 vs 0.75× E18 35.97 and vs 0.75× dense 35.97; climbing 29.76
  bits @800). MATCH survives 4-token means. Seq=512 packed `recall_single`
  inplace r=8 frozen mean **S1 PASS** (99.1% / 47.16 bits @7100 vs 0.75× E18
  35.97 and vs 0.75× dense 35.96; climbing 15.90 bits @800). MATCH survives
  8-token means; wall is between r=8 and r=16. Seq=512 packed `recall_single`
  inplace r=12 frozen mean remainder-off **S1 FAIL** (81.7% / 34.33 bits @8000 vs
  0.75× E18 35.97; climbing 5.79 bits @800, not chance). Seq=512 packed
  `recall_single` inplace r=10 frozen mean remainder-off **0 bits** @800 (chance
  every eval; CE at ln(4); vs 0.75× E18 35.96 and vs 0.75× dense 35.92).
  Unexpected vs r=12 live at 800. Pooling width is not monotone. Seq=512 packed
  `recall_single` inplace r=10 frozen mean remainder-on **S1 PASS** (95.1% /
  44.21 bits @8000 vs 0.75× E18 35.85 and vs 0.75× dense 35.91; climbing 8.65
  bits @800). Leftover 2 tokens were the r=10 floor, not the 10-token grid.
  Seq=512 packed `recall_single` inplace r=12 frozen mean remainder-on **S1
  PASS** (97.9% / 46.06 bits @8000 vs 0.75× E18 35.97 and vs 0.75× dense 35.90;
  climbing 6.14 bits @800). Remainder-off r=12 was 34.33 S1 FAIL. Leftover 8
  was the 1.64-bit miss, not the r=12 grid. Next: 512 recall **r=16 remainder-on**.

## Follow-up (rung 1b — remainder pooling)
Ran. S1 **FAIL** at 2400 (12.7 bits). Geometry stacking did not help.

## Follow-up (rung 1c — align QUERY to r)
Ran seq=132. S1 **FAIL** at 2950 (8.7 bits). Complete coverage ≠ the missing bits at short budget.

## Follow-up (rung 1d — extra steps)
Ran 8000 steps on seq=128 complete-block E21. **NEAR-PASS** (47.01 vs 47.29 bits). Channel is slow INDEX, not dead.

## Follow-up (rung 2 — tiny recall_single)
Ran at 2850 steps. Dense S0 **PASS**. E18 0 bits. E21 **3.19 bits** (not chance; not dense).

## Follow-up (rung 2b — extra steps on tiny recall)
Ran 8000 steps. E21 **9.81 bits / flow 0.307** (best acc 48.5% @7700) vs 0.75× dense 23.53 /
0.735. MATCH wall vs dense; still beats E18's 0 bits. Do not 16k.

## Follow-up (rung 3 — tiny select_1decoy)
Ran at 1900 steps. Dense S0 **PASS**. E18 **23.2 bits**. E21 **1.47 bits / flow 0.046**
(type-cue wall; K3 borderline). Do not extra-step this rung.

## Follow-up (rung 4 — tiny chain_ordered)
Ran at 3200 steps. Dense S0 **PASS** (93.1%). E18 **24.6 bits**. E21 **8.22 bits / flow 0.316**.
S1 fail; still climbing (INDEX-like).

## Follow-up (rung 4b — extra steps on tiny chain)
Ran 8000 steps. E21 **49.4% / 10.02 bits / flow 0.386** (best 51.7% @5450). Acc did not
rise vs 3200. Composition wall. Do not 16k.

## Follow-up (rung 5 — GPU seq=512 far_copy)
Ran on Odra GPU 0, E24 right-align recipe. Dense S0 **PASS** (99.5% / 61.2 bits). E18 **99.4% /
63.2 bits**. E21 **25.1% / 0 bits / flow 0** @800 (chance; not climbing). S1 **FAIL**. K2 **PASS**.
Do not extra-step (curve is a floor). Do not 1024.

## Follow-up (rung 5b — match tiny slot count)
Ran `--message_ratio 64` on Odra. Dense **100% / 63.3 bits**. E18 **100% / 63.9 bits**. E21
**25.1% / 0.01 bits** @800 (chance, not climbing). Eight slots do not rescue 512 INDEX.

## Follow-up (rung 5c — identity slots)
Ran `--message_ratio 1` on Odra. Dense **100% / 63.95 bits**. E18 **100% / 63.94 bits**. E21
**25.1% / 0.00 bits** @800 (chance, not climbing). Identity slots do not rescue 512 INDEX.
Compression is not the killer; the exclusive QUERY cut is.

## Follow-up (rung 5d — raw override)
Ran `--message_override raw` on Odra. Dense **100% / 63.94 bits**. E18 **99.6% / 62.23 bits**.
E21 raw **100% / 63.96 bits / flow 0.999** @450 (chance through 350, 77.8% @400). S1 **PASS**.
K2 **PASS**. Slot routing (concat extra KV) is the 512 killer, not document-start, not pooling.

## Follow-up (rung 5e — in-place slots)
Ran `--message_slots_inplace` + `--message_ratio 1` on Odra (`msg_override=real`). Dense
**100% / 63.95 bits**. E18 **100% / 63.93 bits**. E21 r=1 in-place **25.1% / 0.00 bits /
flow 0** @800 (chance, not climbing). S1 **FAIL**. K2 **PASS**. K3 **triggered**. KV_LEN=S
is not enough; do not stack r=16. Next: exclusive mask vs in-place values.

## Follow-up (rung 5f — mask vs values)
Ran `--message_inplace_raw_kv` + inplace r=1 on Odra (`msg_override=real`). Dense **100% /
63.95 bits**. E18 **100% / 63.90 bits**. E21 inplace raw KV **99.5% / 63.29 bits / flow
0.989** @450 (chance through 350, 81.9% @400). S1 **PASS**. K2 **PASS**. Compressor values
(even r=1) are the remaining 512 killer; exclusive `~replace` is not. Do not stack r=16.

## Follow-up (rung 5g — hard identity at r=1)
Ran `--message_identity_slots` + inplace r=1 on Odra (`msg_override=real`, no raw_kv).
Dense **100% / 63.92 bits**. E18 **100% / 63.94 bits**. E21 inplace identity **99.8% /
62.64 bits / flow 0.979** @750 (chance through 650, 56.6% @700). S1 **PASS**. K2 **PASS**.
Scatter is fine; learned `delta` wrecked claimed r=1 identity (one-token softmax ignores
`u`). Do not run r=16 in this turn.

## Follow-up (rung 5h — inplace r=16 frozen mean-pool)
Ran `--message_identity_slots` + inplace r=16 on Odra (`msg_override=real`, no raw_kv).
Dense **100% / 63.91 bits**. E18 **100% / 63.94 bits**. E21 frozen mean **84.3% /
47.36 bits / flow 0.740** @8000 (42.4% / 8.18 bits @800, climbing; 48.0% / 22.98 @3200;
best 88.7% @7900). S1 **NEAR-PASS** (47.36 vs 47.95). K2 **PASS**. 16-token means carry
INDEX at 512; not chance, not a lost span. Do not 16k. Next: inplace r=16 learned pool
(unfreeze `u`/`delta`), no identity flag, no raw_kv. Not 1024. Not Glyph.

## Follow-up (rung 5i — inplace r=16 learned pool)
Ran inplace r=16 **without** `--message_identity_slots` on Odra (`msg_override=real`,
no raw_kv). Dense **100% / 63.94 bits**. E18 **100% / 63.94 bits**. E21 learned pool
**25.1% / 0.00 bits / flow 0** @800 (chance every eval; CE at floor). S1 **FAIL**. K2
**PASS**. Do not extra-step (floor; frozen mean was already climbing at 800). Learning
wrecks r=16 the way `delta` wrecked r=1. Keep frozen mean. Next: seq=1024 inplace r=16
frozen mean. Not Glyph. Not remainder.

## Follow-up (rung 5j — seq=1024 inplace r=16 frozen mean)
Ran `--scale bridge_1k` H=256 `--global_logit_scale log` + inplace identity r=16 on Odra.
Dense **100% / 63.95 bits**. E18 **99.7% / 63.12 bits** @2750 (live; not dilution). E21
frozen mean **91.2% / 53.82 bits / flow 0.841** @8000 (68.8% / 32.37 @3200, climbing;
best 92.7% @7850). S1 **PASS** (53.82 vs 47.34). K2 **PASS**. Frozen means carry INDEX
at 1024. Do not 16k. Next: 512 `recall_single` frozen mean. Not 4k. Not Glyph.

## Follow-up (rung 5k — seq=512 recall_single inplace r=16 frozen mean)
Ran `--scale bridge` packed `recall_single` H=128 + inplace identity r=16 on Odra.
Dense **99.0% / 46.97 bits**. E18 **100% / 47.94 bits** @400 (live in this JSON; not
E24's 0-bit 512-recall control). E21 frozen mean **25.5% / 0.00 bits / flow 0** @800
(chance every eval; CE at floor). S1 **FAIL** (0 vs 35.95). Content vs 0.75× dense
**FAIL** (0 vs 35.23). K2 **PASS**. Do not extra-step (floor). Frozen means carry
INDEX, not MATCH, at 512. Next: 512 recall inplace r=1 identity. Not 4k. Not Glyph.

## Follow-up (rung 5l — seq=512 recall_single inplace r=1 hard identity)
Ran `--scale bridge` packed `recall_single` H=128 + inplace identity **r=1** on Odra
(no raw_kv). Dense **99.9% / 47.69 bits**. E18 **99.4% / 46.67 bits** @350 (live).
E21 identity **91.6% / 43.08 bits / flow 0.897** @800 (chance through ~350; best
92.1% @750). S1 **PASS** (43.08 vs 35.00). Content vs 0.75× dense **PASS** (43.08 vs
35.77). K2 **PASS**. Do not 8k (S1 already PASS). Exclusive identity binds a key;
r=16 pooling is the MATCH killer. Next: 512 `select_1decoy` r=1 identity. Not 4k.
Not Glyph.

## Follow-up (rung 5m — seq=512 select_1decoy inplace r=1 hard identity)
Ran `--scale bridge` packed `select_1decoy` H=128 + inplace identity **r=1** on Odra
(no raw_kv). Dense **99.9% / 47.80 bits**. E18 **100% / 47.92 bits** @400 (live).
E21 identity **100% / 47.98 bits / flow 1.000** @700 (chance through ~500; early
stop). S1 **PASS** (47.98 vs 35.94). Content vs 0.75× dense **PASS** (47.98 vs
35.85). K2 **PASS**. Do not 8k. Exclusive identity does type-cue select, not only
MATCH/INDEX. Next: 512 `chain_ordered` r=1 identity. Not 4k. Not Glyph.

## Follow-up (rung 5n — seq=512 chain_ordered inplace r=1 identity, H=128)
Ran `--scale bridge` packed `chain_ordered` H=128 + inplace identity **r=1** on Odra.
Dense **24.2% / 0 bits** @3200 (chance every eval; CE at ln(4)). **K1.** e18 / e21 /
e18_local **skipped**. Do not score S1. Packed 512 chain is not dense-solvable at
this width. Next: 512 chain **dense S0 hunt** H=256 `--global_logit_scale log`.
Not 4k. Not Glyph. Do not score E21 until S0.

## Follow-up (rung 5o — seq=512 chain_ordered dense S0 hunt)
Ran packed `chain_ordered` dense-first on Odra. H=256 `--global_logit_scale log`:
dense **32.4% / 1.10 bits** @3200 (**K1**). One bump `--hidden 512 --kv_heads 0`:
dense **24.1% / 0 bits** @3200 (**K1**). e18 / e21 skipped both times. Width is
not the lever. Next: 512 chain **`--key_len 13`** (tiny packed 26-bit keys) H=256
log, dense-first. Not 4k. Not Glyph. Do not invent a third width. Do not score
E21 until S0.

## Follow-up (rung 5p — seq=512 chain_ordered --key_len 13)
Ran packed `chain_ordered --key_len 13` (tiny 26-bit keys, hops=2) H=256
`--global_logit_scale log` dense-first on Odra. Dense **22.5% / 0 bits** @3200
(**K1**; chance floor). e18 / e21 skipped. Tiny keys still fail at seq=512 —
context, not 24-token packing. Next: seq=**256** chain `--key_len 13` hops=2
(H=256 log). `--hops 1` is illegal (`chain needs hops >= 2`). Not 4k. Not Glyph.
Do not bump width. Do not score E21 until S0.

## Follow-up (rung 5q — seq=256 chain_ordered --key_len 13)
Ran `--scale bridge --seq_len 256` (min_gap 64 > window 16; not tiny_wide)
`--key_len 13` hops=2 H=256 `--global_logit_scale log` on Odra. Dense **96.4% /
24.12 bits** @3200 (**S0 PASS**). E18 **26.2% / 0 bits**. E21 identity **24.9% /
0 bits** (chance floor). K2 **PASS**. Do not extra-step (floor). Do not pass S1
via 0.75×0; vs 0.75× dense 18.09 **FAIL**. Dense hops wall is between 256 and
512. Next: stop hops extra-steps. Not seq=192. Not 4k. Not Glyph.

## Follow-up (rung 5r — seq=512 recall_single r=4 frozen mean)
Ran packed `recall_single` inplace `--message_identity_slots --message_ratio 4`
(no raw_kv, `u`/`delta` frozen) H=128 dense-first on Odra. Dense **100% / 47.96
bits** (**S0 PASS**). E18 **100% / 47.96 bits**. E21 **78.3% / 29.76 bits** @800
(climbing) then **99.3% / 47.04 bits** @1900 (**S1 PASS** vs 0.75× E18 35.97 and
vs 0.75× dense 35.97). K2 **PASS**. MATCH survives 4-token means (r=16 was 0 bits;
r=1 was 43.08). Next: 512 recall **r=8 frozen mean**. Not hops. Not 4k. Not Glyph.
Do not unfreeze `u`/`delta`.

## Follow-up (rung 5s — seq=512 recall_single r=8 frozen mean)
Ran packed `recall_single` inplace `--message_identity_slots --message_ratio 8`
(no raw_kv, `u`/`delta` frozen) H=128 dense-first on Odra. Dense **100% / 47.95
bits** (**S0 PASS**). E18 **100% / 47.96 bits**. E21 **56.5% / 15.90 bits** @800
(climbing) then **99.1% / 47.16 bits** @7100 (**S1 PASS** vs 0.75× E18 35.97 and
vs 0.75× dense 35.96). K2 **PASS**. MATCH survives 8-token means; wall is
between r=8 and r=16. Next: 512 recall **r=12 frozen mean**. Not hops. Not 4k.
Not Glyph. Do not unfreeze `u`/`delta`.

## Follow-up (rung 5t — seq=512 recall_single r=12 frozen mean)
Ran packed `recall_single` inplace `--message_identity_slots --message_ratio 12`
(remainder **off**, no raw_kv, `u`/`delta` frozen) H=128 dense-first on Odra.
Dense **100% / 47.91 bits** (**S0 PASS**). E18 **100% / 47.96 bits**. E21
**39.8% / 5.79 bits** @800 (climbing) then **81.7% / 34.33 bits** @8000 (**S1
FAIL** vs 0.75× E18 35.97 and vs 0.75× dense 35.93). K2 **PASS**. Not chance.
S1-passing MATCH dies between r=8 and r=12 at the 8k budget. Next: 512 recall
**r=10 frozen mean**. Do not 16k. Not hops. Not 4k. Not Glyph. Do not remainder.
Do not unfreeze `u`/`delta`.

## Follow-up (rung 5u — seq=512 recall_single r=10 frozen mean)
Ran packed `recall_single` inplace `--message_identity_slots --message_ratio 10`
(remainder **off**, no raw_kv, `u`/`delta` frozen) H=128 dense-first on Odra.
Dense **99.9% / 47.89 bits** (**S0 PASS**). E18 **100% / 47.94 bits**. E21
**25.5% / 0.00 bits** @800 (**S1 FAIL** vs 0.75× E18 35.96 and vs 0.75× dense
35.92). K2 **PASS**. Chance every eval; CE at ln(4). Do not extra-step (floor).
Unexpected vs r=12 live at 800. Pooling width is not monotone. Stop stacking
MATCH r-sweeps. Do not r=9. Do not r=11. Do not remainder-on. Do not 16k.
Not hops. Not 4k. Not Glyph. Do not unfreeze `u`/`delta`.

## Follow-up (rung 5v — seq=512 recall_single r=10 remainder-on frozen mean)
Ran packed `recall_single` inplace `--message_identity_slots --message_ratio 10`
`--message_pool_remainder` (no raw_kv, `u`/`delta` frozen) H=128 dense-first on
Odra. Dense **99.9% / 47.88 bits** (**S0 PASS**). E18 **100% / 47.79 bits**.
E21 **44.2% / 8.65 bits** @800 (climbing) then **95.1% / 44.21 bits** @8000
(best 96.4% @7800; **S1 PASS** vs 0.75× E18 35.85 and vs 0.75× dense 35.91).
K2 **PASS**. Remainder-off r=10 was 0 bits at 800. Leftover 2 tokens were the
floor, not the r=10 grid. Do not 16k. Next: 512 recall **r=12 remainder-on**.
Not r=9/11. Not hops. Not 4k. Not Glyph. Do not unfreeze `u`/`delta`.

## Follow-up (rung 5w — seq=512 recall_single r=12 remainder-on frozen mean)
Ran packed `recall_single` inplace `--message_identity_slots --message_ratio 12`
`--message_pool_remainder` (no raw_kv, `u`/`delta` frozen) H=128 dense-first on
Odra. Dense **100% / 47.87 bits** (**S0 PASS**). E18 **100% / 47.96 bits**.
E21 **41.0% / 6.14 bits** @800 (climbing) then **97.9% / 46.06 bits** @8000
(best 98.2% @7850; **S1 PASS** vs 0.75× E18 35.97 and vs 0.75× dense 35.90).
K2 **PASS**. Remainder-off r=12 was 34.33 S1 FAIL. Leftover 8 was the 1.64-bit
miss, not the r=12 grid. Do not 16k. Next: 512 recall **r=16 remainder-on**.
Not r=9/11. Not hops. Not 4k. Not Glyph. Do not unfreeze `u`/`delta`.
