# MrCogito — Research Agenda (living)

**Updated:** 2026-09-14 · The daily driver for *current* work. Overarching direction: [vision_and_goals.md](vision_and_goals.md). Results ledger: [master_experiment_log.md](../2_Experiments_Registry/master_experiment_log.md). Specs: [experiments_specs](../experiments_specs/).

> This is **research / exploration** — the direction is genuinely open. This file
> stays small on purpose: how we work, the immediate focus, and a neutral record
> of what we've learned. It is **not** a committed multi-step plan, and nothing
> here is a final verdict.

## How we work (the process — this is the point)
- Go back to fundamentals. Make **small, well-defined increments** — one change at a time.
- One to a few active experiments, each with a frozen spec in
  `docs/experiments_specs/ahead/<ID>.md` (hypothesis · builds-on · single change ·
  success/kill criteria).
- Build on the **existing foundation**; reuse and extend it, don't fork a script per idea
  (see `.cursor/rules/project-overview.mdc`).
- Treat every past run as **evidence that improved our understanding**, not as success or failure. Keep conclusions tentative.

## Guiding direction (open)
We still follow the [Vision](vision_and_goals.md): compress sequences into concepts and **reason in latent space**, working toward a multimodal / audio model eventually. *How* we get there is unsettled and under active exploration. Latent-space reasoning stays a central interest — likely explored with a different approach than before.

## Current focus
- **2026-09-14 — E25 E21 capability ladder (tiny limits; 512 concat wall; raw PASS; inplace identity PASS; r=16 mean INDEX; MATCH remainder-on S1 through r=16; remainder-off r=10/16 chance, r=12 S1 FAIL; SELECT r=8 rem-off S1 PASS, r=12 rem-on S1 PASS, r=16 rem-on live S1 FAIL; SELECT identity PASS; 1024 MATCH r=8 rem-off H=128 S1 FAIL, H=256 log S1 PASS 60.35; 1024 SELECT r=8 rem-off H=256 log chance; 1024 SELECT r=1 identity H=256 log chance; 1024 SELECT SWA-unsever `--message_keep_local_swa` chance 0.01 bits; 1024 SELECT extra exclusive hop `--message_extra_slot_attends 1` chance 0 bits; 1024 SELECT type_marks anchors `--message_global_anchors type_marks` 0 bits @800 / 31.90 @1050 S1 FAIL vs 0.75× dense; 1024 SELECT type_marks 8k extra-step S1 FAIL 6.48 vs live E18 63.96; 1024 SELECT extra hop + unfrozen slot K/V `--message_update_slot_kv` chance 0 bits; 1024 SELECT second exclusive global layer `--global_layers 2` chance 0 bits; 1024 SELECT QUERY-side anchors `--message_global_anchors query_side` chance 0 bits (4 tokens vs 1024, extra 4); 768 SELECT r=1 identity H=256 log chance 0.01 bits; 640 SELECT r=1 identity H=256 log S1 PASS 63.95 vs 47.93 (`--seq_len 640`); 704 SELECT r=1 identity H=256 log chance 0 bits (`--seq_len 704`); 672 SELECT r=1 identity H=256 log S1 PASS 63.96 vs 47.95 (`--seq_len 672`); 688 SELECT r=1 identity H=256 log S1 PASS 62.68 vs 47.97 (`--seq_len 688`; 800 climb 31.57, extra-step 99.5% @400; wall was (688, 704]); 696 SELECT r=1 identity H=256 log S1 FAIL 0 bits (`--seq_len 696`; chance @800; wall was (688, 696]); 692 SELECT r=1 identity H=256 log S1 PASS 62.76 vs 47.91 (`--seq_len 692`; 99.8% @500; wall now (692, 696]); 696 SELECT pack_stride 32 S1 FAIL 31.67 vs 47.50 @800 / 0 bits @8k (`--message_pack_stride 32`; leftover drop to 640 slots does not rescue 696); 696 SELECT `--evidence_align spread` K1 (dense 24.7% / 0 bits @3200; e18/e21 skipped); 4k INDEX r=16 rem-off H=256 log chance vs 0.75× dense; 2048 INDEX r=16 rem-off H=256 log chance vs 0.75× dense; 1536 INDEX r=16 rem-off H=256 log chance vs 0.75× dense (shared wall (1024, 1536]); 512 chain K1; 256 hops FAIL vs dense; 256 hops `--global_layers 2` S1 PASS 25.85 vs 4.17 (E18 live 5.56; 8k not run); 512 hops `--global_layers 2` K1 dense 22.5% / 0 bits @3200 (e18/e21 skipped; 8k not run); 384 hops `--global_layers 2` K1 dense 24.5% / 0 bits @3200 (e18/e21 skipped; 8k not run); 320 hops `--global_layers 2` K1 dense 26.2% / 0 bits @3200 (e18/e21 skipped; 8k not run); 288 hops `--global_layers 2` S0 PASS / E21 FAIL dense 93.1% / 22.20 bits @3200, E18 0, E21 0 (chance @800/@3200; vs 0.75× dense 16.65; dense wall (288 S0 PASS, 320 K1]; E21 wall (256 S1 PASS, 288 FAIL]; 8k not run)).**
  Tiny INDEX near-pass at 8k. GPU seq=512 exclusive concat slots **0 bits** (r=16/64/1).
  `--message_override raw` **100% / 63.96 bits**. r=1 in-place learned compressor **0 bits**.
  In-place raw KV **63.29 bits**. In-place hard identity **99.8% / 62.64 bits** @750.
  In-place r=16 frozen mean-pool **84.3% / 47.36 bits** @8k. In-place r=16 learned pool
  **0 bits** @800. Seq=1024 frozen mean **91.2% / 53.82 bits** @8k. Seq=512
  `recall_single` frozen mean **0 bits**. Seq=512 `recall_single` r=1 identity
  **91.6% / 43.08 bits**. Seq=512 `recall_single` r=4 frozen mean **99.3% / 47.04
  bits** @1900. Seq=512 `recall_single` r=8 frozen mean **99.1% / 47.16 bits**
  @7100. Seq=512 `recall_single` r=12 frozen mean **81.7% / 34.33 bits** @8000
  (S1 FAIL, not chance). Seq=512 `recall_single` r=10 frozen mean **25.5% / 0
  bits** @800 (chance floor remainder off; unexpected vs r=12 live). Seq=512
  `recall_single` r=10 remainder-on **95.1% / 44.21 bits** @8000 (S1 PASS;
  leftover 2 was the floor). Seq=512 `recall_single` r=12 remainder-on **97.9% /
  46.06 bits** @8000 (S1 PASS; leftover 8 was the 1.64-bit miss). Seq=512
  `recall_single` r=16 remainder-on **92.6% / 40.92 bits** @8000 (S1 PASS vs
  0.75× E18 35.89; remainder-off r=16 was 0 bits chance; leftover/alignment
  was the MATCH wall through r=16). Seq=512 `select_1decoy` r=16 remainder-on
  **84.5% / 35.94 bits** @8000 (S1 FAIL vs 0.75× E18 35.95 by 0.016; live, not
  chance; type-cue slightly harder than MATCH). Seq=512 `select_1decoy` r=12
  remainder-on **93.8% / 43.29 bits** @8000 (S1 PASS vs 0.75× E18 35.92; r=16
  is the SELECT S1 edge). Seq=512 `select_1decoy` r=8 remainder-off **96.4% /
  45.53 bits** @8000 (S1 PASS vs 0.75× E18 35.93; default MATCH grid enough).
  Seq=1024 `recall_single` r=8 remainder-off H=128 **26.8% / 0.01 bits** @8000
  (S1 FAIL vs 0.75× E18 46.98). Seq=1024 `recall_single` r=8 rem-off **H=256
  SSMax log 96.5% / 60.35 bits** @8000 (S1 PASS vs 0.75× E18 46.96; MATCH at
  1024 needs INDEX width).   Seq=1024 `select_1decoy` r=8 rem-off H=256 SSMax
  log **25.7% / 0 bits** @800 (S1 FAIL vs 0.75× E18 47.89; chance floor;
  SELECT does not scale with MATCH). Seq=1024 `select_1decoy` r=1 identity
  H=256 SSMax log **25.7% / 0 bits** @800 (S1 FAIL vs 0.75× E18 47.85; chance
  floor; exclusive SELECT dead at 1024 even with identity slots). Seq=1024
  `select_1decoy` r=1 identity SWA-unsever `--message_keep_local_swa` H=256
  SSMax log **25.8% / 0.01 bits** @800 (S1 FAIL vs 0.75× E18 47.95; chance
  floor; unsevering SWA does not rescue 1024 SELECT). Seq=1024
  `select_1decoy` r=1 identity extra exclusive hop
  `--message_extra_slot_attends 1` H=256 SSMax log **25.7% / 0 bits** @800
  (S1 FAIL vs 0.75× E18 47.94; chance floor; two exclusive reads over frozen
  slots do not rescue 1024 SELECT). Seq=1024
  `select_1decoy` r=1 identity type_marks anchors
  `--message_global_anchors type_marks` H=256 SSMax log **25.7% / 0 bits**
  @800 floor then **48.1% / 31.90 bits** @1050 (**S1 FAIL** vs 0.75× dense
  47.99; this-JSON E18 0 — do not pass via 0.75×0; vs 0.75× prior live E18
  47.94). Type-mark leak is 2 tokens vs seq=1024; at r=1 identity extra
  non-slot count is 0. Seq=1024 type_marks **8k extra-step** (`--no-dense_first`)
  E18 **100% / 63.96 bits** @3250 (live); E21 **39.3% / 6.48 bits** @8000
  (**S1 FAIL** vs 0.75× live E18 47.97; 0 @800/1050; best 14.43 bits; did
  not climb past 31.90). Seq=1024 `select_1decoy` r=1 identity extra hop +
  unfrozen slot K/V `--message_extra_slot_attends 1 --message_update_slot_kv`
  H=256 SSMax log **25.7% / 0 bits** @800 (**S1 FAIL** vs 0.75× live E18
  47.85; chance every eval). Extra hop is real updated-Q / frozen-KV (not a
  no-op); rewriting slot K/V between hops still does not bind type-cue at
  1024. Seq=1024 `select_1decoy` r=1 identity **second exclusive global
  layer** `--global_layers 2` (extra hops 0) H=256 SSMax log **25.7% / 0
  bits** @800 (**S1 FAIL** vs 0.75× live E18 47.97; chance every eval). Two
  full exclusive attend+FFN Blocks over slots — not extra hops and not extra
  SWA. Seq=1024 `select_1decoy` r=1 identity **QUERY-side anchors**
  `--message_global_anchors query_side` H=256 SSMax log **25.7% / 0 bits**
  @800 (**S1 FAIL** vs 0.75× this-JSON E18 35.07 / dense 47.97; chance every
  eval). Leak is QUERY + 2 asked-key symbols + ANSWER (**4 vs seq=1024**;
  extra non-slot count 4; not r=1 prefix slots). Existing `query_nbhd` extra
  count is 0 at r=1. This-JSON E18 late-clicked **75.1% / 46.77 bits** @800
  (live). Seq=512
  `select_1decoy` r=1 identity **100% / 47.98 bits** @700. Seq=512 packed
  `chain_ordered` **K1**.   Seq=256 `--key_len 13` hops=2: dense **96.4% / 24.12
  bits** (S0 PASS); E18 **0 bits**; E21 **0 bits**. Hops FAIL vs 0.75× dense.
  Seq=256 hops `--global_layers 2` (same recipe): dense **99.2% / 25.46 bits**
  @1800 (**S0 PASS**); E18 **42.3% / 5.56 bits** (live, climbing); E21
  **99.6% / 25.85 bits** @1650 (**S1 PASS** vs 0.75× E18 4.17 and vs 0.75×
  dense 19.10). glob=1 was 0/0. 8k not run.
  Seq=512 hops `--global_layers 2` (same compressor): dense **22.5% / 0 bits**
  @3200 (**S0 FAIL / K1**; e18/e21 skipped; glob=1 was also K1). Do not score
  E21. 8k not run.
  Seq=384 hops `--global_layers 2` (`--scale bridge --seq_len 384`): dense
  **24.5% / 0 bits** @3200 (**S0 FAIL / K1**; e18/e21 skipped; best 29.1%
  @1450). Do not score E21. 8k not run.
  Seq=320 hops `--global_layers 2` (`--scale bridge --seq_len 320`): dense
  **26.2% / 0 bits** @3200 (**S0 FAIL / K1**; e18/e21 skipped; best 27.9%
  @2400). Do not score E21. 8k not run.
  Seq=288 hops `--global_layers 2` (`--scale bridge --seq_len 288`): dense
  **93.1% / 22.20 bits** @3200 (**S0 PASS**); E18 **26.8% / 0 bits**; E21
  **24.5% / 0 bits** (chance @800 and @3200; **S1 FAIL** vs 0.75× dense
  16.65; do not pass via 0.75×0). 8k not run. Dense hops wall at glob=2
  is **(288 S0 PASS, 320 K1]**. E21/E18 hops wall at glob=2 is
  **(256 S1 PASS, 288 FAIL]**.
  Seq=4096 packed `far_copy` r=16 rem-off H=256 SSMax log **25.7% / 0 bits**
  @800 (**S1 FAIL** vs 0.75× dense 47.96; E18 ~0 live; chance floor). Exclusive
  INDEX that passed at 1024 is chance at 4k. No 4k scale code change (`medium`).
  Seq=2048 packed `far_copy` `--scale bridge_1k --seq_len 2048` r=16 rem-off
  H=256 SSMax log **25.0% / 0 bits** @800 (**S1 FAIL** vs 0.75× dense 47.84;
  E18 ~0 live; chance floor). Exclusive INDEX that passed at 1024 is chance
  at 2048 with the same gap=64.
  Seq=1536 packed `far_copy` `--scale bridge_1k --seq_len 1536` r=16 rem-off
  H=256 SSMax log **24.8% / 0 bits** @800 (**S1 FAIL** vs 0.75× dense 46.81;
  E18 ~0 live; chance floor). Exclusive INDEX that passed at 1024 is chance
  at 1536 with the same gap=64 / leftover 12. Shared INDEX wall is
  **(1024 PASS, 1536 FAIL]**. INDEX length extra-steps are **stopped**.
  8k not run.
  Seq=768 packed `select_1decoy` `--scale bridge_1k --seq_len 768` r=1
  identity H=256 SSMax log **25.6% / 0.01 bits** @800 (**S1 FAIL** vs 0.75×
  live E18 47.97; E18 **100% / 63.97 bits** @350; chance every eval).
  Exclusive SELECT that passed at 512 is chance at 768 on the 1024-passing
  width. Do not extra-step (floor).
  Seq=640 packed `select_1decoy` `--scale bridge_1k --seq_len 640` r=1
  identity H=256 SSMax log **100% / 63.95 bits** @450 (**S1 PASS** vs 0.75×
  live E18 47.93; E18 **100% / 63.91 bits** @300; chance through 350, click
  51.3% @400). Exclusive SELECT is live at 640 and chance at 768. Do not
  extra-step (S1 already PASS).
  Seq=704 packed `select_1decoy` `--scale bridge_1k --seq_len 704` r=1
  identity H=256 SSMax log **25.7% / 0 bits** @800 (**S1 FAIL** vs 0.75×
  live E18 47.92; E18 **100% / 63.89 bits** @350; chance every eval).
  Exclusive SELECT that passed at 640 is chance at 704. Do not extra-step
  (floor).
  Seq=672 packed `select_1decoy` `--scale bridge_1k --seq_len 672` r=1
  identity H=256 SSMax log **100% / 63.96 bits** @500 (**S1 PASS** vs 0.75×
  live E18 47.95; E18 **100% / 63.93 bits** @350; chance through 400, click
  47.6% @450). Exclusive SELECT is live at 672 and chance at 704. Do not
  extra-step (S1 already PASS).
  Seq=688 packed `select_1decoy` `--scale bridge_1k --seq_len 688` r=1
  identity H=256 SSMax log **99.5% / 62.68 bits** @400 extra-step (**S1
  PASS** vs 0.75× live E18 47.97; E18 **100% / 63.96 bits** @350; 800 climb
  72.4% / 31.57 bits, best 88.9% @750). Exclusive SELECT is live at 688 and
  chance at 704. Do not 16k (S1 already PASS).
  Seq=696 packed `select_1decoy` `--scale bridge_1k --seq_len 696` r=1
  identity H=256 SSMax log **25.0% / 0 bits** @800 (**S1 FAIL** vs 0.75×
  live E18 47.95; E18 **100% / 63.93 bits** @350; chance every eval).
  Exclusive SELECT that passed at 688 is chance at 696. Do not extra-step
  (floor).
  Seq=692 packed `select_1decoy` `--scale bridge_1k --seq_len 692` r=1
  identity H=256 SSMax log **99.8% / 62.76 bits** @500 (**S1 PASS** vs 0.75×
  live E18 47.91; E18 **100% / 63.89 bits** @350; chance through 450, click
  99.8% @500). Exclusive SELECT is live at 692 and chance at 696. Do not
  extra-step (S1 already PASS).
  Seq=696 packed `select_1decoy` `--message_pack_stride 32` r=1 identity
  H=256 SSMax log **49.7% / 31.67 bits** @800 then **24.1% / 0 bits**
  @8000 (**S1 FAIL** vs 0.75× live E18 47.50 / 47.94; 800 climb did not
  replicate). r=1 remainder is a no-op. Leftover drop to 640 exclusive
  slots does not rescue 696. Do not 16k.
  Seq=696 packed `select_1decoy` `--evidence_align spread` r=1 identity
  H=256 SSMax log **K1** (dense **24.7% / 0 bits** @3200; e18/e21
  skipped; chance every eval). QUERY 658 leftover 18 unchanged; seed-0
  fact0 **202** vs right **528**. Spread at 696 is uncalibrated. Do not
  extra-step (floor). Do not 16k.
  Seq=696 packed `select_1decoy` `--local_window 32` r=1 identity
  H=256 SSMax log **25.0% / 0 bits** @800 (**S1 FAIL** vs 0.75× live
  E18 47.71; E18 **99.9% / 63.62 bits** @600; chance every eval). Banner
  `gap=64 window=32` (32 < 64). Widening SWA does not rescue 696. Do not
  extra-step (floor). Do not 16k.
  Next: seq=**272** packed `chain_ordered --key_len 13` `--global_layers 2`
  (localize E21/E18 hops wall **(256 S1 PASS, 288 FAIL]**; dense already S0
  PASS at 288; do not run here).
  Do not reopen SELECT **(692 PASS, 696 FAIL]**. Do not seq=694. INDEX length
  extra-steps stay stopped.
  Remaining DNA walls:
  SELECT length **(692 PASS, 696 FAIL]** leftover-drop, spread (K1), and window 32 did not close, INDEX **(1024 PASS, 1536 FAIL] shared with E18**, hops glob=2 dense **(288 S0 PASS, 320 K1]**, hops glob=2 E21/E18 **(256 S1 PASS, 288 FAIL]**, 512 chain **K1 at glob=1 and glob=2**, 256 hops
  **FAIL at glob=1 / PASS at glob=2**. Not remainder-on. Not H=512. Not
  hops seq shrink below 256. Not Glyph. Do not restore raw global KV. Default remainder
  stays off. Default `--message_keep_local_swa` stays off. Default
  `--message_extra_slot_attends` stays 0. Default `--message_update_slot_kv`
  stays off. Default `--message_global_anchors` stays `none`. Default
  `--global_layers` stays 1. Default `--message_pack_stride` stays 0.
  Default `--local_window` stays 16.
  Spec [E25](../experiments_specs/ahead/E25_e21_bapo_capability_ladder.md) ·
  [rung 1](../2_Experiments_Registry/run_reports/e25_tiny_far_copy_20260913.md) ·
  [remainder](../2_Experiments_Registry/run_reports/e25_tiny_far_copy_remainder_20260913.md) ·
  [QUERY-align](../2_Experiments_Registry/run_reports/e25_tiny_far_copy_query_align_20260913.md) ·
  [INDEX extra steps](../2_Experiments_Registry/run_reports/e25_tiny_far_copy_e21_steps_20260913.md) ·
  [recall](../2_Experiments_Registry/run_reports/e25_tiny_recall_single_20260913.md) ·
  [recall extra steps](../2_Experiments_Registry/run_reports/e25_tiny_recall_single_e21_steps_20260913.md) ·
  [select](../2_Experiments_Registry/run_reports/e25_tiny_select_1decoy_20260913.md) ·
  [chain](../2_Experiments_Registry/run_reports/e25_tiny_chain_ordered_20260913.md) ·
  [chain extra steps](../2_Experiments_Registry/run_reports/e25_tiny_chain_ordered_e21_steps_20260913.md) ·
  [512 r=16](../2_Experiments_Registry/run_reports/e25_bridge512_far_copy_20260913.md) ·
  [512 r=64](../2_Experiments_Registry/run_reports/e25_bridge512_r64_far_copy_20260913.md) ·
  [512 r=1](../2_Experiments_Registry/run_reports/e25_bridge512_r1_far_copy_20260913.md) ·
  [512 raw](../2_Experiments_Registry/run_reports/e25_bridge512_raw_far_copy_20260913.md) ·
  [512 in-place r=1](../2_Experiments_Registry/run_reports/e25_bridge512_ip_r1_far_copy_20260913.md) ·
  [512 in-place raw KV](../2_Experiments_Registry/run_reports/e25_bridge512_ip_rawkv_far_copy_20260913.md) ·
  [512 in-place identity](../2_Experiments_Registry/run_reports/e25_bridge512_ip_id_far_copy_20260913.md) ·
  [512 in-place r=16 mean](../2_Experiments_Registry/run_reports/e25_bridge512_ip_r16_mean_far_copy_20260914.md) ·
  [512 in-place r=16 learned](../2_Experiments_Registry/run_reports/e25_bridge512_ip_r16_learned_far_copy_20260914.md) ·
  [1024 in-place r=16 mean](../2_Experiments_Registry/run_reports/e25_bridge1k_ip_r16_mean_far_copy_20260914.md) ·
  [512 recall in-place r=16 mean](../2_Experiments_Registry/run_reports/e25_bridge512_ip_r16_mean_recall_20260914.md) ·
  [512 recall in-place r=1 identity](../2_Experiments_Registry/run_reports/e25_bridge512_ip_id_recall_20260914.md) ·
  [512 select in-place r=1 identity](../2_Experiments_Registry/run_reports/e25_bridge512_ip_id_select_20260914.md) ·
  [512 chain H=128 K1](../2_Experiments_Registry/run_reports/e25_bridge512_ip_id_chain_20260914.md) ·
  [512 chain S0 hunt](../2_Experiments_Registry/run_reports/e25_bridge512_chain_s0_20260914.md) ·
  [512 chain key_len=13](../2_Experiments_Registry/run_reports/e25_bridge512_chain_k13_20260914.md) ·
  [256 chain hops](../2_Experiments_Registry/run_reports/e25_bridge256_chain_k13_20260914.md) ·
  [512 recall r=4 mean](../2_Experiments_Registry/run_reports/e25_bridge512_ip_r4_mean_recall_20260914.md) ·
  [512 recall r=8 mean](../2_Experiments_Registry/run_reports/e25_bridge512_ip_r8_mean_recall_20260914.md) ·
  [512 recall r=12 mean](../2_Experiments_Registry/run_reports/e25_bridge512_ip_r12_mean_recall_20260914.md) ·
  [512 recall r=10 mean](../2_Experiments_Registry/run_reports/e25_bridge512_ip_r10_mean_recall_20260914.md) ·
  [512 recall r=10 remainder](../2_Experiments_Registry/run_reports/e25_bridge512_ip_r10_rem_recall_20260914.md) ·
  [512 recall r=12 remainder](../2_Experiments_Registry/run_reports/e25_bridge512_ip_r12_rem_recall_20260914.md) ·
  [512 recall r=16 remainder](../2_Experiments_Registry/run_reports/e25_bridge512_ip_r16_rem_recall_20260914.md) ·
  [512 select r=16 remainder](../2_Experiments_Registry/run_reports/e25_bridge512_ip_r16_rem_select_20260914.md) ·
  [512 select r=12 remainder](../2_Experiments_Registry/run_reports/e25_bridge512_ip_r12_rem_select_20260914.md) ·
  [512 select r=8 rem-off](../2_Experiments_Registry/run_reports/e25_bridge512_ip_r8_select_20260914.md) ·
  [1024 recall r=8 rem-off](../2_Experiments_Registry/run_reports/e25_bridge1k_ip_r8_recall_20260914.md) ·
  [1024 recall r=8 H=256 log](../2_Experiments_Registry/run_reports/e25_bridge1k_ip_r8_h256_recall_20260914.md) ·
  [1024 select r=8 H=256 log](../2_Experiments_Registry/run_reports/e25_bridge1k_ip_r8_h256_select_20260914.md) ·
  [1024 select r=1 identity H=256 log](../2_Experiments_Registry/run_reports/e25_bridge1k_ip_id_h256_select_20260914.md) ·
  [1024 select SWA-unsever](../2_Experiments_Registry/run_reports/e25_bridge1k_ip_id_keepswa_select_20260914.md) ·
  [1024 select extra hop](../2_Experiments_Registry/run_reports/e25_bridge1k_ip_id_extrahop_select_20260914.md) ·
  [1024 select type_marks anchors](../2_Experiments_Registry/run_reports/e25_bridge1k_ip_id_anchors_select_20260914.md) ·
  [1024 select type_marks 8k](../2_Experiments_Registry/run_reports/e25_bridge1k_ip_id_anchors_select_s8k_20260914.md) ·
  [1024 select extra hop + unfrozen KV](../2_Experiments_Registry/run_reports/e25_bridge1k_ip_id_updatekv_select_20260914.md) ·
  [1024 select second global layer](../2_Experiments_Registry/run_reports/e25_bridge1k_ip_id_glob2_select_20260914.md) ·
  [1024 select QUERY-side anchors](../2_Experiments_Registry/run_reports/e25_bridge1k_ip_id_qside_select_20260914.md) ·
  [4k INDEX r=16 H=256 log](../2_Experiments_Registry/run_reports/e25_medium_4k_ip_r16_mean_far_copy_20260914.md) ·
  [2048 INDEX r=16 H=256 log](../2_Experiments_Registry/run_reports/e25_bridge1k_2k_ip_r16_mean_far_copy_20260914.md) ·
  [1536 INDEX r=16 H=256 log](../2_Experiments_Registry/run_reports/e25_bridge1k_1536_ip_r16_mean_far_copy_20260914.md) ·
  [256 chain hops glob2](../2_Experiments_Registry/run_reports/e25_bridge256_chain_k13_glob2_20260914.md) ·
  [512 chain hops glob2](../2_Experiments_Registry/run_reports/e25_bridge512_chain_k13_glob2_20260914.md) ·
  [384 chain hops glob2](../2_Experiments_Registry/run_reports/e25_bridge384_chain_k13_glob2_20260914.md) ·
  [320 chain hops glob2](../2_Experiments_Registry/run_reports/e25_bridge320_chain_k13_glob2_20260914.md) ·
  [288 chain hops glob2](../2_Experiments_Registry/run_reports/e25_bridge288_chain_k13_glob2_20260914.md) ·
  [768 select r=1 identity H=256 log](../2_Experiments_Registry/run_reports/e25_bridge1k_768_ip_id_h256_select_20260914.md) ·
  [640 select r=1 identity H=256 log](../2_Experiments_Registry/run_reports/e25_bridge1k_640_ip_id_h256_select_20260914.md) ·
  [704 select r=1 identity H=256 log](../2_Experiments_Registry/run_reports/e25_bridge1k_704_ip_id_h256_select_20260914.md) ·
  [672 select r=1 identity H=256 log](../2_Experiments_Registry/run_reports/e25_bridge1k_672_ip_id_h256_select_20260914.md) ·
  [688 select r=1 identity H=256 log](../2_Experiments_Registry/run_reports/e25_bridge1k_688_ip_id_h256_select_20260914.md) ·
  [696 select r=1 identity H=256 log](../2_Experiments_Registry/run_reports/e25_bridge1k_696_ip_id_h256_select_20260914.md) ·
  [692 select r=1 identity H=256 log](../2_Experiments_Registry/run_reports/e25_bridge1k_692_ip_id_h256_select_20260914.md) ·
  [696 select pack_stride 32](../2_Experiments_Registry/run_reports/e25_bridge1k_696_pack32_ip_id_h256_select_20260914.md) ·
  [696 select spread](../2_Experiments_Registry/run_reports/e25_bridge1k_696_spread_ip_id_h256_select_20260914.md) ·
  [696 select window 32](../2_Experiments_Registry/run_reports/e25_bridge1k_696_w32_ip_id_h256_select_20260914.md) ·
  [1536 INDEX r=16 H=256 log](../2_Experiments_Registry/run_reports/e25_bridge1k_1536_ip_r16_mean_far_copy_20260914.md) ·
  [256 chain hops glob2](../2_Experiments_Registry/run_reports/e25_bridge256_chain_k13_glob2_20260914.md) ·
  [512 chain hops glob2](../2_Experiments_Registry/run_reports/e25_bridge512_chain_k13_glob2_20260914.md) ·
  [384 chain hops glob2](../2_Experiments_Registry/run_reports/e25_bridge384_chain_k13_glob2_20260914.md) ·
  [320 chain hops glob2](../2_Experiments_Registry/run_reports/e25_bridge320_chain_k13_glob2_20260914.md) ·
  [288 chain hops glob2](../2_Experiments_Registry/run_reports/e25_bridge288_chain_k13_glob2_20260914.md).
  Do not relabel E18 scores as E21.
- **2026-09-13 — E24 BAPO DNA ladder (tiny + GPU bridge 512/1024; 4k S0 open).** Packed tiny
  seq=128 / 0.59M: E18 matches dense on positional `far_copy` (99.2% vs 99.4%) and recovers
  **0 bits** on keyed `recall`. GPU right-align INDEX: E18 **100% / 64 bits at seq=512**,
  **0 bits at seq=1024** while dense is 99.9%. 512 `recall_single` at a fixed offset is still
  E18 0 bits (not a find-the-mark failure). Advertised 4k spread is K1. Spec
  [E24](../experiments_specs/ahead/E24_e18_bapo_capability_ladder.md) ·
  [tiny report](../2_Experiments_Registry/run_reports/e24_tiny_bapo_ladder_20260913.md) ·
  [bridge report](../2_Experiments_Registry/run_reports/e24_bridge512_bapo_ladder_20260913.md) ·
  foundation [bapo_capability_ladder.md](../engineering_specs/bapo_capability_ladder.md).
  **Glyph (same day, not a new E-id):** A=4 DNA stays the exact-floor bandwidth control;
  iid filler is a toy noise model. A second family — typed vocab 16/32, Markov/Dyck/arith
  haystacks, reverse / every-k / filter / Dyck-close / fact-in-Markov — is specified and
  generated in `data/glyph_tasks.py`. Do not score E18 on Glyph until a dense S0 hits 75%.
  Named DNA scales `bridge` (512) and `bridge_1k` (1024) are the first GPU INDEX rungs;
  advertised 4k spread is K1. Note
  [glyph_capability_ladder.md](../4_Research_Notes/glyph_capability_ladder.md).
- **2026-09-12 — E23 Exclusive concept channel (spec ready, not launched).** E22 closed the same
  day (below): the array was live and diverse but CE read it as a document embedding — the far slots
  were worth 0.05 nats and a segment-only decoder matched the bet at half the compute. E23 keeps the
  `perceiver_concept` platform and changes the two things the diagnosis isolates: the cross-attention
  mask (`concept_xattn_scope=exclusive` — a token reads only slots that end before its raw segment,
  so the array is the *only* route for everything it carries) and the objective (natural-text CE with a
  ×8 weight on far-repeat tokens + 30% dense-label long-range rows: keyed recall, far copy, multi-hop
  variable tracking — tokens whose targets are *determined* by far content). Gates on recall/passkey
  and on the far marginal, with segment 0 as a built-in zero. Spec
  [E23](../experiments_specs/ahead/E23_exclusive_concept_channel.md). Odra is free; Polonez is
  free after the `goodwrite_ml` move. Dense control for S3 must be rerun from scratch (E22's was lost).
- **Instrument first (zero GPU-days, before launch):** the far-repeat token mask on the E22 mix —
  what share of tokens is far-copyable, and does that share justify the ×8 weight (target: far-repeat
  tokens ≥ 10% of the weighted loss). The number goes into the spec's Plan before the run starts.

## What we've explored so far
- **2026-09-14 — E25 GPU seq=288 chain_ordered --key_len 13 `--global_layers 2` (Odra, H=256 log).**
  Same 256 glob2 S1 PASS compressor at `--scale bridge --seq_len 288` (bridge
  default is 512; window 16 < gap 64). Dense **93.1% / 22.20 bits** @3200
  (**S0 PASS**; crossed 75% at 2200; best 98.2% @3000; 2.802M). E18 **26.8%
  / 0 bits** (live ~0). E21 **24.5% / 0 bits** (chance @800 and @3200; **S1
  FAIL** vs 0.75× dense 16.65; do not pass via 0.75×0). K2 PASS. Two global
  Blocks that compose hops at 256 do not compose them at 288 for E18 or E21.
  Dense wall **(288 S0 PASS, 320 K1]**. E21 wall **(256 S1 PASS, 288 FAIL]**.
  8k not run. Next: seq=272 `--key_len 13` `--global_layers 2` (do not run
  it). `--hops 1` is illegal.
  [report](../2_Experiments_Registry/run_reports/e25_bridge288_chain_k13_glob2_20260914.md).
- **2026-09-14 — E25 GPU seq=320 chain_ordered --key_len 13 `--global_layers 2` (Odra, H=256 log).**
  Same 256 glob2 S1 PASS compressor at `--scale bridge --seq_len 320` (bridge
  default is 512; window 16 < gap 64). Dense **26.2% / 0 bits** @3200
  (**S0 FAIL / K1**; chance every eval; best 27.9% @2400; 2.802M). e18 / e21
  skipped. glob=2 at 384 and 512 were also K1. Two global Blocks that compose
  hops at 256 do not make packed 320 dense-solvable. Wall **(256 S0 PASS,
  320 K1]**. Do not score E21. 8k not run. Next: seq=288 `--key_len 13`
  `--global_layers 2` (do not run it). `--hops 1` is illegal.
  [report](../2_Experiments_Registry/run_reports/e25_bridge320_chain_k13_glob2_20260914.md).
- **2026-09-14 — E25 GPU seq=384 chain_ordered --key_len 13 `--global_layers 2` (Odra, H=256 log).**
  Same 256 glob2 S1 PASS compressor at `--scale bridge --seq_len 384` (bridge
  default is 512; window 16 < gap 64). Dense **24.5% / 0 bits** @3200
  (**S0 FAIL / K1**; chance every eval; best 29.1% @1450; 2.802M). e18 / e21
  skipped. glob=2 at 512 was also K1. Two global Blocks that compose hops at
  256 do not make packed 384 dense-solvable. Wall **(256 S0 PASS, 384 K1]**.
  Do not score E21. 8k not run. Next: seq=320 `--key_len 13` `--global_layers 2`
  (do not run it). `--hops 1` is illegal.
  [report](../2_Experiments_Registry/run_reports/e25_bridge384_chain_k13_glob2_20260914.md).
- **2026-09-14 — E25 GPU seq=512 chain_ordered --key_len 13 `--global_layers 2` (Odra, H=256 log).**
  Same 256 glob2 S1 PASS compressor at `--scale bridge` seq=512 (no `--seq_len`;
  window 16 < gap 64). Dense **22.5% / 0 bits** @3200 (**S0 FAIL / K1**; chance
  every eval; 2.802M). e18 / e21 skipped. glob=1 at 512 was also K1. Two
  global Blocks that compose hops at 256 do not make packed 512 dense-solvable.
  Do not score E21. 8k not run. Next: seq=384 `--key_len 13` `--global_layers 2`
  (do not run it). `--hops 1` is illegal.
  [report](../2_Experiments_Registry/run_reports/e25_bridge512_chain_k13_glob2_20260914.md).
- **2026-09-14 — E25 GPU seq=696 select `--local_window 32` (Odra).**
  `--scale bridge_1k --seq_len 696` packed SELECT, inplace identity, remainder
  **off**, pack_stride **0**, `--evidence_align right`, `--local_window 32`
  (< min_gap 64), extra hops 0, update_slot_kv **off**, keep_local_swa
  **off**, `global_layers` 1, anchors **none**, `--hidden 256
  --global_logit_scale log`. Banner `gap=64 window=32`. QUERY **658**
  leftover **18** (658%16=2 vs 654%16=14). Dense **99.9% / 63.87 bits**
  @200 (**S0 PASS**). E18 **99.9% / 63.62 bits** @600 (live). E21
  **25.0% / 0 bits** @800 (**S1 FAIL** vs 0.75× E18 47.71; chance
  every eval). K2 **PASS**. 8k not run. Widening SWA does not rescue
  696. Wall still **(692, 696]**. Next: seq=694 right-align same
  identity recipe, pack_stride 0, default window 16 (do not run here).
  [report](../2_Experiments_Registry/run_reports/e25_bridge1k_696_w32_ip_id_h256_select_20260914.md).
- **2026-09-14 — E25 GPU seq=696 select `--evidence_align spread` (Odra).**
  `--scale bridge_1k --seq_len 696` packed SELECT, inplace identity, remainder
  **off**, pack_stride **0**, `--evidence_align spread`, extra hops 0,
  update_slot_kv **off**, keep_local_swa **off**, `global_layers` 1, anchors
  **none**, `--hidden 256 --global_logit_scale log`. Geometry seed 0: QUERY
  **658** leftover **18** on both aligns; right fact0=**528** left-filler
  **527**; spread fact0=**202** left-filler **201**. Dense **24.7% / 0
  bits** @3200 (**S0 FAIL / K1**; chance every eval). e18 / e21 skipped.
  8k not run. Spread at 696 is uncalibrated. Wall still **(692, 696]** on
  right-align. Next: seq=694 right-align same identity recipe, pack_stride
  0 (do not run here).
  [report](../2_Experiments_Registry/run_reports/e25_bridge1k_696_spread_ip_id_h256_select_20260914.md).
- **2026-09-14 — E25 GPU seq=696 select leftover drop `--message_pack_stride 32` (Odra).**
  `--scale bridge_1k --seq_len 696` packed SELECT, inplace identity, remainder
  **off**, `--message_pack_stride 32`, extra hops 0, update_slot_kv **off**,
  keep_local_swa **off**, `global_layers` 1, anchors **none**, `--hidden 256
  --global_logit_scale log`. Geometry: QUERY 650/654/658; leftover 10/14/18;
  2 complete 35-token packs; 693–696 are left-filler after BOS; r=1 remainder
  is a no-op; pack_stride 32 drops leftover → 640 exclusive slots at 692 and
  696. Dense **100% / 63.95 bits** @200 (**S0 PASS**). E18 **99.8% / 63.33
  bits** @550 (live). E21 **49.7% / 31.67 bits** @800 (**S1 FAIL** vs 0.75×
  E18 47.50, climbing) then **24.1% / 0 bits** @8000 (**S1 FAIL** vs 0.75×
  E18 47.94; chance floor). K2 **PASS**. Leftover count is not the 4-token
  cliff. Wall still **(692, 696]**. Do not 16k. Next: seq=696
  `--evidence_align spread` same identity recipe, pack_stride 0 (do not run
  here).
  [report](../2_Experiments_Registry/run_reports/e25_bridge1k_696_pack32_ip_id_h256_select_20260914.md).
- **2026-09-14 — E25 GPU seq=692 select_1decoy r=1 identity length bracket (Odra).**
  `--scale bridge_1k --seq_len 692` packed SELECT, inplace identity, remainder
  **off**, extra hops 0, update_slot_kv **off**, keep_local_swa **off**,
  `global_layers` 1, anchors **none**, `--hidden 256 --global_logit_scale log`
  (1024-passing width). Window 16 < gap 64; 32-token / 64-bit pack (scale
  packing, not seq override). Dense **100% / 63.97 bits** @200 (**S0 PASS**).
  E18 **100% / 63.89 bits** @350 (live). E21 **99.8% / 62.76 bits** @500
  (**S1 PASS** vs 0.75× E18 47.91). K2 **PASS**. Exclusive SELECT that
  passed at 688 is live at 692 and chance at 696. Wall **(692, 696]**. Do
  not extra-step (S1 already PASS). Next: seq=694 same recipe (do not run
  here).
  [report](../2_Experiments_Registry/run_reports/e25_bridge1k_692_ip_id_h256_select_20260914.md).
- **2026-09-14 — E25 GPU seq=696 select_1decoy r=1 identity length bracket (Odra).**
  `--scale bridge_1k --seq_len 696` packed SELECT, inplace identity, remainder
  **off**, extra hops 0, update_slot_kv **off**, keep_local_swa **off**,
  `global_layers` 1, anchors **none**, `--hidden 256 --global_logit_scale log`
  (1024-passing width). Window 16 < gap 64; 32-token / 64-bit pack. Dense
  **100% / 63.95 bits** @200 (**S0 PASS**). E18 **100% / 63.93 bits**
  @350 (live). E21 **25.0% / 0 bits** @800 (chance every eval; **S1 FAIL**
  vs 0.75× E18 47.95). K2 **PASS**. Exclusive SELECT that passed at 688 is
  chance at 696. Wall **(688, 696]**. Do not extra-step (floor). Next:
  seq=692 same recipe (do not run here).
  [report](../2_Experiments_Registry/run_reports/e25_bridge1k_696_ip_id_h256_select_20260914.md).
- **2026-09-14 — E25 GPU seq=688 select_1decoy r=1 identity length bracket (Odra).**
  `--scale bridge_1k --seq_len 688` packed SELECT, inplace identity, remainder
  **off**, extra hops 0, update_slot_kv **off**, keep_local_swa **off**,
  `global_layers` 1, anchors **none**, `--hidden 256 --global_logit_scale log`
  (1024-passing width). Window 16 < gap 64; 32-token / 64-bit pack. Dense
  **100% / 63.96 bits** @200 (**S0 PASS**, 8k JSON). E18 **100% / 63.96 bits**
  @350 (live). E21 **72.4% / 31.57 bits** @800 (climb, best 88.9% @750;
  short of S1 47.80) then **99.5% / 62.68 bits** @400 extra-step (**S1 PASS**
  vs 0.75× E18 47.97). K2 **PASS**. Exclusive SELECT that passed at 672 is
  live at 688 and chance at 704. Wall **(688, 704]**. Do not 16k (S1 already
  PASS). Next: seq=696 same recipe (do not run here).
  [report](../2_Experiments_Registry/run_reports/e25_bridge1k_688_ip_id_h256_select_20260914.md).
- **2026-09-14 — E25 GPU seq=672 select_1decoy r=1 identity length bracket (Odra).**
  `--scale bridge_1k --seq_len 672` packed SELECT, inplace identity, remainder
  **off**, extra hops 0, update_slot_kv **off**, keep_local_swa **off**,
  `global_layers` 1, anchors **none**, `--hidden 256 --global_logit_scale log`
  (1024-passing width). Window 16 < gap 64; 32-token / 64-bit pack. Dense
  **100% / 63.76 bits** @200 (**S0 PASS**). E18 **100% / 63.93 bits** @350
  (live). E21 **100% / 63.96 bits** @500 (chance through 400, 47.6% @450;
  **S1 PASS** vs 0.75× E18 47.95). K2 **PASS**. Exclusive SELECT that
  passed at 640 is live at 672 and chance at 704. Wall **(672, 704]**. Do
  not extra-step (S1 already PASS). Next: seq=688 same recipe (do not run
  here).
  [report](../2_Experiments_Registry/run_reports/e25_bridge1k_672_ip_id_h256_select_20260914.md).
- **2026-09-14 — E25 GPU seq=704 select_1decoy r=1 identity length bracket (Odra).**
  `--scale bridge_1k --seq_len 704` packed SELECT, inplace identity, remainder
  **off**, extra hops 0, update_slot_kv **off**, keep_local_swa **off**,
  `global_layers` 1, anchors **none**, `--hidden 256 --global_logit_scale log`
  (1024-passing width). Window 16 < gap 64; 32-token / 64-bit pack. Dense
  **100% / 63.95 bits** @300 (**S0 PASS**). E18 **100% / 63.89 bits** @350
  (live). E21 **25.7% / 0 bits** @800 (chance every eval; **S1 FAIL** vs
  0.75× E18 47.92). K2 **PASS**. Exclusive SELECT that passed at 640 is
  chance at 704. Wall **(640, 704]**. Do not extra-step (floor). Next:
  seq=672 same recipe (do not run here).
  [report](../2_Experiments_Registry/run_reports/e25_bridge1k_704_ip_id_h256_select_20260914.md).
- **2026-09-14 — E25 GPU seq=640 select_1decoy r=1 identity length bracket (Odra).**
  `--scale bridge_1k --seq_len 640` packed SELECT, inplace identity, remainder
  **off**, extra hops 0, update_slot_kv **off**, keep_local_swa **off**,
  `global_layers` 1, anchors **none**, `--hidden 256 --global_logit_scale log`
  (1024-passing width). Window 16 < gap 64; 32-token / 64-bit pack. Dense
  **99.8% / 63.76 bits** @200 (**S0 PASS**). E18 **100% / 63.91 bits** @300
  (live). E21 **100% / 63.95 bits** @450 (chance through 350, 51.3% @400;
  **S1 PASS** vs 0.75× E18 47.93). K2 **PASS**. Exclusive SELECT that
  passed at 512 is live at 640 and chance at 768. Wall **(640, 768]**. Do
  not extra-step (S1 already PASS). Next: seq=704 same recipe (do
  not run here).
  [report](../2_Experiments_Registry/run_reports/e25_bridge1k_640_ip_id_h256_select_20260914.md).
- **2026-09-14 — E25 GPU seq=768 select_1decoy r=1 identity length bracket (Odra).**
  `--scale bridge_1k --seq_len 768` packed SELECT, inplace identity, remainder
  **off**, extra hops 0, update_slot_kv **off**, keep_local_swa **off**,
  `global_layers` 1, anchors **none**, `--hidden 256 --global_logit_scale log`
  (1024-passing width). Window 16 < gap 64; 32-token / 64-bit pack. Dense
  **100% / 63.95 bits** @250 (**S0 PASS**). E18 **100% / 63.97 bits** @350
  (live). E21 **25.6% / 0.01 bits** @800 (chance every eval; **S1 FAIL** vs
  0.75× E18 47.97). K2 **PASS**. Exclusive SELECT that passed at 512 is
  chance at 768. Do not extra-step (floor). Next: seq=640 same recipe (do
  not run here).
  [report](../2_Experiments_Registry/run_reports/e25_bridge1k_768_ip_id_h256_select_20260914.md).
- **2026-09-14 — E25 GPU seq=1024 select_1decoy QUERY-side anchors (Odra).**
  `--scale bridge_1k` packed SELECT, inplace identity, remainder **off**,
  `--message_global_anchors query_side`, extra hops 0, keep_local_swa **off**,
  `global_layers` 1, `--hidden 256 --global_logit_scale log`. Existing
  `query_nbhd` is prefix-before-QUERY (r=1 slots, extra 0). `query_side` leaks
  QUERY + 2 asked-key symbols + ANSWER (**4 vs seq=1024**, extra non-slot
  count 4). Dense **100% / 63.96 bits** @250 (**S0 PASS**). E18 **75.1% /
  46.77 bits** @800 (live late-click). E21 **25.7% / 0 bits** @800 (chance
  every eval; **S1 FAIL** vs 0.75× E18 35.07). K2 **PASS**. QUERY-side leak
  does not rescue 1024 SELECT. Do not extra-step (floor). Do not restore raw
  global KV. Default `--message_global_anchors` stays `none`. Next: stop
  1024 SELECT architecture hunts.
  [report](../2_Experiments_Registry/run_reports/e25_bridge1k_ip_id_qside_select_20260914.md).
- **2026-09-14 — E25 GPU seq=1024 select_1decoy second exclusive global layer (Odra).**
  `--scale bridge_1k` packed SELECT, inplace identity, remainder **off**,
  `--global_layers 2`, extra hops 0, update_slot_kv **off**, keep_local_swa
  **off**, anchors **none**, `--hidden 256 --global_logit_scale log`. Existing
  flag (default 1); two full exclusive attend+FFN Blocks, not extra hops and
  not extra SWA (`stack_layers` already 2). Dense **100% / 63.95 bits** @200
  (**S0 PASS**). E18 **100% / 63.97 bits** @750 (live). E21 **25.7% / 0 bits**
  @800 (chance every eval; **S1 FAIL** vs 0.75× E18 47.97). K2 **PASS**. A
  second exclusive global Block does not rescue 1024 SELECT. Do not extra-step
  (floor). Do not restore raw global KV. Default `--global_layers` stays 1.
  Next: stop 1024 SELECT architecture hunts.
  [report](../2_Experiments_Registry/run_reports/e25_bridge1k_ip_id_glob2_select_20260914.md).
- **2026-09-14 — E25 GPU seq=1024 select_1decoy extra hop + unfrozen slot K/V (Odra).**
  `--scale bridge_1k` packed SELECT, inplace identity, remainder **off**,
  `--message_extra_slot_attends 1 --message_update_slot_kv`, keep_local_swa
  **off**, anchors **none**, `--hidden 256 --global_logit_scale log`.
  Inspection: extra hop is two sequential attends with updated Q and frozen
  K/V (QUERY Q can contain type); not a no-op. Dense **100% / 63.95 bits**
  @250 (**S0 PASS**). E18 **100% / 63.80 bits** @450 (live). E21 **25.7% /
  0 bits** @800 (chance every eval; **S1 FAIL** vs 0.75× E18 47.85). K2
  **PASS**. Rewriting exclusive slot K/V between hops does not rescue 1024
  SELECT. Do not extra-step (floor). Do not restore raw global KV. Default
  `--message_update_slot_kv` stays off. Next: stop 1024 SELECT architecture
  hunts.
  [report](../2_Experiments_Registry/run_reports/e25_bridge1k_ip_id_updatekv_select_20260914.md).
- **2026-09-14 — E25 GPU seq=1024 select_1decoy type_marks 8k extra-step (Odra).**
  Same identity + `--message_global_anchors type_marks` recipe as the 1050
  hunt; `--arch e18 e21 e18_local --steps 8000 --no-dense_first --k1_mult 1`.
  Dense skipped (prior S0 **63.98**). E18 **100% / 63.96 bits** @3250 (live;
  prior this-JSON E18 0 at 1050 was short). E21 **0 bits** @800 and @1050
  (31.90 @1050 did not replicate), climb @4850, plateau **14.43 bits** @7100
  / 42.9% @7850, final **39.3% / 6.48 bits** @8000 (**S1 FAIL** vs 0.75× live
  E18 47.97). Did not climb past 31.90. K2 **PASS**. Type-mark leak still a
  no-op at r=1 identity. Do not 16k. Next: stop 1024 SELECT extra-steps.
  [report](../2_Experiments_Registry/run_reports/e25_bridge1k_ip_id_anchors_select_s8k_20260914.md).
- **2026-09-14 — E25 GPU seq=1024 select_1decoy r=1 identity type_marks anchors H=256 SSMax log (Odra).**
  `--scale bridge_1k` packed SELECT, inplace identity, remainder **off**,
  `--message_global_anchors type_marks`, extra hops 0, keep_local_swa **off**,
  `--hidden 256 --global_logit_scale log`. Dense **100% / 63.98 bits** @1050
  (**S0 PASS**). E18 **24.9% / 0 bits** (this replica; prior live ~63.92).
  E21 **25.7% / 0 bits** @800 then **48.1% / 31.90 bits** @1050 (**S1 FAIL**
  vs 0.75× dense 47.99; do not pass via 0.75×0). K2 **PASS**. Type-mark leak
  2 tokens vs seq=1024; at r=1 identity extra non-slot count is 0. Do not
  extra-step that 800 floor as a new architecture. 8k extra-step ran as
  `e25_1k_ip_id_anchors_select_s8k` (live E18 63.96; E21 6.48 S1 FAIL).
  Default `--message_global_anchors` stays `none`.
  [report](../2_Experiments_Registry/run_reports/e25_bridge1k_ip_id_anchors_select_20260914.md).
- **2026-09-14 — E25 GPU seq=1024 select_1decoy r=1 identity extra exclusive hop H=256 SSMax log (Odra).**
  `--scale bridge_1k` packed SELECT, inplace identity, remainder **off**,
  `--message_extra_slot_attends 1`, keep_local_swa **off**, `--hidden 256
  --global_logit_scale log`. Dense **100% / 63.95 bits** @200 (**S0 PASS**).
  E18 **100% / 63.92 bits** @350 (live). E21 **25.7% / 0 bits** @800 (chance
  every eval; **S1 FAIL** vs 0.75× E18 47.94). K2 **PASS**. Inspection: slots
  already post-pre-SWA; type-cue tokens already in r=1 identity slots. A
  second exclusive hop over frozen slots does not bind type-cue at 1024.
  Do not extra-step (floor). Do not restore raw global KV. Default
  `--message_extra_slot_attends` stays 0. Next: sparse exclusive-plus-anchors
  (QUERY neighborhood or type markers only), still not full raw prefix.
  [report](../2_Experiments_Registry/run_reports/e25_bridge1k_ip_id_extrahop_select_20260914.md).
- **2026-09-14 — E25 GPU seq=1024 select_1decoy r=1 identity SWA-unsever H=256 SSMax log (Odra).**
  `--scale bridge_1k` packed SELECT, inplace identity, remainder **off**,
  `--message_keep_local_swa`, `--hidden 256 --global_logit_scale log`. Dense
  **100% / 63.03 bits** @200 (**S0 PASS**). E18 **100% / 63.94 bits** @400
  (live). E21 **25.8% / 0.01 bits** @800 (chance every eval; **S1 FAIL** vs
  0.75× E18 47.95). K2 **PASS**. Inspection: r=1 identity already covers every
  sender token on the exclusive global read. Unsevering SWA does not bind
  type-cue at 1024 (window 16 cannot reach gap 100). Do not extra-step
  (floor). Do not restore raw global KV. Default `--message_keep_local_swa`
  stays off. Next: stop 1024 SELECT knobs.
  [report](../2_Experiments_Registry/run_reports/e25_bridge1k_ip_id_keepswa_select_20260914.md).
- **2026-09-14 — E25 GPU seq=2048 far_copy r=16 rem-off H=256 SSMax log (Odra).**
  `--scale bridge_1k --seq_len 2048` packed INDEX, inplace identity mean, remainder
  **off**, no raw_kv, `--hidden 256 --global_logit_scale log`, batch 32. Dense
  **100% / 63.79 bits** @250 (**S0 PASS**). E18 **23.9% / 0 bits** (live ~0).
  E21 **25.0% / 0 bits** @800 (chance every eval; **S1 FAIL** vs 0.75× dense
  47.84). Do not pass S1 via 0.75×0. K2 **PASS**. Exclusive INDEX that passed
  at 1024 is chance at 2048 with the same gap=64. No new scale enum. Do not
  extra-step (floor). Do not remainder-on. Do not 2048 H=512. Next: stop INDEX
  length extra-steps (do not seq=1536).
  [report](../2_Experiments_Registry/run_reports/e25_bridge1k_2k_ip_r16_mean_far_copy_20260914.md).
- **2026-09-14 — E25 GPU seq=256 chain_ordered --key_len 13 `--global_layers 2` (Odra, H=256 log).**
  Same FAIL recipe plus two sequential global attend+FFN Blocks (extra hops 0).
  Window 16 < gap 64. Dense **99.2% / 25.46 bits** @1800 (**S0 PASS**;
  recalibrated at glob=2). E18 **42.3% / 5.56 bits** (live, climbing). E21
  **24.0% / 0.02 bits** @800 then **99.6% / 25.85 bits** @1650 (**S1 PASS**
  vs 0.75× E18 4.17 and vs 0.75× dense 19.10). K2 **PASS**. glob=1 was E18/E21
  0 bits. Exclusive identity composes hops with two global blocks and beats
  live E18. 8k not run. Next: seq=512 `--key_len 13` `--global_layers 2`.
  `--hops 1` is illegal. Do not run it.
  [report](../2_Experiments_Registry/run_reports/e25_bridge256_chain_k13_glob2_20260914.md).
- **2026-09-14 — E25 GPU seq=1536 far_copy r=16 rem-off H=256 SSMax log (Odra).**
  `--scale bridge_1k --seq_len 1536` packed INDEX, inplace identity mean, remainder
  **off**, no raw_kv, `--hidden 256 --global_logit_scale log`, batch 32. Dense
  **99.1% / 62.41 bits** @300 (**S0 PASS**). E18 **25.8% / 0 bits** (live ~0).
  E21 **24.8% / 0 bits** @800 (chance every eval; **S1 FAIL** vs 0.75× dense
  46.81). Do not pass S1 via 0.75×0. K2 **PASS**. Exclusive INDEX that passed
  at 1024 is chance at 1536 with the same gap=64 / leftover 12. Shared wall
  **(1024 PASS, 1536 FAIL]**. No new scale enum. Do not extra-step (floor).
  Do not remainder-on. Next: stop INDEX length extra-steps (do not seq=1280).
  [report](../2_Experiments_Registry/run_reports/e25_bridge1k_1536_ip_r16_mean_far_copy_20260914.md).
- **2026-09-14 — E25 GPU seq=4096 far_copy r=16 rem-off H=256 SSMax log (Odra).**
  `--scale medium` packed INDEX, inplace identity mean, remainder **off**, no
  raw_kv, `--hidden 256 --global_logit_scale log`, batch 8. Dense **100% /
  63.94 bits** @400 (**S0 PASS**). E18 **25.7% / 0 bits** (live ~0). E21
  **25.7% / 0 bits** @800 (chance every eval; **S1 FAIL** vs 0.75× dense
  47.96). Do not pass S1 via 0.75×0. K2 **PASS**. Exclusive INDEX that passed
  at 1024 is chance at 4k. No 4k scale code change. Do not extra-step
  (floor). Do not remainder-on. Do not 4k H=512.
  [report](../2_Experiments_Registry/run_reports/e25_medium_4k_ip_r16_mean_far_copy_20260914.md).
- **2026-09-14 — E25 GPU seq=1024 select_1decoy r=1 identity H=256 SSMax log (Odra).**
  Inplace hard identity, remainder **off**, no raw_kv, `--hidden 256 --global_logit_scale log`.
  Dense **100% / 63.96 bits** (**S0 PASS**). E18 **100% / 63.80 bits** (live).
  E21 **25.7% / 0 bits** @800 (chance every eval; **S1 FAIL** vs 0.75× E18
  47.85 and vs 0.75× dense 47.97). K2 **PASS**. Exclusive SELECT is dead at
  1024 even with identity slots. The 512 MATCH r=16 mean → identity split
  does not hold. Do not extra-step (floor). Next: stop 1024 SELECT.
  [report](../2_Experiments_Registry/run_reports/e25_bridge1k_ip_id_h256_select_20260914.md).
- **2026-09-14 — E25 GPU seq=1024 select_1decoy r=8 remainder-off H=256 SSMax log (Odra).**
  Inplace identity, remainder **off**, no raw_kv, `--hidden 256 --global_logit_scale log`.
  Dense **100% / 63.96 bits** (**S0 PASS**). E18 **100% / 63.86 bits** (live).
  E21 **25.7% / 0 bits** @800 (chance every eval; **S1 FAIL** vs 0.75× E18
  47.89 and vs 0.75× dense 47.97). K2 **PASS**. SELECT does not scale with
  MATCH at 1024. Do not extra-step (floor). Next: 1024 SELECT r=1 identity.
  [report](../2_Experiments_Registry/run_reports/e25_bridge1k_ip_r8_h256_select_20260914.md).
- **2026-09-14 — E25 GPU seq=1024 recall_single r=8 remainder-off H=256 SSMax log (Odra).**
  Inplace identity, remainder **off**, no raw_kv, `--hidden 256 --global_logit_scale log`.
  Dense **99.4% / 62.90 bits** (**S0 PASS**). E18 **100% / 62.61 bits** (live).
  E21 **38.1%** @800 then **49.0% / 24.35 bits** @3200 (climbing) then **96.5%
  / 60.35 bits** @8000 (**S1 PASS** vs 0.75× E18 46.96 and vs 0.75× dense
  47.18; best 97.6% @7500). K2 **PASS**. MATCH at 1024 needs INDEX-passing
  width, not a new pooler. Next: 1024 select_1decoy r=8 rem-off H=256 log.
  Do not 16k.
  [report](../2_Experiments_Registry/run_reports/e25_bridge1k_ip_r8_h256_recall_20260914.md).
- **2026-09-14 — E25 GPU seq=1024 recall_single r=8 remainder-off frozen mean (Odra, H=128).**
  Inplace identity, remainder **off**, no raw_kv, `logit_scale=none`. Dense
  **100% / 63.73 bits** (**S0 PASS**, not K1). E18 **99.5% / 62.64 bits**
  (live in 8k JSON). E21 **43.1% / 13.33 bits** @800 (climbing) then **26.8% /
  0.01 bits** @8000 (**S1 FAIL** vs 0.75× E18 46.98 and vs 0.75× dense 47.79).
  K2 **PASS**. MATCH does not scale to 1024 on the 512 default H=128 recipe.
  Next: 1024 recall r=8 rem-off H=256 SSMax log. Do not 16k.
  [report](../2_Experiments_Registry/run_reports/e25_bridge1k_ip_r8_recall_20260914.md).
- **2026-09-14 — E25 GPU seq=512 select_1decoy r=8 remainder-off frozen mean (Odra, H=128).**
  Inplace identity, remainder **off**, no raw_kv. Dense **100% / 47.91 bits**
  (**S0 PASS**). E18 **100% / 47.90 bits**. E21 **43.3% / 11.73 bits** @800
  (climbing) then **96.4% / 45.53 bits** @8000 (**S1 PASS** vs 0.75× E18 35.93
  and vs 0.75× dense 35.94; best 97.9% @7450). K2 **PASS**. SELECT works on
  the default MATCH r=8 grid; remainder-on is not required. Next: do not
  another SELECT r. Do not 16k.
  [report](../2_Experiments_Registry/run_reports/e25_bridge512_ip_r8_select_20260914.md).
- **2026-09-14 — E25 GPU seq=512 select_1decoy r=12 remainder-on frozen mean (Odra, H=128).**
  Inplace identity, `--message_pool_remainder`, no raw_kv. Dense **100% / 47.93
  bits** (**S0 PASS**). E18 **100% / 47.90 bits**. E21 **45.1% / 8.92 bits**
  @800 (climbing) then **93.8% / 43.29 bits** @8000 (**S1 PASS** vs 0.75× E18
  35.92 and vs 0.75× dense 35.95; best 98.0% @7900). K2 **PASS**. SELECT
  clears at r=12 rem-on; r=16 is the SELECT S1 edge. Next: 512 select_1decoy
  r=8 remainder-off. Do not 16k.
  [report](../2_Experiments_Registry/run_reports/e25_bridge512_ip_r12_rem_select_20260914.md).
- **2026-09-14 — E25 GPU seq=512 select_1decoy r=16 remainder-on frozen mean (Odra, H=128).**
  Inplace identity, `--message_pool_remainder`, no raw_kv. Dense **99.8% / 47.71
  bits** (**S0 PASS**). E18 **100% / 47.94 bits**. E21 **35.6% / 4.56 bits**
  @800 (climbing) then **84.5% / 35.94 bits** @8000 (**S1 FAIL** vs 0.75× E18
  35.95 by 0.016 bits; **PASS** vs 0.75× dense 35.78; best 91.4% @7950). K2
  **PASS**. Live, not chance. Type-cue slightly harder than MATCH on the same
  pooler. Next: 512 select_1decoy r=12 remainder-on. Do not 16k.
  [report](../2_Experiments_Registry/run_reports/e25_bridge512_ip_r16_rem_select_20260914.md).
- **2026-09-14 — E25 GPU seq=512 recall_single r=16 remainder-on frozen mean (Odra, H=128).**
  Inplace identity, `--message_pool_remainder`, no raw_kv. Dense **100% / 47.95
  bits** (**S0 PASS**). E18 **100% / 47.86 bits**. E21 **36.4% / 4.72 bits**
  @800 (climbing) then **92.6% / 40.92 bits** @8000 (**S1 PASS** vs 0.75× E18
  35.89 and vs 0.75× dense 35.96; best 96.4% @7750). K2 **PASS**. Remainder-off
  r=16 was 0 bits chance. Leftover/alignment was the MATCH wall through r=16.
  Next: 512 select_1decoy r=16 remainder-on. Do not 16k.
  [report](../2_Experiments_Registry/run_reports/e25_bridge512_ip_r16_rem_recall_20260914.md).
- **2026-09-14 — E25 GPU seq=512 recall_single r=12 remainder-on frozen mean (Odra, H=128).**
  Inplace identity, `--message_pool_remainder`, no raw_kv. Dense **100% / 47.87
  bits** (**S0 PASS**). E18 **100% / 47.96 bits**. E21 **41.0% / 6.14 bits**
  @800 (climbing) then **97.9% / 46.06 bits** @8000 (**S1 PASS** vs 0.75× E18
  35.97 and vs 0.75× dense 35.90; best 98.2% @7850). K2 **PASS**. Remainder-off
  r=12 was 34.33 S1 FAIL. Leftover 8 was the 1.64-bit miss, not the r=12 grid.
  Next: 512 recall r=16 remainder-on. Do not 16k.
  [report](../2_Experiments_Registry/run_reports/e25_bridge512_ip_r12_rem_recall_20260914.md).
- **2026-09-14 — E25 GPU seq=512 recall_single r=10 remainder-on frozen mean (Odra, H=128).**
  Inplace identity, `--message_pool_remainder`, no raw_kv. Dense **99.9% / 47.88
  bits** (**S0 PASS**). E18 **100% / 47.79 bits**. E21 **44.2% / 8.65 bits**
  @800 (climbing) then **95.1% / 44.21 bits** @8000 (**S1 PASS** vs 0.75× E18
  35.85 and vs 0.75× dense 35.91; best 96.4% @7800). K2 **PASS**. Remainder-off
  r=10 was 0 bits at 800. Leftover 2 tokens were the floor, not the r=10 grid.
  Next: 512 recall r=12 remainder-on. Do not 16k.
  [report](../2_Experiments_Registry/run_reports/e25_bridge512_ip_r10_rem_recall_20260914.md).
- **2026-09-14 — E25 GPU seq=512 recall_single r=10 in-place frozen mean (Odra, H=128).**
  Inplace identity, remainder off, no raw_kv. Dense **99.9% / 47.89 bits** (**S0
  PASS**). E18 **100% / 47.94 bits**. E21 **25.5% / 0 bits** @800 (**S1 FAIL** vs
  0.75× E18 35.96 and vs 0.75× dense 35.92). K2 **PASS**. Chance every eval; CE
  at ln(4). No extra-step (floor). Unexpected vs r=12 live at 800. Pooling
  width is not monotone. Stop stacking MATCH r-sweeps.
  [report](../2_Experiments_Registry/run_reports/e25_bridge512_ip_r10_mean_recall_20260914.md).
- **2026-09-14 — E25 GPU seq=512 recall_single r=12 in-place frozen mean (Odra, H=128).**
  Inplace identity, remainder off, no raw_kv. Dense **100% / 47.91 bits** (**S0
  PASS**). E18 **100% / 47.96 bits**. E21 **39.8% / 5.79 bits** @800 (climbing)
  then **81.7% / 34.33 bits** @8000 (**S1 FAIL** vs 0.75× E18 35.97 and vs 0.75×
  dense 35.93). K2 **PASS**. Not chance. S1-passing MATCH dies between r=8 and
  r=12 at the 8k budget. Next: 512 recall r=10 frozen mean. Do not 16k.
  [report](../2_Experiments_Registry/run_reports/e25_bridge512_ip_r12_mean_recall_20260914.md).
- **2026-09-14 — E25 GPU seq=512 recall_single r=8 in-place frozen mean (Odra, H=128).**
  Inplace identity, no raw_kv. Dense **100% / 47.95 bits** (**S0 PASS**). E18
  **100% / 47.96 bits**. E21 **56.5% / 15.90 bits** @800 (climbing) then
  **99.1% / 47.16 bits** @7100 (**S1 PASS** vs 0.75× E18 35.97 and vs 0.75×
  dense 35.96). K2 **PASS**. MATCH survives 8-token means; wall is between r=8
  and r=16. Next: 512 recall r=12 frozen mean. Do not unfreeze `u`/`delta`.
  [report](../2_Experiments_Registry/run_reports/e25_bridge512_ip_r8_mean_recall_20260914.md).
- **2026-09-14 — E25 GPU seq=512 recall_single r=4 in-place frozen mean (Odra, H=128).**
  Inplace identity, no raw_kv. Dense **100% / 47.96 bits** (**S0 PASS**). E18
  **100% / 47.96 bits**. E21 **78.3% / 29.76 bits** @800 (climbing) then
  **99.3% / 47.04 bits** @1900 (**S1 PASS** vs 0.75× E18 35.97 and vs 0.75×
  dense 35.97). K2 **PASS**. MATCH survives 4-token means (r=16 was 0 bits).
  Next: 512 recall r=8 frozen mean. Do not unfreeze `u`/`delta`.
  [report](../2_Experiments_Registry/run_reports/e25_bridge512_ip_r4_mean_recall_20260914.md).
- **2026-09-14 — E25 GPU seq=256 chain_ordered --key_len 13 (Odra, H=256 log).**
  Bridge `--seq_len 256`, min_gap 64 > window 16, tiny 26-bit keys, hops=2. Dense
  **96.4% / 24.12 bits** @3200 (**S0 PASS**). E18 **26.2% / 0 bits**. E21
  identity **24.9% / 0 bits** (chance floor). K2 **PASS**. Do not extra-step.
  Do not pass S1 via 0.75×0; vs 0.75× dense 18.09 **FAIL**. Dense hops wall is
  between 256 and 512. Next: stop hops extra-steps (not seq=192).
  [report](../2_Experiments_Registry/run_reports/e25_bridge256_chain_k13_20260914.md).
- **2026-09-14 — E25 GPU seq=512 chain_ordered --key_len 13 dense S0 (Odra, H=256 log).**
  Tiny packed 26-bit keys, hops=2, right-align. Dense **22.5% / 0 bits** @3200
  (**K1**; chance floor). e18 / e21 skipped. Key packing is not the remaining
  S0 lever — 512 context is. Next: seq=256 `--key_len 13` hops=2. Do not bump width.
  `--hops 1` is illegal. Do not run it.
  [report](../2_Experiments_Registry/run_reports/e25_bridge512_chain_k13_20260914.md).
- **2026-09-14 — E25 GPU seq=512 chain_ordered dense S0 hunt (Odra, H=256 log + H=512 MHA).**
  H=256 SSMax log: dense **32.4% / 1.10 bits** @3200 (**K1**). One bump H=512
  `--kv_heads 0`: dense **24.1% / 0 bits** @3200 (**K1**, harder floor). e18 / e21
  skipped. Packed 48-bit keys are not dense-solvable at allowed widths. Do not
  invent a third width. Next: 512 chain `--key_len 13` (tiny pack) H=256 log.
  [report](../2_Experiments_Registry/run_reports/e25_bridge512_chain_s0_20260914.md).
- **2026-09-14 — E25 GPU seq=512 chain_ordered r=1 in-place hard identity (Odra, H=128).**
  Dense **24.2% / 0 bits** @3200 (chance every eval; CE at ln(4)). **K1 / S0 FAIL.**
  e18 / e21 / e18_local skipped. S1 and K2 not scored. Packed 512 chain is not
  dense-solvable at this width — instrument failure, not a hops verdict. Do not
  extra-step (floor, not a ~50% plateau). Next: 512 chain dense S0 hunt H=256
  `--global_logit_scale log`. Skip E21 until S0.
  [report](../2_Experiments_Registry/run_reports/e25_bridge512_ip_id_chain_20260914.md).
- **2026-09-14 — E25 GPU seq=512 select_1decoy r=1 in-place hard identity (Odra, H=128).**
  Dense **99.9% / 47.80 bits** (S0). E18 **100% / 47.92 bits** @400 (live). E21
  identity **100% / 47.98 bits / flow 1.000** @700 (early stop). S1 **PASS**
  (47.98 vs 35.94). Content vs 0.75× dense **PASS** (47.98 vs 35.85). K2 **PASS**.
  Exclusive identity does type-cue select, not only MATCH/INDEX. Do not 8k.
  Next: 512 `chain_ordered` r=1 identity.
  [report](../2_Experiments_Registry/run_reports/e25_bridge512_ip_id_select_20260914.md).
- **2026-09-14 — E25 GPU seq=512 recall_single r=1 in-place hard identity (Odra, H=128).**
  Dense **99.9% / 47.69 bits** (S0). E18 **99.4% / 46.67 bits** @350 (live). E21
  identity **91.6% / 43.08 bits / flow 0.897** @800 (best 92.1% @750). S1 **PASS**
  (43.08 vs 35.00). Content vs 0.75× dense **PASS** (43.08 vs 35.77). K2 **PASS**.
  Exclusive identity binds a key; r=16 pooling is the MATCH killer. Do not 8k.
  Next: 512 `select_1decoy` r=1 identity.
  [report](../2_Experiments_Registry/run_reports/e25_bridge512_ip_id_recall_20260914.md).
- **2026-09-14 — E25 GPU seq=512 recall_single r=16 in-place frozen mean (Odra, H=128).**
  Dense **99.0% / 46.97 bits** (S0). E18 **100% / 47.94 bits** @400 (live in this JSON).
  E21 frozen mean **25.5% / 0.00 bits / flow 0** @800 (chance floor). S1 **FAIL**
  (0 vs 35.95). Content vs 0.75× dense **FAIL** (0 vs 35.23). K2 **PASS**. Frozen
  means carry INDEX, not MATCH, at 512. Do not extra-step. Next: 512 recall r=1 identity.
  [report](../2_Experiments_Registry/run_reports/e25_bridge512_ip_r16_mean_recall_20260914.md).
- **2026-09-14 — E25 GPU seq=1024 far_copy r=16 in-place frozen mean (Odra, H=256 SSMax log).**
  Dense **100% / 63.95 bits** (S0). E18 **99.7% / 63.12 bits** @2750 (live; not dilution).
  E21 frozen mean **91.2% / 53.82 bits / flow 0.841** @8000 (68.8% / 32.37 @3200,
  climbing; best 92.7% @7850). S1 **PASS** (53.82 vs 47.34). K2 **PASS**. Frozen
  16-token means carry INDEX at 1024. Do not 16k. Next: 512 `recall_single` frozen mean.
  [report](../2_Experiments_Registry/run_reports/e25_bridge1k_ip_r16_mean_far_copy_20260914.md).
- **2026-09-14 — E25 GPU seq=512 far_copy r=16 in-place learned pool (Odra).** Dense
  **100% / 63.94 bits** (S0). E18 **100% / 63.94 bits**. E21 learned pool **25.1% /
  0.00 bits / flow 0** @800 (chance floor). S1 **FAIL**. K2 **PASS**. Learning wrecks
  r=16; frozen mean stays the working recipe. Do not extra-step.
  [report](../2_Experiments_Registry/run_reports/e25_bridge512_ip_r16_learned_far_copy_20260914.md).
- **2026-09-14 — E25 GPU seq=512 far_copy r=16 in-place frozen mean-pool (Odra).** Dense
  **100% / 63.91 bits** (S0). E18 **100% / 63.94 bits**. E21 frozen mean **84.3% /
  47.36 bits / flow 0.740** @8000 (42.4% / 8.18 @800, climbing; best 88.7% @7900). S1
  **NEAR-PASS** (47.36 vs 47.95). K2 **PASS**. 16-token means carry INDEX at 512, slow
  like tiny. Next: inplace r=16 learned pool.
  [report](../2_Experiments_Registry/run_reports/e25_bridge512_ip_r16_mean_far_copy_20260914.md).
- **2026-09-13 — E25 GPU seq=512 far_copy r=1 in-place hard identity (Odra).** Dense
  **100% / 63.92 bits** (S0). E18 **100% / 63.94 bits**. E21 inplace identity **99.8% /
  62.64 bits / flow 0.979** @750. S1 **PASS**. K2 **PASS**. Scatter is fine; learned
  `delta` wrecked r=1 (one-token softmax ignores `u`). Next: inplace r=16 frozen mean-pool.
  [report](../2_Experiments_Registry/run_reports/e25_bridge512_ip_id_far_copy_20260913.md).
- **2026-09-13 — E25 GPU seq=512 far_copy r=1 in-place raw KV (Odra).** Dense **100% /
  63.95 bits** (S0). E18 **100% / 63.90 bits**. E21 inplace raw KV **99.5% / 63.29 bits /
  flow 0.989** @450. S1 **PASS**. K2 **PASS**. Compressor values (even r=1) are the
  remaining 512 killer; exclusive `~replace` is not.
  [report](../2_Experiments_Registry/run_reports/e25_bridge512_ip_rawkv_far_copy_20260913.md).
- **2026-09-13 — E25 GPU seq=512 far_copy r=1 in-place (Odra).** Dense **100% / 63.95 bits**
  (S0). E18 **100% / 63.93 bits**. E21 r=1 in-place **25.1% / 0.00 bits** @800. S1 **FAIL**.
  K2 **PASS**. K3: chance, not climbing. KV_LEN=S is not enough; do not stack r=16.
  Next: exclusive mask vs in-place values (not flex-vs-sdpa first; both this FAIL and the
  raw PASS used sdpa).
  [report](../2_Experiments_Registry/run_reports/e25_bridge512_ip_r1_far_copy_20260913.md).
- **2026-09-13 — E25 GPU seq=512 far_copy raw override (Odra).** Dense **100% / 63.94 bits**
  (S0). E18 **99.6% / 62.23 bits**. E21 raw **100% / 63.96 bits / flow 0.999** @450. S1
  **PASS**. Local cut still on (`e18_local` 0 bits). Concat slot routing is the 512 killer.
  [report](../2_Experiments_Registry/run_reports/e25_bridge512_raw_far_copy_20260913.md).
- **2026-09-13 — E25 GPU seq=512 far_copy r=1 (Odra, identity slots).** Dense **100% / 63.95
  bits** (S0). E18 **100% / 63.94 bits**. E21 **25.1% / 0.00 bits**. Concat identity slots
  still chance. Superseded as “document-start is the killer” by the raw-override pass.
  [report](../2_Experiments_Registry/run_reports/e25_bridge512_r1_far_copy_20260913.md).
- **2026-09-13 — E25 GPU seq=512 far_copy r=64 (Odra, ~8 slots).** Dense **100% / 63.3 bits**
  (S0). E18 **100% / 63.9 bits**. E21 **25.1% / 0.01 bits**. Same floor as r=16. Slot count
  is not the 512 wall. `e18_local` 0 bits (K2).
  [report](../2_Experiments_Registry/run_reports/e25_bridge512_r64_far_copy_20260913.md).
- **2026-09-13 — E25 GPU seq=512 far_copy (Odra, r=16, right-align).** Dense **99.5% / 61.2
  bits** (S0). E18 **99.4% / 63.2 bits**. E21 **25.1% / 0 bits**. `e18_local` 0 bits (K2).
  Chance, not climbing; unlike tiny E21 at 800 steps. Do not 1024.
  [report](../2_Experiments_Registry/run_reports/e25_bridge512_far_copy_20260913.md).
- **2026-09-13 — E25 tiny chain extra steps (CPU, 8k).** E21 **49.4% / 10.02 bits / flow
  0.386** (best 51.7% @5450). Acc stuck vs 3200; composition wall. Do not 16k.
  [report](../2_Experiments_Registry/run_reports/e25_tiny_chain_ordered_e21_steps_20260913.md).
- **2026-09-13 — E25 tiny chain_ordered (CPU, 3200).** Dense **93.1% / 23.3 bits** (S0).
  E18 **97.5% / 24.6 bits**. E21 **49.3% / 8.22 bits / flow 0.316**. INDEX-like half-channel;
  still climbing. `e18_local` 0 bits.
  [report](../2_Experiments_Registry/run_reports/e25_tiny_chain_ordered_20260913.md).
- **2026-09-13 — E25 tiny select_1decoy (CPU, 1900).** Dense **99.2% / 31.3 bits** (S0).
  E18 **82.3% / 23.2 bits**. E21 **33.0% / 1.47 bits / flow 0.046**. Type-cue wall; K3
  borderline. `e18_local` 0 bits.
  [report](../2_Experiments_Registry/run_reports/e25_tiny_select_1decoy_20260913.md).
- **2026-09-13 — E25 tiny recall extra steps (CPU, 8k).** E21 **45.1% / 9.81 bits / flow
  0.307** (best 48.5% @7700) vs 0.75× dense 23.5 / 0.735. MATCH wall vs dense; still beats
  E18's 0 bits. Do not 16k.
  [report](../2_Experiments_Registry/run_reports/e25_tiny_recall_single_e21_steps_20260913.md).
- **2026-09-13 — E25 tiny recall_single (CPU, 2850).** Dense **99.2% / 31.4 bits** (S0).
  E18 **25.3% / 0 bits**. E21 **37.9% / 3.19 bits / flow 0.100**. `e18_local` 0 bits (K2).
  Slots weakly beat E18's MATCH wall; 0.75× dense (23.5 bits) fails. Still climbing.
  [report](../2_Experiments_Registry/run_reports/e25_tiny_recall_single_20260913.md).
- **2026-09-13 — E25 tiny far_copy extra steps (CPU, 8k).** Complete-block E21 **84.6% /
  47.01 bits / flow 0.734** (best acc 86.6% @7700) vs 0.75× E18 47.29 / 0.739. Near-pass;
  2400-step S1 fail was budget.
  [report](../2_Experiments_Registry/run_reports/e25_tiny_far_copy_e21_steps_20260913.md).
- **2026-09-13 — E25 tiny far_copy QUERY-align seq=132 (CPU).** QUERY at 96 = 6×16. Dense
  **99.1% / 62 bits**, E18 **99.2% / 63**, E21 **40.0% / 8.7 bits** (flow 0.136). Complete
  sender coverage did not recover INDEX at the short budget.
  [report](../2_Experiments_Registry/run_reports/e25_tiny_far_copy_query_align_20260913.md).
- **2026-09-13 — E25 tiny far_copy remainder pooling (CPU).** Same recipe as rung 1 with
  `message_pool_remainder`. Dense/E18 replica-match. E21 **47.6% / 12.7 bits** (flow 0.199)
  vs complete-block-only 17.7 bits. Remainder did not close INDEX. K3 not hit.
  [report](../2_Experiments_Registry/run_reports/e25_tiny_far_copy_remainder_20260913.md).
- **2026-09-13 — E25 tiny far_copy (CPU, complete-block-only E21).** Dense 99.4% / 63 bits
  (S0 = E24 replica). E18 99.2% / 63 bits. E21 **51.5% / 17.7 bits** (flow 0.277 vs 0.75×
  gate). `e18_local` 0 bits (K2). Remainder tokens next to QUERY were not pooled.
  [report](../2_Experiments_Registry/run_reports/e25_tiny_far_copy_20260913.md).
- **2026-09-13 — E24 GPU bridge (Polonez/Odra, right-align INDEX).** Seq=512 `far_copy`:
  E18 **100% / 63.9 bits** vs dense 99.7% (S1 pass). Seq=1024 `far_copy`: dense **99.9% /
  64 bits**, E18 **0 bits** (S1 fail). 512 `recall_single` at a *fixed* offset: dense 32 bits,
  E18 **0 bits** @2500 steps — the content wall is not “find the mark”. 4k spread is K1;
  4k right-align 16M once hit 74% (missed the gate by 1 pt). 
  [report](../2_Experiments_Registry/run_reports/e24_bridge512_bapo_ladder_20260913.md).
- **2026-09-13 — E24 tiny BAPO ladder (CPU, 0.59M).** Packed DNA rungs with a 75% dense control:
  E18 copies 63/64 bits (`far_copy` 99.2% ≈ dense 99.4%) and follows in-order hops (`chain_ordered`
  97.5%), but recovers **0 bits** on single-fact keyed recall (dense 99.2% / 31 bits). `select_1decoy`
  is a marker-type cue (E18 87%), not MATCH2. Shuffled chain and 2–3-item MATCH2 are uncalibrated
  (dense ~30–34%). Encoder-decoder at 0.8M is not a copy baseline.
  [report](../2_Experiments_Registry/run_reports/e24_tiny_bapo_ladder_20260913.md).
- **2026-09-12 — E22 Perceiver Concept LM (from scratch, 32k; killed same day).** First ledger design
  with positional slots (1 / 16 tokens), a transformer *over* the slots and a decoder with no raw route
  past its 1024-token segment, trained under plain CE (+5% keyed recall) to 0.44B tokens (88% of budget) on Odra. The
  array is **live** (Δ_none 0.25 nats, > 10σ) and **diverse** (RankMe 265/768, learned-query pooler
  alive), yet **arm A = arm C** on far tokens (4.167 vs 4.172; C is an 8-layer segment-only decoder at half
  the compute), passkey 0.0, keyed recall 4.8% vs a 3.5% floor. A `near`/`far` ablation added after the
  run decomposes the 0.25: **0.17 is document-level content present redundantly in every slot** (a
  document embedding — a wrong book's array beats no array at 16k–32k), **0.05 is the far slots'
  marginal — the memory the bet was about, flat from 1k to 32k**, 0.03 is a local bypass through the
  `cpos ≤ pos` mask (same-segment slots). Root causes: CE on natural text pays ≈ 0.05 nats for far
  context at this scale — the array captured exactly that and the S1 gate (0.30) was unreachable by
  construction; and the mask never made the array the only route for anything. Two laws added to the
  revisit synthesis' three: *the objective must pay for the channel* (gate on the far marginal, never
  Δ_none) and *exclusivity is two-sided*. Banked: the `perceiver_concept` family, the `near`/`far`
  instrument, `concept_xattn_scope`. Lost: the dense control (disk-full crash + cleanup-sweep bug).
  Spec [E22](../experiments_specs/done_failed/E22_perceiver_concept_lm.md) ·
  [report](../2_Experiments_Registry/run_reports/e22_pilot_verdict_20260912.md) ·
  [root cause](../4_Research_Notes/e22_root_cause_20260912.md).
- **2026-09-13 — the array *can* be addressed, and a zero-init artefact was hiding it** (CPU-hours,
  zero GPU-days, `data/symbolic_tasks.py` + `verification/symbolic_channel_probe.py`). On symbolic
  rows whose information floor is *exact* (`far_copy`, alphabet 4, floor 1.3863 nats), three arms:
  full raw access **0.0000** nats / 100% acc, segment-confined **1.3863** / 25% (pinned at the floor
  for 3000 steps — the task provably does not leak), and the array as the *only* route **1.1726** /
  42%. So the channel carries addressable far content — **the first positive evidence in this
  family** — but recovers only 15% of what raw access does. The write and mask are not at fault: the
  evidence moves the slots (`max|Δz|=1.68`) and the mask exposes exactly those slots, yet the answer
  logits are bit-identical at init. Cause: **two zero-init residual gates in series** on the concept
  path (`pooler.wo`, the only order-sensitive part of the write, and `xattn.wo`, the read's output),
  each one's gradient proportional to the other — so the channel's only early escape is the
  content-free *mean* of its slots, i.e. a document embedding. Seeding both recovers **17× more
  information at matched steps (3/3 seeds)** and removes a ~2000-step plateau. This plausibly
  reframes E22's headline result (0.17 nats of document content, 0.05 far marginal) as an **init
  artefact rather than an architecture limit**; magnitude is seed-variable at 1.3M params and must be
  re-measured at scale. Now config-selectable (`pcl_xattn_wo_init_std`, `pcl_pooler_wo_init_std`,
  default `0.0` = E22). **Consequence: every new read of a new memory must have a warm output
  projection, and the order-sensitive part of a write must be live at init.**
  [note](../4_Research_Notes/concept_channel_cold_start_20260913.md) ·
  [suite](../engineering_specs/symbolic_long_context_suite.md).
- **2026-09-13 — the family's compute win is a constant, not an asymptote** (analytic, zero GPU-days,
  `analysis/geometry_cost_model.py`). The E22 geometry cuts decode state 192× (18 KB/token of dense KV
  → 96 B/token of array; 180 GB → 0.96 GB at 10M) and that *is* structural. But the *read* is dense —
  every token scores every visible slot — so cross-attention stays O(S²/r) and is 94% of arm A's FLOPs
  at 10M; the whole-model saving converges to `r·L_dense/L_dec` = **36×** at any context length. A
  selective top-k read raises the ceiling to `r²·L_dense/L_latent` = 1152× but then the full-causal
  latent stack becomes the wall; only with a windowed/hierarchical latent stack does the model become
  per-token-work bound (1288× at 10M and growing with S). **Consequence for the 1M/10M goal: a
  selective read and a non-global latent stack are not optimisations, they are requirements** — and
  since E22 showed the dense read fails to *use* far slots anyway, forcing it to name what it wants is
  plausibly the same fix twice. Design constraint for E23's successor; E23 stays one bet.
- **2026-09-12 — E18 / E18b (one global read):** a from-scratch 125M LM whose only unbounded layer is
  a single full-causal read. It is **free** (eval 3.790 vs matched dense 3.786, 1.02× throughput) and
  does exact **positional** retrieval (plain copy @32k offset 16k: **99.9998%**; cutting its reach two
  tokens short → 0.4%). But a model with **no read at all** reaches the same loss (arm C **4.091** vs
  arm A **4.090**), and placement does not change that (arm B, read at layer 7, also 4.090 and depended
  on 5.7× less): next-token prediction never supervises long-range addressing, so the read is *used but
  not useful*. E18b then tried to supply that gradient with dense-label keyed-recall rows: first-token
  accuracy moved 2.4% → 4.2% (5% mix) → 4.4% (+ value embedding on the read) → **4.49%** (100% task
  data, ~20× supervision), passkey 0.0 throughout. The **dense control on the identical task reached
  99.33% and transferred to passkey 0.725 @32k** — so the task is learnable at this scale and the
  architecture is the cause. Read structurally, the pilot is an encoder-decoder with a **1-layer
  encoder**, 1 cross-attention and a 12-layer decoder; we asked one local layer to produce keys
  discriminative enough to be content-addressed. Banked regardless: the context-extension protocol fix
  (weights-only restart at peak lr cost ~2%; resuming at ≤20% lr with decay gains ~12% at 32k), a
  reusable paired **reach-ablation** instrument, and a 1 KB/token cache that is 23× under dense.
  Specs [E18](../experiments_specs/done_failed/E18_perceiver_ar_v2_baseline.md) ·
  [E18b](../experiments_specs/done_failed/E18b_retrieval_trained_read.md) ·
  [verdict report](../2_Experiments_Registry/run_reports/e18_family_verdict_20260912.md).
  **One cheap open question:** move the read to mid-depth (`PAR_GLOBAL_POSITIONS=7` makes everything
  below it a 7-layer encoder, deepening queries *and* keys, with no change to the cache and no LM cost).
  Staged as `Cache/jobs/e18b_mid_taskonly.sh`, ~1.4 GPU-h, not launched.
  [E18c](../experiments_specs/ahead/E18c_concept_compressed_read.md) (compress the read's K/V) is
  **blocked**: it needs a functional retrieval channel, which we do not have.
- **E17e 300M closed (train 2026-08-22, eval 2026-08-25).** Late-half Δperm
  **0.104** CI [0.095, 0.114] on best `checkpoint-2660` (last **0.097** miss);
  RankMe **31.5–57.4** and eval_loss **2.464** passed; gen `real`@256
  **0.162/0.686** (`real=shuffle`). Starve lifted E17d's **0.044**, same
  gist-not-memory decay. **Do not launch 1B.** Do not another half-window in this
  ID. Spec:
  [E17e](../experiments_specs/done_failed/E17e_starved_local_window.md) ·
  [report](../2_Experiments_Registry/run_reports/e17e_starved_local_window_20260825.md).
- **E17d 300M closed (train 2026-08-17, eval 2026-08-18).** Late-bin 256–512 Δperm
  **0.044** CI [0.039, 0.049] missed ≥0.10; RankMe **43.2–76.8** and eval_loss **2.365**
  passed; gen `real`@256 **0.185/0.595** (`real=shuffle`). First-64 Δperm **0.75** is
  multi-bank (unlike E17c). **Do not launch 1B.** Spec:
  [E17d](../experiments_specs/done_failed/E17d_global_concept_assimilation.md) ·
  [report](../2_Experiments_Registry/run_reports/e17d_global_concept_assimilation_20260818.md).
- **E17c 300M closed (2026-08-15).** Carryless first-64 Δpermutation **0.594** CI [0.543, 0.645]
  cleared the registered ≥0.20 gate, almost entirely in bank 0 / layer 5. Geometry collapsed
  (RankMe **6.75**, bank 1 **1.84**) and normal-context Δpermutation_beyond **0.013** missed the
  0.02 stop, so **do not launch 1B**. Free-run `real`@256 **0.23/0.53** stays in the E17 family
  (`real≈shuffle`). Carry dropout only trains a block-boundary gist (see
  [five-whys](../4_Research_Notes/e17c_failure_five_whys_20260815.md)). Spec:
  [E17c](../experiments_specs/done_failed/E17c_depth_private_working_memory.md) ·
  [report](../2_Experiments_Registry/run_reports/e17c_depth_private_working_memory_20260815.md).
- **Still open / still wanted:** [E08 Concept-Flow reasoner](../experiments_specs/ahead/E08_concept_flow_reasoner.md)
  (latent reasoning composition — preferably on a platform that already carries
  concepts), diffusion revive from `parked/` if a materially new ingredient appears,
  and design-only [E11](../experiments_specs/ahead/E11_memtoken_concept_memory.md) /
  [E12](../experiments_specs/ahead/E12_perlayer_kv_prefix_concepts.md) /
  [E13](../experiments_specs/ahead/E13_layerwise_recurrent_kv_memory.md).
  Priorities shifted; exploration stays multi-path.
- **Background references:** E02-long remains the from-scratch semantic reference
  (STS-B 0.714). E10–E16a / E14–E15 remain valid evidence about short-ctx / sparse
  recall regimes — lower priority for the next budget, not erased. E17 init-0.01
  ([done_success/E17](../experiments_specs/done_success/E17_four_bank_concept_memory.md) ·
  [1B gen](../2_Experiments_Registry/run_reports/e17_lowinit_1b_generation_20260810.md))
  remains the per-layer free-run baseline. E17c showed that causal carry dropout can
  force concept use without making that use survive ordinary CE or keep RankMe healthy.

### Series roadmap (genealogy; each step is one registered coherent bet)
1. **E01 — AR decoder from scratch** *(done 2026-06-14, mixed).*
2. **E02 — objective:** [prefix→suffix AR generation](../experiments_specs/done_success/E02_ar_prefix_suffix.md)
   *(done 2026-06-14, mixed/positive; STS-B 0.702).*
3. **E03 — de-collapse via frozen-encoder anchor** *(done).*
4. **E04 — concept-only parallel decoder** [(spec)](../experiments_specs/done_success/E04_concept_only_parallel_decoder.md)
   *(done 2026-06-20, mixed).*
5. **E05 — windowed decoder** [(spec)](../experiments_specs/done_failed/E05_windowed_decoder_concept_memory.md)
   *(done_failed — used but semantically empty at <200M; motivated the Gemma pivot).*
6. **E10–E16a short-ctx Gemma line** *(done_failed / mixed — useful regime evidence; see ledger).*
7. **E16b long-ctx Muon** *(done_success 2026-07-25 — validated long-context path; follow-ups + other routes still open).*
8. **E17 four-bank per-layer (init 0.01)** *(done_success mixed 2026-08-10 — relative free-run win; writes dead).*
9. **E17b per-layer mid write-init 0.1** *(done_failed 2026-08-13 — mid-init not sticky; free-run ≈E17).*
10. **E17c depth-private gated working memory + causal carry pressure** *(done_failed mixed 2026-08-15 — carryless Δperm 0.59 PASS; RankMe 6.7 / Δbeyond 0.013 kill 1B).*
11. **E17d depth-private concept layers as global-attention replacement** *(done_failed mixed 2026-08-18 — RankMe 43–77 PASS; late-bin Δperm 0.044 miss; no 1B).*
12. **E17e starve the local window to K=256** *(done_failed mixed 2026-08-25 — late-half 0.104 on best, last 0.097, gen 0.16/0.69; no 1B).*

## What we've explored so far (evidence, not verdicts)
- **E17e starve local window K=256 (per_layer_banks, Polonez, train 2026-08-22, eval 2026-08-25):**
  300M run `…20260822_120601`. Late-half of each 256-token window Δperm **0.104**
  CI [0.095, 0.114] on best (last **0.097**). RankMe **31.5 / 34.9 / 31.2 / 57.4**.
  First-64 Δperm **1.34**. Free-run `real` greedy @256 **0.162/0.686** (`real=shuffle`;
  E17d **0.185/0.595**). Halving the window lifted late-half vs E17d **0.044**;
  per-bank late-half stayed ~0.025 and generation did not improve. Do not 1B. See
  [report](../2_Experiments_Registry/run_reports/e17e_starved_local_window_20260825.md).
- **E17d attn-residual global mix, no token carry (per_layer_banks, Polonez, train 2026-08-17, eval 2026-08-18):**
  300M run `…20260817_141227`. Late-bin 256–512 Δperm **0.044** CI [0.039, 0.049]
  (E17c **0.026**). RankMe **43.2 / 58.7 / 65.9 / 76.8**. Carryless first-64 Δperm **0.75**
  (banks 0.13 / 0.21 / 0.08 / 0.05 — not a bank-0 monopoly). Free-run `real` greedy @256
  **0.185/0.595** (`real=shuffle`; E17c **0.23/0.53**). Attn-residual + dropped carry
  keeps geometry healthy and spreads the *block-start* gist; it does not assimilate
  late-page tokens. Do not 1B. See
  [report](../2_Experiments_Registry/run_reports/e17d_global_concept_assimilation_20260818.md).
- **E17c gated cell + carry pressure (per_layer_banks, Polonez, train 2026-08-14, eval 2026-08-15):**
  300M run `…20260814_133241`. Carryless first-64 Δpermutation **0.594** CI [0.543, 0.645]
  (bank 0 **0.38**; others ≤0.03). RankMe **6.75** (bank 1 **1.84**). Normal-context
  Δpermutation_beyond **0.013**. Free-run `real` greedy @256 **0.23/0.53** (E17b **0.20/0.60**;
  E17 **0.21/0.59**). Pressure forces concept use when carry is dropped; it does not
  transfer to ordinary CE or keep geometry healthy. Do not 1B. See
  [report](../2_Experiments_Registry/run_reports/e17c_depth_private_working_memory_20260815.md)
  and [five-whys](../4_Research_Notes/e17c_failure_five_whys_20260815.md).
- **E17b mid-init 0.1 (per_layer_banks, Polonez, train 2026-08-10→13, Tier-1+1.5 2026-08-13):**
  write gates opened near ~100M (max \|tanh\| 0.14) then closed to ~0.05 by 1B; RankMe 68;
  Δshuf/static≥1024 **0.0055/0.0033**; free-run `real` greedy @256 **0.20/0.60** (E17 **0.21/0.59**;
  E16b **0.04/0.94**). Mid write-init alone is not sticky under plain CE. See
  [report](../2_Experiments_Registry/run_reports/e17b_per_layer_mid_write_init_20260813.md).
- **E17 low-init 1B (per_layer_banks, Polonez, train 2026-08-07→10, Tier-1.5 2026-08-10):**
  matched init 0.01 vs E16b finished 1B with writes still dead (`|tanh|≤0.033`), Δbeyond ~0.004,
  RankMe 98. Free-run `real` greedy @256 **0.21/0.59** (E16b **0.04/0.94**; base **0.16/0.71**) —
  absolute success bar missed; relative lift vs E16b (prose, `real≈zero`, long prompts help).
  The topology is block-causal, but its write mechanism never engaged. See
  [report](../2_Experiments_Registry/run_reports/e17_lowinit_1b_generation_20260810.md).
- **E16b free-run generation vs base Gemma (Odra, Tier-1.5, 2026-08-01) — FAIL on generation; teacher-forced mechanism interpretation revised 2026-08-14:**
  matched continuation bank + context sweep on `checkpoint-7900` vs `gemma-3-1b-pt`.
  E16b `real` greedy @256: distinct-1 **0.04** / REP-3 **0.94** (digit/punctuation
  attractors); base sample @256 REP-3 **0.03**. Longer prompt prefixes help base and
  hurt E16b free-run. Chat template is not the fix.
  **Layer-0 decode probe (same day, rp=1.2 + `frozen` mode, commit `8a6bafa`):** `zero`
  is the *only* fluent mode (greedy d1@256 **0.74** / r3 **0.01**, base-like); `frozen`
  degenerates like `real` → **refutes "self-writes poison free-run"**; the driver is the
  **concept read pathway** reading a near-static `z` (write gates ≈0 → z ≈ learned
  constant). `repetition_penalty` doesn't help (turns loops into structured junk);
  sampling does. Backbone + LoRA + windowed-global-attention are sound (`zero` is
  fluent) — we did **not** break Gemma. See
  [report](../2_Experiments_Registry/run_reports/e16b_generation_quality_assessment_20260801.md).
- **E16b long-context Muon 1B (Gemma-3-1B, Odra, train 2026-07-18→20, Tier-1 2026-07-25) — recorded success; causal interpretation revised 2026-08-14:**
  shared-depth workspace at seq 4096 on `e16b_long_4k_v1` with Muon for 1B tokens
  reached offline RankMe **101** and Δshuffle/Δstatic≥1024 **2.47/2.35** (clears the
  0.01 gate by a large margin; E16a Muon was 0.0028 at 100M/2K). Δone-block≥1024
  **0.58** showed accumulated multi-block state under the registered protocol. **Revised
  reading:** the implementation rereads non-token-causal current-block writes at later
  same-block depths, so these deltas establish predictive use but not clean causal
  cross-block memory. Free-run separately failed; retain the numbers as historical
  evidence, but do not build new work on the shared-depth topology.
  See [run report](../2_Experiments_Registry/run_reports/e16b_longctx_muon_1b_20260725.md).
- **E16 shared depth-recurrent workspace (Gemma-3-1B, Odra, 2026-07-14):**
  the 50M run kept healthy geometry (within-sample RankMe 62.2; centered 125.0) and
  eval CE 1.8122, but beyond-local static/shuffle deltas were only
  +0.000499/+0.001018 nats, both below 0.01. Interleaved tied writes at four Gemma
  depths did not establish persistent concept use *under 2K CE*; the same architecture
  later cleared the registered metric under E16b’s long-context regime, whose causal
  interpretation was revised on 2026-08-14. See
  [spec](../experiments_specs/done_failed/E16_shared_depth_recurrent_concepts.md).
- **E15 supervision-calibrated delayed recall (Gemma-3-1B, Odra, 2026-07-13):** after
  resuming E14 to 12,000 total answer labels, the block-2 explicit-carry control was still
  below chance (0.98% versus 1.56%; required ≥80%). Thus more exposure alone does not make the
  one-answer-per-2K sparse task learnable, and E15 did not test the E10e memory interface. See
  [run report](../2_Experiments_Registry/run_reports/e15_supervision_calibrated_delayed_recall_20260713.md).
- **E14 forced delayed recall (Gemma-3-1B, Odra, 2026-07-13):** the registered 2M-token stop
  fired (all block-4 memory margins <0.0036 nats), while healthy geometry persisted (RankMe 91.4).
  Because the block-2 explicit-carry control also stayed at chance after only 984 supervised
  answers, the run exposed an input-token-vs-supervision budgeting flaw rather than isolating
  writer retention or read integration. See
  [run report](../2_Experiments_Registry/run_reports/e14_forced_delayed_recall_gate_20260713.md).
- **E10e calibrated concept memory at 100M (Gemma-3-1B, Odra, 2026-07-13):** versus the
  same-budget E10 pilot, CE fell 1.8150→1.7972 and within-sample RankMe rose 77.1→99.9, but
  beyond-local static/shuffle deltas reached only +0.000962/+0.001613 nats; more calibrated
  plain-CE exposure did not yield persistent memory use. See
  [run report](../2_Experiments_Registry/run_reports/e10e_calibrated_memory_100m_20260713.md).
- **E10b normalized concept read (Gemma-3-1B, Odra, 2026-07-12):** at the ~25M decision
  checkpoint, geometry remained healthy (RankMe 112.2; centered 125.1) and local CE matched E10,
  but static−real was only +0.000371 and Δshuffle +0.000179 at positions ≥1024; read normalization
  alone did not create persistent multi-block usage. See
  [run report](../2_Experiments_Registry/run_reports/e10b_normalized_concept_read_20260712.md).
- **E10 100M concept-arm pilot (Gemma-3-1B, Odra, 2026-07-11):** stable and non-collapsed
  (RankMe 77.1; centered 123.1), but every beyond-local recurrent-state ablation was <0.001 nats;
  the matched control is still required before judging the primary recovery criterion. See
  [run report](../2_Experiments_Registry/run_reports/e10_100m_concept_pilot_20260711.md).
- **Reference baseline:** `perceiver_mlm_H512L6C128_20260208_211633` — just a comparison anchor (MRPC 82.7 / STS-B 0.650 via ViaDecoder; concept effective rank ~5/128). Not a target, not "good."
- **MLM + concept losses** (combined / kendall_gal / fixed): pushing concept diversity tended to cost downstream semantics — a tension worth remembering.
- **Diffusion (self-reconstruction, ELBO, VICReg) and prefix diffusion:** explored on MiniPile / WikiText-103; concept effective rank stayed low so far. Code in `parked/`. **2026-06-13 lit scan (CALM/ELF/Cosmos/LDLM/Nemotron):** our 5 failures match a *known* failure mode — an **unvalidated bottleneck + decoder bypass**, not just bugs. Reviving needs a materially-new ingredient (frozen-encoder MSE anchor + concept-dropout/CFG, ideally warm-start), not a re-run. For "do concepts carry semantics?", **AR + ΔCE is the cleaner probe** than diffusion.
- **Recursive / latent-reasoning (Ouro/Huginn/TRM lit scan, 2026-06-13):** recurrent-depth is real but **task-selective** — gains show on multi-step/compositional benches, often flat on plain denoise/STS; **measurement is the bottleneck**. Use **Ouro**, not TRM (its ARC headline was audited down to ensemble + puzzle-ID lookup + shallow step-1). Only worth running on **de-collapsed** concepts (hence E03 first).
- **Perceiver denoise reconstruction:** strongest zero-shot STS-B at the time (~0.607) with still-low-rank geometry and mixed supervised signal. Now superseded by E02.
- **E01 — AR denoising reconstruction (FineWeb-Edu, 1 epoch, 2026-06-14):** AR plumbing confirmed; decoder uses concepts early (Δshuffle 1.50 at step 4000). Eval CE rises monotonically thereafter (overfitting); rank collapses 14.64 → 4.64; best STS-B 0.556. Reconstruction + word-dropout insufficient to sustain concept quality over full training. Best checkpoint is an early checkpoint (4000 steps).
- **E02 — prefix→suffix AR generation (FineWeb-Edu, 1 epoch, 2026-06-14):** STS-B **0.702** — new project best, well above prior best (0.607) and E01-best (0.556). Prefix→suffix creates better semantic pressure than reconstruction. Rank still collapsed (11.57/128), suffix-CE ablation modest (Δshuffle 0.50). Key insight: compact geometry can coexist with high STS-B — the active subspace is semantically loaded even when rank is low.
- **Collapse root-cause + measurement reframe (2026-06-14):** deep dive (code + data + lit) concluded "concept collapse" is mostly **(a) a measurement artifact** and **(b) strong-AR-decoder posterior collapse**, not a capacity problem. (a) The headline "effective rank" SVDs the **batch-averaged** concepts, so it measures *slot redundancy*, not representation dimensionality; zero-shot STS-B **mean-pools 128 slots to one vector**, so it's nearly blind to slot rank. New per-sample manifold metric (RankMe) on a live E03 checkpoint: slot-rank **2.7** vs **RankMe ≈24**, anisotropy 0.28, 100% active slots — the usable geometry is far healthier than the slot-rank number implied. (b) The teacher-forced AR decoder bypasses the bottleneck via local context, so required rate through `z`→0 (Bowman 2015; Chen VLAE 2016; Alemi 2018). **Evidence:** E02 all-position ablation `Δzero=0.50` but **early-position `Δzero=1.43`/`Δshuffle=1.04`** (concepts strongly used where bypass is impossible); `gap_clean_vs_wd=0.037` rules out the word-dropout protocol artifact. New tooling (manifold RankMe, anisotropy, per-slot activity, early-Δ as primary gate) committed in `analysis/`. **Implication:** judge de-collapse on the per-sample manifold + early-Δ, not slot-mean rank; the levers are anchor (E03) + decoder weakening, not more concepts or longer training.
- **E03 anchor-ON warmup (FineWeb-Edu, 0.3 epoch, 2026-06-15):** All kill gates pass. Anchor MSE decreasing (0.512), AR CE stable (4.446 < E01-best 4.676), concept ablation healthy (Δshuffle 1.345, Δshuffle_early 3.342 — notably higher early-position signal than prior runs). Slot rank 10.34 held steady at 19k steps (no collapse seen unlike E01). STS-B 0.556 on par with E01-best on the same reconstruction objective. See [run report](../2_Experiments_Registry/run_reports/e03a_anchor_on_warmup_20260615.md).
- **E03 matched control (anchor-OFF, FineWeb-Edu, 0.3 epoch, eval 2026-06-18):** completes the matched pair. Reconstruction at 0.3 ep **collapses without the anchor** — slot rank peaks 17.4 (0.06 ep) then falls to 5.1, STS-B 0.485 (below E01-best 0.556), `gap_clean_vs_wd 1.677` (the decoder bypasses its collapsing concepts via local context). The anchor arm beats the control on every relative metric — RankMe 167 vs 150 (+16.7), STS-B 0.556 vs 0.485 (+0.071), AR CE 4.45 vs 4.79, and decisively gap_clean_vs_wd 0.128 vs 1.677 (13×). **But** absolute gates are unmet at 0.3 ep (STS-B < 0.62; slot rank +4.4 < +16) and the control's higher early-Δ (5.58 vs 3.34) is a *collapse symptom* (fewer directions used harder), not health. Verdict **mixed/promising**. See [run report](../2_Experiments_Registry/run_reports/e03_control_anchor_off_20260618.md).
- **E02-long — prefix→suffix, 5 epochs (FineWeb-Edu, Polonez, eval 2026-06-18; Tier-2.5 probe 2026-06-20):** **the most important reframe so far.** Longer prefix→suffix training **de-collapses** concepts: slot rank rises 5.9 → 11.6 → 16.7 across 0.3/1/5 epochs — the *opposite* of E01 reconstruction (rank 14.6 → 4.6 over 1 epoch). Upgraded metrics show genuinely healthy geometry (RankMe 245.9, anisotropy 0.32, mean concept cosine 0.124, 63 dims for 95% var). STS-B 0.714 (new project best, +0.012 over 1-ep E02) plateaus despite 5× budget + much richer geometry. Tier-2.5 confirms the extra structure is partly distributed across slots: SICK relatedness improves from mean **P −0.203** to attention **P 0.133** (Δ**+0.336**), while PAWS is mixed (accuracy +0.037, F1 −0.051). **Takeaway:** "concept collapse" is **objective-dependent, not universal** — prefix→suffix improves with scale and is the right objective basis for E05. See [run report](../2_Experiments_Registry/run_reports/e02_long_5epoch_20260618.md).
- **E04 — parallel Perceiver-IO decoder, reconstruction (FineWeb-Edu, Odra, eval 2026-06-20):** removes the AR bypass (no token self-attention). **Within-sample RankMe 107.8**, cross-sample RankMe 177.8 (+27 vs E03 control); STS-B 0.532 > control 0.485 but << E02 0.702. Tier-2.5 pool probe: SICK ΔPearson **+0.22** (mean −0.07 → attn 0.16) — distributed geometry partially hidden from mean pool; PAWS inconclusive; absolute semantics still weak. See [run report](../2_Experiments_Registry/run_reports/e04_parallel_decoder_20260620.md).
- **Anchor status after E03/E02-long:** anchoring concepts to frozen pretrained per-token hidden states helps reconstruction relative to a matched control, but it is an auxiliary/de-risking lever, not the main research direction. The architecture-first path is E05 from scratch with prefix→suffix.
- **Long-context engineering, round 2 (2026-06-27):** the real memory wall was the **output head**, not the encoder — F2's chunked CE secretly retained `[B,N,V]` in the autograd graph (fixed by `ChunkedLMHeadCE`; **256K now fits on one 3090**). **Sequence parallelism (F6) reaches 1M context on 3× 3090 at 22.6 GB/GPU** (validated ≡ single-GPU to ~1e-6 in loss + all grads). **Muon** converges ~2× faster than AdamW on wikitext-103. 10M is the hardware ceiling for 24 GB cards (needs 80 GB cards / ~30 GPUs). Full note: [long_context_memory_optimization_round2_2026_06_27.md](../4_Research_Notes/long_context_memory_optimization_round2_2026_06_27.md).
- **E05 attempt 2 — windowed decoder, diverged at step 40k (Odra, killed + fast-evaled 2026-06-28):** the LR 1e-4 / warmup 1500 retune of attempt 1 (which diverged at step ~20 under LR 3e-4) **delayed but did not prevent** divergence — same signature, just at step 40k / epoch 0.19 / 5.2B tokens instead of step 20: eval_loss 3.32 → 4.03 over 12k steps, pre-clip grad_norm escalated 9 → 56 → 219 → 903 while cosine LR was still ~8.5e-5. **Architecture is sound** — best checkpoint-40000 fast-evaled clears Stage 1 floor on every gate except the beyond-window Δshuffle target (within-sample RankMe **59.8**, early-Δshuffle **0.85**, beyond-window Δshuffle **0.35** ≥ floor 0.3 but < Stage 2 target 0.5). Compute: **81.3 GPU-h / 17.78 kWh / 5.21B tokens** (`compute/max_tokens_b`). **Takeaway:** divergence is optimization-side, not architectural — cosine-kept-hot + HF-default `max_grad_norm=1.0` let bad-direction updates dominate the sharpening late-run loss landscape. Retune for attempt 3: LR 5e-5, `max_grad_norm` 0.5 (now wired through launcher), batch 12, re-scoped to 0.5 ep (~7B tokens). See [run report](../2_Experiments_Registry/run_reports/e05_attempt2_diverged_20260628.md).
- **E05 attempt 3 — windowed decoder, completed 0.5 ep + evaluated (Odra, 2026-06-30):** the LR 5e-5 / clip 0.5 retune of attempt 2 **held — first fully completed E05 run**, and the eval is now in. Training: 0.5 ep / 10.2B tokens / 68.2 GPU-h / 18.24 kWh; eval_loss fell monotonically 5.40 → 3.83 across 17 evals; grad_norm held 0.4–0.55 through step ~48k, then rose to 40–75 in the cosine-tail region (LR ≈ 1e-6) without hurting eval_loss — the *opposite* signature from attempt 2's escalation. **Stage 1 PASS:** within-sample RankMe **37.67** (not collapsed), Δzero_beyond **6.99** (decoder reads concepts), Δshuffle_beyond **0.39** (≥ floor 0.3). **Stage 2 NOT YET MET:** Δshuffle_beyond < target 0.5; **STS-B zero-shot 0.452 is below both trivial floors** (token-embed-mean 0.486, teacher-hidden-mean 0.460) — the concept bottleneck currently *destroys* semantic-similarity signal vs averaging raw token embeddings. Free-running generations are grammatical but semantically empty repetition loops (token-F1 0.149, exact-match 0.015). SICK-R 0.183, SICK-E acc 0.634, PAWS 0.550/0.253, GLUE MRPC 0.669/0.778, GLUE STSB 0.354/0.341 (full-finetune, demoted evidence). **Takeaway:** optimization succeeded and the architecture is stable — this is a "more training / stronger objective" signal, not an architectural dead-end (cf. attempt 2's divergence). Matched A/B now justified. Two eval-script bugs fixed during eval (wandb tag truncation `730e607`, SmolLM2 pad_token `70e1fd2`). See [run report](../2_Experiments_Registry/run_reports/e05_attempt3_completed_20260630.md).
- **E05 Muon A/B — optimizer arm (Odra, 2026-07-02):** stabilized Muon (wd=0.1, adamw_lr=2e-4, LR 0.01) **converges ~5× faster + to a 1.22-nat-lower eval_loss (2.606 vs Adam's 3.83)**, stable end-to-end (no divergence). Naive Muon diverged at LR 0.02 and 0.01 — root-caused (Moonlight arXiv:2502.16982: full-rank orthogonalized updates grow weight spectral norms in the Q·K + lm_head bilinear couplings, with no weight decay to curb it + an over-hot `adamw_lr=2e-3`) and fixed (wd=0.1 + `adamw_lr=2e-4`) + a **sustained-LR (`constant_with_warmup`) calibration protocol** (a short cosine hid the delayed onset — a calibration-fidelity lesson). **Takeaway:** the optimizer is a first-class, ready lever — a compute-efficiency win for E06–E08 (re-validate Muon per objective). **Caveat:** eval_loss ≠ concept semantics (Adam had 3.83 but STS-B 0.45); the Muon semantic/geometry verdict (STS-B, RankMe, Δshuffle) awaits its eval suite. **Eval (2026-07-04, ⚠️ TENTATIVE): split** — downstream semantics up (STS-B **0.518** clearing floors, SICK-R 0.302, GLUE MRPC/STSB improved) BUT concept geometry + long-range gate **regressed** (within-sample RankMe **10.57** vs 37.67, Δshuffle_beyond **0.209** vs 0.39, Δzero_beyond 0.41 vs 6.99) → the lower loss came from the decoder's within-window bypass, not richer concepts. **Not decisive** (wd confound Muon 0.1 vs Adam 0.0; authoritative prefix→suffix `concept_ablation/*` not yet read; single seed) — open for discussion + literature on optimizer-vs-representation-collapse. Adam wd=0.0 vs Muon wd=0.1 confound (wd is part of "what Muon needs to be stable"). See [run report](../2_Experiments_Registry/run_reports/e05_muon_divergence_rootcause_20260701.md).
- **E05 Muon long (2 ep) — compute-matched to E02-long, evaluated (Odra, 2026-07-09):** the 2-ep Muon run (`concept_ar_prefix_H768L6C128D4_20260704_225659`, **300.88 GPU-h / 85.25 kWh / 40.78 B tokens** ≈ E02-long's 290.7 GPU-h) tested whether more compute de-collapses the 0.5-ep Muon bottleneck. **Result: REGRESSION — more compute collapsed it harder.** Stable end-to-end (grad_norm 0.4–0.8 mid, ~3.7 cosine tail, no divergence), eval_loss fell to a project-low **2.581** (best ckpt-272000). But **every concept gate regressed vs the 0.5-ep Muon arm**: within-sample RankMe **4.96/128** (was 10.57; centered 4.61 → genuine collapse, not offset), slot-mean 1.66, mean concept cosine 0.892; Δshuffle_beyond **0.227** (was 0.209 — flat from step 4k in W&B `concept_ablation/*`, so the decoder's long-range concept use was stationary for the whole 2-ep run); **STS-B zero-shot 0.062** (was 0.518 — now 0.42 *below* the token-embed-mean floor 0.486). Frozen-probe SICK-R mean→attn 0.048→0.160 (Δ+0.112, small distributed component). SICK-R 0.111, SICK-E 0.626, PAWS 0.562/0.305, GLUE MRPC 0.699/0.815, STSB 0.341, QQP 0.807, MNLI-m 0.613 (full-finetune, demoted). **Takeaway:** the "is it under-trained?" hypothesis for the windowed+Muon regime is **falsified** — the K=128 within-window bypass is the attractor; extra optimization makes the bypass better, not the concepts richer (opposite of E02-long's full-causal de-collapse where 5→16.7). eval_loss is orthogonal to concept quality (loss ↓ 2.606→2.581 while RankMe ↓ 10.6→5.0 and STS-B ↓ 0.518→0.062). **Closes the E05 from-scratch "more compute" branch.** Open confounds (do not block the pivot): ~~wd (Muon 0.1 vs Adam 0.0)~~ — **RESOLVED 2026-07-11 by [E05b](../experiments_specs/done_success/E05b_wd_confound_control.md): wd is innocent**; Tier-1 protocol split (0.5-ep old / 2-ep new — seq-512 flattens collapse so the regression holds a fortiori; recompute of the 0.5-ep arm under the new protocol queued). Reinforces the E10 pretrained-backbone pivot. **Mechanism deep-dive (2026-07-09):** no weight corruption (0 NaN/Inf, 229 tensors); the collapse is rank-1 of `enc.L5.bixt.rv_lat` (834/1536 dead rows) — 128 diverse slots all fed the same single-rank document summary. The LR=0.003 "grad-norm-rises-while-loss-falls" event is a real Edge-of-Stability threshold crossing (descendable-sharpness ceiling `2/(η·s)` lifts as η falls, crossing the bypass-gorge curvature → loss breaks a 116k-step plateau); wd is the proximate collapse driver (selective shrinkage of bypass-redundant directions, `muon.py:101`) — ~~hypothesis~~ **FALSIFIED 2026-07-11 by [E05b](../experiments_specs/done_success/E05b_wd_confound_control.md)** (Adam@wd=0.1 stays healthy: RankMe 30.88 vs Muon's 1.8–3.6 at identical wd=0.1); the collapse is Muon's full-rank whitened updates converging fast into the intrinsically low-rank bypass minimum, not wd. Turns the wd confound into a decisive cheap test (Adam@wd=0.1 control, [E05b](../experiments_specs/done_success/E05b_wd_confound_control.md)) + a minimal anti-collapse objective extension ([E05c](../experiments_specs/ahead/E05c_anticollapse_extension.md); lit [concept_bottleneck_collapse_mitigation](../literature_review/concept_bottleneck_collapse_mitigation.md)). **Status (2026-07-11): [E05b](../experiments_specs/done_success/E05b_wd_confound_control.md) EVALUATED — DECISIVE, wd innocent.** Adam@wd=0.1 (68.62 GPU-h / 19.13 kWh / 10.2 B tok) within-sample RankMe **30.88** (centered 32.17) — **3–6× Muon's 1.8–3.6 at identical wd=0.1**, Δshuffle_beyond **0.50** (clears Stage-2), active-slot 1.000, 0 NaN → the collapse is Muon-specific (full-rank whitened updates), not wd. [E05c](../experiments_specs/ahead/E05c_anticollapse_extension.md) (non-bypassable objective, config-only) is the next fix; [E05d](../experiments_specs/ahead/E05d_concept_vicreg.md) (VICReg) stays queued. Run report [e05b_wd_confound_control_20260711](../2_Experiments_Registry/run_reports/e05b_wd_confound_control_20260711.md). See [run report](../2_Experiments_Registry/run_reports/e05_muon_long_2ep_collapsed_20260709.md).
- Full history (with caveats): [master_experiment_log.md](../2_Experiments_Registry/master_experiment_log.md); older roadmap + TODO diary in [5_Archive/](../5_Archive/).

## Not active right now (still part of the Vision)
Recursive concept refinement and latent reasoning remain Vision goals — E08 and related
ideas stay in play; compose them only after a strictly causal platform demonstrably carries
content. E17c's carryless signal is not that platform. From-scratch and other bases are not ruled out. Diffusion
decode stays parked/revivable from `parked/`. Instruction SFT, long-context, and audio
remain long-term Vision only. Multi-agent latent communication stays the Stage-2
headline (see [team_brief](../sprind_frontier_ai/team_brief.md)).

## Engineering notes (not live experiments)
Canonical eval protocol, Tier-1 data-protocol upgrade, compute audit, and training-pipeline
modularization are done — see `docs/engineering_specs/` and
[evaluation_protocol.md](../3_Evaluations_and_Baselines/evaluation_protocol.md).
**2026-09-12 — `perceiver_ar` eval layer on `dev`:** lm-evaluation-harness adapter + SmolLM2-card
0-shot tiers, teacher-forced RULER-lite (`passkey`, `multikey`, `vt`, `fwe`, `buckets`, `reach`),
health check, two-GPU runner `scripts/eval_perceiver_ar_suite.sh`. Spec:
[long_context_reasoning_eval_layer.md](../engineering_specs/long_context_reasoning_eval_layer.md).
