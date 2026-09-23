# HARDER concept-slot scaling frontier (2026-09-13)

**Class:** dated measurement (append-only). CPU only. Does **not** rewrite
[the seq128 ~100% limits note](symbolic_arm_a_100pct_limits_20260913.md).
Ledger summary:
[exclusive_slot_under10m_law_20260915](../2_Experiments_Registry/run_reports/exclusive_slot_under10m_law_20260915.md).
**Instrument:** `verification/symbolic_channel_probe.py` over
`data/symbolic_tasks.py`. Campaign runner: `verification/run_scale_hard_campaign.py`.
JSON: `/workspace/Cache/scale_hard/` (durable inventory:
`docs/4_Research_Notes/exclusive_slot_law_inventory.json`). Plots:
[`exclusive_slot_task_comparisons.png`](figures/exclusive_slot_task_comparisons.png)
(per-task A=concepts vs D=dense vs C=leak),
[`exclusive_slot_working_law.png`](figures/exclusive_slot_working_law.png),
[`exclusive_slot_law_steps_sizes_acc.png`](figures/exclusive_slot_law_steps_sizes_acc.png).
The Cursor `/opt/cursor/artifacts` FUSE store wiped mid-campaign (seq1024 A
hit 95.9% then `Path.write_text` raised `FileNotFoundError`).
First merge-to-dev snapshot (2026-09-13, rebuilt from logs, no retraining):
JSON `/opt/cursor/artifacts/scale_hard/` (inventory
`/opt/cursor/artifacts/harder_campaign_inventory.json`); panels
[`harder_accuracy_vs_examples.png`](/opt/cursor/artifacts/harder_accuracy_vs_examples.png),
[`harder_accuracy_vs_steps.png`](/opt/cursor/artifacts/harder_accuracy_vs_steps.png),
[`harder_difficulty_vs_accuracy.png`](/opt/cursor/artifacts/harder_difficulty_vs_accuracy.png),
[`harder_lr_probe.png`](/opt/cursor/artifacts/harder_lr_probe.png),
[`harder_params_vs_max_seq.png`](/opt/cursor/artifacts/harder_params_vs_max_seq.png),
[`concept_slot_scaling_frontier.png`](/opt/cursor/artifacts/concept_slot_scaling_frontier.png)
(alias [`concept_slot_harder_learning_curves.png`](/opt/cursor/artifacts/concept_slot_harder_learning_curves.png)).

## Hypothesis

The exclusive-scope concept array has a measurable **length × compression × composition**
frontier **below 10M params**, scored at **95%** on supervised tokens (not 100%).

A <10M exclusive-slot model (Arm A) should copy a 32-letter span that its decoder cannot
see as text, using only the slot array, at a harder exam than seq=128 / r=8. Matched
controls at the same hidden size and decoder depth: **C** (no array, floor) and **D**
(dense full-causal). If D cannot hit 95%, the exam is too hard for this compute/data
budget, not a concept failure. If C leaves the floor, the exam leaks.

The model is **not** shrunk. Reference A is frozen at hidden=256, enc=2, latent=2,
dec=4, tok_emb=32 → **5.11M**. C/D drop encoder/pooler/latent → **2.27M**.

## Protocol

- Exam: `far_copy` unless the cell is `chain`. `n_symbols=4`, chance **25%**, floor
  **1.3863 nats**. `span_len=32`, `min_gap=32`, `dec_segment=32`, exclusive scope,
  warm `wo` init 0.02, `warmup_constant` LR (warmup 200), batch 32, seed 0, CPU.
- Bar: **95%** eval accuracy on supervised tokens (128 held-out rows).
- LR search on the first hard cell (seq256 r=8 A), then freeze the winner for the grid.
- Early-stop at 95%; floor-kill if still at chance after `floor_patience_steps`.
- Easy end (already measured, do not rerun): seq=128, r=8, hidden=128, ~1.35M A /
  0.60M C,D. See `/opt/cursor/artifacts/scale/` and the 100% limits note.

## Results (campaign closed)

Winner LR on seq256 r=8 = **1e-3**. 3e-3 and 6e-3 (the easy-end LRs) floor-kill that
geometry. Seq512 A needed a further drop to **3e-4** (1e-3 stuck at chance for 2250
steps). **Tune LR per geometry; it is not portable from the 1.35M toy.**

| run | arm | params | LR | seq | r | task | hops | examples | steps | acc | CE | 95%? | stop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| easy seq128 r=8 | A | 1.35M | 3e-3 | 128 | 8 | far_copy | 2 | 56k (interp ~51k) / 96k | 1,750 / 3,000 | 98.2% → 99.9% @ 96k | 0.053 | **yes** | target 99% |
| easy seq128 r=8 | C | 0.60M | 3e-3 | 128 | 8 | far_copy | 2 | 96k | 3,000 | 25% | 1.386 | no (floor) | budget |
| easy seq128 r=8 | D | 0.60M | 3e-3 | 128 | 8 | far_copy | 2 | 32k (interp ~25k) / 64k | 1,000 / 2,000 | 99.7% → 100% @ 64k | 0.012 | **yes** | budget |
| easy seq128 r=16 | A | 1.35M | 3e-3 | 128 | 16 | far_copy | 2 | 120k (interp ~112k) / 184k | 3,750 / 5,750 | 96.2% → 99.1% @ 184k | 0.025 | **yes** | target 99% |
| lr1e03_a_seq256_r8 | A | 5.11M | 1e-3 | 256 | 8 | far_copy | 2 | 25.6k | 800 | 63.6% | 0.830 | no (probe) | budget |
| lr3e03_a_seq256_r8 | A | 5.11M | 3e-3 | 256 | 8 | far_copy | 2 | 25.6k | 800 | 25.1% | 1.386 | no | floor_patience |
| lr6e03_a_seq256_r8 | A | 5.11M | 6e-3 | 256 | 8 | far_copy | 2 | 25.6k | 800 | 25.1% | 1.386 | no | floor_patience |
| cell_seq256_r8_A | A | 5.11M | 1e-3 | 256 | 8 | far_copy | 2 | **96k** | 3,000 | **95.02%** | 0.135 | **yes** | target_acc |
| **cell_seq256_r16_A** | A | 5.11M | 1e-3 | 256 | 16 | far_copy | 2 | **224k** | 7,000 | **95.7%** | 0.110 | **yes** | target_acc |
| cell_seq256_r8_C | C | 2.27M | 1e-3 | 256 | 8 | far_copy | 2 | 48k | 1,500 | 25.4% | 1.386 | no (floor) | budget |
| cell_seq256_r8_D | D | 2.27M | 1e-3 | 256 | 8 | far_copy | 2 | **72k** (interp ~65k) | 2,250 | **99.17%** | 0.025 | **yes** | target_acc |
| cell_seq256_r32_A | A | 5.11M | 1e-3 | 256 | 32 | far_copy | 2 | 256k | 8,000 | **87.9%** | 0.258 | **no**, still climbing | budget |
| cell_seq256_r32_D | D | 2.27M | 1e-3 | 256 | — | far_copy | 2 | 72k | 2,250 | 99.17% | 0.025 | **yes** (D ignores r) | reuse |
| cell_chain_h3_A | A | 5.11M | 1e-3 | 256 | 8 | chain | 3 | 128k | 4,000 | 23.7% | 1.387 | no | floor_patience |
| cell_chain_h3_D | D | 2.27M | 1e-3 | 256 | 8 | chain | 3 | 128k | 4,000 | 23.7% | 1.386 | no | floor_patience |
| cell_chain_h3_C | C | 2.27M | 1e-3 | 256 | 8 | chain | 3 | 48k | 1,500 | 25.1% | 1.386 | no (floor) | budget |
| **cell_chain_h2_D9** | D | **4.97M** | 3e-4 | 512 | 8 | chain | 2 | 128k | 4,000 | 25.7% | 1.386 | no | floor_patience |
| cell_seq512_r8_A | A | 5.11M | **3e-4** | 512 | 8 | far_copy | 2 | **72k** | 2,250 | **97.05%** | 0.082 | **yes** | target_acc |
| cell_seq512_r8_C | C | 2.27M | 1e-3 | 512 | 8 | far_copy | 2 | 38k | 1,200 | 25.4% | 1.386 | no (floor) | budget |
| cell_seq512_r8_D | D | 2.27M | 3e-4 | 512 | 8 | far_copy | 2 | 96k | 3,000 | **29.9%** | 1.337 | **no** | budget |
| **cell_seq512_r8_D9** | D | **4.97M** | **3e-4** | 512 | 8 | far_copy | 2 | **120k** | 3,750 | **98.5%** | 0.042 | **yes** | target_acc |
| **reach_seq512_A** | A | 5.11M | **3e-4** | 512 | 8 | far_copy gap128 | 2 | **48k** | 1,500 | **95.0%** | 0.140 | **yes** | target_acc |
| **reach_seq512_D9** | D | **4.97M** | **3e-4** | 512 | 8 | far_copy gap128 | 2 | **32k** | 1,000 | **98.3%** | 0.047 | **yes** | target_acc |
| **reach_seq512_C** | C | 2.27M | 3e-4 | 512 | 8 | far_copy gap128 | 2 | 38k | 1,200 | 25.4% | 1.386 | no (floor) | budget |
| **reach_seq1024_A** | A | 5.11M | **1e-4** | 1024 | 8 | far_copy gap256 | 2 | **96k** | 3,000 | **95.9%** | 0.109 | **yes** | target_acc |
| **reach_seq1024_D9** | D | **4.97M** | **1e-4** | 1024 | 8 | far_copy gap256 | 2 | **136k** | 4,250 | **97.1%** | 0.080 | **yes** | target_acc |
| **reach_seq1024_C** | C | 2.27M | 1e-4 | 1024 | 8 | far_copy gap256 | 2 | 38k | 1,200 | 26.0% | 1.386 | no (floor) | budget |
| seq512 D lr=1e-3 | D | 2.27M | 1e-3 | 512 | 8 | far_copy | 2 | 104k | 3,250 | 27.8% | 1.343 | no | retuned |
| seq512 D lr=3e-3 | D | 2.27M | 3e-3 | 512 | 8 | far_copy | 2 | 64k | 2,000 | 25.2% | 1.368 | no | retuned |
| seq512 A lr=1e-3 | A | 5.11M | 1e-3 | 512 | 8 | far_copy | 2 | 72k | 2,250 | 24.5% | 1.386 | no | retuned to 3e-4 |

Array ablation on every finished A that left chance: removing the array returns ~24–26%.
The skill is in the slots. C stays at the floor on every new (seq, min_gap) and on chain.

## Kill / success against the 95% bar

- **Success (copy, 2× length, r=8):** 5.11M exclusive-slot A hits 95% at 96k unique
  examples. C stays at the floor. D hits 95% earlier (~65–72k). Doubling seq at
  fixed r costs A about **1.7×** examples versus the seq128 95% point (~51k → 96k).
- **Success (true reach, seq=512 min_gap=128):** 5.11M A hits **95.0% at 48k / 40 min**
  (ablation → 24.5%). Param-matched 4.97M D hits **98.3% at 32k / 32 min**. C stays
  at 25.4% (no leak). On this farther-span exam D is *more* data-efficient than A
  (opposite of padded min_gap=32).
- **Success (true reach, seq=1024 min_gap=256, A):** 5.11M A hits **95.9% at 96k / 3000
  steps / ~5.5 h CPU** (lr=1e-4, 6.63 s/step, 128 slots). Ablation → 23.8%. Same
  example-count as seq256 r=8. Durable record:
  `docs/4_Research_Notes/exclusive_slot_law_inventory.json`.
- **Success (true reach, seq=1024 min_gap=256, matched D9 + C, 2026-09-15):** param-matched
  4.97M D hits **97.1% at 136k / 4250 steps / 887 min** (lr=1e-4, 12.52 s/step,
  early-stop). A is cheaper on the same exam: **0.71× examples and 0.37× wall**.
  Leak-check C stays at **26.0% / CE 1.3861 (−0.0002 nats vs floor)** through 38k /
  1200 steps / 73 min. JSON:
  `/workspace/Cache/scale_hard/reach_seq1024_D9.json`,
  `/workspace/Cache/scale_hard/reach_seq1024_C.json`.
- **A beats width-matched D at seq512, and is more efficient than param-matched D.**
  The 2.27M 4-layer dense decoder never left ~25–30% across 1e-3 / 3e-4 / 3e-3 through
  64–104k examples. Deepening D to **9 layers / 4.97M** (same parameter band as A's 5.11M)
  hits **98.5% at 120k examples / 121 min**. A on the same exam hit **97% at 72k / 76 min**.
  Same compute class, same data generator, ~5M params: exclusive slots need **0.60× examples
  and 0.63× wall**. Width matching was the false "D cannot solve seq512" claim; param matching
  is an **efficiency** win for the array, not a solvability claim.
- **Success (copy, 2× compression, r=16):** 5.11M A hits **95.7% at 224k / 77 min**
  (16 slots, span occupies 2). Ablation: array removed → 25.1%. ~**2.3×** the
  examples r=8 needed on the same seq. r=32 (1 slot for the span) still misses
  95% at 256k (87.9%).
- **Exam kill (composition, packed hops=2):** chain hops=2, key_len=value_len=32,
  seq=512, param-matched 4.97M D: **25.7% at 128k / 4000 steps**, CE glued to the
  floor. Same budget where this D hit 98.5% on far_copy. Packing the loss does
  not make composition D-green at <10M. A was not run (exam-too-hard protocol).
- **Exam kill (composition, hops=3 / 8-token answers):** chain hops=3, key_len=8,
  seq=256: **both A and D** floor-killed at chance after 128k. C at floor (no leak).
- **LR is not portable.** 3e-3 trains 1.35M/seq128; 1e-3 trains 5.11M/seq256; 3e-4
  trains 5.11M/seq512. Search LR per (hidden, seq) or the channel looks dead.

## Working law (measured, 95% bar, hidden frozen at 256 / 5.11M)

Not a Kaplan-style fit — too few cells. The measured pattern is:

1. **Length is cheap at r=8 through seq=1024 true reach, vs both C and D.** A
   solves `far_copy` seq=512 min_gap=32 (72k / 76 min), min_gap=128 (48k / 40 min),
   **and seq=1024 min_gap=256 (96k / 5.5 h, lr=1e-4)**. Param-matched D9 also
   solves seq=1024 (**97.1% at 136k / 887 min**); A uses **0.71× examples and
   0.37× wall**. C stays at chance. E95 stays on the order of 10^5 examples.
   Wall grows with seq (A 1.6 s → 6.6 s; D9 12.5 s/step at seq=1024).
2. **Compression is the binding axis.** r=8 solved at 96k; r=16 solved at 224k
   (~2.3× data); r=32 missed at 256k (87.9%). Doubling tokens/slot more than
   doubles examples-to-95%; another doubling does not finish in 256k. Do not
   shrink width to chase this.
3. **Composition is an exam kill at this budget**, including packed hops=2 /
   32-token answers on param-matched 4.97M D (25.7% @ 128k) and hops=3 / 8-token
   answers on A and D. Not an exclusive-slot failure.
4. **C stays on the floor** on every new (seq, min_gap), including seq=1024
   min_gap=256 (26.0%, −0.0002 nats), and on chain. The exam does not leak.
5. **D matching is exam-dependent.** Width-matched 2.27M D misses padded seq512.
   Param-matched 4.97M D **solves padded seq512, true-reach seq512, and
   true-reach seq=1024**. A is cheaper on padded seq512 (0.60× examples) and on
   seq=1024 (0.71× examples); D9 is cheaper on true-reach seq512 (32k vs A's 48k).
   Exclusive slots are a competitive <10M solver, not uniformly the data-cheapest one.

## Still unknown

- Whether r=32 A crosses 95% with more than 256k examples at the same 5.11M
  (r=16 now hits 95.7% at 224k).
- hops=2 packed chain is now measured: param-matched D stays at chance through
  128k. Not D-green; A skipped per protocol.

## Closed — param-matched D on seq512 (2026-09-14)

Width-matched D (4 layers, 2.27M) missed seq512 at ~30%. D deepened to **9 decoder
layers = 4.97M** (closest to A's 5.11M) on the same exam, LR **3e-4**, CPU.
JSON: `/opt/cursor/artifacts/scale_hard/cell_seq512_r8_D9.json`.

| examples | steps | acc | CE | wall |
|---|---|---|---|---|
| 8k | 250 | 25.9% | 1.390 | 8 min |
| 32k | 1,000 | 30.8% | 1.317 | 32 min |
| 56k | 1,750 | 45.9% | 1.034 | 57 min |
| 64k | 2,000 | 51.4% | 0.909 | 65 min |
| 72k | 2,250 | 53.4% | 0.905 | 73 min |
| 96k | 3,000 | 60.5% | 0.698 | 97 min |
| 104k | 3,250 | 62.6% | 0.649 | 105 min |
| 112k | 3,500 | 82.2% | 0.362 | 113 min |
| **120k** | **3,750** | **98.5%** | 0.042 | **121 min** |

Early-stop at the 95% bar. Takeoff was sawtooth until 104k, then a sharp drop
(62.6% → 82.2% → 98.5% in 16k examples). A was already 97% at 72k / 76 min.
**Same-parameter verdict:** both arms solve seq512 r=8; exclusive slots use
**0.60× examples and 0.63× wall**.

Closed 2026-09-14: seq=1024 min_gap=256 A hits 95.9% at 96k. Plots:
`docs/4_Research_Notes/figures/exclusive_slot_working_law.png`,
`docs/4_Research_Notes/figures/exclusive_slot_law_steps_sizes_acc.png`.

## In flight — seq1024 matched D9 and C (2026-09-14)

Live snapshot kept as a historical trace (PID 79146,
`/workspace/Cache/scale_hard/reach_seq1024_dc.log`). Closed the next day; see
the 2026-09-15 section.

| arm | steps | examples | acc | eval CE | s/step |
|---|---|---|---|---|---|
| D9 dense 9L | 250 | 8k | 25.1% | 1.391 | 10.16 |
| D9 dense 9L | 500 | 16k | 26.4% | 1.386 | 13.03 |

## Closed — seq1024 matched D9 and C (2026-09-15)

Same exam as A (far_copy seq=1024, min_gap=256, span=32, r=8, 128 slots, lr=1e-4).
A was not rerun. JSON:
`/workspace/Cache/scale_hard/reach_seq1024_D9.json`,
`/workspace/Cache/scale_hard/reach_seq1024_C.json`.

D9 takeoff (4.97M, 9 decoder layers):

| examples | steps | acc | CE | s/step |
|---|---|---|---|---|
| 8k | 250 | 25.1% | 1.391 | 10.16 |
| 64k | 2,000 | 24.9% | 1.369 | 15.65 |
| 72k | 2,250 | 34.1% | 1.301 | 16.82 |
| 96k | 3,000 | 50.4% | 1.077 | 14.54 |
| 104k | 3,250 | 62.4% | 0.851 | 13.99 |
| 112k | 3,500 | 91.0% | 0.243 | 13.51 |
| 120k | 3,750 | 93.2% | 0.171 | 13.14 |
| 128k | 4,000 | 94.6% | 0.144 | 12.80 |
| **136k** | **4,250** | **97.1%** | **0.080** | **12.52** |

Early-stop at the 95% bar. Same sawtooth-then-cliff as padded seq512 D9, delayed
by ~16k examples (98.5% @ 120k there; 97.1% @ 136k here). A on this exam was
already 95.9% at 96k / 331 min. **Same-parameter verdict at seq=1024:** both
arms solve it; exclusive slots use **0.71× examples and 0.37× wall**.

C leak-check (2.27M, 4 decoder layers, 1200 steps, no 95% target):

| examples | steps | acc | CE |
|---|---|---|---|
| 8k | 250 | 25.0% | 1.392 |
| 16k | 500 | 25.9% | 1.387 |
| 24k | 750 | 26.9% | 1.386 |
| 32k | 1,000 | 25.2% | 1.386 |
| 38k | 1,200 | 26.0% | 1.386 |

Verdict `[ok] task does not leak`: −0.0002 nats vs floor. Acc oscillates around
chance. The seq=1024 exam is not a local-window giveaway.

## One-sentence frontier

A **<10M exclusive-slot** model (5.11M, hidden 256) hits the 95% bar on
**`far_copy` through seq=1024, r=8, min_gap=256 (128 slots, 95.9% @ 96k)** —
**cheaper than param-matched 4.97M D9 (97.1% @ 136k)** and with C on the floor
(26.0%, no leak) — and on seq=512 both padded and true-reach; **r=16 hits 95.7%
at 224k**; it **misses 95% at r=32** (87.9% @ 256k); **composition is unsolvable
for param-matched D** at this budget; width-matched 4-layer D misses padded
seq512.

**What a <10M model can handle vs C/D:** length through seq=1024 true-reach at
r=8 (A and D9 both clear 95%; C does not); compression through r=16 not r=32;
composition not at all at this budget.

## What not to do next

Do not shrink hidden size. The easy campaign already showed 0.12M solves seq128; the
harder campaign is a **length / compression / hops / LR** law at a frozen <10M
width. Keep A at hidden=256 (5.11M). The length axis is closed through seq=1024
true reach **against matched D9 and leak-check C**. Next spend after a positive
signal is r=32 with more than 256k examples, or composition that is D-green —
not a tinier model.
