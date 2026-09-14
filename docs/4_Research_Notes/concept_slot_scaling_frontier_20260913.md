# HARDER concept-slot scaling frontier (2026-09-13)

**Class:** dated measurement (append-only). CPU only. Does **not** rewrite
[the seq128 ~100% limits note](symbolic_arm_a_100pct_limits_20260913.md).
**Instrument:** `verification/symbolic_channel_probe.py` over
`data/symbolic_tasks.py`. Campaign runner: `verification/run_scale_hard_campaign.py`.
JSON: `/opt/cursor/artifacts/scale_hard/` (inventory:
`/opt/cursor/artifacts/harder_campaign_inventory.json`). Plots rebuilt after seq512
closed (2026-09-13):
[`harder_accuracy_vs_examples.png`](/opt/cursor/artifacts/harder_accuracy_vs_examples.png),
[`harder_accuracy_vs_steps.png`](/opt/cursor/artifacts/harder_accuracy_vs_steps.png),
[`harder_difficulty_vs_accuracy.png`](/opt/cursor/artifacts/harder_difficulty_vs_accuracy.png),
[`harder_lr_probe.png`](/opt/cursor/artifacts/harder_lr_probe.png),
[`harder_params_vs_max_seq.png`](/opt/cursor/artifacts/harder_params_vs_max_seq.png),
[`concept_slot_scaling_frontier.png`](/opt/cursor/artifacts/concept_slot_scaling_frontier.png)
(alias of the examples curve:
[`concept_slot_harder_learning_curves.png`](/opt/cursor/artifacts/concept_slot_harder_learning_curves.png)).

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
| easy seq128 r=8 | A | 1.35M | 3e-3 | 128 | 8 | far_copy | 2 | ~51k / 96k | 1,750 / 3,000 | 95% / 99.9% | 0.053 | **yes** | target 99% |
| easy seq128 r=8 | C | 0.60M | 3e-3 | 128 | 8 | far_copy | 2 | 96k | 3,000 | 25% | 1.386 | no (floor) | budget |
| easy seq128 r=8 | D | 0.60M | 3e-3 | 128 | 8 | far_copy | 2 | ~25k / 64k | 1,000 / 2,000 | 99.7% / 100% | 0.012 | **yes** | budget |
| easy seq128 r=16 | A | 1.35M | 3e-3 | 128 | 16 | far_copy | 2 | ~112k / 184k | 3,750 / 5,750 | 96% / 99.1% | 0.025 | **yes** | target 99% |
| lr1e03_a_seq256_r8 | A | 5.11M | 1e-3 | 256 | 8 | far_copy | 2 | 25.6k | 800 | 63.6% | 0.830 | no (probe) | budget |
| lr3e03_a_seq256_r8 | A | 5.11M | 3e-3 | 256 | 8 | far_copy | 2 | 25.6k | 800 | 25.1% | 1.386 | no | floor_patience |
| lr6e03_a_seq256_r8 | A | 5.11M | 6e-3 | 256 | 8 | far_copy | 2 | 25.6k | 800 | 25.1% | 1.386 | no | floor_patience |
| cell_seq256_r8_A | A | 5.11M | 1e-3 | 256 | 8 | far_copy | 2 | **96k** | 3,000 | **95.02%** | 0.135 | **yes** | target_acc |
| cell_seq256_r8_C | C | 2.27M | 1e-3 | 256 | 8 | far_copy | 2 | 48k | 1,500 | 25.4% | 1.386 | no (floor) | budget |
| cell_seq256_r8_D | D | 2.27M | 1e-3 | 256 | 8 | far_copy | 2 | **72k** | 2,250 | **99.17%** | 0.025 | **yes** | target_acc |
| cell_seq256_r32_A | A | 5.11M | 1e-3 | 256 | 32 | far_copy | 2 | 256k | 8,000 | **87.9%** | 0.258 | **no** | budget |
| cell_seq256_r32_D | D | 2.27M | 1e-3 | 256 | — | far_copy | 2 | 72k | 2,250 | 99.17% | 0.025 | **yes** (D ignores r) | reuse |
| cell_chain_h3_A | A | 5.11M | 1e-3 | 256 | 8 | chain | 3 | 128k | 4,000 | 23.7% | 1.387 | no | floor_patience |
| cell_chain_h3_D | D | 2.27M | 1e-3 | 256 | 8 | chain | 3 | 128k | 4,000 | 23.7% | 1.386 | no | floor_patience |
| cell_chain_h3_C | C | 2.27M | 1e-3 | 256 | 8 | chain | 3 | 48k | 1,500 | 25.1% | 1.386 | no (floor) | budget |
| cell_seq512_r8_A | A | 5.11M | **3e-4** | 512 | 8 | far_copy | 2 | **72k** | 2,250 | **97.05%** | 0.082 | **yes** | target_acc |
| cell_seq512_r8_C | C | 2.27M | 1e-3 | 512 | 8 | far_copy | 2 | 38k | 1,200 | 25.4% | 1.386 | no (floor) | budget |
| cell_seq512_r8_D | D | 2.27M | 3e-4 | 512 | 8 | far_copy | 2 | 96k | 3,000 | **29.9%** | 1.337 | **no** | budget |
| seq512 D lr=1e-3 | D | 2.27M | 1e-3 | 512 | 8 | far_copy | 2 | 104k | 3,250 | 27.8% | 1.343 | no | retuned |
| seq512 D lr=3e-3 | D | 2.27M | 3e-3 | 512 | 8 | far_copy | 2 | 64k | 2,000 | 25.2% | 1.368 | no | retuned |
| seq512 A lr=1e-3 | A | 5.11M | 1e-3 | 512 | 8 | far_copy | 2 | 72k | 2,250 | 24.5% | 1.386 | no | retuned to 3e-4 |

Array ablation on every finished A that left chance: removing the array returns ~24–26%.
The skill is in the slots. C stays at the floor on every new (seq, min_gap) and on chain.

## Kill / success against the 95% bar

- **Success (copy, 2× length, r=8):** 5.11M exclusive-slot A hits 95% at 96k unique
  examples. C stays at the floor. D hits 95% earlier (~65–72k). Doubling seq at
  fixed r costs A about **1.7×** examples versus the seq128 95% point (~51k → 96k).
- **Success (copy, 4× length, r=8, lower LR):** 5.11M A hits **97% at 72k examples**
  on seq=512 (64 slots) once LR is 3e-4. Example-count did **not** grow with length;
  the binding knob is LR. 1e-3, which won seq256, is lethal at seq512.
- **A beats width-matched D at seq512.** The 2.27M 4-layer dense decoder never left
  ~25–30% across 1e-3 / 3e-4 / 3e-3 through 64–104k examples. Caveat: D is
  **width-matched, not param-matched** (no encoder/latent). A param-matched dense
  stack might still solve it. Under the stated matching rule, the exclusive array
  is the only arm that carries a 32-letter span across a 512-token row at this
  depth.
- **Miss, not a kill (copy, 4× compression):** r=32 (8 slots, span occupies 1 slot)
  reaches **87.9% at 256k / 8k steps** and is still slowly climbing. Same 5.11M.
  The 95% bar is not met at this budget. D ignores r and is the seq256 D curve.
- **Exam kill (composition):** chain hops=3, key_len=8, seq=256: **both A and D**
  floor-killed at chance after 128k. C at floor (no leak). Per protocol this is
  **too hard for this budget**, not a concept-architecture failure. Short answers
  (8 supervised tokens) plus 3-hop lookup is the likely starve, same family as
  span=8 far_copy on the easy exam.
- **LR is not portable.** 3e-3 trains 1.35M/seq128; 1e-3 trains 5.11M/seq256; 3e-4
  trains 5.11M/seq512. Search LR per (hidden, seq) or the channel looks dead.

## Working law (measured, 95% bar, hidden frozen at 256 / 5.11M)

Not a Kaplan-style fit — too few cells. The measured pattern is:

1. **Length is cheap at r=8.** A solves `far_copy` through seq=512. Examples-to-95% did
   **not** grow with seq (96k at 256 → 72k at 512 once LR dropped to 3e-4). Wall did
   (36 min → 76 min). Lengthening seq without stretching `min_gap` adds slots, not reach.
2. **Compression is the binding axis.** r=8 solved; r=32 (32 letters in 1 of 8 slots)
   is **87.9% at 256k**, miss. Do not shrink width to chase this — spend examples or
   change the write.
3. **Composition at hops=3 / 8-token answers is an exam kill**, including for dense D.
   Not an exclusive-slot failure.
4. **C stays on the floor** on every new (seq, min_gap) and on chain. The exam does not leak.
5. **D matching matters.** Width-matched 2.27M D solves seq256 and misses seq512.
   Param-matched 4.97M D is in flight (60.5% @ 96k, still climbing). Until that cell
   hits or misses 95%, "A beats D at seq512" is not a same-parameter claim.

## Still unknown

- Whether r=32 A crosses 95% with more than 256k examples at the same 5.11M.
- hops=2 chain with longer answers (pack the loss like span=32) — hops=3/key=8
  was an exam kill.
- Seq>512 far_copy on GPU; nothing here is a language-model result.
- True reach (`min_gap = seq/4`) is queued in `verification/run_scale_hard_reach.py`
  after the live D9 / r=16 / hops=2 queue. Do not treat seq512 min_gap=32 as a
  480-token memory exam.

## In flight — param-matched D on seq512 (2026-09-13, live)

Width-matched D (4 layers, 2.27M) missed seq512 at ~30%. The goal needs the same
parameter regime, so D was deepened to **9 decoder layers = 4.97M** (closest to
A's 5.11M without shrinking A). Same exam, LR **3e-4**, CPU.

Live log `/opt/cursor/artifacts/scale_hard/cell_seq512_r8_D9.log` (not finished):

| examples | steps | acc | CE |
|---|---|---|---|
| 8k | 250 | 25.9% | 1.390 |
| 24k | 750 | 28.5% | 1.349 |
| 32k | 1,000 | 30.8% | 1.317 |
| 40k | 1,250 | 35.1% | 1.258 |
| 48k | 1,500 | 41.1% | 1.111 |
| 56k | 1,750 | 45.9% | 1.034 |
| 64k | 2,000 | 51.4% | 0.909 |
| 72k | 2,250 | 53.4% | 0.905 |
| 80k | 2,500 | 56.4% | 0.807 |
| 88k | 2,750 | 57.0% | 0.798 |
| 96k | 3,000 | **60.5%** | 0.698 |

Takeoff is sawtooth (eval CE stalls then drops) but the trend is up. 4-layer D
was 28% at 56k. A on this cell was **83% at 32k and 97% at 72k**. D9 is in the
same parameter band but **behind on data**: at 96k A was already over the 95%
bar and D9 is at 60.5%. Wall is similar per step (A 2.03 s/step, D9 1.94 s/step).
From 64k→96k the slope is ~2.3 pp / 8k examples; if that holds, 95% lands near
~220k (inside the 256k / 8k-step budget, ~3× A's 72k). Train CE 0.677 vs eval
0.698, so this is not an overfit stall.

Queued after this cell (frozen hidden=256): r=16, packed hops=2 chain, seq=1024.

## One-sentence frontier

A **<10M exclusive-slot** model (5.11M, hidden 256) hits the 95% bar on
**`far_copy` through seq=512, r=8, span=32 (64 slots)** at ~10^5 examples if LR is
tuned down with length; it **misses 95% at r=32** (87.9% @ 256k); **3-hop chain
with 8-token answers is unsolvable for both A and D** at this budget; and at
seq=512 the exclusive array **succeeds where a width-matched 4-layer dense
decoder does not**.

## What not to do next

Do not shrink hidden size. The easy campaign already showed 0.12M solves seq128; the
harder campaign is a **length / compression / hops / LR** law at a frozen <10M
width. Keep A at hidden=256 (5.11M). Next spend is more examples at r=32, a
param-matched D at seq512, or packed-loss chain — not a tinier model.
