# HARDER concept-slot scaling frontier (2026-09-13)

**Class:** dated measurement (append-only). CPU only. Does **not** rewrite
[the seq128 ~100% limits note](symbolic_arm_a_100pct_limits_20260913.md).
**Instrument:** `verification/symbolic_channel_probe.py` over
`data/symbolic_tasks.py`. Campaign runner: `verification/run_scale_hard_campaign.py`.
JSON: `/opt/cursor/artifacts/scale_hard/` (inventory:
`/opt/cursor/artifacts/harder_campaign_inventory.json`). Plots rebuilt 2026-09-13
from those logs (no retraining):
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

## Results so far

Winner LR = **1e-3**. 3e-3 and 6e-3 (the easy-end LRs) floor-kill this geometry.

| run | arm | params | LR | seq | r | task | hops | examples | steps | acc | CE | 95%? | stop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| easy seq128 r=8 | A | 1.35M | 3e-3 | 128 | 8 | far_copy | 2 | 56k (interp ~51k) | 1,750 | 98.2% → 99.9% @ 96k | 0.053 | **yes** | target 99% |
| easy seq128 r=8 | C | 0.60M | 3e-3 | 128 | 8 | far_copy | 2 | 96k | 3,000 | 25% | 1.386 | no (floor) | budget |
| easy seq128 r=8 | D | 0.60M | 3e-3 | 128 | 8 | far_copy | 2 | 32k (interp ~25k) | 1,000 | 99.7% → 100% @ 64k | 0.012 | **yes** | budget |
| easy seq128 r=16 | A | 1.35M | 3e-3 | 128 | 16 | far_copy | 2 | 120k (interp ~112k) | 3,750 | 96.2% → 99.1% @ 184k | 0.025 | **yes** | target 99% |
| lr1e03_a_seq256_r8 | A | 5.11M | 1e-3 | 256 | 8 | far_copy | 2 | 25.6k | 800 | 63.6% | 0.830 | no (probe) | budget |
| lr3e03_a_seq256_r8 | A | 5.11M | 3e-3 | 256 | 8 | far_copy | 2 | 25.6k | 800 | 25.1% | 1.386 | no | floor_patience |
| lr6e03_a_seq256_r8 | A | 5.11M | 6e-3 | 256 | 8 | far_copy | 2 | 25.6k | 800 | 25.1% | 1.386 | no | floor_patience |
| cell_seq256_r8_A | A | 5.11M | 1e-3 | 256 | 8 | far_copy | 2 | **96k** | 3,000 | **95.02%** | 0.135 | **yes** | target_acc |
| cell_seq256_r8_C | C | 2.27M | 1e-3 | 256 | 8 | far_copy | 2 | 48k | 1,500 | 25.4% | 1.386 | no (floor) | budget |
| cell_seq256_r8_D | D | 2.27M | 1e-3 | 256 | 8 | far_copy | 2 | **72k** (interp ~65k) | 2,250 | **99.17%** | 0.025 | **yes** | target_acc |
| cell_seq256_r32_A | A | 5.11M | 1e-3 | 256 | 32 | far_copy | 2 | 256k | 8,000 | **87.9%** | 0.258 | **no**, still climbing | budget |
| cell_seq256_r32_D | D | 2.27M | 1e-3 | 256 | — | far_copy | 2 | 72k | 2,250 | 99.17% | 0.025 | **yes** (reuse: D ignores r) | target_acc |
| cell_chain_h3_A | A | 5.11M | 1e-3 | 256 | 8 | chain | 3 | 128k | 4,000 | 23.7% | 1.387 | no | floor_patience |
| cell_chain_h3_D | D | 2.27M | 1e-3 | 256 | 8 | chain | 3 | 48k+ | 1,500+ | ~25% | ~1.387 | unknown | **in flight** (still chance at 48k; D far_copy took off ~48–56k) |
| cell_chain_h3_C | C | — | — | 256 | 8 | chain | 3 | — | — | — | — | — | not started |
| cell_seq512_r8_A/C/D | — | — | — | 512 | 8 | far_copy | 2 | — | — | — | — | — | not started |

Array ablation on every finished A that left chance: removing the array returns ~25%.
The skill is in the slots.

## Kill / success against the 95% bar

- **Success (copy, 2× length, r=8):** 5.11M exclusive-slot A hits 95% at 96k unique
  examples. C stays at the floor (exam does not leak). D hits 95% earlier (~65–72k)
  and 99% by 72k, cheaper per step. Doubling seq at fixed r costs A about **1.7×**
  examples versus the seq128 95% point (~51k → 96k).
- **Open, not a kill (copy, 4× compression):** r=32 (8 slots, span occupies 1 slot)
  reaches 87.9% at the 256k / 8k-step budget and is still rising. Same 5.11M; do **not**
  shrink. More examples at this width is the next measurement, not a smaller model.
- **Fail so far (composition):** chain hops=3, key/value length 8, is floor-killed
  for A at 128k. That is not yet a concept-architecture kill: D has not finished,
  so we do not know whether the exam is solvable at this budget.
- **LR is not portable from the easy end.** 3e-3, which trained the 1.35M seq128
  model, is lethal at hidden=256 / seq=256. Search LR per geometry.

## Still unknown / in flight

- `cell_chain_h3_D` (running): if D hits 95%, hops=3 is a real exclusive-slot miss
  at 5.11M. If D also stays at chance, kill the cell as "exam too hard", not an A fail.
- `cell_chain_h3_C` and the whole **seq=512 r=8** triple have not started.
- Whether r=32 A crosses 95% with more than 256k examples at the same 5.11M.
- Seq>256 far_copy, hops>3, and a param-matched (not just width-matched) C/D.
- Nothing here is a language-model result. It is a closed-form channel exam.

## One-sentence frontier

The longest / hardest task a **<10M exclusive-slot** model has actually solved so far
is **`far_copy` seq=256, r=8, span=32 (32 slots), 5.11M params, 95.0% at 96k examples**;
r=32 is close but unsolved at 256k, and 3-hop chain is still at chance.

## What not to do next

Do not shrink hidden size. The easy campaign already showed 0.12M solves seq128; the
harder campaign is a **length / compression / hops** law at a frozen <10M width.
Keep A at hidden=256 (5.11M) and spend examples, not width cuts.
