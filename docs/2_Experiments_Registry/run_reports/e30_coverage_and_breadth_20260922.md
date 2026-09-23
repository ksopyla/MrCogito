# E30 coverage and breadth — how much each note should cover

**Date:** 2026-09-22
**Machine:** Odra 3×RTX 3090 and Polonez 4×RTX 3090, one probe per GPU
**Run ID:** `e30_broad`
**WandB:** n/a (on-the-fly probes; no `compute/*` audit)
**Best checkpoint:** none (models discarded)
**Git tag:** —
**Related:** [E30 spec](../../experiments_specs/ahead/E30_sliding_window_perceiver.md) · [length limits](e30_length_hardness_limits_20260922.md) · [training inventory](e30_training_inventory_20260923.md)

---

## Goal

The length ladder held coverage at the default (about 8 tokens per learned question) and stopped several curves early. This pass asks three things the ladder did not: how many tokens each note should cover, whether a missed exam was a step-size or a budget miss, and whether the chain and the far copy survive a longer book.

Passing means 75% of answer tokens. Prize is 64 bits unless noted. 31M is H=960, 4 layers, step 1e-4, warm residuals. Coverage 4 is a 128-token window, 32 questions, 352 notes at 1024 tokens (about 3 tokens per note). Coverage 8 is a 256-token window, 160 notes (about 6). Coverage 16 is a 512-token window, 96 notes (about 11).

## Coverage at 1024 tokens

| exam | coverage | full read | averaged notebook | sliding window |
|---|---|---|---|---|
| in-order 4-hop chain | 4 (4800 steps) | 63 bits, 99% | 4 bits, 39% | **36 bits, 76%** |
| in-order 4-hop chain | 8 (9600 steps, prior) | 63 bits, 99% | 4 bits, 39% | **40 bits, 77%** |
| in-order 4-hop chain | 16 (4800 steps) | 63 bits, 99% | 5 bits, 39% | **0 bits, 28%** |
| lookalike | 4 | 63 bits, 99% | 15 bits, 45% | 25 bits, 54% |
| lookalike | 8 (prior, 4800) | 63 bits, 99% | 19 bits, 52% | 26 bits, 56% |
| lookalike | 16 | 63 bits, 99% | 12 bits, 41% | **0 bits, 26%** |
| single lookup | 4 (extended) | 0 | 0 | **26 bits, 55%** |
| single lookup | 8 (prior, extended) | 0 | 0 | **26 bits, 54%** |
| single lookup | 16 (extended) | 0 | 20 bits, 49% | **0** |

The lookup row where the average scores 20 bits did not repeat. An earlier run with the same flags stayed at 0 bits through 4800 steps. Treat that 20-bit average as one escape from chance, not as a stable win over the sliding window.

## Window versus number of questions (chain, 1024)

Same 128-token window as the pass, but 16 questions instead of 32: 176 notes, **36 bits / 74%** after an extended budget. Just under the bar. Same 512-token window as the miss, but 64 questions instead of 32: 192 notes, **0 bits**. A long window writes nothing useful. More questions do not rescue it. Fewer questions in a short window almost pass, slower.

## Longer book, copy, shuffle, and the 50M step

| exam | setting | full read | averaged notebook | sliding window |
|---|---|---|---|---|
| in-order 4-hop chain, 2048 | coverage 4, 1e-4 | **63 bits, 99%** | 0 | 0 |
| in-order 4-hop chain, 2048 | coverage 4, 2e-4 | 0 | 0 | 0 |
| single lookup, 2048 | coverage 4, extended | 0 | 0 | **25 bits, 55%** |
| far copy, 1024 | coverage 8, 1e-4 | 2 bits, 27% | 9 bits, 36% | 3 bits, 29% |
| far copy, 1024 | coverage 4, 1e-4 | 2 bits | 4 bits | 2 bits |
| far copy, 1024 | coverage 8, 3e-4 | 0 | 0 | 2 bits |
| shuffled 2-hop, 1024 | coverage 4, no extra distractors | 0 | 0 | 0 |
| in-order 4-hop chain, 1024 | 50M, 5e-5, coverage 8 | **63 bits, 99%** | **46 bits, 84%** (budget doubled) | **42 bits, 79%** (stopped at the first budget, already flat) |

At matched steps the 50M sliding window was ahead: it had passed by 9600 steps, while the average was still near 67% of tokens around step 8800 and only reached 84% after the budget doubled.

## Reading

Concepts should cover a short stretch. About 3 tokens per note (a 128-token window) passes the in-order chain at 1024 tokens in half the steps the default needed. About 11 tokens per note (a 512-token window) is a kill on the chain, the lookalike, and the lookup. The kill is the window, not the note count: 192 notes in a long window still score nothing, and 176 notes in a short window reach 74%.

The chain does not survive 2048 tokens. Both notebooks stay at chance with the coverage that passed at 1024, while the full read is still perfect. Doubling the step to 2e-4 zeros the full read as well, so the miss is not a timid step. Finer notes do lift the 2048 lookup to the same half-answer plateau already seen at 1024 (25 bits, 55%). That is not a pass.

Far copy, solved at 512 tokens, is unsolved at 1024 by the full read. A larger step makes it worse. A 2-hop shuffled chain with no extra distractors is also unsolved by the full read, so that zero is the exam.

The earlier 50M story (full read collapses, only the sliding window passes) was too few steps at 1e-4. A slower step restores the full read. The averaged notebook can pass the same chain if it is trained about twice as long. The sliding window still gets there first, then flattens a few points short of that longer average.

Do not widen the model and do not start a 1T pretrain. The open limit is the in-order chain dying when the book doubles, at a coverage that already works at 1024 tokens.
