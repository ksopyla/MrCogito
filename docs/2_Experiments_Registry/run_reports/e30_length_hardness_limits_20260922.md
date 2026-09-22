# E30 length and hardness limits — 31M and 50M

**Date:** 2026-09-22
**Machine:** Odra 3×RTX 3090 and Polonez 4×RTX 3090, one probe per GPU
**Run ID:** `e30_limits`
**WandB:** n/a (on-the-fly probes; no `compute/*` audit)
**Best checkpoint:** none (models discarded)
**Git tag:** —
**Related:** [E30 spec](../../experiments_specs/ahead/E30_sliding_window_perceiver.md) · [31M hunt](e30_30m_gpu_odra_polonez_20260922.md)

---

## Goal

Find where the sliding-window write stops, against the full read and the averaged notebook, once the text is longer than 512 tokens and the exam is harder than a single lookup. Ask whether 50M parameters or a longer step budget buys an exam that 31M misses.

Scored rows use the slow step (1e-4) and warm residuals. A faster step (3e-4) scored zero for every architecture on both the 1024-token lookup and the lookalike. Those rows are a false kill and are not in the table.

Prize on these rungs is 64 bits. Passing token accuracy is 75%.

## Results

| exam | size | full read | averaged notebook | sliding window |
|---|---|---|---|---|
| in-order 4-hop chain, 1024 | 31M | 63.3 bits, 99% tokens | 4.0 bits, 39% | **40.1 bits, 77%** |
| in-order 4-hop chain, 1024 | 50M | 3.0 bits, 34% | 4.0 bits, 39% | **44.1 bits, 80%** |
| lookalike, 1024, 3200 steps | 31M | 62.3 bits, 99% | 9.5 bits | 22.0 bits, 51% |
| lookalike, 1024, 4800 steps | 31M | 62.6 bits, 99% | 18.7 bits, 52% | 25.9 bits, 56% |
| lookalike, 1024 | 50M | 62.9 bits, 99% | 16.6 bits | 25.3 bits, 53% |
| lookup, 1024, 3200 steps | 31M | 0 | 0 | 9.4 bits, 42% |
| lookup, 1024, 4800 steps | 31M | 0 | 0 | **25.7 bits, 54%** |
| lookup, 1024 | 50M | 0 | 0 | 24.3 bits, 53% |
| lookup, 2048 | 31M | 0 | 0 | 8.3 bits, 38% |
| lookup, 4096 | 31M | 0 | 0 | 0 |
| lookup, 16384 | 31M | 0 | 0 | 0 |
| shuffled 4-hop chain, 1024 | 31M and 50M | 0 | 0 | 0 |

## Reading

The sliding window learns the in-order chain. The average does not. At 50M the full read on that chain collapses and the sliding window still passes, a few bits higher than at 31M.

Longer training helps and then flattens. On the 1024-token lookup the sliding window went from 9 bits to 26 bits when the budget grew from 3200 to 4800 steps, and the other two stayed at zero. On the lookalike it went from 22 to 26 bits. Neither crossed a pass.

Width from 31M to 50M does not finish a missed exam. Lookup and lookalike at 50M match the longer 31M run. Shuffled chains are unsolved by the full read as well, so that zero is the exam, not the notebook.

The length wall for a single lookup is between 512 (previously solved) and 1024 (full read at zero, sliding window partial). By 4096 tokens nobody recovers the fact.
