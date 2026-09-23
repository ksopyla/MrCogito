# E30 training inventory — what was actually trained

**Date:** 2026-09-23
**Scope:** every scored GPU probe of the sliding-window notebook (E30), the averaged notebook (E21, one frozen mean per 16 tokens), and the full read (dense). Models were discarded after each probe. No text pretraining.
**Sources:** [31M hunt](e30_30m_gpu_odra_polonez_20260922.md) · [length limits](e30_length_hardness_limits_20260922.md) · [coverage sweep](e30_coverage_and_breadth_20260922.md)

Passing means 75% of the answer tokens correct.

## Shared architecture

Four layers, always in this order: one local layer (sees 16 tokens), one global layer (can see every note, or every token in the full read), then two more local layers (16 tokens). The notes do not talk to each other while they are written. Multi-hop has to happen in that single global layer.

| size | hidden | query heads | KV heads | head dim | feed-forward | parameters |
|---|---|---|---|---|---|---|
| 31M | 960 | 15 | 1 | 64 | 1920 | full read 31.00M · averaged 31.12M · sliding window 31.13M |
| 50M | 1216 | 19 | 1 | 64 | 2432 | full read 49.54M · averaged 49.69M · sliding window 49.70M |

Write side, sliding window: 8 heads, question width 128, 32 learned questions per window unless a row says otherwise. Windows overlap by a quarter (stride is 75% of the window). Token embedding width is 32. The averaged notebook is not this write: it stores one average per 16 tokens, so its per-note reach is always 16, with no overlap.

**Concept reach** here means how many tokens one note can see while it is written. It is the window, not the whole book. The notebook as a whole still covers the book, because the windows tile it.

| book length | coverage | questions / window | window (reach) | windows | notes | tokens / note |
|---|---|---|---|---|---|---|
| 128 | 8 (auto-shrunk) | 8 | 64 | 3 | 24 | 5.3 |
| 256 | 8 (auto-shrunk) | 16 | 128 | 3 | 48 | 5.3 |
| 512 | 8 | 32 | 256 | 3 | 96 | 5.3 |
| 1024 | 4 | 32 | **128** | 11 | **352** | 2.9 |
| 1024 | 8 | 32 | **256** | 5 | **160** | 6.4 |
| 1024 | 16 | 32 | **512** | 3 | **96** | 10.7 |
| 1024 | 8 | 16 | 128 | 11 | 176 | 5.8 |
| 1024 | 16 | 64 | 512 | 3 | 192 | 5.3 |
| 2048 | 4 | 32 | 128 | 21 | 672 | 3.0 |
| 2048 | 8 | 32 | 256 | 11 | 352 | 5.8 |
| 4096 | 8 | 32 | 256 | 21 | 672 | 6.1 |
| 16384 | 8 | 32 | 256 | 85 | 2720 | 6.0 |

The averaged notebook at the same lengths has 8, 16, 32, 64, 128, 256, 1024 notes (length ÷ 16). Each of those notes reaches 16 tokens.

## Tasks, by length

Step size is the learning rate that was actually scored. "Short window" is reach 128 (coverage 4). "Medium" is reach 256 (coverage 8). "Long" is reach 512 (coverage 16).

| task | length | size | reach | notes | step | sliding window | averaged notebook | full read |
|---|---|---|---|---|---|---|---|---|
| copy a marked span | 128 | 31M | 64 | 24 | 3e-4 | 60 / 28 bits (copy / match) | 43 / 15 | 63 / 31 |
| pick the right fact vs a lookalike | 128 | 31M | 64 | 24 | 3e-4 | 28 bits | 9 bits | 31 bits |
| copy a marked span | 256 | 31M | 128 | 48 | 3e-4 | 46 / 47 bits | 42 / 41 | 47 / 47 |
| copy a marked span | 512 | 31M | 256 | 96 | 3e-4 | 63 / 48 bits | 61 / 44 | 64 / 48 |
| in-order 4-hop chain | 1024 | 31M | 128 | 352 | 1e-4 | **76% tokens, pass** | 39% | 99% |
| in-order 4-hop chain | 1024 | 31M | 256 | 160 | 1e-4 | **77% tokens, pass** (twice the steps) | 39% | 99% |
| in-order 4-hop chain | 1024 | 31M | 512 | 96 | 1e-4 | 28%, 0 bits | 39% | 99% |
| in-order 4-hop chain | 1024 | 31M | 128 | 176 | 1e-4 | 74%, just under the bar | 38% | 99% |
| in-order 4-hop chain | 1024 | 31M | 512 | 192 | 1e-4 | 0 bits | 40% | 99% |
| in-order 4-hop chain | 1024 | 50M | 256 | 160 | 5e-5 | **79%**, then flat | **84%** after twice the steps | 99% |
| lookalike | 1024 | 31M | 128 or 256 | 352 or 160 | 1e-4 | ~54–56%, not a pass | ~45–52% | 99% |
| lookalike | 1024 | 31M | 512 | 96 | 1e-4 | 0 bits | 41% | 99% |
| lookalike | 1024 | 50M | 256 | 160 | 1e-4 | 53% | ~half or less | 99% |
| single lookup | 1024 | 31M | 128 or 256 | 352 or 160 | 1e-4 | **~55%**, stuck | 0 (one run later reached 49%; it did not repeat) | 0 |
| single lookup | 1024 | 50M | 256 | 160 | 1e-4 | 53% | 0 | 0 |
| single lookup | 2048 | 31M | 128 | 672 | 1e-4 | **55%**, same ceiling | 0 | 0 |
| single lookup | 2048 | 31M | 256 | 352 | 1e-4 | 38%, still moving when stopped | 0 | 0 |
| single lookup | 4096 and 16384 | 31M | 256 | 672 / 2720 | 1e-4 | 0 | 0 | 0 |
| in-order 4-hop chain | 2048 | 31M | 128 | 672 | 1e-4 | **0** | 0 | **99%** |
| in-order 4-hop chain | 2048 | 31M | 128 | 672 | 2e-4 | 0 | 0 | 0 (step too large) |
| copy a far span | 1024 | 31M | 128 or 256 | 352 or 160 | 1e-4 | ~2–3 bits | 4–9 bits | ~2 bits |
| copy a far span | 1024 | 31M | 256 | 160 | 3e-4 | ~2 bits | 0 | 0 |
| shuffled 4-hop chain | 1024 | 31M and 50M | 256 | 160 | 1e-4 | 0 | 0 | 0 |
| shuffled 2-hop, no extra distractors | 1024 | 31M | 128 | 352 | 1e-4 | 0 | 0 | 0 |

Copy at 128–512 is the exam that already passed, including against the averaged notebook on the lookalike at 128. From 1024 up, the only clear pass is the in-order chain, and only when each note's reach stays at 128 or 256 tokens.

## What is missing

Already answered, do not spend another sweep on it:

- More width. 50M matches 31M on lookup and lookalike.
- A longer window. Reach 512 writes nothing on the chain, the lookalike, and the lookup. Extra questions in that window do not help.
- A bigger step at 1024 or 2048. It zeros the full read.
- Shuffled hops at 1024. The full read cannot do a 2-hop shuffle either, so that zero is the exam.
- Lookup at 4096 and beyond. Everyone is at chance, including the full read.

Not answered:

1. **Where the in-order chain dies between 1024 and 2048.** At 2048 the per-note reach was the same 128 tokens that passed at 1024. What changed is the notebook the global layer must search: 352 notes became 672. A book of about 1500 tokens, same short window, tells us whether the reader is drowning in notes or the write itself loses the hops.
2. **Whether 2 hops, still in order, survive 2048.** Four hops failed while the full read passed. If two hops pass, the wall is composition depth. If two hops also fail, the wall is length.
3. **No text has been trained.** Every row above is a tiny alphabet with a planted fact. Next-word text is a different objective.

## Are we ready for multi-hop, or for text?

**Multi-hop, on this synthetic exam:** partly. Four in-order hops pass at 1024 tokens when each note sees 128–256 tokens. They fail at 2048 even though each note still sees 128. Shuffled hops are out of reach of the full read, so they are not a fair next test of the notebook. The next multi-hop test is the two questions above, not a new architecture.

**Text:** not a pretrain. A lookup of one planted fact is still only about half the answer at 1024 tokens, and the full read scores zero there. Copying a far span, which was solved at 512, is unsolved at 1024 for everyone. A million-token text run would be training past a wall we have already measured.

A short text smoke is fair only at **512 tokens**, the longest length where copy already passes for the sliding window and the full read. That smoke asks whether the notebook is a trick that works only on the tiny alphabet. It is not a license to scale data or length.
