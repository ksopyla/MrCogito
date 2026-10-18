# E36 — a trained writer for a frozen reader

- **Status:** S0 + S1 done 2026-10-18 (Polonez): the span ladder 512 → 256 → 128 stays below the pass mark (18–27 % of the raw-text gain on books, no facts cross); the writer, not the entry format, is the limit (54 vs 781 bits per entry). The author asked (18 Oct) to train the memory (the writer) and
  keep a frozen, well-evaluated small decoder as the reader, with no extra labels; entries start at 512 tokens per
  vector ("aligned with Cramming 1568 tokens"), shrinking to 256 then 128 if a stage fails; "implement the training
  and evaluation and do a training and evaluation on Odra or Polonez".
- **Research note:** [e36_frozen_reader_writer_20261018.md](../../4_Research_Notes/e36_frozen_reader_writer_20261018.md)
  (reader choice, writer alternatives, label-free stages, evaluation suite, uncertainty).
- **Plan:** [E36_plan.md](E36_plan.md)
- **Builds on:** E35 (a frozen reader opens the channel; broadcast vectors carry nothing), E33f (fetched notes work;
  a notes-only read stores facts without labels), the multi-hop review F1–F7.
- **Owner / dates:** ksopyla · opened 2026-10-18

## The bet
A small writer (10.6M trainable layers from SmolLM2-135M, bidirectional, one mean-pooled vector per 512-token span),
trained only on public text against a **frozen SmolLM2-360M**, makes the reader use earlier text it cannot see:
the entries recover a large share of what the raw earlier text gives, with 512× fewer reader positions, and do not
hurt the reader when they come from another document.

## Stage S0 (no training): the reader
- Floor (no text) and ceiling (raw text in context) on the continuation measure and SQuAD; the Cramming probe:
  one optimised input vector and one optimised key/value entry for texts of 128 / 256 / 512 tokens.
- Readers: SmolLM2-360M (main), Qwen3-0.6B-Base (transfer, later).

## Stage S1: task-free writer
- Rows: 4,610 tokens of one long document (PG-19 train, FinePDFs). Continuation: entries of spans 0–7 → the reader
  predicts span 8 (tokens 1–511 scored); 20 % of steps rebuild a span from its own entry.
- Two arms, one change: **A** entries as reader input positions; **B** entries as a key and value in every reader
  layer.

## Success / kill (S1, from the note §6)
- **Pass:** memory recovers ≥ 50 % of the raw-text cross-entropy gain on held-out rows at 512 tokens per entry;
  another document's entries cost < 0.02 nats against no memory.
- **Stop for this span:** < 20 % on both arms → rerun S1 at 256, then 128 tokens per entry (the span ladder).
- Reported per source (books / PDFs), with bits carried per entry and SQuAD none / raw / memory / other.

## Result

### S0 — the frozen reader alone (Polonez GPU3, 17:10–17:28 UTC, 18 Oct)
Reader SmolLM2-360M; the first 256 held-out rows per source; tokens 1–511 of the target span scored.

| measure | no text | raw text in context | last 512 raw tokens only |
|---|---|---|---|
| continuation, PG-19 books (nats/token) | 2.959 | 2.748 (155 bits per span) | 2.806 |
| continuation, FinePDFs (nats/token) | 2.239 | 1.747 (363 bits per span) | 1.920 |
| SQuAD v1.1, 500 items, answer contained / exact | 8.8 % / 4.4 % | 51.6 % / 14.4 % | — |

The Cramming probe optimises one entry per text (8 texts, 800 Adam steps). For each way in it reports the share of
tokens the reader then predicts correctly, the exact prefix in tokens, and the bits the entry adds:

| text length | one input vector | one key/value entry (all 32 layers) |
|---|---|---|
| 128 tokens | 61 %, 20 tokens, 227 bits | 99.9 %, 111 tokens (7/8 exact), 547 bits |
| 256 tokens | 53 %, 15 tokens, 238 bits | 99.95 %, 223 tokens (7/8 exact), 1,044 bits |
| 512 tokens | 49 %, 12 tokens, 264 bits | 99.98 %, 448 tokens (7/8 exact), 1,994 bits |

**Reading:** an input vector carries about 250 bits whatever the length. A key/value entry carries a whole
512-token span. So the channel allows 512 tokens per entry only through the key/value way.

### S1 round 1 — 10.6M writer, 512 tokens per entry, 4,000 steps × 32 rows (≈ 0.59B row tokens)
- Arm B (key/value): GPU3, 17:28:58–19:07:27 UTC. W&B `frozen_reader_SmolLM2-360M_w3_s512m8_kv_20261018_172911`.
- Arm A (input positions): GPU1, 17:34:01–19:01:53 UTC. W&B `frozen_reader_SmolLM2-360M_w3_s512m8_input_20261018_173412`.
- Evaluation: `evaluation/frozen_reader_eval.py`. Results in `Cache/e36/<run>/eval_final.json` (Polonez ~/dev/MrCogito-e36).

| measure | none | raw | arm B memory | arm B other doc | arm A memory | arm A other doc |
|---|---|---|---|---|---|---|
| PG-19 continuation | 2.959 | 2.748 | 2.921 | 2.930 | 2.952 | 2.952 |
| FinePDFs continuation | 2.239 | 1.747 | 2.177 | 2.216 | 2.231 | 2.235 |
| trainer's 256 eval rows (89 % books) | 2.781 | 2.515 | 2.738 | 2.824 | 2.773 | 2.784 |
| share of the raw-text gain recovered (books / PDFs / trainer rows) | — | 100 % | 18 / 13 / 16 % | — | 4 / 2 / 3 % | — |
| span rebuilt from its own entry (bits per 511 tokens, books / PDFs) | — | — | 32 / 48 | — | 8 / 9 | — |
| SQuAD, passage only in memory (contains) | 8.8 % | 51.6 % | 9.8 % | 8.4 % | 9.4 % | 8.8 % |

- **Verdict against the marks:** both arms are below the 20 % stop line, so the 512 span fails. By the ladder, S1
  reruns at 256.
- **What crosses is document-specific but small.** Another document's entries are worse than none on the trainer
  rows (2.824 vs 2.781). The gain is about 4–6 bits per entry. No SQuAD facts cross.
- **The writer is the bottleneck, not the channel.** Arm B's entries let the reader rebuild about 35 bits of their
  own span; an optimised key/value entry carries about 2,000 bits for the same span (S0), roughly 50× more.
  Arm B's held-out loss was still falling at step 4,000 (2.762 at step 500, 2.738 at 4,000).
- **Correction:** at 17:53 UTC I reported arm B "recovering ~93 %". I had compared the trainer's eval loss (a
  shuffled sample over both sources) with S0's baseline on different rows. On the same rows the share is 16 %.

### S1 round 2 — key/value way only, 4,000 steps each (Polonez, 18 Oct)
Two runs, one change each from round 1's arm B:
- Span ladder, 256 tokens per entry (16 entries). GPU1, 19:12:01–20:15:28 UTC; W&B
  `frozen_reader_SmolLM2-360M_w3_s256m16_kv_20261018_191213`. The target span is 256 tokens here, so its
  none / raw baselines differ from the 512 rows.
- Writer ladder, 8 layers = 28.3M at 512. GPU3, 19:12:01–21:17:42 UTC; W&B
  `frozen_reader_SmolLM2-360M_w8_s512m8_kv_20261018_191213`.

| run | books: none / raw / memory / other | PDFs: none / raw / memory / other | recovered (books / PDFs) | span rebuilt (bits, books / PDFs) | SQuAD from memory |
|---|---|---|---|---|---|
| 512, 3 layers (round 1) | 2.959 / 2.748 / 2.921 / 2.930 | 2.239 / 1.747 / 2.177 / 2.216 | 18 / 13 % | 32 / 48 | 9.8 % |
| **256, 3 layers** | 3.074 / 2.745 / 3.000 / 3.027 | 2.455 / 1.749 / 2.350 / 2.435 | 22 / 15 % | 35 / 42 | 9.0 % |
| **512, 8 layers** | 2.959 / 2.748 / 2.920 / 2.932 | 2.239 / 1.747 / 2.175 / 2.219 | 18 / 13 % | 31 / 48 | 9.0 % |

- **Neither change moves the result.** Shorter spans and a writer 2.7× larger leave the share recovered within
  4 points. Each entry carries ~35–45 bits of its own span at every setting. The total useful information about the
  next span is ~28 bits in all three runs (8 × 3.5 at 512; 16 × 1.7 at 256).
- **The cap sits in how the writer is trained or pooled, not in its size or the span length.** The reader takes
  ~2,000 bits through one key/value entry (S0).
- 256 is still below the pass mark (50 %) and below the stop line on PDFs. The 128 rung and a capacity diagnostic
  follow after the cooling pause (Polonez loaded 16:01–21:17 UTC; pause 21:18–21:38).

### S1 round 3 — the last rung, 128 tokens per entry (32 entries), key/value way
Polonez GPU1, 21:38:11–22:29:08 UTC. W&B `frozen_reader_SmolLM2-360M_w3_s128m32_kv_20261018_213822`.

| corpus | none | raw | memory | other doc | recovered | span rebuilt from its own entry |
|---|---|---|---|---|---|---|
| books | 3.245 | 2.739 | 3.109 | 3.163 | 27 % | 36 bits |
| PDFs | 2.732 | 1.777 | 2.562 | 2.723 | 18 % | 36 bits |

SQuAD, passage only in memory: 8.6 % (none 8.8 %, with the passage 51.6 %).

**The span ladder is done: 512 → 256 → 128 recovers 18 → 22 → 27 % (books) and 13 → 15 → 18 % (PDFs).**
- Every rung is below the 50 % pass mark, and no rung moves SQuAD.
- Each entry carries ~36 bits of its own span at every span length. The useful information about the next span
  stays ~25–31 bits in total. The share recovered rises only because a shorter target span has less to gain.
- By the plan, E36's S1 stops here. The capacity diagnostic below shows the format would allow more, so the next
  step is a better writer, not a shorter span.

### Capacity diagnostic (Polonez GPU3, 21:38–21:44 UTC)
8 held-out texts of 512 tokens (4 books, 4 PDFs). The table gives the bits one entry adds about its own span when
the reader rebuilds it (`Cache/e36/subspace_probe.json`):

| entry | bits |
|---|---|
| the trained writer's entry (round 1, key/value way) | 54 |
| optimised in the writer's 576-number output space, through the trained expander (800 Adam steps) | 781 |
| the same through a fresh, untrained expander | 290 |
| a free key/value entry, 20,480 numbers (S0) | ~2,000 |

**Reading:** the entry format and the trained expander can hold ~14× what the writer puts in. The gap is the
writer's: its encoder and mean pooling, and an objective in which continuation on books pays for little. The next
fixes target the writer:
- attention pooling instead of a mean over 512 tokens;
- reconstruction-first training at growing spans (REFRAG's curriculum, with the reader frozen);
- then teacher matching on passages with questions (S2).
