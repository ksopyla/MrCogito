# E36 — a trained writer for a frozen reader

- **Status:** S0 + S1 started 2026-10-18 (Polonez). The author asked (18 Oct) to train the memory (the writer) and
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
_Pending._
