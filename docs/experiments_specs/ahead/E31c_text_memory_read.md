# E31c — The E31 notebook on plain text (closed-window read, reader-relative slot keys)

- **Status:** draft (2026-10-03). The code changes are approved with the condition "no breaking
  changes". Text runs wait for the E31c ladder check (below) and the author's go-ahead.
- **Serves:** the Vision (compress long context into latent concepts and read them back) and the
  agenda focus "explore E31 further". The author's order (2026-10-03): first prove the notebook on
  text at about 256k context, with no two-level notebook. After that comes the compression ladder
  6 → 12 → 16 → 32 … tokens per latent, one rung per experiment.
- **Implementation plan:** [E31c_text_memory_read_plan.md](E31c_text_memory_read_plan.md)
- **Owner / dates:** ksopyla · opened 2026-10-03

## Hypothesis
If every token reads the E31 notes of the windows that have **already closed**, plus the last 256
tokens verbatim, then on long documents the notebook will carry far-back facts in ordinary
next-token prediction. The E31 writer, reader and main path stay exactly as tested on the exams.
On held-out keyed-recall rows at 16k tokens the model recovers at least half of the gap between
the no-notebook control and the dense ceiling. Trained at 16k, it keeps at least 75 % of that
recall at 256k. The reason: the exam already showed the notes hold whole facts and can be found
by content at any length (E31b, lookup to 128k). Text only changes **who may read** them and
**when**.

## Builds-on
- **Foundation:**
  - `nn/latent_memory.py` (E31 writer, unchanged);
  - `nn/perceiver_ar_lm.py` (global read and masks; two new config values);
  - the shared text trainer `scripts/train_concept_pretraining_multigpu.sh` →
    `training/concept_pretraining_factories.py` (new pass-through args);
  - `evaluation/long_context_probes.py` (passkey, multikey, keyed recall, buckets);
  - `scripts/build_retrieval_mix_dataset.py` (E18b keyed-recall rows in real text);
  - the capability suite and length battery for the no-harm check.
  No new trainer and no new model file.
- **Init / checkpoint:** random init. The point is that the text objective, not an exam, teaches the
  notebook.
- **Baseline to beat:**
  - on the exams: the champion `e31_li_m1` (suite 30M full tier, frontier L5, lookup to 128k in
    E31b);
  - on text: the two controls trained in this experiment, the same model without a notebook and the
    dense full-attention model.
  - Past text evidence that the bar is real: E22 (notebook used as a document embedding; far-context
    value 0.05 nats, passkey 0.0, recall 4.8 %) and E18b (dense control 99.3 % keyed recall vs 4.5 %
    for a one-read model).
- **Materially new:** the first E31 run where the notebook is part of ordinary next-token
  prediction:
  - every token reads it, not only the tokens after one QUERY;
  - the read is causal by window close;
  - slot keys are positioned relative to the reading token, so nothing anchors on a question
    boundary that text does not have.
  E22 had a notebook on text, but every token read a positional array and nothing forced
  content-addressed facts. E31c keeps E31's stateful, content-addressed latents and the
  far-content objective E23 called for.

## The architectural bet
Two config values. Defaults reproduce E31 bit for bit.

1. **`lm_read = "closed"`** (default `"exclusive"` = E31 as tested).
   - **The rule:** token `t` may read slot `j` iff the slot belongs to `t`'s document and its window
     has closed: the last token the window wrote from is at or before `t`.
   - **The writer is unchanged.** Its two-way attention stays inside one window, so the rule is
     causal. Raw keys keep the E31 rule: causal, same document and side, within
     `message_raw_window` = 256 tokens.
   - **No gap between the two sources.** Windows are 256 tokens at stride 192, so the newest closed
     window always ends within 192 tokens of `t`. Notes and the raw 256 tokens therefore cover the
     whole past with no gap.
   - **Text batches no longer drop the notebook.** Today a batch without a QUERY token skips it, and
     the global read silently becomes full attention. In closed mode the notebook is always on, and
     no boundary token is needed.
   - On exam rows the closed rule gives the answer tokens **exactly** the notes E31 gave them. Every
     book window closes before QUERY, and answer regions are shorter than one window. Book tokens
     now also read earlier notes, but nothing an answer reads depends on book hidden states: raw
     keys stay on their side and local layers stop at QUERY. So answer logits are unchanged. A
     test checks this on real suite rows.

2. **`lm_slot_pos = "reader"`**: slot keys carry no position, and the reading query meets them
   unrotated.
   - **What changes from E31:** E31's champion rotates every slot key at the QUERY position, so all
     notes sit "just before" the reader. Matching is by content, the same at every book length.
     Text has no QUERY. The faithful translation is to put every note at the reader's own position,
     which is RoPE-free (NoPE) matching for slots.
   - **Raw keys keep RoPE exactly.** One attention call does both, with query `[R(p)q, q]`, raw keys
     `[R(j)k, 0]`, slot keys `[0, k_s]` and the scale `1/√dh`. One softmax, exact.
   - **Cost:** the global read's query–key width doubles (64 → 128). Values are unchanged.
   - **Fallback, no new code:** `lm_slot_pos = "read"` puts slots at the true distance of what they
     read. It keeps order but risks length transfer.

What does **not** change: the writer, window 256 / stride 192, 32 latents, one reader entry per
latent (N/6 entries), the main path (local → global read → local → local, U-net skips, x0
re-injection), the E33a loop code.

**Use cases this serves (the author's picture).**
- **A long prompt is a book:** it is written in one parallel pass, and while the answer is generated
  a new page closes every 192 tokens.
- **A long chat** keeps adding pages.
- **Generation:** the training mask is exactly the causal semantics of incremental generation, so
  no retraining is needed. Incremental writing during generation (a KV cache plus a page writer) is
  engineering for later, not part of this bet.

**Arms (text, same data, budget and seeds):**

| arm | what it is | role |
|---|---|---|
| E31c notebook | E31 champion shape + closed read + reader slot keys | the bet |
| no-notebook control | same model trained with the notebook removed (`message_override none`: raw 256-token read only) | floor: is the notebook load-bearing? |
| dense ceiling | same shape, global read = full causal attention, no notebook | ceiling at the training length |
| *(later)* E31c + loop | the E33a loop on top (4 tied read-think-reread rounds, answer exits) | run only after E33a's ladder verdict; same protocol |

## Why this is not a safe retread
It is not E22 again:
- E22's array was positional, read by every token through cross-attention, with no far-content
  objective.
- E31c keeps E31's content-addressed, two-way-written latents, which were proven on exams to 128k.
- It trains on rows whose loss pays for far content (keyed recall in real text, the E23 lesson).

The analogy is a student taking notes while reading a long book. Each finished page is summarised
in the notebook. While reading the current page, the student sees it in full and can look up any
earlier note by what it says, not by its page number.

## Compatibility with the capability ladder (checked before any run)
- **Defaults unchanged:** with `lm_read = "exclusive"` and the existing slot positions the model is
  bit-identical to E31 and E33a. That is tested, and existing tests stay green. E33a's runs use
  their own branch and are not touched.
- **The closed read on exam rows:** answer logits are identical to exclusive mode. This is tested
  on rows from every suite recipe, for the E31 champion and the E33a loop config.
- **Reader slot keys on exam rows: not identical.** Slot keys lose the small QUERY-relative rotation.
  This is the one real change to exam behaviour. The ladder must measure it (below), and the bet
  fails if it costs a capability.
- **Cost on exams:** book tokens now also read the notebook. That costs more global-read compute,
  but no more for the answers. The mask's block lists cover it (tested). The length battery to
  128k stays feasible: each query reads only the closed notes, so about half the notebook on
  average.

**Verified on the branch (2026-10-03, CPU):**
- **Old code against new code, same seed and rows: bit-identical.** Logits, loss and every
  gradient match for `e31_page`, `e31_li_m1`, `e33a_loop`, `e31_mix_m1`, `e30_li`, `e21` and
  `dense`, on a lookup row and a 4-hop chain row.
- **The closed read on exam rows: answers identical.** Answer logits are equal for `e31_li_m1` and
  `e33a_loop` on lookup, lookalike, chain-1k and Glyph-fact rows, while book-token logits do
  change.
- **The precondition holds on every suite cell.** All 19 cells have exactly one QUERY per row and
  an answer region of at most 68 tokens, well under one 256-token window.
- **The rest of the test suite:** 200 related tests pass. The only failures, in the collator and
  tokenizer tests, fail the same way on the base branch.
- **Not yet verified:** the flex kernel with query–key width 128 on the RTX 3090. Odra's GPUs are
  busy with E33a, so it waits for a free GPU. It is a check of under a minute.

## Success criteria (set BEFORE running)
- **S0, no harm on the exams (gate before any text run):**
  - `e31c_m1` (the E31 champion + closed read + reader slot keys) runs the suite full tier, 30M,
    3 seeds, `--message_raw_window 256`, plus the E31b length battery;
  - it passes the no-harm rule vs `e31_li_m1` (from `docs/engineering_specs/capability_checks.md`):
    no exam or length where E31 passes drops by more than 5 points, comparing medians over seeds.
- **S1, the notebook is load-bearing on text (16k, the training length):** held-out keyed-recall
  accuracy (E18b format, first-letter accuracy)
  ≥ no-notebook control + 0.5 × (dense ceiling − no-notebook control).
  - The ceiling is measured by the dense arm in this experiment, not assumed. E18b's dense control
    reached 99.3 % on the same row format.
- **S2, far-context value in ordinary text:** on PG-19 / long-PDF held-out rows, the loss on
  positions beyond 2k tokens must improve by
  CE(no-notebook) − CE(E31c) ≥ 0.5 × (CE(no-notebook) − CE(dense)).
  - Same weights with the notebook removed at eval: Δ > 0 with a paired standard error below Δ/2.
- **S3, length transfer (weights trained at 16k):** passkey and keyed recall at 32k, 64k, 128k and
  256k keep at least 75 % of E31c's own 16k accuracy at 256k.
  - No dense ceiling exists beyond the training length; the dense arm is reported for reference and
    is expected to fall off.

## Kill criteria (set BEFORE running)
- **K0, before any text GPU time:** S0 fails on L1–L3 (lookup, lookalike, long reach). Then reader
  slot keys broke something E31 can do. Stop, and either run `lm_slot_pos = "read"` or fix the
  position scheme.
- **K1, the E22 failure again, at 50 % of the budget:** keyed recall at 16k ≤ no-notebook + 5 points
  **and** removing the notebook at eval costs < 0.01 nats on positions beyond 2k. The notebook is
  being ignored. Stop and fix the objective (more far-content rows), not the architecture.
- **K2, cost:** a training step at 16k is more than 3× slower than the no-notebook control on the
  same GPUs (measured in the smoke). The read is not affordable as built; stop and profile.

## Plan
- **Data:**
  - long documents at 16k: the E22 long-document sources (PG-19 chunked + long PDFs), at about 70 %
    of tokens;
  - E18b-format keyed-recall rows built at `--context 16384` from the same train splits, at about
    25 % of tokens (dense labels: the loss pays for far content);
  - a short fluency tier (fineweb-edu), at about 5 %.
  - Eval rows are held out; PG-19 test/validation are decontaminated.
- **Model:** the suite's 30M shape, the one E31 was tested at:
  - hidden 960, head_dim 64, 1 KV head, layers local(16) → global read → local → local;
  - token embedding 128, no n-gram tables, value embeddings on layers 0 and 1, sinks on;
  - memory: W 256 / stride 192 / 32 latents / one reader entry / `lm_addr none` / raw window 256.
  - Larger models come only after S1–S3 pass.
- **Compute:** Odra 3 × RTX 3090. Three arms × about 0.5B tokens at 16k (E22's budget). The
  estimate is about a day per arm; the smoke step measures it before launch.
- **Order:**
  1. CPU tests (this branch);
  2. a 2-minute GPU check of the flex kernel with query–key 128 / value 64;
  3. **S0 ladder** for `e31c_m1` (Odra, through the `capability-checks` skill; queued behind E33a,
     never interleaved with it);
  4. build the 16k data;
  5. smoke 30 steps per arm;
  6. three text arms;
  7. eval S1–S3.
- **Launch (text, per arm), on the shared launcher with env overrides:**
  ```
  PAR_MESSAGE_WRITE=latent_memory PAR_LM_READ=closed PAR_LM_SLOT_POS=reader PAR_LM_ADDR=none \
  PAR_LM_READER_TOKENS=1 PAR_MESSAGE_RAW_WINDOW=256 MAX_SEQ_LENGTH=16384 \
  bash scripts/train_concept_pretraining_multigpu.sh            # + the 30M shape overrides (plan)
  # no-notebook control: + PAR_MESSAGE_OVERRIDE=none ; dense ceiling: PAR_MESSAGE_WRITE unset, raw window 0
  ```
- **Ladder launch (S0):** the suite variant `e31c_m1` through `scripts/run_capability_suite.py`,
  and the battery entry `e31c` in the study plan's `BATTERY_VARIANTS`, added when the
  capability-process branch merges.
- **New foundation code:** see the plan. It amounts to:
  - two config values (`lm_read`, `lm_slot_pos = "reader"`) and a block-list update for the long
    read;
  - trainer pass-through args;
  - suite variants `e31c_m1` and `e31c_loop` (the E33a loop + the E31c read, for later);
  - a notebook-removed switch in the long-context probes.

**Not in scope (follow-ups, one at a time):**
- the two-level notebook (deferred by the author);
- the compression ladder (12, 16, 32 … tokens per latent; each its own flavour);
- an order-aware slot scheme for text, unless S3 or the chat use case shows order confusion;
- incremental generation with a page writer;
- bigger models.

## Result
*Filled in by experiment-track.*
