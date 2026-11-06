# E33n stage 1 — E33 notebook on natural long text at 30M (LSA testbed), four arms

**Date:** 2026-11-05 (training 2026-11-05 00:33 – 19:52 UTC; long-document and copy-split evals 2026-11-05 19:00 – 21:30 UTC)
**Machine:** Odra (window control, full note read, top-k note read; one RTX 3090 each) · Polonez (dense ceiling, one RTX 3090, 5.5 h bursts with 20 min cooldowns)
**Run ID:** trunk / decay per arm — dense `perceiver_ar_perceiver_H576L2g1s5N128_20261105_003327` (+ resumes `…_062344`, `…_121401`) / `…_123824` · window `…_013323` / `…_155729` · e33 (full read) `…_013322` / `…_163042` · e33n (top-k) `…_013323` / `…_182321`
**WandB (training):** https://wandb.ai/ksopyla/MrCogito — the window and e33n trunks share the id `…_20261105_013323` (both started the same second), so their W&B curves are mixed; local logs are not (fixed for later runs: `RUN_ID_SUFFIX`)
**Raw log:** `trunk.log` / `decay.log` / `events.log` in each arm's run folder on its server (not archived)
**Best checkpoint:** each arm's `decay/<run>/final` on its server (not on the NAS)
**Git commit:** `fed2963` (training launch) · evals `8c0beb6`–`686b7dc` on `e33n-lsa-testbed`
**Git tag:** —
**Related TODO:** E33n spec stage 1 (K1, K2)

---

## Goal
First test of the E33 notebook (32 notes per closed 256-token page, a reading layer over the last 512 raw tokens plus
the notes) on natural long text with a published protocol (LSA, arXiv 2610.42147): FineWeb-Edu trunk at 4k, then a
long-document decay on LongDocs-lite at 16k. Four arms of one 30M model, identical except for the reading layer: a
window-only control, a dense ceiling, the full note read (E33) and the top-64 note read (E33n). Stage gate: K1 (both
notebook arms recover < 15 % of the window − dense gap → stop them) and K2 (top-k > 0.02 nats worse than the full read).

## Configuration
| Item | Value |
|---|---|
| Family | perceiver_ar, width 576, 2 pre + reading layer + 5 stack layers, E33 loop 4 rounds, FFN 1472, GPT-2 vocabulary |
| Parameters | 30.2M non-embedding (dense 26.3M), 72.7M total |
| Writer (notebook arms and window) | E31c: 32 notes / 256-token page, stride 192, closed read; window arm runs it with notes masked |
| Data | trunk: FineWeb-Edu GPT-2 shards, 4k rows, 1.5B tokens; decay: LongDocs-lite (gutenberg, arxiv, doab, usgpo, hansard), one document per 16k row, 0.2B tokens |
| Schedule | trunk constant LR 1e-3 after 1 % warmup; decay 3 % warmup, 1 − sqrt to 0.1 × over 97 %, fresh optimizer |
| Batch | 0.26M tokens per step (micro 4 × 4k trunk, 1 × 16k decay, accumulated) |
| Throughput (3090, tokens/s) | trunk: dense 35k, window 27.8k, e33 26.7k, e33n 24.9k |
| Compute | GPU-hours from the run events: dense 13.1, window 16.4, e33 17.1, e33n 19.3 (≈ 66 RTX 3090 hours, no cash cost). The W&B compute audit refuses these runs (`gpu_count_mismatch`: one GPU used on a multi-GPU server) |

## Training Outcome
All four arms trained without loss spikes or restarts beyond the planned Polonez bursts (resume from checkpoint
verified on the dense arm). Trunk training losses at the end: 3.38–3.41 for the three writer arms, the dense arm
lowest (trunk validation 3.434). The top-k arm tracked the full read within ~0.02 throughout.

## Concept Health
Notebook value inside each notebook model (16k, positions ≥ 2k, 64 rows per source): switching the notes off at eval
costs the full-read model **+0.093** nats (2.700 vs 2.607) and the top-k model **+0.040** (2.646 vs 2.606). Both
notes-off losses are worse than the window control's (2.615): the models lean on the notebook (the full-read model
more), yet gain little over the window control.

## Evaluation
Files: `docs/2_Experiments_Registry/results/evaluations/E33n_s30/`.

**1. Held-out long-document loss, 16k, positions 2k–16k** (64 rows per source, 156 documents, pooled; same rows for
every arm; window from the copy-split eval, see Notes):

| arm | loss | share of G (95 % CI, bootstrap over documents) |
|---|---|---|
| dense ceiling | 2.4811 | — |
| full note read (E33) | 2.6073 | **5.5 %** (3.9–7.5) |
| top-k note read (E33n) | 2.6064 | **6.1 %** (4.7–7.8) |
| window control | 2.6146 | — |

G = window − dense = **0.1335** nats/token (0.099–0.176). Top-k − full read = −0.0008 (−0.0025 to +0.0009).
Per source G: doab 0.334, arxiv 0.179, usgpo 0.117, gutenberg 0.028, hansard 0.010.

**2. Copy split** (`evaluation/copy_split.py`, same rows): each scored token is *far* if its 4-gram (3 tokens of
context + itself) occurred earlier only beyond the 512-token window, *near* if inside it, *none* otherwise.

| group | tokens | window − dense | part of G | share closed by e33 / e33n |
|---|---|---|---|---|
| far copy | 9.6 % | +1.361 | **97 %** | **1.6 % / 1.8 %** |
| near copy | 17.7 % | −0.009 | −1 % | — |
| no copy | 72.7 % | +0.007 | 4 % | 94 % / 106 % |

With an 8-gram test (7 tokens of context): far copy 5.2 % of tokens, 63 % of G, closed 1.0–1.2 %; the rest of G
(38 %) sits on "no copy" tokens that still contain shorter repeats, closed 12–14 %.

**3. RULER (13 tasks, 100 documents, string match), 4k only:** dense 0.061 · top-k 0.041 · window 0.030 · full read
0.020 — all near the floor at this size. Lengths above 4k were not scored by any run (lm-eval's `limit` takes the
first documents of a dataset that holds every length, first length first) — fixed for later runs (`b143938`).

**4. Core benchmarks** (acc_norm where defined; WikiText word perplexity, lower is better):

| arm | HellaSwag | ARC-e | ARC-c | PIQA | WinoGrande | SciQ | LAMBADA | WikiText |
|---|---|---|---|---|---|---|---|---|
| dense (SDPA) | 0.278 | 0.394 | 0.235 | 0.580 | 0.506 | 0.588 | 0.140 | 71.5 |
| window (flex) | 0.279 | 0.386 | 0.259 | 0.581 | 0.521 | 0.567 | 0.159 | 72.8 |
| full read (flex) | 0.280 | 0.394 | 0.249 | 0.569 | 0.500 | 0.599 | 0.171 | 72.6 |

Ties within noise (these tasks never close a notebook page). Top-k arm rerunning (its pipeline used batch 32 and ran
out of memory).

**5. Length transfer** (positions ≥ 2k; window from the fixed eval): at 64k (64 rows, 61 documents) G = 0.102
(0.071–0.148) and the shares are 3.7 % (1.6–6.4) full read, 4.5 % (2.9–6.5) top-k; with 8 rows: 4k 4.6 / 3.4 %,
32k 4.9 / 5.5 %, 128k −2.6 / 1.1 % (G 0.081, 4 sources). The notebook's share does not grow with length.

## Interpretation
K1 is met: both notebook arms recover about 6 % of the far-context gap at the training length, far below the 15 % stop
line (and the 40 % success bar). K2 is not met and S2 is: the top-k read costs nothing against the full read. The copy
split explains the failure: at 30M, nearly all the value of far context on natural long documents is **exact copying**
of earlier text (a tenth of the tokens, where the window control is 1.36 nats worse than the dense model). A note
averages about 8 tokens' values, so it cannot return the exact next token; the notebook recovers about 2 % of that
part. On tokens that are not copies the far-context gap is tiny (0.007 nats) and the notebook recovers nearly all of
it. The capacity to copy exists at this size — the 30M dense model does it — so the missing piece is access to exact
past tokens, not model size.

## Decision
Stop the notebook arms (K1); no 100M AWS stage for them (the author put it on hold on 2026-11-05 before K1 was
computed correctly). Next: a note-indexed exact read — notes as the index (keys tied to the reading layer's own raw
keys), the reading layer opens the raw keys and values of the tokens behind its best notes, LSA-style — tested at 30M
on the home servers with the same protocol and the copy split. AWS: nothing launched (no on-demand H100 in us-east-1
since 2026-11-04 21:50 UTC); spend $0.17 (data preparation).

## Notes
- **Eval bug (fixed, `8c0beb6`):** `long_doc_loss.py` reset the notebook switch to "on" after each source, so the
  window control (trained with notes off) was scored reading notes for 4 of 5 sources and at every later length. It
  inflated G (0.153 instead of 0.134) and the notebook shares (17 % instead of 6 %, and 37–49 % instead of ≤ 5 % at
  64k–128k) reported in chat at 19:01 UTC. The copy-split eval never had the bug; the window's long-document files in
  the registry are from the fixed rerun (2026-11-05 22:45 – 23:10 UTC, Odra).
- **RULER HotpotQA:** lm-eval downloads it from `curtis.ml.cmu.edu`, which no longer resolves; served from
  `hotpotqa/hotpot_qa` (same 7,405 questions, same order).
- **Core benchmarks:** batch 32 ran out of memory on 24 GB; flex attention's compiler then failed on their ragged
  batches (ptxas segfault, PassManager failure); SDPA works.
- **Polonez bit flip:** at about 14:24 UTC one bit of a freshly built LAMBADA cache file flipped (0x74 → 0xf4) and a
  tokenizer thread crashed in the same minute; six rebuilds were correct; the dense checkpoint re-evaluated
  bit-identically (118 bucket losses). A one-off, but Polonez is not ECC.

*Related: `master_experiment_log.md`, `docs/experiments_specs/done_failed/E33n_lsa_testbed_topk_notes.md`, `agenda.md`*
