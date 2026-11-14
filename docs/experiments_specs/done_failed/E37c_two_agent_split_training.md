# E37c — Two-agent split training: the answering side starts cold and reads the document only through the memory

- **Status:** done 2026-11-14 — main pass mark missed (run 06:30–11:45 UTC, one L40S, $9.06). Approved to spec and
  run by the author on 2026-11-14 06:10 UTC ("Yes, spec it", answering "Shall I spec and run the short split
  fine-tune (about $15)?"). Spec written 2026-11-14 06:30 UTC by the agent-to-agent session.
- **Serves:** the Vision's last step (agents exchange latent memory instead of text) and E35's goal, on the E37
  100M models. A flavour of E37: one mask added in training, same model, same parameters. ID checked on every remote
  branch, local branch and worktree on 2026-11-14 (E37, E37a, e37b used; the SFT stage is a run of E37, not an ID).
- **Implementation plan:** this spec's "Plan" section (one config switch and one job; no separate plan file).
- **Owner / dates:** ksopyla · opened 2026-11-14 · closed —
- **Alignment page:** `docs/3_Evaluations_and_Baselines/e37c_architecture.html`

## Why now (measured 2026-11-13, eval only, no training)
One trained 100M model was run as two agents (`nn/agent_split.py`): agent A reads the document and keeps the reading
layer's memory (one key and one value per token and loop pass); agent B gets only the question, and its only view of
the document is the reading layer's read of that memory. B's local (sliding-window) layers see B's own tokens only.

Concept-read model (E37a, 3.0B + 0.4B tokens), RULER needle with a number in a filler haystack, % found:

| length | docs | one agent | B with memory only | + A's last 16 tokens' local caches | no memory / another document's |
|---|---|---|---|---|---|
| 4k | 30 | 100 | 63 | 100 | 0 / 0 |
| 16k | 30 | 97 | 43 | 93 | 0 / 0 |
| 32k | 20 | 90 | 35 | 60 | 0 / 0 |
| 64k | 12 / 20 | 83 / 100 | 17 / 40 | 58 (last 32: 95, last 64: 100) | 0 / 0 |

- Handing over A's whole state reproduces one agent (23 of 24 cells; float32 exact), so the split code is right.
- The memory holds the answer (the same memory plus A's last tokens' local caches gives 93–100 %).
- What fails is B's **cold start**: in training, a token whose local window is empty was always the first token of a
  document, where the memory is empty too. A token with an empty local window and a full memory was never seen.
- Why A's last tokens help so much is not established (B's own 148-token preamble does not help: 90 / 40 / 20 / 8 %).

## Hypothesis
If rows are cut at random points during training, and the tokens after a cut may use their local layers only on
their own side of the cut while the reading layer still reads the whole document, the model learns to build its
reads from a cold start. Then agent B, given only A's memory, finds the needle about as often as one agent does
(≥ 80 % of the gap closed at 16k and 32k), at no cost to the one-agent model.

Falsified if the memory-only score at 16k stays below 61 % (less than a third of the gap closed) after the budget.

## Builds-on
- **Foundation:** `PerceiverARLM` (`nn/perceiver_ar_lm.py`) with `par_global_read = "concept"`; the trainer
  `training/train_concept_pretraining.py` through `scripts/run_e33n.py` (new phase `split`); the two-agent decoder
  `nn/agent_split.py` and its probe `scripts/a2a_split_probe.py` (unchanged, the exam).
- **Init / checkpoint:** E37a at 100M, the 16k stage's final weights
  (`runs/e37/e33n_m100_e37a_tok3.0e9_seed42/decay/…/final`, 162.4M parameters; training loss ~2.25 at its end).
  Fresh optimizer. No new parameters.
- **Baseline to beat:** the same checkpoint untuned, table above: memory only 63 / 43 / 35 % at 4k / 16k / 32k.
- **Ceiling (measured):** the same checkpoint as one agent: 100 / 97 / 90 %.
- **Materially new:** E33f made the question the boundary on synthetic question rows of the notebook model (the
  wire was the page notes). E37c cuts plain long documents at random points for the concept-memory model, whose wire
  is the reading layer's per-token memory. No question shape is taught, so the needle tests stay unseen prompts.

## The architectural bet
One training row with a cut **is** the two-agent setting. Attention is causal, so the tokens before a cut are computed
exactly as agent A alone computes them. The tokens after the cut are agent B:

```
tokens      t0 … t(c-1) | t(c) … t(e-1) | t(e) …
side             0      |      1        |   2
local layers  own side only (window 128, clipped at the cut)          ← changed in training
reading layer the whole document so far: 4 sinks, own 512-block and the block before exactly,
              512 opened tokens of the best concepts, one summary per other concept   ← unchanged
positions     continue through the cuts (RoPE in local layers; none in the reading layer) ← unchanged
loss          next token at every position                                               ← unchanged
```

- **Cuts.** Every row; the gap between cuts is log-uniform in [32, 4,096] tokens (mean 838): about 20 cuts per
  16k row, about 15 % of tokens within 128 tokens after a cut. Short gaps look like B's question, long gaps like A's
  document. A chain: side k reads sides 0…k−1 through the reading layer.
- **Switches** (`PerceiverARConfig`, off by default): `par_cut_gap_min`, `par_cut_gap_max`, `par_cut_prob` (1.0).
  Cuts are drawn in the forward in training mode only; an explicit `cut_ids` [B, S] bool can be passed for tests and
  exams. Eval and every existing run are unchanged (no cuts unless asked).
- **Analogy.** A shift hand-over: the incoming nurse did not see the last hour, only the chart. Today's model was
  only ever a nurse who stayed the whole shift; the fine-tune makes it start mid-shift from the chart, many times.

## Why this is not a safe retread
It changes what is trained (who may see what), not an optimiser knob, and tests a claim the eval-only probe could
not: whether the cold start is learnable or A's recent raw states are needed on the wire. Both outcomes decide the
shape of the demo (memory only vs memory + A's last tokens).

## Success criteria (set BEFORE running)
Exam: `scripts/a2a_split_probe.py`, same settings and documents as 2026-11-13 (30 documents at 4k and 16k, 20 at 32k
and 64k, 12 answer tokens), on the tuned checkpoint. Number haystack (`niah_single_1`):
- **S1 (main):** B with memory only ≥ **93 % at 4k, 86 % at 16k, 79 % at 32k** — 80 % of the gap from the untuned
  memory-only score to the untuned one-agent score (ceiling measured: 100 / 97 / 90).
- **S2 (controls):** no memory and another document's memory ≤ 5 % in every cell (untuned: 0 %).
- **S3 (no harm):** the tuned model as one agent ≥ untuned − 7 points per cell (2 documents of 30), and the
  long-document loss at 16k without cuts (8 held-out rows, `evaluation/long_doc_loss.py`) ≤ untuned + 0.01.
- **Reported, no pass mark:** 64k (trained at 16k only); the essay haystack (`niah_single_2`; one agent itself scores
  63 / 10 / 10 %); memory on 8 and 4 bits; A's last 16 tokens; the cut loss by distance (below).

## Kill criteria (set BEFORE running)
- **K1:** memory only at 16k < 61 % after 0.2B tokens → a short fine-tune does not teach the cold start at this
  rate; the hand-over keeps A's last 32–64 tokens' local caches (0.1–0.2 MB, already 95–100 % with no training).
- **K2 (in-run, a bug not a result):** mean training loss of steps 40–60 above **2.39** → stop the node.
  Reference corrected on 2026-11-14 06:41 UTC, after the pre-flight and before any training loss was logged: the rule
  first compared training rows with the pre-flight's held-out rows, which differ in level (plain loss 2.19 on the
  40 held-out rows, 2.26–2.28 over the 16k stage's last 50–100 training steps). It is now the 16k stage's last
  training loss (2.27) + the measured cost of training-style cuts on the untuned model (+0.07: 2.187 → 2.258) + 0.05.
- **K3:** S3 fails by more than twice its margin → the result is not usable as a demo model whatever S1 says.

## Measures
- **Cut loss by distance** (`scripts/a2a_cut_loss.py`, new, small): held-out 16k rows, a cut every 1,024 tokens;
  extra loss of the tokens 0–7, 8–31, 32–127 and 128+ after a cut against the same tokens without cuts; untuned and
  tuned. Thousands of tokens per bin, so it reads small changes the 30-document exam cannot. The first tokens after
  a cut cannot be fully recovered (the reading layer has no position code, so word order just before the cut is
  not addressable); the 32–127 bin is the one expected to close.

## Plan
- **Data:** LongDocs-lite, rows of 16,384, one document per row (the 16k stage's data; manifest seed 3042, another
  order than the 16k stage's 1042). 0.2B tokens = 12,207 rows.
- **Recipe:** the 64k stage's shape at 16k: from the 16k stage's final weights, fresh optimizer, 32 rows a step
  (micro 1 × accumulation 32), 381 steps, 6.4e-5 (the rate the 16k stage ended on) → 6.4e-6, 3 % warmup, 1−sqrt
  decay, bf16, flex attention. One seed (1).
- **Compute:** one L40S (g6e.xlarge, spot). Measured speed of this model at 16k: 13.3k tokens/s per L40S (0.4B in
  2 h 05 min on 4 cards, 10 Nov) → about 4.2 h of training + about 1.5 h of exams ≈ 6 h ≈ $11 at the on-demand
  price ($1.86/h; spot is lower). Hard limit 9 h; checkpoints to S3 every 50 steps, a relaunch resumes.
- **Launch:**
  `MC_SPOT=1 bash infra/aws/launch.sh e37c-split 9 "infra/aws/jobs/e37c_split_ft.sh e37a" claude/agent-latent-communication-0479aa g6e.xlarge`
  The job: pull the checkpoint (read only), data and tokenizer → cut loss of the untuned model → training
  (`scripts/run_e33n.py cmd --arm e37a --size m100 --phase split`) → cut loss, the two-agent exam (4k, 16k, 32k, then
  64k), the long-document loss → power off. Results: `s3://BUCKET/runs/e37c/e37c_m100_e37a_seed1/` (its own folder;
  the shared E37 run folder is only read).
- **New foundation code:** the cut switches in `PerceiverARLM._run_layers` (local layers get a side id, layers with
  the full pattern keep the document id), reusable by any arm without the message boundary (dense, LSA port, concept
  read); `ModelArguments` + factory pass-through; phase `split` in `scripts/run_e33n.py`; the job; the cut-loss
  script; unit tests (a cut row equals the two-agent decoder; no cut = unchanged outputs).

## Not in scope (follow-ups, only after S1)
- The same fine-tune with cuts off (same tokens): attributes the gain to the cuts rather than to more 16k training.
- A second seed, and the LSA port and dense arms.
- Question-shaped rows, a 64k split stage, coarser memory on the wire, remote read as a service.

## Result
- Run id: `perceiver_ar_perceiver_H896L3g1s7N128_20261114_064152_e37a_m100_split` (AWS node `e37c-split`, one L40S,
  training 2026-11-14 06:41 → 10:37 UTC, exams until 11:43 UTC, $9.06)
- WandB: offline run in `s3://BUCKET/runs/e37c/e37c_m100_e37a_seed1/wandb/` (not synced)
- Run report: `docs/2_Experiments_Registry/run_reports/e37c_two_agent_split_training_20261114.md`
- Result files: `docs/2_Experiments_Registry/results/evaluations/E37c_split_ft/`
- Verdict: **mixed, main pass mark missed** — B with memory only 63 / 70 / 60 % at 4k / 16k / 32k (marks 93 / 86 / 79;
  untuned 63 / 43 / 35 on other documents); K1 not hit; controls 0 %; the cut loss fell by about three quarters at
  32–127 tokens; loss on plain long text unchanged or better; but the tuned model as one agent finds 84 of 100
  needles against 97 of 100 (a new habit: a blank line after the colon), which fails S3 and trips K3.
- Protocol flaw found afterwards: the two exams drew different documents (the generator was not seeded; `--seed`
  added to `scripts/a2a_split_probe.py` on 2026-11-14), so untuned and tuned are two independent samples.
